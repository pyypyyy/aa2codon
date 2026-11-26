"""PyTorch AA-to-codon encoder-decoder. Independent of legacy TensorFlow code."""
from dataclasses import dataclass, asdict
from itertools import product
import math
import torch
from torch import nn
from torch.nn import functional as F
from Bio.Data import CodonTable

PAD, BOS, EOS = 0, 1, 2
AA = ["<pad>", "<bos>", "<eos>"] + list("ACDEFGHIKLMNPQRSTVWY*")
CODONS = ["<pad>", "<bos>", "<eos>"] + ["".join(c) for c in product("ACGT", repeat=3)]
AA_ID = {s:i for i,s in enumerate(AA)}
CODON_ID = {s:i for i,s in enumerate(CODONS)}
TABLE = CodonTable.unambiguous_dna_by_id[11]
CODE = dict(TABLE.forward_table, **{c:"*" for c in TABLE.stop_codons})
ALLOWED = torch.zeros(len(AA), len(CODONS), dtype=torch.bool)
for i in range(3):
    ALLOWED[i, i] = True
for codon, aa in CODE.items():
    ALLOWED[AA_ID[aa], CODON_ID[codon]] = True

def encode_protein(protein):
    protein = "".join(protein.split()).upper()
    if not protein or any(a not in AA_ID or len(a)!=1 for a in protein) or "*" in protein[:-1]:
        raise ValueError("Provide standard amino acids, with optional terminal *.")
    return [BOS] + [AA_ID[a] for a in protein] + [EOS]

def encode_cds(dna):
    dna = "".join(dna.split()).upper().replace("U", "T")
    if len(dna)%3 or not dna:
        raise ValueError("CDS must be nonempty and divisible by three.")
    codons = [dna[i:i+3] for i in range(0,len(dna),3)]
    if any(c not in CODE for c in codons):
        raise ValueError("Ambiguous codon.")
    protein = "".join(CODE[c] for c in codons)
    return encode_protein(protein), [BOS]+[CODON_ID[c] for c in codons]+[EOS], protein

class XIELU(nn.Module):
    """xIELU, arXiv:2411.13010; two learned scalars per FFN."""
    def __init__(self):
        super().__init__()
        self.raw_p = nn.Parameter(torch.tensor(math.log(math.expm1(.8))))
        self.raw_n = nn.Parameter(torch.tensor(math.log(math.expm1(.3))))
    def coefficients(self):
        return F.softplus(self.raw_p), .5+F.softplus(self.raw_n)
    def forward(self, x):
        # Float32 evaluation is useful in mixed precision; clamp inactive
        # exponential branch to prevent overflow before torch.where.
        dtype = x.dtype
        z = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        p,n = self.coefficients()
        neg = z.clamp(max=0)
        return torch.where(z>0, p*z.square()+.5*z,
                           n*(torch.expm1(neg)-neg)+.5*neg).to(dtype)

@dataclass
class Config:
    d_model: int = 256
    heads: int = 8
    layers: int = 3
    d_ff: int = 1024
    dropout: float = .05
    max_length: int = 1024
    activation: str = "xielu"
    synonym_mask: bool = True
    architecture_version: int = 3

class FFN(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.up=nn.Linear(c.d_model,c.d_ff)
        self.down=nn.Linear(c.d_ff,c.d_model)
        self.drop=nn.Dropout(c.dropout)
        self.gate=nn.Linear(c.d_model,c.d_ff) if c.activation=="swiglu" else None
        if c.activation not in ("xielu","relu","swiglu"):
            raise ValueError("Unknown activation")
        self.act=XIELU() if c.activation=="xielu" else nn.ReLU()
    def forward(self,x):
        z=self.up(x)
        z=z*F.silu(self.gate(x)) if self.gate is not None else self.act(z)
        return self.drop(self.down(z))

class Rotary(nn.Module):
    """Cached RoPE; query positions can be offset for aligned cross-attention."""
    def __init__(self,head_dim,max_length):
        super().__init__()
        if head_dim%2: raise ValueError("RoPE requires even head dimension")
        phase=torch.outer(torch.arange(max_length+2,dtype=torch.float32),
            torch.exp(-math.log(10000)*torch.arange(0,head_dim,2,dtype=torch.float32)/head_dim))
        self.register_buffer("cos",phase.cos(),persistent=False)
        self.register_buffer("sin",phase.sin(),persistent=False)
    def forward(self,x,offset=0):
        c=self.cos[offset:offset+x.shape[-2]].to(x.dtype)
        s=self.sin[offset:offset+x.shape[-2]].to(x.dtype)
        a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)

class Attention(nn.Module):
    """QK-RMSNorm, rotary positions, SDPA and query-dependent head gates."""
    def __init__(self,c,cross=False):
        super().__init__()
        self.heads=c.heads
        self.dim=c.d_model//c.heads
        self.cross=cross
        self.dropout=c.dropout
        self.q=nn.Linear(c.d_model,c.d_model,bias=False)
        self.k=nn.Linear(c.d_model,c.d_model,bias=False)
        self.v=nn.Linear(c.d_model,c.d_model,bias=False)
        self.out=nn.Linear(c.d_model,c.d_model,bias=False)
        self.qnorm=nn.RMSNorm(self.dim,eps=1e-6)
        self.knorm=nn.RMSNorm(self.dim,eps=1e-6)
        self.rope=Rotary(self.dim,c.max_length)
        self.gate=nn.Linear(c.d_model,c.heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,1.)
    def heads_view(self,x):
        return x.reshape(x.shape[0],x.shape[1],self.heads,self.dim).transpose(1,2)
    def forward(self,x,context,pad,causal=False,attention=False):
        q=self.rope(self.qnorm(self.heads_view(self.q(x))),int(self.cross))
        k=self.rope(self.knorm(self.heads_view(self.k(context))))
        v=self.heads_view(self.v(context))
        q,k=q.to(v.dtype),k.to(v.dtype)
        # SDPA bool masks use True for allowed keys, unlike MultiheadAttention.
        mask=~pad[:,None,None,:]
        if causal:
            mask=mask & torch.ones(q.shape[-2],k.shape[-2],dtype=torch.bool,device=q.device).tril()
        if attention:
            # Explicit scores are needed only by the existing analysis interface.
            weights=(q.float() @ k.float().transpose(-1,-2) / math.sqrt(self.dim))
            weights=weights.masked_fill(~mask,float("-inf")).softmax(-1)
            out=F.dropout(weights,p=self.dropout,training=self.training).to(v.dtype) @ v
        else:
            weights=None
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.)
        gate=self.gate(x).sigmoid().transpose(1,2).unsqueeze(-1).to(out.dtype)
        out=(out*gate).transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
        return self.out(out),weights

class Block(nn.Module):
    def __init__(self,c,decoder=False):
        super().__init__()
        self.self_attn=Attention(c)
        self.cross=Attention(c,cross=True) if decoder else None
        self.n1=nn.RMSNorm(c.d_model,eps=1e-6)
        self.n2=nn.RMSNorm(c.d_model,eps=1e-6)
        self.n3=nn.RMSNorm(c.d_model,eps=1e-6) if decoder else None
        self.ff=FFN(c)
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x,pad,context=None,context_pad=None,causal=False,attention=False):
        z=self.n1(x)
        out,_=self.self_attn(z,z,pad,causal=causal)
        x=x+self.drop(out)
        weights=None
        if self.cross is not None:
            out,weights=self.cross(self.n3(x),context,context_pad,attention=attention)
            x=x+self.drop(out)
        return x+self.ff(self.n2(x)),weights

class Model(nn.Module):
    def __init__(self,c=Config()):
        super().__init__()
        if c.architecture_version!=3 or c.d_model%(2*c.heads):
            raise ValueError("Expected v3 architecture and even head dimension")
        self.config=c
        self.aa_embed=nn.Embedding(len(AA),c.d_model,padding_idx=PAD)
        self.codon_embed=nn.Embedding(len(CODONS),c.d_model,padding_idx=PAD)
        self.encoder=nn.ModuleList(Block(c) for _ in range(c.layers))
        self.decoder=nn.ModuleList(Block(c,True) for _ in range(c.layers))
        self.output=nn.Linear(c.d_model,len(CODONS))
        self.enc_norm=nn.RMSNorm(c.d_model,eps=1e-6)
        self.dec_norm=nn.RMSNorm(c.d_model,eps=1e-6)
        self.align=nn.Linear(c.d_model,c.d_model,bias=False)
        self.align_gate=nn.Parameter(torch.tensor(-1.))
        self.drop=nn.Dropout(c.dropout)
        for emb in (self.aa_embed,self.codon_embed):
            nn.init.normal_(emb.weight,std=c.d_model**-.5)
            with torch.no_grad(): emb.weight[PAD].zero_()
        self.register_buffer("allowed",ALLOWED.clone())
    def embed(self,ids,layer):
        return layer(ids)*math.sqrt(self.config.d_model)
    def with_pos(self,x):
        if x.shape[1]>self.config.max_length+2:
            raise ValueError("Sequence exceeds configured context length")
        return self.drop(x)
    def encode(self,aa,emb=None):
        x=self.with_pos(self.embed(aa,self.aa_embed) if emb is None else emb)
        for block in self.encoder:
            x,_=block(x,aa.eq(PAD))
        return self.enc_norm(x)
    def decode(self,aa,prefix,context,emb=None,attention=False):
        x=self.with_pos(self.embed(prefix,self.codon_embed) if emb is None else emb)
        # Shift by one: BOS predicts the first amino acid, not BOS itself.
        aligned=context[:,1:prefix.shape[1]+1]
        x=x+self.align_gate.sigmoid()*self.align(aligned)
        weights=[]
        for block in self.decoder:
            x,w=block(x,prefix.eq(PAD),context,aa.eq(PAD),True,attention)
            if attention: weights.append(w)
        logits=self.output(self.dec_norm(x))
        if self.config.synonym_mask:
            target=aa[:,1:prefix.shape[1]+1]
            logits=logits.masked_fill(~self.allowed[target],float("-inf"))
        return logits,weights
    def forward(self,aa,prefix,aa_emb=None,codon_emb=None,attention=False):
        context=self.encode(aa,aa_emb)
        return self.decode(aa,prefix,context,codon_emb,attention)
    def activation_parameters(self):
        return {n: [float(v.detach().cpu()) for v in m.coefficients()]
                for n,m in self.named_modules() if isinstance(m,XIELU)}
    def save(self,path,**metadata):
        torch.save({"config":asdict(self.config),"state":self.state_dict(),
                    "aa_vocab":AA,"codon_vocab":CODONS,"metadata":metadata},path)

def load_model(path,device="cpu"):
    ckpt=torch.load(path,map_location=device,weights_only=True)
    if ckpt["aa_vocab"]!=AA or ckpt["codon_vocab"]!=CODONS:
        raise ValueError("Checkpoint vocabulary mismatch")
    if ckpt["config"].get("architecture_version")!=3:
        raise ValueError("Earlier architecture checkpoint; retrain with v3.")
    m=Model(Config(**ckpt["config"])).to(device)
    m.load_state_dict(ckpt["state"])
    return m.eval(),ckpt["metadata"]
