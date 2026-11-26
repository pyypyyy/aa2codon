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
    d_model: int = 128
    heads: int = 8
    layers: int = 3
    d_ff: int = 256
    dropout: float = .1
    max_length: int = 1024
    activation: str = "xielu"
    synonym_mask: bool = True

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

class Block(nn.Module):
    def __init__(self,c,decoder=False):
        super().__init__()
        self.self_attn=nn.MultiheadAttention(c.d_model,c.heads,dropout=c.dropout,batch_first=True)
        self.cross=nn.MultiheadAttention(c.d_model,c.heads,dropout=c.dropout,batch_first=True) if decoder else None
        self.n1=nn.LayerNorm(c.d_model)
        self.n2=nn.LayerNorm(c.d_model)
        self.n3=nn.LayerNorm(c.d_model) if decoder else None
        self.ff=FFN(c)
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x,pad,context=None,context_pad=None,causal=None,attention=False):
        out,_=self.self_attn(x,x,x,key_padding_mask=pad,attn_mask=causal,need_weights=False)
        x=self.n1(x+self.drop(out))
        weights=None
        if self.cross is not None:
            out,weights=self.cross(x,context,context,key_padding_mask=context_pad,
                need_weights=attention,average_attn_weights=False)
            x=self.n3(x+self.drop(out))
        return self.n2(x+self.ff(x)),weights

class Model(nn.Module):
    def __init__(self,c=Config()):
        super().__init__()
        if c.d_model%c.heads or c.d_model%2:
            raise ValueError("d_model must be even and divisible by heads")
        self.config=c
        self.aa_embed=nn.Embedding(len(AA),c.d_model,padding_idx=PAD)
        self.codon_embed=nn.Embedding(len(CODONS),c.d_model,padding_idx=PAD)
        self.encoder=nn.ModuleList(Block(c) for _ in range(c.layers))
        self.decoder=nn.ModuleList(Block(c,True) for _ in range(c.layers))
        self.output=nn.Linear(c.d_model,len(CODONS))
        self.drop=nn.Dropout(c.dropout)
        pos=torch.arange(c.max_length+2).float()[:,None]
        freq=torch.exp(-math.log(10000)*torch.arange(c.d_model//2).float()/(c.d_model//2))
        angle=pos*freq
        self.register_buffer("pos",torch.cat([angle.sin(),angle.cos()],-1))
        self.register_buffer("allowed",ALLOWED.clone())
    def embed(self,ids,layer):
        return layer(ids)*math.sqrt(self.config.d_model)
    def with_pos(self,x):
        if x.shape[1]>len(self.pos):
            raise ValueError("Sequence exceeds configured context length")
        return self.drop(x+self.pos[:x.shape[1]].to(x.dtype))
    def encode(self,aa,emb=None):
        x=self.with_pos(self.embed(aa,self.aa_embed) if emb is None else emb)
        for block in self.encoder:
            x,_=block(x,aa.eq(PAD))
        return x
    def decode(self,aa,prefix,context,emb=None,attention=False):
        x=self.with_pos(self.embed(prefix,self.codon_embed) if emb is None else emb)
        causal=torch.ones(prefix.shape[1],prefix.shape[1],dtype=torch.bool,device=prefix.device).triu(1)
        weights=[]
        for block in self.decoder:
            x,w=block(x,prefix.eq(PAD),context,aa.eq(PAD),causal,attention)
            if attention: weights.append(w)
        logits=self.output(x)
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
    m=Model(Config(**ckpt["config"])).to(device)
    m.load_state_dict(ckpt["state"])
    return m.eval(),ckpt["metadata"]
