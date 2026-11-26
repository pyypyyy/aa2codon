"""Contrastive integrated gradients and retained cross-attention plots."""
import torch
from captum.attr import IntegratedGradients
from model import encode_protein, AA_ID, CODON_ID, ALLOWED, BOS, PAD

def explain(model,protein,codons,position,alternative,steps=64,temperature=1.):
    """Zero-based position. Fixed shared prefix; contrast chosen vs alternative.
    Zero embedding baseline, positional encodings and padding masks unchanged.
    Attribution describes a model logit contrast, not an expression effect.
    """
    model.eval()
    device=next(model.parameters()).device
    aa=torch.tensor([encode_protein(protein)],device=device)
    if not 0<=position<len(codons) or len(codons)!=aa.shape[1]-2:
        raise ValueError("Invalid position or codon count")
    chosen=CODON_ID[codons[position]]; alt=CODON_ID[alternative]
    if chosen==alt or not ALLOWED[aa[0,position+1].item(),alt]:
        raise ValueError("Alternative must be a different synonymous codon")
    prefix=torch.tensor([[BOS]+[CODON_ID[c] for c in codons[:position]]],device=device)
    aemb=model.embed(aa,model.aa_embed).detach()
    cemb=model.embed(prefix,model.codon_embed).detach()
    def f(a,c):
        n=a.shape[0]
        logits,_=model(aa.expand(n,-1),prefix.expand(n,-1),aa_emb=a,codon_emb=c)
        return (logits[:,-1,chosen]-logits[:,-1,alt])/temperature
    # Disable inference fast paths where needed so autograd can see attention.
    previous=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        attrs,delta=IntegratedGradients(f).attribute((aemb,cemb),
            baselines=(torch.zeros_like(aemb),torch.zeros_like(cemb)),
            n_steps=steps,internal_batch_size=4,return_convergence_delta=True)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    with torch.no_grad():
        contrast=float(f(aemb,cemb).item())
        baseline=float(f(torch.zeros_like(aemb),torch.zeros_like(cemb)).item())
    return dict(amino=attrs[0].sum(-1)[0].detach().cpu(),
                prefix=attrs[1].sum(-1)[0].detach().cpu(),
                convergence_delta=float(delta.item()),contrast=contrast,baseline=baseline)

@torch.no_grad()
def attention_maps(model,protein,codons):
    device=next(model.parameters()).device
    aa=torch.tensor([encode_protein(protein)],device=device)
    prefix=torch.tensor([[BOS]+[CODON_ID[c] for c in codons[:-1]]],device=device)
    model.eval()
    _,weights=model(aa,prefix,attention=True)
    return torch.stack([w[0].cpu() for w in weights])
