"""Shared-prefix ensemble inference, calibration and uncertainty."""
import torch
from torch.nn import functional as F
from model import AA, CODONS, ALLOWED, BOS, EOS, PAD, encode_protein

def entropy(p):
    return -torch.special.xlogy(p,p).sum(dim=-1)

def summarize(p, allowed):
    """p: [members, positions, codons], already synonym-normalized."""
    if p.ndim!=3 or not p.shape[0]:
        raise ValueError("Expected nonempty [members, positions, codons]")
    mean=p.mean(0)
    denom=allowed.sum(-1).clamp(min=2).float().log()
    total=entropy(mean)/denom
    within=entropy(p).mean(0)/denom
    between=(total-within).clamp(min=0)
    chosen=mean.argmax(-1)
    votes=p.argmax(-1)
    return dict(mean=mean,total=total,aleatoric=within,epistemic=between,
        sd=p.std(0,unbiased=False),chosen=chosen,
        consensus=(votes==chosen).float().mean(0),
        margin=mean.topk(2,dim=-1).values.diff(dim=-1).abs().squeeze(-1),
        members=p)

@torch.no_grad()
def predict(models, protein, temperatures=None):
    if not models: raise ValueError("No models")
    device=next(models[0].parameters()).device
    aa=torch.tensor([encode_protein(protein)],device=device)
    temperatures=[1.]*len(models) if temperatures is None else list(temperatures)
    if len(temperatures)!=len(models) or any(t<=0 for t in temperatures):
        raise ValueError("One positive temperature per model required")
    contexts=[]
    for m in models:
        m.eval()
        contexts.append(m.encode(aa))
    prefix=torch.tensor([[BOS]],device=device)
    all_p=[]
    for i in range(aa.shape[1]-2):
        allowed=ALLOWED.to(device)[aa[:,i+1]]
        p=torch.stack([(m.decode(aa,prefix,ctx)[0][:,-1].float()
            .masked_fill(~allowed,float("-inf"))/t).softmax(-1)[0]
            for m,ctx,t in zip(models,contexts,temperatures)])
        all_p.append(p)
        prefix=torch.cat([prefix,p.mean(0).argmax().reshape(1,1)],1)
    p=torch.stack(all_p,dim=1)
    result=summarize(p,ALLOWED.to(device)[aa[0,1:-1]])
    result["protein"]="".join(AA[i] for i in aa[0,1:-1].tolist())
    result["codons"]=[CODONS[i] for i in result["chosen"].tolist()]
    result["dna"]="".join(result["codons"])
    return {k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in result.items()}

def fit_temperature(logits,labels):
    """Fit scalar temperature using a dedicated held-out calibration split.
    Inputs already masked to synonymous codons; ignore padding."""
    x=logits.detach().float().reshape(-1,logits.shape[-1])
    y=labels.detach().reshape(-1)
    keep=(y!=PAD)&(y!=EOS)
    x,y=x[keep],y[keep]
    if not y.numel(): raise ValueError("No calibration tokens")
    # Finite sentinel avoids undefined gradients through -inf / temperature.
    x=x.masked_fill(~torch.isfinite(x),-1e4)
    logt=torch.zeros((),device=x.device,requires_grad=True)
    opt=torch.optim.LBFGS([logt],lr=.1,max_iter=60,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss=F.cross_entropy(x/logt.clamp(-4,4).exp(),y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.detach().clamp(-4,4).exp().cpu())

def reliability(probs,labels,bins=15):
    """Top-label ECE, NLL and multiclass Brier score; observed codons only."""
    p=probs.float().reshape(-1,probs.shape[-1])
    y=labels.reshape(-1)
    keep=(y!=PAD)&(y!=EOS)
    p,y=p[keep],y[keep]
    if not y.numel(): raise ValueError("No evaluation tokens")
    conf,pred=p.max(-1)
    correct=pred.eq(y).float()
    rows=[]
    ece=0.
    for i in range(bins):
        mask=(conf>=i/bins)&((conf<(i+1)/bins) if i<bins-1 else (conf<=1))
        n=int(mask.sum())
        if n:
            c=float(conf[mask].mean()); a=float(correct[mask].mean())
            rows.append(dict(confidence=c,accuracy=a,count=n))
            ece+=n/len(y)*abs(c-a)
    return dict(ece=ece,nll=float(-p.gather(1,y[:,None]).clamp_min(1e-12).log().mean()),
        brier=float((p-F.one_hot(y,p.shape[-1])).square().sum(-1).mean()),bins=rows)
