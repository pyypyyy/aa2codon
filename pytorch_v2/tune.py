"""Small, resumable coordinate search for exactly five hyperparameters."""
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Config, Model, ALLOWED, PAD, EOS
from train import records, split, loader, epoch

BASELINE = dict(d_model=256, d_ff=1024, dropout=.05, lr=3e-4, weight_decay=.01)
# Start with LR to avoid comparing architectures at an obviously poor LR.
SPACE = {
    "lr": [5e-5, 1e-4, 2e-4, 3e-4, 6e-4],
    "d_model": [128, 192, 256, 384, 512],
    "d_ff": [256, 512, 768, 1024, 1536],
    "dropout": [.05, .1, .2, .3, .4],
    "weight_decay": [0., 1e-4, 1e-3, .01, .05],
}

def atomic_json(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)

@torch.no_grad()
def synonymous_nll(model, dl, device):
    model.eval()
    total=0.; count=0
    for a,p,y in dl:
        a,p,y=[v.to(device) for v in (a,p,y)]
        keep=(ALLOWED.to(device)[a[:,1:]].sum(-1)>1)&y.ne(PAD)&y.ne(EOS)
        if keep.any():
            z,_=model(a,p)
            total+=float(F.cross_entropy(z[keep].float(),y[keep],reduction="sum"))
            count+=int(keep.sum())
    if not count: raise ValueError("No degenerate codons in validation data")
    return total/count

def model_config(params,max_length):
    return Config(d_model=params["d_model"],d_ff=params["d_ff"],dropout=params["dropout"],
                  heads=8,layers=3,activation="xielu",max_length=max_length,synonym_mask=True)

def signature(paths,batch,max_length,epochs,seed,prune,groups):
    files=[]
    for p in paths:
        h=hashlib.sha256()
        with open(p,"rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        files.append([str(p),h.hexdigest()])
    group_hash=None if groups is None else hashlib.sha256(json.dumps(groups,sort_keys=True).encode()).hexdigest()
    return dict(version=1,files=files,batch=batch,max_length=max_length,epochs=epochs,
                seed=seed,prune=prune,groups=group_hash,space=SPACE,baseline=BASELINE)

def tune(paths,out,batch=16,max_length=1024,epochs=8,seed=2026,prune=True,groups=None):
    """At most 21 unique trials; one coordinate sweep, no Cartesian grid.
    Finished trials persist; interrupted trials restart from their beginning.
    Validation only; calibration and test are reserved for later training.
    """
    if epochs<4 or batch<1: raise ValueError("Use at least four epochs and a positive batch size")
    paths=list(paths)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    state_path=out/"tuning_trials.json"
    sig=signature(paths,batch,max_length,epochs,seed,prune,groups)
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(signature=sig,trials=[])
    if state["signature"]!=sig:
        raise ValueError("Tuning settings/data changed. Choose a new output directory.")
    tr,va,_,_=split(records(paths,max_length),groups=groups)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dl=loader(va,batch)
    cache={json.dumps(r["params"],sort_keys=True):r for r in state["trials"]}

    def evaluate(params,incumbent=None):
        key=json.dumps(params,sort_keys=True)
        if key in cache: return cache[key]
        torch.manual_seed(seed)
        m=Model(model_config(params,max_length)).to(device)
        dl=loader(tr,batch,True,seed)
        opt=torch.optim.AdamW(m.parameters(),lr=params["lr"],weight_decay=params["weight_decay"])
        steps=len(dl)*epochs; warm=max(1,int(.05*steps))
        def schedule(s):
            return min(1.,(s+1)/warm) if s<warm else .1+.9*(1+math.cos(math.pi*min(1.,(s-warm)/max(1,steps-warm))))/2
        sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
        curve=[]; status="complete"; start=time.perf_counter()
        for e in range(epochs):
            epoch(m,dl,device,opt,sched)
            nll=synonymous_nll(m,val_dl,device)
            if not math.isfinite(nll):
                status="nonfinite"; break
            curve.append(nll)
            print("trial",len(state["trials"])+1,"epoch",e+1,"NLL",round(nll,5),params,flush=True)
            # Conservative optional pruning: compare at the same epoch.
            if prune and incumbent and e+1==max(3,epochs//2):
                ref=incumbent["curve"][e]
                if nll>ref*1.15:
                    status="pruned"; break
        row=dict(params=dict(params),curve=curve,status=status,
                 score=min(curve) if status=="complete" else None,
                 seconds=time.perf_counter()-start,
                 parameters=sum(p.numel() for p in m.parameters()))
        del m,opt,sched,dl
        if device.type=="cuda": torch.cuda.empty_cache()
        state["trials"].append(row); cache[key]=row
        atomic_json(state_path,state)
        return row

    best=evaluate(BASELINE)
    if best["status"]!="complete":
        raise RuntimeError("Baseline failed; inspect training before tuning.")
    for name,values in SPACE.items():
        anchor=dict(best["params"])
        incumbent=best
        candidates=[best]
        for value in values:
            if value!=anchor[name]:
                candidates.append(evaluate(dict(anchor,**{name:value}),incumbent))
        best=min((r for r in candidates if r["status"]=="complete"),key=lambda r:r["score"])
        print("Selected",name,best["params"][name],"NLL",best["score"],flush=True)
    result=dict(config=asdict(model_config(best["params"],max_length)),
                lr=best["params"]["lr"],weight_decay=best["params"]["weight_decay"],
                batch=batch,validation_nll=best["score"],trials=len(state["trials"]),
                signature=sig)
    atomic_json(out/"best_config.json",result)
    return result,state["trials"]

def plot_trials(trials):
    import plotly.graph_objects as go
    fig=go.Figure()
    for i,r in enumerate(trials,1):
        p=r["params"]
        label=f'{i}: d={p["d_model"]}, ff={p["d_ff"]}, drop={p["dropout"]}, lr={p["lr"]}, wd={p["weight_decay"]}'
        fig.add_trace(go.Scatter(x=list(range(1,len(r["curve"])+1)),y=r["curve"],
            name=label,mode="lines+markers",line=dict(dash="solid" if r["status"]=="complete" else "dot"),
            hovertemplate=label+"<br>NLL %{y:.5f}<extra></extra>"))
    fig.update_layout(title="Validation synonymous-codon NLL by trial",xaxis_title="Epoch",yaxis_title="NLL")
    return fig
