"""Training, split preparation and checkpoint calibration."""
import argparse, json, random
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from model import Model, Config, encode_cds, PAD, EOS, ALLOWED
from uncertainty import fit_temperature, reliability

def records(paths,max_length):
    seen=set()
    rows=[]
    for path in paths:
        for record in SeqIO.parse(str(path),"fasta"):
            if "pseudogene" in record.description.lower(): continue
            try: aa,cd,protein=encode_cds(str(record.seq))
            except ValueError: continue
            if len(protein)>max_length: continue
            key=tuple(cd)
            if key in seen: continue
            seen.add(key)
            rows.append((aa,cd,protein))
    return rows

def split(rows,seed=2026,groups=None):
    """Exact protein grouping by default; supply homolog cluster IDs for strict evaluation."""
    by={}
    for a,c,p in rows:
        key=groups[p] if groups is not None else p
        by.setdefault(key,[]).append((a,c))
    keys=sorted(by); random.Random(seed).shuffle(keys)
    n=len(keys)
    if n<20: raise ValueError("Need at least 20 protein groups")
    k=max(1,int(.1*n))
    return [[row for key in ks for row in by[key]] for ks in
            (keys[3*k:],keys[:k],keys[k:2*k],keys[2*k:3*k])]

def collate(rows):
    a,c=zip(*rows)
    a=pad_sequence([torch.tensor(x) for x in a],batch_first=True,padding_value=PAD)
    c=pad_sequence([torch.tensor(x) for x in c],batch_first=True,padding_value=PAD)
    return a,c[:,:-1],c[:,1:]

def loader(rows,batch,shuffle=False,seed=0):
    return DataLoader(rows,batch_size=batch,shuffle=shuffle,collate_fn=collate,
        generator=torch.Generator().manual_seed(seed))

def epoch(model,dl,device,opt=None,scheduler=None):
    model.train(opt is not None)
    total=0.; count=0; correct=0
    for a,p,y in dl:
        a,p,y=[v.to(device) for v in (a,p,y)]
        if opt: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(opt is not None):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                 enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
                logits,_=model(a,p)
                loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=PAD,reduction="sum")
                n=int(y.ne(PAD).sum())
            if opt:
                (loss/max(n,1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                if scheduler: scheduler.step()
        total+=float(loss.detach()); count+=n
        correct+=int((logits.argmax(-1).eq(y)&y.ne(PAD)).sum())
    return dict(loss=total/max(count,1),accuracy=correct/max(count,1))

@torch.no_grad()
def collect(model,dl,device):
    model.eval()
    xs=[]; ys=[]
    for a,p,y in dl:
        a,p,y=[v.to(device) for v in (a,p,y)]
        logits,_=model(a,p)
        # Exclude deterministic codons and special tokens for calibration.
        allowed=ALLOWED.to(device)[a[:,1:]]
        keep=(allowed.sum(-1)>1)&y.ne(PAD)&y.ne(EOS)
        xs.append(logits[keep].float().cpu()); ys.append(y[keep].cpu())
    return torch.cat(xs),torch.cat(ys)

def run(paths,out,config,ensemble=5,batch=32,epochs=40,lr=3e-4,groups=None,weight_decay=.01):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train,val,cal,test=split(records(paths,config.max_length),groups=groups)
    print("Device:",device,"split sizes:",*[len(x) for x in (train,val,cal,test)])
    manifest=[]
    test_probs=[]
    for seed in range(ensemble):
        torch.manual_seed(1000+seed)
        m=Model(config).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=weight_decay)
        dl=loader(train,batch,True,1000+seed)
        steps=len(dl)*epochs
        warm=max(1,int(.05*steps))
        import math
        def schedule(s):
            return min(1.,(s+1)/warm) if s<warm else .1+.9*(1+math.cos(math.pi*min(1.,(s-warm)/max(1,steps-warm))))/2
        sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
        best=float("inf"); stale=0; history=[]
        path=out/f"member_{seed}.pt"
        for e in range(epochs):
            tr=epoch(m,dl,device,opt,sched)
            va=epoch(m,loader(val,batch),device)
            row=dict(epoch=e+1,train=tr,validation=va,activation=m.activation_parameters())
            history.append(row); print(seed,row)
            if va["loss"]<best:
                best=va["loss"]; stale=0; m.save(path,seed=seed)
            else:
                stale+=1
                if stale>=5: break
        from model import load_model
        m,_=load_model(path,device)
        x,y=collect(m,loader(cal,batch),device)
        temperature=fit_temperature(x,y)
        xt,yt=collect(m,loader(test,batch),device)
        test_probs.append((xt/temperature).softmax(-1))
        m.save(path,seed=seed,temperature=temperature)
        manifest.append(dict(path=path.name,temperature=temperature,
             test=reliability(test_probs[-1],yt),history=history))
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    report=reliability(torch.stack(test_probs).mean(0),yt)
    (out/"ensemble_test.json").write_text(json.dumps(report,indent=2))
    return manifest,report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("fasta",nargs="+"); p.add_argument("--out",required=True)
    p.add_argument("--ensemble",type=int,default=5); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--activation",choices=["relu","xielu","swiglu"],default="xielu")
    p.add_argument("--max-length",type=int,default=1024)
    p.add_argument("--tuned-config",type=str,default=None)
    a=p.parse_args()
    if a.tuned_config:
        t=json.loads(Path(a.tuned_config).read_text())
        run(a.fasta,a.out,Config(**t["config"]),a.ensemble,t["batch"],a.epochs,
            lr=t["lr"],weight_decay=t["weight_decay"])
    else:
        run(a.fasta,a.out,Config(activation=a.activation,max_length=a.max_length),a.ensemble,a.batch,a.epochs)
