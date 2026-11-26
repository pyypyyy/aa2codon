"""Training, split preparation and checkpoint calibration."""
import argparse, json, random, time, math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from model import Model, Config, encode_cds, PAD, EOS, ALLOWED, AA
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
    a=pad_sequence([torch.as_tensor(x,dtype=torch.long) for x in a],batch_first=True,padding_value=PAD)
    c=pad_sequence([torch.as_tensor(x,dtype=torch.long) for x in c],batch_first=True,padding_value=PAD)
    return a,c[:,:-1],c[:,1:]

class BucketSampler(Sampler):
    """Shuffle windows, sort by length locally, shuffle resulting batches."""
    def __init__(self,rows,batch,seed):
        self.lengths=[len(a) for a,_ in rows]
        self.batch=batch
        self.generator=torch.Generator().manual_seed(seed)
    def __len__(self):
        return (len(self.lengths)+self.batch-1)//self.batch
    def __iter__(self):
        indices=torch.randperm(len(self.lengths),generator=self.generator).tolist()
        window=self.batch*32
        batches=[]
        for i in range(0,len(indices),window):
            ordered=sorted(indices[i:i+window],key=self.lengths.__getitem__)
            batches.extend(ordered[j:j+self.batch] for j in range(0,len(ordered),self.batch))
        for i in torch.randperm(len(batches),generator=self.generator).tolist():
            yield batches[i]

def loader(rows,batch,shuffle=False,seed=0):
    # Tokenize into tensors once per loader, not on every epoch.
    rows=[(torch.as_tensor(a,dtype=torch.long),torch.as_tensor(c,dtype=torch.long)) for a,c in rows]
    kwargs=dict(collate_fn=collate,pin_memory=torch.cuda.is_available())
    if shuffle:
        return DataLoader(rows,batch_sampler=BucketSampler(rows,batch,seed),**kwargs)
    # Stable validation/calibration/test order is required for ensemble aggregation.
    return DataLoader(rows,batch_size=batch,shuffle=False,**kwargs)

def make_optimizer(model,lr,weight_decay):
    # Biases, norm weights and xIELU scalars should not receive weight decay.
    decay=[]; no_decay=[]
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim>=2 else no_decay).append(p)
    device=next(model.parameters()).device
    return torch.optim.AdamW([dict(params=decay,weight_decay=weight_decay),
        dict(params=no_decay,weight_decay=0.)],lr=lr,fused=device.type=="cuda")

def make_scaler(device):
    return torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and not torch.cuda.is_bf16_supported())

def epoch(model,dl,device,opt=None,scheduler=None,scaler=None,monitor=None,epoch_number=1,aa_breakdown=False):
    training=opt is not None
    model.train(training)
    if training and scaler is None: scaler=make_scaler(device)
    amp=device.type=="cuda"
    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    total=0.; count=0; correct=0; grad_sum=0.; grad_steps=0; skipped=0
    started=time.perf_counter()
    if aa_breakdown:
        aa_sums=torch.zeros(len(AA),device=device)
        aa_counts=torch.zeros(len(AA),device=device)
    if monitor: monitor.start_epoch(epoch_number,len(dl),'Training' if training else 'Validation')
    for batch_idx,(a,p,y) in enumerate(dl,1):
        a,p,y=[v.to(device,non_blocking=True) for v in (a,p,y)]
        if model.config.synonym_mask:
            keep=(model.allowed[a[:,1:]].sum(-1)>1)&y.ne(PAD)&y.ne(EOS)
        else:
            keep=y.ne(PAD)
        n=int(keep.sum())
        if not n:
            if monitor: monitor.batch(batch_idx,len(dl),total/max(count,1),correct/max(count,1))
            continue
        if training: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                logits,_=model(a,p)
                # Mask deterministic positions as well as padding.
                labels=y.masked_fill(~keep,PAD)
                loss=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),
                                     ignore_index=PAD,reduction="sum")
            if training:
                scaler.scale(loss/n).backward()
                scaler.unscale_(opt)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                if math.isfinite(norm): grad_sum+=norm; grad_steps+=1
                else:
                    skipped+=1
                    if not scaler.is_enabled():
                        raise RuntimeError('Non-finite gradients in unscaled training')
                old_scale=scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scheduler and scaler.get_scale()>=old_scale: scheduler.step()
        total+=float(loss.detach()); count+=n
        correct+=int((logits.argmax(-1).eq(y)&keep).sum())
        if aa_breakdown:
            with torch.no_grad():
                ids=a[:,1:][keep]
                losses=F.cross_entropy(logits[keep].float(),y[keep],reduction='none')
                aa_sums.scatter_add_(0,ids,losses)
                aa_counts.scatter_add_(0,ids,torch.ones_like(losses))
        if monitor: monitor.batch(batch_idx,len(dl),total/count,correct/count,opt.param_groups[0]['lr'] if training else None)
    if monitor: monitor.close_bar()
    if not count: raise ValueError("No informative codon positions")
    seconds=max(time.perf_counter()-started,1e-9)
    result=dict(loss=total/count,accuracy=correct/count,tokens_per_second=count/seconds,seconds=seconds)
    if training: result.update(grad_norm=grad_sum/max(grad_steps,1),nonfinite_grad_batches=skipped)
    if aa_breakdown:
        sums,counts=aa_sums.cpu().tolist(),aa_counts.cpu().tolist()
        result['aa_nll']={AA[i]:sums[i]/n for i,n in enumerate(counts) if n}
    return result

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

def run(paths,out,config,ensemble=5,batch=32,epochs=40,lr=3e-4,groups=None,weight_decay=.01,live=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train,val,cal,test=split(records(paths,config.max_length),groups=groups)
    print("Device:",device,"split sizes:",*[len(x) for x in (train,val,cal,test)])
    manifest=[]
    test_sum=None
    val_sum=None
    from training_monitor import TrainingMonitor, diagnostics, gpu_utilization
    monitor=TrainingMonitor(out,ensemble,epochs,live=live)
    val_dl=loader(val,batch)
    cal_dl=loader(cal,batch)
    test_dl=loader(test,batch)
    if device.type=="cuda": torch.set_float32_matmul_precision("high")
    try:
        for seed in range(ensemble):
            monitor.start_member(seed+1)
            torch.manual_seed(1000+seed)
            m=Model(config).to(device)
            opt=make_optimizer(m,lr,weight_decay)
            scaler=make_scaler(device)
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
                started=time.perf_counter()
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                tr=epoch(m,dl,device,opt,sched,scaler,monitor,e+1)
                va=epoch(m,val_dl,device,monitor=monitor,epoch_number=e+1,aa_breakdown=True)
                gates,magnitudes=diagnostics(m,val_dl,device)
                if not math.isfinite(va['loss']) or not math.isfinite(tr['loss']):
                    raise RuntimeError('Non-finite training or validation loss; inspect logs.')
                if va["loss"]<best:
                    best=va["loss"]; stale=0; m.save(path,seed=seed)
                else:
                    stale+=1
                row=dict(epoch=e+1,train=tr,validation=va,activation=m.activation_parameters(),
                    attention_gates=gates,activation_rms=magnitudes,validation_aa_nll=va['aa_nll'],
                    lr=opt.param_groups[0]['lr'],best_val=best,stale=stale,seconds=time.perf_counter()-started)
                if device.type=='cuda':
                    gib=1024**3
                    row.update(gpu_allocated_gib=torch.cuda.memory_allocated(device)/gib,
                        gpu_reserved_gib=torch.cuda.memory_reserved(device)/gib,
                        gpu_peak_gib=torch.cuda.max_memory_allocated(device)/gib,
                        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory/gib)
                    util=gpu_utilization(device)
                    if util is not None: row['gpu_utilization_pct']=util
                history.append(row)
                monitor.record(row)
                if stale>=5: break
            from model import load_model
            m,_=load_model(path,device)
            x,y=collect(m,cal_dl,device)
            temperature=fit_temperature(x,y)
            xv,yv=collect(m,val_dl,device)
            vp=(xv/temperature).softmax(-1)
            val_sum=vp.clone() if val_sum is None else val_sum.add_(vp)
            v_report=reliability(val_sum/(seed+1),yv)
            del xv,yv,vp
            xt,yt=collect(m,test_dl,device)
            probs=(xt/temperature).softmax(-1)
            test_sum=probs.clone() if test_sum is None else test_sum.add_(probs)
            m.save(path,seed=seed,temperature=temperature)
            manifest.append(dict(path=path.name,temperature=temperature,
                 test=reliability(probs,yt),history=history))
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
            monitor.member_done({k:v_report[k] for k in ("nll","ece","brier")})
        report=reliability(test_sum/len(manifest),yt)
        (out/"ensemble_test.json").write_text(json.dumps(report,indent=2))
        return manifest,report
    finally:
        monitor.close()

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
