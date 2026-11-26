"""Live notebook progress and durable training diagnostics."""
import json
import csv
import time
from pathlib import Path
import torch

class TrainingMonitor:
    def __init__(self,out,total,epochs,live=False):
        self.out=Path(out)/"training_logs"/time.strftime("%Y%m%d-%H%M%S")
        self.out.mkdir(parents=True,exist_ok=False)
        self.total,self.epochs,self.live=total,epochs,live
        self.writer=None; self.display=None; self.bar=None
        self.member=0; self.history=[]; self.completed=[]
        self.started=time.perf_counter()
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(str(self.out/"tensorboard"))
        except ImportError:
            print("TensorBoard unavailable; JSONL and CSV logging remain enabled.")
        print("Training diagnostics:",self.out)

    def start_member(self,member):
        self.member=member
        self.history=[]
        self.display=None

    def start_epoch(self,epoch,batches,phase):
        self.close_bar()
        if self.live:
            from tqdm.auto import tqdm
            self.bar=tqdm(total=batches,desc=f"Member {self.member}/{self.total} | Epoch {epoch}/{self.epochs} | {phase}",leave=False,mininterval=1.)

    def batch(self,step,total,loss,accuracy,lr=None):
        if self.bar is not None:
            self.bar.update(step-self.bar.n)
            self.bar.set_postfix(loss=f"{loss:.4f}",acc=f"{accuracy:.1%}",
                **({"lr":f"{lr:.2e}"} if lr is not None else {}),refresh=False)

    def close_bar(self):
        if self.bar is not None:
            self.bar.close(); self.bar=None

    def _append(self,name,row):
        with (self.out/name).open("a") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")

    def record(self,row):
        self.history.append(row)
        self._append("epochs.jsonl",dict(member=self.member,**row))
        flat={"member":self.member,"epoch":row["epoch"],"lr":row["lr"],
              "seconds":row["seconds"],"best_val":row["best_val"],"stale":row["stale"]}
        for phase in ("train","validation"):
            for key,val in row[phase].items():
                if isinstance(val,(float,int)):
                    flat[f"{phase}_{key}"]=val
        for key in ("gpu_allocated_gib","gpu_reserved_gib","gpu_total_gib"):
            flat[key]=row.get(key,0)
        file=self.out/"epochs.csv"
        with file.open("a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(flat))
            if f.tell()==0: w.writeheader()
            w.writerow(flat)
        if self.writer:
            prefix=f"member_{self.member:02d}"
            for key,val in flat.items():
                if key not in ("member","epoch"):
                    self.writer.add_scalar(f"{prefix}/{key}",val,row["epoch"])
            for name,(ap,an) in row["activation"].items():
                self.writer.add_scalar(f"{prefix}/xielu/{name}/alpha_p",ap,row["epoch"])
                self.writer.add_scalar(f"{prefix}/xielu/{name}/alpha_n",an,row["epoch"])
            for name,values in row.get("attention_gates",{}).items():
                for h,v in enumerate(values):
                    self.writer.add_scalar(f"{prefix}/gates/{name}/head_{h+1}",v,row["epoch"])
            for aa,nll in row.get("validation_aa_nll",{}).items():
                self.writer.add_scalar(f"{prefix}/aa_nll/{aa}",nll,row["epoch"])
            self.writer.flush()
        if self.live: self.render()
        else:
            print(f"Member {self.member}/{self.total}, epoch {row['epoch']}: "
                  f"train NLL {row['train']['loss']:.4f}, validation NLL {row['validation']['loss']:.4f}")

    def member_done(self,metrics):
        self.completed.append(dict(member=self.member,**metrics))
        self._append("ensemble.jsonl",self.completed[-1])
        if self.writer:
            for k,v in metrics.items(): self.writer.add_scalar(f"ensemble_validation/{k}",v,self.member)
            self.writer.flush()
        if self.live: self.render()
        print(f"Completed member {self.member}/{self.total}; cumulative validation NLL {metrics['nll']:.4f}")

    def render(self):
        from IPython.display import display, HTML
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        rows=self.history
        if not rows: return
        x=[r["epoch"] for r in rows]; last=rows[-1]
        fig=make_subplots(rows=3,cols=2,subplot_titles=(
            "Synonymous codon NLL","Synonymous codon accuracy","Learning rate",
            "Gradient norm (before clipping)","xIELU coefficients","Ensemble validation NLL"))
        def line(xx,yy,name,r,c,dash=None):
            fig.add_trace(go.Scatter(x=xx,y=yy,name=name,mode="lines+markers",
                line=dict(dash=dash or "solid"),marker=dict(size=4)),row=r,col=c)
        for phase in ("train","validation"):
            line(x,[r[phase]["loss"] for r in rows],phase+" NLL",1,1)
            line(x,[r[phase]["accuracy"] for r in rows],phase+" accuracy",1,2)
        line(x,[r["lr"] for r in rows],"LR",2,1)
        line(x,[r["train"]["grad_norm"] for r in rows],"Gradient norm",2,2)
        for name in last["activation"]:
            line(x,[r["activation"][name][0] for r in rows],name+" alpha_p",3,1)
            line(x,[r["activation"][name][1] for r in rows],name+" alpha_n",3,1,"dot")
        if self.completed:
            line([r["member"] for r in self.completed],[r["nll"] for r in self.completed],
                 "Cumulative ensemble",3,2)
        fig.update_yaxes(tickformat=".0%",row=1,col=2)
        fig.update_layout(height=720,margin=dict(t=40,b=25,l=35,r=15),legend=dict(font=dict(size=9)))
        gpu=(f"{last['gpu_allocated_gib']:.2f} GiB allocated / {last['gpu_total_gib']:.2f} GiB total"
             if last.get("gpu_total_gib",0) else "CPU")
        done=len(self.completed)
        # Progress is a completed-member count, not a time estimate.
        header=(f"<h3>aa2codon training | member {self.member}/{self.total} | "
                f"epoch {last['epoch']}/{self.epochs}</h3>"
                f"<p>Completed members: {done}/{self.total} &nbsp; "
                f"<progress value='{done}' max='{self.total}'></progress><br>"
                f"Best validation NLL: {last['best_val']:.4f} | "
                f"Early stopping: {last['stale']}/5 | "
                f"Epoch: {last['seconds']:.1f} s | "
                f"Informative codons/s: {last['train']['tokens_per_second']:,.0f}<br>"
                f"Memory: {gpu} | Elapsed: {(time.perf_counter()-self.started)/60:.1f} min</p>")
        html=HTML(header+fig.to_html(full_html=False,include_plotlyjs=True))
        if self.display is None: self.display=display(html,display_id=True)
        else: self.display.update(html)

    def close(self):
        self.close_bar()
        if self.writer: self.writer.close()

def diagnostics(model,dl,device,max_batches=4):
    """Fixed validation sample, no dropout; real gate activations, not weights."""
    from model import Attention, AA, PAD, EOS
    model.eval()
    sums={}; counts={}; magnitudes={}; hooks=[]
    current={}
    def hook(name,module):
        def fn(mod,inputs,output):
            vals=output.detach().float().sigmoid()
            mask=current["prefix"] if name.startswith("decoder") else current["aa"]
            vals=vals[mask]
            if vals.numel():
                sums[name]=sums.get(name,torch.zeros(vals.shape[-1],device=device))+vals.sum(0)
                counts[name]=counts.get(name,0)+vals.shape[0]
        return module.gate.register_forward_hook(fn)
    def activation_hook(name,module):
        def fn(mod,inputs,output):
            magnitudes[name]=magnitudes.get(name,0.)+float(output.detach().float().square().mean().sqrt())
        return module.register_forward_hook(fn)
    for name,m in model.named_modules():
        if isinstance(m,Attention): hooks.append(hook(name,m))
        if name.endswith(".ff.act"): hooks.append(activation_hook(name,m))
    n=0
    try:
        with torch.no_grad():
            for a,p,y in dl:
                a,p=a.to(device),p.to(device)
                current.update(aa=a.ne(PAD)&a.ne(EOS)&a.ne(1),prefix=p.ne(PAD))
                model(a,p)
                n+=1
                if n>=max_batches: break
    finally:
        for h in hooks: h.remove()
    return ({k:(v/counts[k]).cpu().tolist() for k,v in sums.items()},
            {k:v/max(n,1) for k,v in magnitudes.items()})
