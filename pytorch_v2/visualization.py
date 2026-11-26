"""Interactive sequence-level uncertainty, codon detail and attribution plots."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import AA_ID, ALLOWED, CODONS

def uncertainty_plot(result):
    x=np.arange(1,len(result["codons"])+1)
    labels=[f"{i}: {a} / {c}" for i,a,c in zip(x,result["protein"],result["codons"])]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
        subplot_titles=("Normalized synonymous entropy","Ensemble confidence and consensus"))
    for name,color in [("aleatoric","#4775a8"),("epistemic","#d17b35")]:
        fig.add_trace(go.Scatter(x=x,y=result[name].numpy(),name=name,
            stackgroup="entropy",line=dict(color=color),text=labels,
            hovertemplate="%{text}<br>"+name+": %{y:.3f}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=result["total"].numpy(),name="Total",
        line=dict(color="#222222",dash="dot"),text=labels),row=1,col=1)
    for name,values in [("Chosen probability",result["mean"].max(-1).values),
                        ("Consensus",result["consensus"]),("Top-two margin",result["margin"])]:
        fig.add_trace(go.Scatter(x=x,y=values.numpy(),name=name,text=labels),row=2,col=1)
    fig.update_yaxes(range=[0,1.02]); fig.update_xaxes(title_text="Residue position",row=2,col=1)
    fig.update_layout(height=570,hovermode="x unified")
    return fig

def codon_detail(result,position):
    i=position-1
    if not 0<=i<len(result["codons"]): raise ValueError("Position out of range")
    ids=ALLOWED[AA_ID[result["protein"][i]]].nonzero().flatten().tolist()
    ids.sort(key=lambda j:float(result["mean"][i,j]),reverse=True)
    names=[CODONS[j] for j in ids]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=names,y=result["mean"][i,ids].numpy(),name="Ensemble mean",
        error_y=dict(type="data",array=result["sd"][i,ids].numpy())))
    for m,p in enumerate(result["members"][:,i,ids]):
        fig.add_trace(go.Scatter(x=names,y=p.numpy(),mode="markers",name=f"Member {m+1}"))
    fig.update_layout(title=f"Position {position}: {result['protein'][i]} / {result['codons'][i]}",
        yaxis=dict(range=[0,1.05],title="Probability"),height=420)
    return fig

def attribution_plot(explanations,protein,codons,position,alternative):
    import torch
    a=torch.stack([r["amino"] for r in explanations])
    c=torch.stack([r["prefix"] for r in explanations])
    fig=make_subplots(rows=2,cols=1,subplot_titles=("Amino acid context","Decoder prefix"))
    for row,data,labels in [(1,a,["BOS"]+list(protein)+["EOS"]),
                            (2,c,["BOS"]+codons[:position-1])]:
        mean=data.mean(0)
        fig.add_trace(go.Bar(x=list(range(len(labels))),y=mean.numpy(),text=labels,
            marker_color=["#27866a" if v>=0 else "#d17b35" for v in mean],
            error_y=dict(type="data",array=data.std(0,unbiased=False).numpy()),
            name="Mean signed attribution",showlegend=False),row=row,col=1)
    fig.update_layout(height=500,title=f"{codons[position-1]} versus {alternative} at position {position}")
    return fig

def calibration_plot(report):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Ideal",line=dict(dash="dot")))
    rows=report["bins"]
    fig.add_trace(go.Scatter(x=[r["confidence"] for r in rows],y=[r["accuracy"] for r in rows],
        text=[f"n={r['count']}" for r in rows],mode="markers+lines",name="Observed"))
    fig.update_layout(title=f"Reliability: ECE={report['ece']:.3f}, NLL={report['nll']:.3f}",
        xaxis=dict(title="Confidence",range=[0,1]),yaxis=dict(title="Accuracy",range=[0,1]))
    return fig

def cross_attention_plot(weights,protein,codons,layer=-1,head=0):
    return go.Figure(go.Heatmap(z=weights[layer,head].numpy(),
        x=["BOS"]+[f"{i+1}:{a}" for i,a in enumerate(protein)]+["EOS"],
        y=[f"{i+1}:{c}" for i,c in enumerate(codons)],colorscale="Viridis"))
