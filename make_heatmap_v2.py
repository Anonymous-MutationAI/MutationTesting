import sys
R=sys.argv[1]; out=sys.argv[2]; sys.path.insert(0,R); sys.path.insert(0,R+'/src')
from pathlib import Path
from analyze_v2 import load, witnesses, OPS, FAM
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
B=load(Path(R)/'results/v2', set())
SU=["T-Dev*","T-Rand*","T-E2E*","T-Struct"]; SL=["T-Dev$^\\star$","T-Rand$^\\star$","T-E2E$^\\star$","T-Struct"]
CR=["K-Struct","K-Outcome","K-Safety"]
NAME={"Del-Rem":"Deleg-Rem","Del-Red":"Deleg-Red","Del-Inv":"Deleg-Inv","Del-Add":"Deleg-Add"}
def ms(op,s,K):
    kd=kb=0
    for b,d in B.items():
        S=set(d['meta']['suites'][s])
        for m in d['mutants']:
            if m['operator']!=op: continue
            w=witnesses(m,K)
            if w: kb+=1; kd+=bool(w&S)
    return kd/kb if kb else None
M=[[ms(op,s,K) for K in CR for s in SU] for op in OPS]
cmap=LinearSegmentedColormap.from_list("seq",["#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"])
plt.rcParams.update({"font.family":"serif","font.size":8})
fig,ax=plt.subplots(figsize=(7.2,3.9))
for i,row in enumerate(M):
    for j,v in enumerate(row):
        if v is None:
            ax.add_patch(plt.Rectangle((j,i),1,1,fc="#e4e3df",ec="white",lw=1.5))
            ax.text(j+.5,i+.5,"n/a",ha="center",va="center",color="#77756f",fontsize=6.5)
        else:
            ax.add_patch(plt.Rectangle((j,i),1,1,fc=cmap(v),ec="white",lw=1.5))
            ax.text(j+.5,i+.5,f"{v:.2f}",ha="center",va="center",fontsize=6.5,color="white" if v>0.55 else "#1f1e1c")
ax.set_xlim(0,12); ax.set_ylim(len(OPS),0)
ax.set_xticks([j+.5 for j in range(12)]); ax.set_xticklabels(SL*3,rotation=45,ha="right",fontsize=7)
ax.set_yticks([i+.5 for i in range(len(OPS))]); ax.set_yticklabels([f"{NAME.get(o,o)} ({FAM[o]})" for o in OPS],fontsize=7)
for x in (4,8): ax.axvline(x,color="#1f1e1c",lw=1.2)
for y,f in enumerate(OPS[1:],1):
    if FAM[f]!=FAM[OPS[y-1]]: ax.axhline(y,color="#1f1e1c",lw=1.2)
for k,K in enumerate(CR): ax.text(4*k+2,-0.4,K,ha="center",va="bottom",fontsize=8,fontweight="bold")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1)); cb=fig.colorbar(sm,ax=ax,fraction=0.025,pad=0.01)
cb.set_label("$MS_{strict}$",fontsize=7); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(out,bbox_inches="tight"); fig.savefig(out.replace(".pdf",".png"),dpi=200,bbox_inches="tight")
print("saved",out)
