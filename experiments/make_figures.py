"""All manuscript figures and tables derive from retained result files."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path('results');O=Path('paper/figures');O.mkdir(exist_ok=True)
A=json.loads((P/'kernels.json').read_text());S=json.loads((P/'sdp.json').read_text());G=json.loads((P/'scaling.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'legend.fontsize':8,'savefig.bbox':'tight','pdf.fonttype':42})
colors=['#16697a','#d87b25','#7851a9'];names=['Digits','Breast cancer','Diabetes'];dsets=['digits','breast_cancer','diabetes']
def save(fig,name):
 fig.savefig(O/(name+'.pdf'));fig.savefig(O/(name+'.png'),dpi=180);plt.close(fig)
# Concrete identical-transcript residual pair and exact spectra.
fig,axs=plt.subplots(1,3,figsize=(10,2.6),gridspec_kw={'width_ratios':[1,1,1.7]})
n=32;r=np.r_[0.,np.ones(n-1)];R1=np.diag(r);R2=np.outer(np.sqrt(r),np.sqrt(r))
for ax,R,title in zip(axs[:2],[R1,R2],['Spread residual','Aligned residual']):
 ax.imshow(R,cmap='Blues',vmin=0,vmax=1,interpolation='none');ax.set_title(title);ax.set_xticks([0,31]);ax.set_yticks([0,31]);ax.set_xlabel('Coordinate')
axs[0].set_ylabel('Coordinate')
axs[2].plot(np.arange(1,n+1),np.r_[1.,np.repeat(2.,n-1)],'o-',ms=3,label='Spread: condition = 2',color=colors[0])
axs[2].plot(np.arange(1,n+1),np.r_[np.ones(n-1),32.],'s-',ms=3,label='Aligned: condition = 32',color=colors[1])
axs[2].set_yscale('log');axs[2].set_xlabel('Ordered eigenvalue index');axs[2].set_ylabel('Preconditioned eigenvalue');axs[2].legend(loc='upper left');fig.tight_layout();save(fig,'same_diagonal')
# Certificate ratios (excess) by rank, with seed and matrix-size variation.
fig,axs=plt.subplots(1,3,figsize=(10,2.7),sharey=True)
for ax,ds,name,c in zip(axs,dsets,names,colors):
 for key,style,label in [('floor_excess','--','Trace-of-relative-residual floor'),('top_rank_floor_excess','-','Top-rank balancing floor'),('floating_lower_excess',':','Constructed sign witness')]:
  means=[];mins=[];maxs=[]
  for k in [8,16,32]:
   v=[x[key]/x['trace_excess'] for x in A if x['dataset']==ds and x['k']==k]
   means.append(np.median(v));mins.append(min(v));maxs.append(max(v))
  ax.plot([8,16,32],means,style,color=c,label=label,marker='o',ms=3)
  if key=='top_rank_floor_excess':ax.fill_between([8,16,32],mins,maxs,color=c,alpha=.12)
 ax.set_title(name);ax.set_xlabel('Approximation rank k');ax.set_xticks([8,16,32]);ax.set_ylim(.75,1.01);ax.axhline(1,color='#666666',lw=.7)
axs[0].set_ylabel('Lower bound / trace bound (excess)');axs[1].legend(loc='lower left',fontsize=6.5);fig.tight_layout();save(fig,'barrier')
# Actual spectrum versus information floor.
fig,axs=plt.subplots(1,2,figsize=(9,3.2))
for ds,name,c in zip(dsets,names,colors):
 rows=[x for x in A if x['dataset']==ds]
 axs[0].scatter([x['actual_condition'] for x in rows],[1+x['top_rank_floor_excess'] for x in rows],s=12,alpha=.55,label=name,color=c)
 axs[1].scatter([x['actual_condition'] for x in rows],[x['pcg_iterations'] for x in rows],s=12,alpha=.55,color=c)
axs[0].plot([1,1e5],[1,1e5],color='#555555',lw=.7,ls='--');axs[0].set_xscale('log');axs[0].set_yscale('log');axs[0].set_xlabel('Actual condition number');axs[0].set_ylabel('Minimum possible universal diagonal certificate');axs[0].legend()
axs[1].set_xscale('log');axs[1].set_xlabel('Actual condition number');axs[1].set_ylabel('PCG iterations to relative residual 1e-9');fig.tight_layout();save(fig,'information_gap')
# SDP and scale study.
fig,axs=plt.subplots(1,2,figsize=(9,3))
rat=[min(x['sdp_upper_excess'],x['trace_excess'])/x['lower_excess']-1 for x in S]
axs[0].hist(np.array(rat)*100,bins=15,color=colors[0]);axs[0].set_xlabel('SDP/trace upper above sign witness (%)');axs[0].set_ylabel('Small-kernel cases (n = 64)')
for n,marker,c in [(1024,'o',colors[0]),(1797,'s',colors[1])]:
 for seed in range(3):
  rows=[x for x in G if x['n']==n and x['seed']==seed and x['mu']==.1]
  axs[1].plot([x['k'] for x in rows],[x['top_rank_floor_excess']/x['trace_excess'] for x in rows],marker=marker,ms=3,lw=.7,color=c,alpha=.65,label=f'n = {n}' if seed==0 else None)
axs[1].set_xlabel('Approximation rank k');axs[1].set_ylabel('Top-rank floor / trace bound (excess)');axs[1].set_xticks([16,64,128]);axs[1].set_ylim(.96,1.002);axs[1].legend();fig.tight_layout();save(fig,'sdp_scaling')
# TeX numeric tables; raw rows remain in JSON.
lines=[]
for ds,name in zip(dsets,names):
 rows=[x for x in A if x['dataset']==ds]
 med=lambda key:np.median([x[key] for x in rows])
 ratio=np.median([x['top_rank_floor_excess']/x['trace_excess'] for x in rows])
 lines.append(f'{name} & {len(rows)} & {med("actual_condition"):.1f} & {1+med("trace_excess"):.1f} & {100*ratio:.2f}\\% & {med("pcg_iterations"):.0f} \\\\')
(O/'kernel_table.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for target in [10,100,1000]:
 actual=sum(x['actual_condition']<=target for x in A)
 impossible=sum(x['actual_condition']<=target and 1+x['top_rank_floor_excess']>target for x in A)
 accepted=sum(1+x['trace_excess']<=target for x in A)
 lines.append(f'{target} & {actual} & {impossible} & {accepted} \\\\')
(O/'decision_table.tex').write_text('\n'.join(lines)+'\n')
print('four figures and two tables generated')
