import sys,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import load_digits,load_breast_cancer,load_diabetes
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import pivoted_cholesky,balancing_matrix,floating_sign_witness
with threadpool_limits(limits=1):
 rows=json.loads(Path('results/kernels.json').read_text())
 for ds,loader in [('digits',load_digits),('breast_cancer',load_breast_cancer),('diabetes',load_diabetes)]:
  data=loader().data
  for n in (128,256,512):
   if n>len(data):continue
   for seed in range(5):
    rng=np.random.default_rng(1720406+seed+100*n)
    X=data[rng.choice(len(data),n,replace=False)].copy()
    X=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1.)
    dist=squareform(pdist(X,'sqeuclidean'))
    K=np.exp(-dist/np.median(dist[np.triu_indices(n,1)]))
    F,piv=pivoted_cholesky(K,32,rng)
    for k in (8,16,32):
     U=F[:,:k];r=np.maximum(np.diag(K-U@U.T),0);r[piv[:k]]=0
     for mu in (.01,.1,1.):
      row=next(z for z in rows if (z['dataset'],z['n'],z['seed'],z['k'],z['mu'])==(ds,n,seed,k,mu))
      t=time.perf_counter();q=floating_sign_witness(U,r,mu,seed);elapsed=time.perf_counter()-t
      row.update(top_rank_floor_excess=float((r.sum()-q['top_rank_penalty'])/mu),
       floating_lower_excess=q['lower'],floating_fractional=q['fractional_count'],
       floating_lp_residual=q['lp_residual'],floating_seconds=elapsed)
      assert q['penalty']<=q['top_rank_penalty']+1e-7
    print(ds,n,seed,flush=True)
 Path('results/kernels.json').write_text(json.dumps(rows,indent=2))
 # Larger real-data examples; factor and diagonal only after construction.
 scale=[]
 data=load_digits().data
 for n in (1024,1797):
  for seed in range(3):
   rng=np.random.default_rng(8000+seed)
   X=data[rng.choice(len(data),n,replace=False)].copy()
   X=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1.)
   dist=squareform(pdist(X,'sqeuclidean'))
   K=np.exp(-dist/np.median(dist[np.triu_indices(n,1)]))
   F,piv=pivoted_cholesky(K,128,rng)
   for k in (16,64,128):
    U=F[:,:k];r=np.maximum(np.diag(K-U@U.T),0);r[piv[:k]]=0
    for mu in (.1,1.):
     t=time.perf_counter();q=floating_sign_witness(U,r,mu,seed);elapsed=time.perf_counter()-t
     scale.append(dict(n=n,k=k,seed=seed,mu=mu,trace_excess=float(r.sum()/mu),
      top_rank_floor_excess=float((r.sum()-q['top_rank_penalty'])/mu),
      floating_lower_excess=q['lower'],seconds=elapsed,
      fractional=q['fractional_count'],lp_residual=q['lp_residual']))
   print('scale',n,seed,flush=True)
 Path('results/scaling.json').write_text(json.dumps(scale,indent=2))
