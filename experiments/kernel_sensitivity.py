"""Sensitivity to kernel family and pivot rule, with shared inputs."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import eigh
from sklearn.datasets import load_digits,load_breast_cancer,load_diabetes
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import pivoted_cholesky,balancing_matrix

def greedy(A,rank):
 F=np.zeros((len(A),rank));r=np.diag(A).copy();piv=[]
 for j in range(rank):
  if r.sum() <= 1e-12*max(1.,float(np.trace(A))): break
  p=int(np.argmax(r));v=A[:,p]-F[:,:j]@F[p,:j]
  F[:,j]=v/np.sqrt(r[p]);piv.append(p)
  r=np.maximum(r-F[:,j]**2,0);r[piv]=0
 return F,piv
rows=[]
with threadpool_limits(limits=1):
 for ds,loader in [('digits',load_digits),('breast_cancer',load_breast_cancer),('diabetes',load_diabetes)]:
  data=loader().data
  for seed in range(3):
   rng=np.random.default_rng(9200+seed)
   X=data[rng.choice(len(data),256,replace=False)].copy()
   X=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1.)
   dist=squareform(pdist(X,'sqeuclidean'));med=np.median(dist[np.triu_indices(256,1)])
   t=np.sqrt(3*dist/med)
   for kernel,K in [('rbf',np.exp(-dist/med)),('matern32',(1+t)*np.exp(-t)),('linear',X@X.T/X.shape[1])]:
    for rule in ('random','greedy'):
     F,piv=pivoted_cholesky(K,32,np.random.default_rng(seed)) if rule=='random' else greedy(K,32)
     for k in (8,32):
      U=F[:,:k];r=np.maximum(np.diag(K-U@U.T),0);r[piv[:k]]=0
      for mu in (.1,1.):
       V=balancing_matrix(U,r,mu);c=np.sum(V*V,axis=0)
       h=r.sum()/mu;floor=(r.sum()-np.sort(c)[-k:].sum())/mu
       ev=eigh(K+mu*np.eye(256),U@U.T+mu*np.eye(256),eigvals_only=True)
       rows.append(dict(dataset=ds,seed=seed,kernel=kernel,rule=rule,n=256,k=k,mu=mu,
        trace_excess=float(h),top_rank_floor_excess=float(floor),actual_condition=float(ev[-1]/ev[0])))
  print(ds,'done',flush=True)
Path('results/sensitivity.json').write_text(json.dumps(rows,indent=2))
