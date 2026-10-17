"""Paired real-data kernel study; no model performance claims."""
import sys,json,time,hashlib,platform,os
from pathlib import Path
import numpy as np, scipy
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.sparse.linalg import cg,LinearOperator
from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import load_digits,load_breast_cancer,load_diabetes
from threadpoolctl import threadpool_limits,threadpool_info
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import relative_matrix,overlap_floor,pivoted_cholesky,sign_witness,cheap_upper,sdp_bracket

def pcg(A,P,b):
    c=cho_factor(P,lower=True)
    M=LinearOperator(A.shape,matvec=lambda x:cho_solve(c,x))
    count=[0]
    def cb(x): count[0]+=1
    t=time.perf_counter()
    x,status=cg(A,b,M=M,rtol=1e-9,atol=0.,maxiter=5000,callback=cb)
    elapsed=time.perf_counter()-t
    residual=np.linalg.norm(A@x-b)/np.linalg.norm(b)
    if status!=0 or residual>2e-8: raise RuntimeError((status,residual))
    return count[0],float(residual),elapsed

rows=[]; small=[]
start=time.perf_counter()
with threadpool_limits(limits=1):
 for dataset,loader in [('digits',load_digits),('breast_cancer',load_breast_cancer),('diabetes',load_diabetes)]:
  data=loader().data
  for n in (128,256,512):
   if n>len(data): continue
   for seed in range(5):
    rng=np.random.default_rng(1720406+seed+100*n)
    indices=rng.choice(len(data),n,replace=False)
    X=data[indices].copy()
    X=(X-X.mean(axis=0))/np.where(X.std(axis=0)>0,X.std(axis=0),1.)
    distances=squareform(pdist(X,'sqeuclidean'))
    bandwidth=np.median(distances[np.triu_indices(n,1)])
    K=np.exp(-distances/bandwidth)
    h=hashlib.sha256(K.tobytes()).hexdigest()
    F,pivots=pivoted_cholesky(K,32,rng)
    for k in (8,16,32):
     U=F[:,:k]
     R=K-U@U.T
     r=np.maximum(np.diag(R),0)
     # Exact model has zero landmark residual. Enforce those theoretical zeros.
     r[pivots[:k]]=0.
     psdmin=float(eigh(R,eigvals_only=True)[0])
     if psdmin < -1e-10: raise RuntimeError(psdmin)
     for mu in (.01,.1,1.):
      P=U@U.T+mu*np.eye(n); A=K+mu*np.eye(n)
      ts=time.perf_counter(); o=overlap_floor(U,r,mu); floor_time=time.perf_counter()-ts
      B=relative_matrix(P,r)
      ts=time.perf_counter(); lower,s=sign_witness(B,seed=seed,starts=8); sign_time=time.perf_counter()-ts
      upper=min(o['trace'],cheap_upper(B))
      ev=eigh(A,P,eigvals_only=True); actual=float(ev[-1]/ev[0])
      v=np.sqrt(r)*s; adversary=P+np.outer(v,v)
      adv_ev=eigh(adversary,P,eigvals_only=True)
      adv=float(adv_ev[-1]/adv_ev[0])
      if abs(adv-(1+lower))>1e-6*(1+lower): raise RuntimeError('witness identity')
      if actual>1+upper+1e-6: raise RuntimeError('upper violation')
      b=rng.standard_normal(n)
      it,rel,pcg_time=pcg(A,P,b)
      ait,arel,atime=pcg(adversary,P,b)
      row=dict(dataset=dataset,n=n,k=k,mu=mu,seed=seed,kernel_sha256=h,
       trace_excess=o['trace'],floor_excess=o['floor'],overlap=o['overlap'],
       effective_rank=o['effective_rank'],effective_support=float(r.sum()/r.max()),
       lower_excess=lower,cheap_upper_excess=upper,actual_condition=actual,
       adversarial_condition=adv,pcg_iterations=it,pcg_relres=rel,
       adversarial_pcg_iterations=ait,adversarial_relres=arel,
       floor_seconds=floor_time,sign_seconds=sign_time,pcg_seconds=pcg_time,
       residual_psd_min=psdmin)
      rows.append(row)
    print(dataset,n,seed,'done',flush=True)
 # Separate SDP study at small sizes. It is not included in the larger study.
 for dataset,loader in [('digits',load_digits),('breast_cancer',load_breast_cancer),('diabetes',load_diabetes)]:
  data=loader().data
  for seed in range(3):
   rng=np.random.default_rng(6000+seed)
   X=data[rng.choice(len(data),64,replace=False)].copy()
   X=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1.)
   dist=squareform(pdist(X,'sqeuclidean'))
   K=np.exp(-dist/np.median(dist[np.triu_indices(64,1)]))
   F,piv=pivoted_cholesky(K,16,rng)
   for k in (4,8,16):
    U=F[:,:k]; r=np.maximum(np.diag(K-U@U.T),0);r[piv[:k]]=0
    for mu in (.1,1.):
     P=U@U.T+mu*np.eye(64)
     active=r>0
     B=relative_matrix(P,r)[np.ix_(active,active)]
     ts=time.perf_counter(); q=sdp_bracket(B,seed); elapsed=time.perf_counter()-ts
     o=overlap_floor(U,r,mu)
     small.append(dict(dataset=dataset,seed=seed,n=64,k=k,mu=mu,
      trace_excess=o['trace'],floor_excess=o['floor'],lower_excess=q['lower'],
      sdp_upper_excess=q['upper'],raw_primal=q['raw_primal'],
      repaired_dual_min_eig=q['dual_min_eig'],seconds=elapsed,status=q['status']))
  print('SDP',dataset,'done',flush=True)
Path('results').mkdir(exist_ok=True)
Path('results/kernels.json').write_text(json.dumps(rows,indent=2))
Path('results/sdp.json').write_text(json.dumps(small,indent=2))
meta=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
 platform=platform.platform(),processor=platform.processor(),threads=1,
 total_seconds=time.perf_counter()-start,main_cases=len(rows),sdp_cases=len(small),
 threadpools=threadpool_info())
Path('results/environment.json').write_text(json.dumps(meta,indent=2))
print(meta)
