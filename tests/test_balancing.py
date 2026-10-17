import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import floating_sign_witness,relative_matrix,enumerate_signs,balancing_matrix
rng=np.random.default_rng(20406); cases=0; worst=0.
for n in (4,8,12,32,100):
 for k in (1,2,3):
  for trial in range(10):
   F=rng.normal(size=(n,k));r=rng.uniform(.01,1,n);mu=.1
   q=floating_sign_witness(F,r,mu,seed=trial)
   V=balancing_matrix(F,r,mu)
   B=relative_matrix(F@F.T+mu*np.eye(n),r)
   assert np.allclose(B,np.diag(r)/mu-V.T@V/mu)
   assert q['fractional_count']<=k
   assert q['penalty'] <= q['top_rank_penalty']+1e-7
   assert np.isclose(q['lower'],q['sign']@B@q['sign'],rtol=1e-10)
   if n<=12:
    f,s=enumerate_signs(B)
    assert f >= (r.sum()-q['top_rank_penalty'])/mu-1e-7
   worst=max(worst,q['lp_residual']);cases+=1
# Rank-one Nyström embedding of number partition: u=(1,2,3).
u=np.array([1.,2.,3.]);F=np.r_[1.,u][:,None];r=np.r_[0.,np.ones(3)]
B=relative_matrix(F@F.T+np.eye(4),r)
f,s=enumerate_signs(B)
assert np.isclose(f,3.)
# Tight top-rank barrier: nonzero mutually orthogonal columns of V.
F=np.zeros((8,3));F[:3,:]=np.diag([1.,2.,3.]);r=np.ones(8)
q=floating_sign_witness(F,r,1.)
assert np.isclose(q['penalty'],q['top_rank_penalty'])
out=dict(cases=cases,partition_and_tightness_cases=2,max_lp_residual=worst,status='passed')
Path('results/balancing_verification.json').write_text(json.dumps(out,indent=2));print(out)
