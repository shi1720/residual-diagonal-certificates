import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import relative_matrix,pivoted_cholesky,enumerate_signs,floating_sign_witness
checks=0
for A in [np.zeros((5,5)),np.ones((5,5)),np.diag([1.,0,0,0,0])]:
 F,p=pivoted_cholesky(A,4,np.random.default_rng(1))
 assert np.all(np.isfinite(F))
 assert np.linalg.norm(A-F@F.T)<1e-10
 checks+=1
B=relative_matrix(np.eye(3),np.zeros(3));f,s=enumerate_signs(B)
assert f==0;checks+=1
try:relative_matrix(np.eye(2),[-1.,1.])
except ValueError:checks+=1
else:raise AssertionError('negative diagonal accepted')
F=np.zeros((5,2));r=np.zeros(5)
q=floating_sign_witness(F,r,1.)
assert q['lower']==0 and q['penalty']==0;checks+=1
Path('results/edge_cases.json').write_text(json.dumps(dict(checks=checks,status='passed'),indent=2))
print(checks,'edge cases passed')
