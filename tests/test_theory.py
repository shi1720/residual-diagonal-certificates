"""Independent algebraic checks, exhaustive sign cases, and edge cases."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from diagcert import relative_matrix, enumerate_signs, overlap_floor, sdp_bracket

rng = np.random.default_rng(1720406)
checks = 0
max_witness_error = 0.
for n in range(2, 13):
    for trial in range(12):
        U = rng.normal(size=(n,n))
        P = U@U.T + .5*np.eye(n)
        r = rng.uniform(.01, 2., n)
        if trial % 3 == 0:
            r[0] = 0.
        B = relative_matrix(P, r)
        f, s = enumerate_signs(B)
        v = np.sqrt(r)*s
        vals = eigh(P+np.outer(v,v),P,eigvals_only=True)
        error = abs(vals[-1]/vals[0] - (1+f)) / (1+f)
        max_witness_error = max(max_witness_error, error)
        assert error < 1e-10
        # Independent random PSD completions with exactly the same diagonal.
        for rank in (1,2,n):
            Z = rng.normal(size=(n,rank))
            Z *= (np.sqrt(r)/np.linalg.norm(Z,axis=1))[:,None]
            ev = eigh(P+Z@Z.T,P,eigvals_only=True)
            assert ev[-1]/ev[0] <= 1+f+1e-9
            checks += 1
        assert f >= np.trace(B)-1e-9
        if trial == 0:
            bracket = sdp_bracket(B, seed=n)
            assert bracket['lower'] <= f+1e-7
            assert f <= bracket['upper']+1e-7
            assert bracket['upper'] <= np.pi/2*f + 1e-5
# Structured overlap identity and diagonal special case.
for n in (8,16,32):
    for k in (1,3,5):
        F = rng.normal(size=(n,k))
        r = rng.uniform(0,1,n)
        for mu in (.01,.1,1.):
            P=F@F.T+mu*np.eye(n)
            B=relative_matrix(P,r)
            o=overlap_floor(F,r,mu)
            assert np.isclose(o['floor'],np.trace(B),rtol=1e-10)
            assert o['overlap'] <= r.max()*k/r.sum()+1e-9
            checks+=1
for n in (2,5,10):
    p=rng.uniform(.1,3,n); r=rng.uniform(0,1,n)
    f,_=enumerate_signs(np.diag(r/p))
    assert np.isclose(f,np.sum(r/p))
    checks+=1
# Exact Max-Cut identity, integers, all 32768 cuts on 16 vertices.
n=16
W=rng.integers(0,4,(n,n)); W=np.triu(W,1); W=W+W.T
L=np.diag(W.sum(axis=1))-W
B=L+np.eye(n)
f,s=enumerate_signs(B)
cut=sum(int(W[i,j]) for i in range(n) for j in range(i+1,n) if s[i]!=s[j])
assert f==4*cut+n
checks+=1
out=dict(random_completion_checks=132*3, structured_and_other_checks=checks-396,
         exhaustive_instances=132, sign_dimensions=list(range(2,13)),
         max_relative_witness_error=max_witness_error, status='passed')
Path('results').mkdir(exist_ok=True)
Path('results/verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
