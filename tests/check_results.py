"""Validate retained results and recompute reported summary quantities."""
from pathlib import Path
import json
import numpy as np
p=Path('results');a=json.loads((p/'kernels.json').read_text());s=json.loads((p/'sdp.json').read_text());g=json.loads((p/'scaling.json').read_text());z=json.loads((p/'sensitivity.json').read_text())
assert (len(a),len(s),len(g),len(z))==(360,54,36,216)
for r in a:
 T=r['trace_excess'];L=r['top_rank_floor_excess']
 assert -1e-7<=r['floor_excess']<=L+1e-7<=T+2e-7
 assert L<=r['floating_lower_excess']+1e-6
 assert r['floating_lower_excess']<=T+1e-6
 assert r['lower_excess']<=r['cheap_upper_excess']+1e-6
 assert r['actual_condition']<=1+r['cheap_upper_excess']+1e-6
 assert abs(r['adversarial_condition']-(1+r['lower_excess']))<1e-6*(1+T)
 assert r['pcg_relres']<2e-8 and r['adversarial_relres']<2e-8
 assert r['adversarial_pcg_iterations']==2
for r in s:
 assert r['repaired_dual_min_eig']>=-1e-7
 assert r['lower_excess']<=r['sdp_upper_excess']+1e-6
for r in g:
 assert r['top_rank_floor_excess']<=r['floating_lower_excess']+1e-6
 assert r['floating_lower_excess']<=r['trace_excess']+1e-6
for target,expected in [(10,(107,107,0)),(100,(237,166,71)),(1000,(347,156,191))]:
 actual=sum(r['actual_condition']<=target for r in a)
 blocked=sum(r['actual_condition']<=target and 1+r['top_rank_floor_excess']>target for r in a)
 accepted=sum(1+r['trace_excess']<=target for r in a)
 assert (actual,blocked,accepted)==expected
out=dict(main_cases=len(a),sdp_cases=len(s),scaling_cases=len(g),sensitivity_cases=len(z),status='passed')
(p/'result_validation.json').write_text(json.dumps(out,indent=2));print(out)
