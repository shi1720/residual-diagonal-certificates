"""Six-dimensional teaching example; separate from the manuscript benchmarks.

Exact Fraction arithmetic establishes the sign optimum for this example.
NumPy computes illustrative spectra; this is not an interval eigensolver.
Run with Python and NumPy. No data download is required.
"""
from fractions import Fraction as Q
from itertools import product
import json
import numpy as np


def main():
    n, mu = 6, Q(1, 4)
    roots = [Q(0), Q(1, 5), Q(2, 5), Q(3, 5), Q(4, 5), Q(1)]
    residual_diagonal = [x*x for x in roots]
    # P = 11^T + mu I; its inverse follows from Sherman--Morrison.
    # The zero first residual row preserves the observed Nyström column.
    choices = []
    for tail in product((-1, 1), repeat=n-2):
        signs = (1, 1, *tail)  # Fix global sign at the first nonzero coordinate.
        v = [a*s for a, s in zip(roots, signs)]
        objective = sum(x*x for x in v)/mu - sum(v)**2/(mu*(mu+n))
        choices.append((objective, signs))
    optimum, signs = max(choices)
    exact_worst = 1 + optimum
    trace = 1 + sum(residual_diagonal)/mu
    leverage = Q(1)/(n+mu)
    top_rank_barrier = 1 + (sum(residual_diagonal)
                              - max(residual_diagonal)*leverage)/mu
    assert len(choices) == 16
    assert exact_worst == Q(6109, 625)
    assert top_rank_barrier <= exact_worst <= trace
    F = np.ones((n, 1))
    P = F@F.T + float(mu)*np.eye(n)
    r = np.array([float(x) for x in residual_diagonal])
    v = np.array([float(x)*s for x, s in zip(roots, signs)])
    R_diagonal, R_worst = np.diag(r), np.outer(v, v)
    Linv = np.linalg.solve(np.linalg.cholesky(P), np.eye(n))

    def condition(R):
        values = np.linalg.eigvalsh(np.eye(n) + Linv@R@Linv.T)
        return float(values[-1]/values[0])

    assert np.allclose(np.diag(R_diagonal), np.diag(R_worst))
    assert np.allclose((F@F.T+R_diagonal)[:, 0],
                       (F@F.T+R_worst)[:, 0])
    observed, adversarial = condition(R_diagonal), condition(R_worst)
    assert abs(adversarial-float(exact_worst)) < 1e-10
    assert observed < float(top_rank_barrier)  # The barrier concerns certificates.
    return dict(example='Six-dimensional teaching scenario, not kernel benchmark',
                unique_rank_one_completions=len(choices),
                exact_worst_condition=str(exact_worst),
                worst_condition_numeric=adversarial,
                diagonal_residual_condition=observed,
                top_rank_barrier=str(top_rank_barrier), trace_upper=str(trace),
                same_residual_diagonal=True, same_observed_nystrom_column=True,
                caveat='Condition-number information is not a PCG iteration count.')


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
