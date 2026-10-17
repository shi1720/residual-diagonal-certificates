"""Residual-diagonal uncertainty for SPD preconditioners (real arithmetic)."""
import itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


def relative_matrix(P, diagonal):
    """D P^{-1} D; zero diagonal coordinates are retained."""
    r = np.asarray(diagonal, dtype=float)
    if np.min(r) < 0:
        raise ValueError('Residual diagonal must be nonnegative')
    D = np.sqrt(r)
    Z = cho_solve(cho_factor(P, lower=True), np.eye(len(r)))
    B = D[:, None] * Z * D[None, :]
    return (B + B.T) / 2


def enumerate_signs(B):
    """Exact search family; floating-point objectives, global sign fixed."""
    n = len(B)
    if n > 22:
        raise ValueError('Exhaustive enumeration capped at 22 variables')
    best, sign = -np.inf, None
    for rest in itertools.product((-1., 1.), repeat=n-1):
        s = np.array((1.,) + rest)
        v = float(s @ B @ s)
        if v > best:
            best, sign = v, s.copy()
    return best, sign


def sign_witness(B, seed=0, starts=32):
    """Deterministic seeded lower bound with monotone coordinate ascent."""
    rng = np.random.default_rng(seed)
    _, V = eigh(B, subset_by_index=[len(B)-1, len(B)-1])
    candidates = [np.ones(len(B)), np.where(V[:, 0] >= 0, 1., -1.)]
    candidates += [rng.choice([-1., 1.], len(B)) for _ in range(starts)]
    best, best_s = -np.inf, None
    for s in candidates:
        q = B @ s
        for sweep in range(100):
            changed = False
            for i in range(len(s)):
                # Exclude diagonal, whose contribution cannot change.
                if s[i] * (q[i] - B[i, i] * s[i]) < -1e-13:
                    old = s[i]
                    s[i] = -old
                    q -= 2 * old * B[:, i]
                    changed = True
            if not changed:
                break
        value = float(s @ B @ s)
        if value > best:
            best, best_s = value, s.copy()
    return best, best_s


def cheap_upper(B):
    m = np.count_nonzero(np.diag(B) > 0)
    return min(float(np.abs(B).sum()), m * float(eigh(B, eigvals_only=True)[-1]))


def sdp_bracket(B, seed=0, rounds=512):
    """Classical PSD Grothendieck SDP; numerically repaired dual upper bound.

    This is a floating-point implementation, not interval arithmetic.
    Returned signs are always admissible adversarial residual witnesses.
    """
    import cvxpy as cp
    n = len(B)
    C = cp.Variable((n, n), symmetric=True)
    diag_constraint = cp.diag(C) == 1
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(B, C))),
                      [C >> 0, diag_constraint])
    scale = max(float(np.linalg.norm(B, 2)), 1e-30)
    # Rescale objective for a stable solver tolerance across regularizers.
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(B / scale, C))),
                      [C >> 0, diag_constraint])
    prob.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_feas=1e-8,
               tol_gap_rel=1e-8, max_iter=200)
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(prob.status)
    y = np.asarray(diag_constraint.dual_value) * scale
    M = np.diag(y) - B
    mineig = float(eigh(M, eigvals_only=True)[0])
    shift = max(0., -mineig) + 1e-10 * max(1., np.linalg.norm(B, 'fro'))
    y += shift
    repaired_min = float(eigh(np.diag(y)-B, eigvals_only=True)[0])
    evals, V = eigh((C.value + C.value.T) / 2)
    F = V * np.sqrt(np.maximum(evals, 0))[None, :]
    rng = np.random.default_rng(seed)
    S = np.where(F @ rng.standard_normal((n, rounds)) >= 0, 1., -1.)
    scores = np.einsum('ij,ij->j', S, B @ S)
    i = int(np.argmax(scores))
    lower, s = sign_witness(B, seed=seed)
    if scores[i] > lower:
        lower, s = float(scores[i]), S[:, i]
    return dict(lower=lower, upper=float(y.sum()), sign=s,
                dual=y, dual_min_eig=repaired_min,
                raw_primal=float(prob.value*scale), status=prob.status)


def overlap_floor(F, r, mu):
    """O(n k^2) exact-arithmetic floor; does not construct a dense inverse."""
    G = F.T @ F + mu * np.eye(F.shape[1])
    W = cho_solve(cho_factor(G, lower=True), F.T).T
    leverage = np.einsum('ij,ij->i', F, W)
    trace_bound = float(np.sum(r) / mu)
    overlap = float(r @ leverage / np.sum(r)) if np.sum(r) > 0 else 0.
    return dict(trace=trace_bound, floor=trace_bound*(1-overlap),
                overlap=overlap, effective_rank=float(np.sum(leverage)))


def pivoted_cholesky(A, rank, rng):
    """Unaccelerated RPCholesky, residual-diagonal sampling (Chen et al.)."""
    n = len(A)
    F = np.zeros((n, rank))
    r = np.diag(A).copy()
    pivots = []
    for j in range(rank):
        if r.sum() <= 1e-12 * max(1., float(np.trace(A))):
            break
        p = int(rng.choice(n, p=r/r.sum()))
        v = A[:, p] - F[:, :j] @ F[p, :j]
        F[:, j] = v / np.sqrt(r[p])
        pivots.append(p)
        r = np.maximum(r-F[:, j]**2, 0.)
        r[pivots] = 0.
    return F, pivots


def balancing_matrix(F, r, mu):
    """V with B = diag(r)/mu - V.T@V/mu."""
    from scipy.linalg import cholesky, solve_triangular
    G = F.T @ F + mu * np.eye(F.shape[1])
    return solve_triangular(cholesky(G, lower=True), F.T*np.sqrt(r)[None,:], lower=True)


def floating_sign_witness(F, r, mu, seed=0):
    """Classical floating-variable rounding, applied to residual uncertainty.

    Solve a bounded LP to find a vertex of {x: Vx=0, |x_i|<=1},
    then round remaining fractional coordinates by conditional expectation.
    Returns an actual sign witness; LP inaccuracies cannot invalidate its
    admissibility, but may prevent achieving the analytic existence bound.
    """
    from scipy.optimize import linprog
    V=balancing_matrix(F,r,mu)
    rng=np.random.default_rng(seed)
    sol=linprog(rng.standard_normal(len(r)),A_eq=V,b_eq=np.zeros(len(V)),
                bounds=[(-1.,1.)]*len(r),method='highs')
    if not sol.success: raise RuntimeError(sol.message)
    x=np.clip(sol.x,-1.,1.)
    # Snap near-boundary LP entries, retaining actual final objective checks.
    x[np.abs(x)>1-1e-8]=np.sign(x[np.abs(x)>1-1e-8])
    fractional=np.flatnonzero(np.abs(x)<1)
    z=V@x
    lp_residual=float(np.linalg.norm(z))
    for i in fractional:
        plus=z+(1-x[i])*V[:,i]
        minus=z+(-1-x[i])*V[:,i]
        chosen=1. if plus@plus <= minus@minus else -1.
        z += (chosen-x[i])*V[:,i]
        x[i]=chosen
    s=np.where(x>=0,1.,-1.)
    penalty=float(np.linalg.norm(V@s)**2)
    colnorm=np.sum(V*V,axis=0)
    top=float(np.sort(colnorm)[-min(F.shape[1],len(r)):].sum())
    return dict(lower=float((np.sum(r)-penalty)/mu),sign=s,
                penalty=penalty,top_rank_penalty=top,
                fractional_count=len(fractional),lp_residual=lp_residual)
