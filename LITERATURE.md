# Literature review and contribution boundary

Search date: 17 October 2026. The review used web search, primary publisher pages, author-hosted documents, and full-text arXiv versions. This is a focused prior-art search, not proof that no equivalent observation exists anywhere and not a patent clearance opinion.

## Topic selection

The initial candidate concerned reuse of preconditioners under diagonal Hessian reweighting. It was discarded as insufficiently differentiated: low-rank preconditioner updates, ratio-based index selection, and preconditioner recycling are well established. Relevant work included *Preconditioning Parametrized Linear Systems* (Grim-McNally, de Sturler, Gugercin, arXiv:1601.05883), surveys of low-rank preconditioner updates, and prior work on reweighted normal equations.

The retained question is the exact deterministic information content of a realized column-Nyström factor and its PSD residual diagonal. Prior author manuscripts on assignment-dual lattice fusion and periodic cellular-automaton experiment design have different mathematical objects, algorithms, and evidence; none supplies the results in this paper.

## Search themes

Queries covered: randomized Nyström preconditioning and adaptive residual estimation; preconditioner reuse and diagonal weight updates; worst-case PSD perturbations with prescribed diagonal; preconditioning and Max-Cut/Grothendieck; residual-diagonal stopping criteria; vector balancing and basic feasible solutions; current RPCholesky spectral guarantees. Direct searches for “Nyström diagonal impossibility preconditioner” and “preconditioner residual diagonal certificate” did not surface a paper with the same transcript-optimal characterization and top-rank residual-weighted barrier. Search absence is not an originality guarantee.

## Closest primary sources

| Source | Established contribution used here | Boundary of the present contribution |
| --- | --- | --- |
| Frangella, Tropp, Udell, [SIMAX 2023](https://doi.org/10.1137/21M1466244), full arXiv text inspected | Randomized Nyström PCG, effective dimension, adaptive error estimates using residual products | We condition on a fixed column transcript and do not get further residual products |
| Chen et al., [CPAM 2025](https://doi.org/10.1002/cpa.22234), full arXiv text inspected | RPCholesky, column Nyström equivalence, residual-diagonal sampling and trace stopping | The approximation method and trace stopping are not new; we characterize their realized information limit |
| Díaz et al., [arXiv:2304.12465v6](https://arxiv.org/abs/2304.12465), full text inspected | RPCholesky and KRILL preconditioning for KRR | No claim of a stronger randomized rank guarantee or faster KRR solver |
| Epperly, [arXiv:2608.20633](https://arxiv.org/abs/2608.20633), full text inspected | Improved expected and high-probability RPCholesky error guarantees, preconditioning corollary | Universal completion conditioning is a different, stronger requirement than a high-probability sampling guarantee |
| Abedsoltan et al., [AISTATS 2024](https://proceedings.mlr.press/v238/abedsoltan24a.html) | Approximation/overhead tradeoffs for spectral preconditioners | We do not analyze their rescaled spectral preconditioner |
| Rohn, [LMA 2000](https://doi.org/10.1080/03081080008818644), author survey inspected | NP-hard mixed norms and robust linear-algebra questions | Our rank-one column-Nyström Partition embedding is the specialized hardness statement |
| Nesterov, [OMS 1998](https://doi.org/10.1080/10556789808805690), publisher record and Nesterov–Wolkowicz–Ye treatment inspected | Classical PSD binary quadratic SDP and pi/2 approximation | The SDP, rounding, and constant are explicitly attributed rather than claimed as new |
| Gupta et al., [SODA 2022](https://arxiv.org/abs/2111.06308) | Basic feasible solution / near-integral signing principle in dynamic discrepancy | We apply the classical principle to the Woodbury balancing vectors and obtain the top-rank weighted-leverage barrier |

## What the paper claims

1. Exact optimal universal condition-number certificate over PSD residuals with prescribed diagonal.
2. Preservation of the full observed-column transcript by attaining rank-one completions.
3. Low-rank vector-balancing representation and the top-rank residual-weighted leverage barrier, including a tightness construction.
4. Exact-evaluation hardness at rank one, with a valid one-column Nyström embedding.
5. A concrete, tested information-acquisition protocol and evidence separating actual spectra, universal guarantees, and PCG work.

## What it does not claim

No new Nyström approximation algorithm, SDP relaxation, floating-variable technique, universal novelty certification, new deep-learning architecture, guaranteed acceptance, patentability, production-scale runtime advantage, or state-of-the-art predictive result. The mathematical article is suitable as material supporting an ICERM workshop participation/poster application; it has not been peer reviewed or accepted.
