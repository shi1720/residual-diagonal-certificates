# Residual-diagonal certificates for Nyström preconditioners

Research artifact for **Diagonal Certificates for Nyström Preconditioners: Exact Worst Cases and a Vector-Balancing Barrier**, Shivam Gupta, October 2026.

The paper asks what a known factor and residual diagonal can certify **universally over PSD completions**. It proves an exact sign-optimization characterization and a top-rank vector-balancing barrier. It is not a new Nyström approximation algorithm or a claim of faster kernel training.

- [Paper PDF](paper/Diagonal_Certificates_Shivam_Gupta.pdf)
- [LaTeX source](paper/main.tex)
- [Numerical core](src/diagcert.py)
- [Literature and novelty boundary](LITERATURE.md)
- [Retained results](results/)

## Reproduce

Python 3.9 or newer; the retained run used Python 3.9.6. Work from this repository's root. The bundled scikit-learn datasets need no external download. They are used for kernel geometry; labels are not used.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python tests/test_theory.py
python tests/test_balancing.py
python tests/test_edge_cases.py
python experiments/run_kernels.py
python experiments/augment_balancing.py
python experiments/kernel_sensitivity.py
python tests/check_results.py
python experiments/make_figures.py
```

`augment_balancing.py` runs after `run_kernels.py`: it augments its retained main records and creates the larger factor study. `make_figures.py` reads retained records; it does not refit or select favorable cases. Its generated table row files are also retained; the manuscript includes those values explicitly.

Compile the paper with the official [Tectonic](https://tectonic-typesetting.github.io/) distribution:

```sh
cd paper
tectonic main.tex
```

The rendered PDF will be `main.pdf`. The distributed named PDF is the visually reviewed copy. No publisher template or proceedings submission format is prescribed by the ICERM participation form; the manuscript uses a self-contained mathematical article format.

## Results inventory

| File | Scope |
| --- | --- |
| `results/verification.json` | 132 exhaustive sign instances; 396 additional PSD-completion checks; 31 structured checks |
| `results/balancing_verification.json` | 150 balancing checks and two regression examples |
| `results/edge_cases.json` | Input validation and zero/rank-deficient factor cases |
| `results/kernels.json` | 360 real-data kernel systems, including analytic barriers, witnesses, actual spectra, and PCG residuals |
| `results/sdp.json` | 54 small kernel SDP brackets and repaired dual minimum eigenvalues |
| `results/scaling.json` | 36 larger Digits factor-based audits, n=1024 and 1797 |
| `results/sensitivity.json` | 216 kernel-family and pivot-rule sensitivity cases |
| `results/environment.json` | Main run's software, platform, thread libraries, and total time |

Twenty-four sensitivity cases have essentially zero linear-kernel residuals; ratios with trace excess at most 1e-8 are excluded from ratio summaries. All records remain available.

## Interpret the guarantees correctly

`K(P,r)` is the **smallest possible universal upper certificate**. A lower bound on it is not a lower bound on the actual condition number. A sign witness constructs an admissible PSD residual with the recorded diagonal, preserving observed landmark columns. This residual need not be an RBF kernel on the original input coordinates.

The top-rank barrier is proved in real arithmetic. The implementation uses double precision, numerical LP solutions, and numerical PSD checks; it is not an interval-verified certificate library. A primal feasible SDP objective is a lower bound on the SDP maximum, not an upper certificate. The implementation obtains and numerically repairs the **dual** for the upper bracket.

A worst condition number is not a worst PCG iteration count. The rank-one extremizers in the main study take two iterations because their preconditioned matrices have only two distinct eigenvalues. The artifact does not claim production-scale speedups, GPU performance, predictive-model accuracy improvements, patentability, or peer-reviewed acceptance.

## Data and attribution

RPCholesky follows the residual-diagonal sampling rule in Chen, Epperly, Tropp, and Webber (CPAM, 2025). The quadratic SDP and Gaussian rounding are classical (Nesterov, 1998). Floating variables, basic feasible solutions, and conditional expectation are established discrepancy/rounding techniques. Full citations and proofs are in the paper.

Digits, Breast Cancer Wisconsin Diagnostic, and Diabetes are distributed with scikit-learn. See its dataset documentation and the paper's references for original provenance. Raw datasets are not copied into this repository.

Code is licensed under MIT. The manuscript and original figures are licensed under CC BY 4.0. Retained numeric result files are licensed under CC BY 4.0. Third-party datasets and cited works retain their own terms.
