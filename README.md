# Ensemble semi-Markov framework

Implements the Methods section of *Beyond a Single Electric Vehicle Battery
Life: A Semi-Markov Framework for Stochastic Circular Pathways* (Sec. 2.2–2.7):
an absorbing semi-Markov chain over the EV-battery lifecycle
`{M, U1, U2, RM, RC, RO, L}`, evaluated across a large Dirichlet-sampled
ensemble of feasible transition matrices rather than one assumed point estimate.

## Files

| File | Purpose |
|---|---|
| [lifecycle.py](lifecycle.py) | Topology, Dirichlet sampling, and the analytics (N, B, ν, ρ(Q), V, τ, F). No informed/point-estimate prior — every system is Dirichlet(α) over the admissible topology. |
| [run_ensemble.py](run_ensemble.py) | Generates the base ensemble (α=1) plus α-sensitivity runs, checks convergence, computes drivers (SRRC) and the λ_G/λ_H sweeps, writes everything to `results/`. |
| [plot_results.py](plot_results.py) | Five figures from `results/` into `results/figures/`. |
| [test_lifecycle.py](test_lifecycle.py) | Unit tests: a hand-derivable deterministic case, ensemble-wide algebraic identities, reproducibility, and error handling. |

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest test_lifecycle.py -v   # unit tests
python lifecycle.py                     # analytic-vs-Monte-Carlo self-check
python run_ensemble.py                  # ~1 min, writes results/
python plot_results.py                  # writes results/figures/
```

## Notation (matches the manuscript exactly)

- **Topology (Sec. 2.6):** `M→U1` fixed at 1; `U1→{U2,RM,RC,L}`; `U2→{RC,L}`;
  `RM→{U1,U2,RC,L}`; `RC→{M,RO,L}`. Everything else is a structural zero.
- **Sampling (Sec. 2.6):** each row is `Dirichlet(α_i)` with `α_i = 1_{K_i}`
  (the manuscript's notation for the all-ones vector of length K_i) — i.e.
  α=1 for every component, which is exactly uniform over the simplex. The
  base ensemble uses this. α=0.5/2/5 are additional sensitivity runs, not
  alternative "informed" priors.
- **λ_H = μ_U2/μ_U1, λ_G = g_U2/g_U1 (Sec. 2.7):** normalized second-life
  duration/service, with μ_U1 = g_U1 = 1 as the first-life reference. These
  are `lambda_H`/`lambda_G` in the code (older `r_T`/`r_E` names were renamed
  to match the manuscript).

## Outputs (`results/`)

- `ensemble_<prior>.csv.gz` — one row per sampled system: every sampled
  probability, the full `N_M,*` row, `B_M,RO`/`B_M,L`, `ρ(Q)`, `V_M`, `τ_M`,
  `service_ratio`, and every `F_i→j(M)`.
- `summary_by_prior.csv` — mean/quantiles of the key outputs, per prior.
- `summary_report.md` — the same, as a paste-ready markdown table for Results.
- `convergence_base.csv` — quantiles vs. ensemble size (base prior); this is
  what Sec. 2.6's "convergence assessment" refers to.
- `drivers_alpha_1.csv` — Spearman + standardized rank regression coefficients
  (SRRC) of each key output on each sampled transition probability, baseline
  = probability to L (dropped from the regression since every row sums to 1).
  Outputs: N_M,M, N_M,U2, B_M,RO, B_M,L, V_M, service_ratio (V*), ρ(Q), F_RC→M, τ_M.
- `srrc_r2_alpha_1.csv` / `.md` — R² of each outcome's rank-regression model
  (supplement table): how well the SRRCs describe that outcome.
- `service_sweep_base.csv` — V_M/service_ratio vs λ_G, τ_M vs λ_H, recomputed
  exactly from stored visit counts (no resampling needed).
- `manifest.json` — seed, n, α/λ values, package versions, timestamp, for
  reproducibility.
- `figures/` — all figures (600 dpi PNG); see the [plot_results.py](plot_results.py) header.

## Known redundancies (not bugs — algebraic identities, kept for legibility)

- `F_RC_M == N_M_M - 1` exactly (the only path back to M is through RC→M).
- `service_ratio` is `V_M` rescaled by a constant.
- `B_M_RO + B_M_L == 1` exactly (the two columns of one absorption row).

`test_lifecycle.py` asserts these hold across the ensemble; if they ever break,
something is wrong.

## Using your own data instead of the Dirichlet ensemble

`build_frame` doesn't care where the transition matrix came from — hand it any
valid 7×7 matrix (built from field data, a degradation model, or a technology
assessment) instead of a sampled one:

```python
import numpy as np
from lifecycle import build_frame, validate_matrices, STATES, IDX

P = np.zeros((len(STATES), len(STATES)))
# ... fill in your own transition probabilities, respecting the topology ...
validate_matrices(P[None])          # checks row sums, nonnegativity, topology
df = build_frame(P[None], lambda_G=..., lambda_H=...)
```

## What this does *not* do

No degradation model is implemented here. `λ_H`/`λ_G` are user-supplied
ratios, not derived from a capacity-fade curve — Sec. 2.7 deliberately
normalizes the ensemble to be independent of any specific chemistry or
degradation dataset. If a worked example anchored to real degradation physics
(e.g., the Nájera et al. 2023 model referenced in Supplement S.2) is wanted
later, that belongs in a separate `degradation.py`, not here.

## Reproducing the paper's results

```bash
pip install -r requirements.txt     # pinned versions used for the paper
python -m pytest test_lifecycle.py  # 16 unit tests
python run_ensemble.py              # seed 2026, 100,000 systems per prior (~1 min); writes results/
python plot_results.py              # writes results/figures/ (600 dpi)
```

The four ensemble files (`results/ensemble_alpha_*.csv.gz`, about 33 MB each) are **not** stored in this
repository: they are regenerated exactly by `run_ensemble.py` from the fixed seed (see `results/manifest.json`).
The small summary tables (`summary_by_prior.csv`, `drivers_alpha_1.csv`, `srrc_r2_alpha_1.csv`,
`service_sweep_base.csv`, `convergence_base.csv`) and all figures are included for convenience.

## Citation

Lygizos, A., Keroglou, I., Srinivasan, L., Nath, C., Sutherland, J.W., Zhao, F. (in preparation).
*Beyond a Single Electric Vehicle Battery Life: A Semi-Markov Framework for Stochastic Circular Pathways.*

## License

MIT (see `LICENSE`).
