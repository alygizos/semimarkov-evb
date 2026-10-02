"""Generate and analyse the ensemble of feasible lifecycle systems.

Runs the base ensemble (symmetric Dirichlet, alpha = 1) plus alpha sensitivity runs
-- no informed/point-estimate prior; every system is a Dirichlet draw over the
admissible topology (Sec. 2.6) -- then writes everything to ./results:
    ensemble_<prior>.csv.gz     one row per sampled system
    summary_by_prior.csv        mean / quantiles of key outputs per prior
    convergence_base.csv        running quantiles vs ensemble size (base prior)
    drivers_<base>.csv          Spearman and SRRC of outputs on transition probabilities
    srrc_r2_<base>.csv/.md      R^2 of each outcome's rank-regression model (supplement table)
    service_sweep_base.csv      V_M/service_ratio vs lambda_G and tau_M vs lambda_H (base prior)
    summary_report.md           plain-language table of key outputs, ready to paste
                                 into Results (median [5-95%])
    manifest.json               run parameters, package versions, timestamp (reproducibility)

The results/ directory is cleared of stale ensemble_*.csv.gz before each run, so
plot_results.py never mixes outputs from a previous --alphas selection.

Usage:  python run_ensemble.py [--n 100000] [--seed 2026] [--out results]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lifecycle import (DRIVER_OUTPUTS, KEY_OUTPUTS, PARAM_NAMES, build_frame,
                       sample_transition_matrices, validate_matrices)

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
BASE = "alpha_1"


def run_prior(n, seed_seq, lambda_G, lambda_H, alpha=1.0):
    rng = np.random.default_rng(seed_seq)
    P = sample_transition_matrices(n, rng, alpha=alpha)
    validate_matrices(P)
    return build_frame(P, lambda_G=lambda_G, lambda_H=lambda_H)


def summarize(frames):
    rows = []
    for name, df in frames.items():
        for c in KEY_OUTPUTS:
            q = df[c].quantile(QUANTILES)
            rows.append({"prior": name, "output": c, "mean": df[c].mean(),
                         **{f"q{round(k * 100):02d}": q[k] for k in QUANTILES}})
    return pd.DataFrame(rows)


def convergence_table(df, cols, n_sizes=8, start=1000, qs=(0.05, 0.5, 0.95)):
    """Quantiles of each output over growing prefixes (rows are i.i.d., so prefixes are valid)."""
    sizes = np.unique(np.geomspace(min(start, len(df)), len(df), n_sizes).astype(int))
    rows = []
    for s in sizes:
        q = df.iloc[:s][cols].quantile(list(qs))
        for c in cols:
            for k in qs:
                rows.append({"n": int(s), "output": c, "q": k, "value": q.loc[k, c]})
    return pd.DataFrame(rows)


def convergence_report(conv, tol):
    """Max change of any tracked quantile across the last two size increments, expressed
    as a fraction of that output's 5-95% spread at the final size. (Relative to the
    quantile itself would be dominated by tiny tail values, e.g. a 5th percentile near 0.)"""
    pv = conv.pivot_table(index=["output", "q"], columns="n", values="value")
    sizes = pv.columns.to_list()
    if len(sizes) < 3:
        raise ValueError("convergence_report needs at least 3 distinct ensemble sizes")
    spread = (pv[sizes[-1]].xs(0.95, level="q") - pv[sizes[-1]].xs(0.05, level="q")).clip(lower=1e-6)
    scale = spread.reindex(pv.index.get_level_values("output")).to_numpy()

    def rel(a, b):
        return (pv[b] - pv[a]).abs() / scale

    worst = np.maximum(rel(sizes[-3], sizes[-2]), rel(sizes[-2], sizes[-1]))
    per_output = worst.groupby(level="output").max().sort_values(ascending=False)
    return per_output, bool((worst < tol).all())


def driver_table(df, outputs=DRIVER_OUTPUTS):
    """Spearman correlation and standardised rank regression coefficients (SRRC).

    Within each row the sampled probabilities sum to 1, so the regression drops the
    probability to L in every row: an SRRC reads as the effect of shifting routing
    probability toward that transition at the expense of loss.
    """
    ranks = df[PARAM_NAMES + outputs].rank()
    spearman = ranks.corr().loc[outputs, PARAM_NAMES]
    predictors = [p for p in PARAM_NAMES if not p.endswith("_L")]
    Z = (ranks - ranks.mean()) / ranks.std()
    X = Z[predictors].to_numpy()
    rows = []
    for out in outputs:
        y = Z[out].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - (y - X @ beta).var() / y.var()
        for p in PARAM_NAMES:
            rows.append({"output": out, "param": p, "spearman": spearman.loc[out, p],
                         "srrc": beta[predictors.index(p)] if p in predictors else np.nan,
                         "r2_srrc_model": r2})
    return pd.DataFrame(rows)


# Outputs reported in the SRRC heatmap (Fig. 1 order). F_RC->M, B_M,L and the unnormalised
# V_M are omitted: they are exact rescalings/mirrors of N_M,M, B_M,RO and V*, so their
# rank-regression R^2 values and SRRCs are identical.
R2_ROWS = [("N_M_M", "N_M,M"), ("N_M_U2", "N_M,U2"), ("B_M_RO", "B_M,RO"),
           ("V_M", "V*_M"), ("rho", "rho(Q)"), ("tau_M", "tau_M")]


def write_r2_table(drivers, out, n):
    """R^2 of the rank regression behind each outcome's SRRCs (supplement table).
    SRRCs are most informative when the ranked input-output relationship is well described
    by the regression; a low R^2 means interactions/non-monotone effects carry the rest."""
    r2 = drivers.groupby("output")["r2_srrc_model"].first()
    tab = pd.DataFrame({"outcome": [lab for _, lab in R2_ROWS],
                        "output_column": [k for k, _ in R2_ROWS],
                        "rank_regression_R2": [r2[k] for k, _ in R2_ROWS]})
    tab.to_csv(out / f"srrc_r2_{BASE}.csv", index=False)
    lines = [f"Rank-regression R^2 of the SRRC models (base ensemble, n = {n:,}; nine predictors).", "",
             "| Outcome | Rank-regression R^2 |", "|---|---|"]
    lines += [f"| {o} | {v:.3f} |" for o, v in zip(tab["outcome"], tab["rank_regression_R2"])]
    (out / f"srrc_r2_{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tab


def service_sweep(df, lambda_G_values, lambda_H_values, mu_other=(0.0, 0.0, 0.0)):
    """The reward/time ratios do not change N, so V_M and tau_M are recomputed exactly
    from the stored expected visits for any (lambda_G, lambda_H)."""
    rows = []
    for lam in lambda_G_values:
        V = df["N_M_U1"] + lam * df["N_M_U2"]
        for metric, x in (("V_M", V), ("service_ratio", V / (1 + lam))):
            q = x.quantile(QUANTILES)
            rows.append({"metric": metric, "param": "lambda_G", "value": lam, "mean": x.mean(),
                         **{f"q{round(k * 100):02d}": q[k] for k in QUANTILES}})
    mu_M, mu_RM, mu_RC = mu_other
    for lam in lambda_H_values:
        tau = (mu_M * df["N_M_M"] + df["N_M_U1"] + lam * df["N_M_U2"]
               + mu_RM * df["N_M_RM"] + mu_RC * df["N_M_RC"])
        q = tau.quantile(QUANTILES)
        rows.append({"metric": "tau_M", "param": "lambda_H", "value": lam, "mean": tau.mean(),
                     **{f"q{round(k * 100):02d}": q[k] for k in QUANTILES}})
    return pd.DataFrame(rows)


def write_summary_report(path, base_summary, n, tol_ok):
    s = base_summary.set_index("output")
    lines = ["# Base ensemble summary (symmetric Dirichlet, alpha = 1)", "",
             f"n = {n:,} systems. Convergence at 2% tolerance: "
             f"{'PASSED' if tol_ok else 'NOT MET -- rerun with larger --n'}.", "",
             "| Output | Median | 5% | 95% |", "|---|---|---|---|"]
    for c in KEY_OUTPUTS:
        lines.append(f"| {c} | {s.loc[c, 'q50']:.4g} | {s.loc[c, 'q05']:.4g} | {s.loc[c, 'q95']:.4g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_stale_outputs(out, keep_names):
    """Remove ensemble_*.csv.gz files from a previous run that no longer correspond
    to the current --alphas selection, so plot_results.py (which globs ensemble_*.csv.gz)
    never mixes stale priors into fig2_prior_comparison.png."""
    keep = {f"ensemble_{n}.csv.gz" for n in keep_names}
    for f in out.glob("ensemble_*.csv.gz"):
        if f.name not in keep:
            print(f"  removing stale {f.name}")
            f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="systems per prior")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 2.0, 5.0],
                    help="extra symmetric concentrations for the sensitivity runs")
    ap.add_argument("--lambda-g", type=float, default=0.5, help="g_U2/g_U1 for row-level V_M")
    ap.add_argument("--lambda-h", type=float, default=1.0, help="mu_U2/mu_U1 for row-level tau_M")
    ap.add_argument("--sweep-lambda-g", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--sweep-lambda-h", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--tol", type=float, default=0.02, help="convergence tolerance (relative)")
    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError(f"--n must be positive, got {args.n}")
    if any(a <= 0 for a in args.alphas):
        raise ValueError(f"--alphas must all be positive, got {args.alphas}")
    if args.lambda_g <= 0 or args.lambda_h <= 0 or any(v <= 0 for v in args.sweep_lambda_g + args.sweep_lambda_h):
        raise ValueError("lambda_G/lambda_H values must all be positive")

    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)

    priors = [(BASE, dict(alpha=1.0))]
    priors += [(f"alpha_{a:g}", dict(alpha=a)) for a in args.alphas]
    clean_stale_outputs(out, [name for name, _ in priors])
    seeds = np.random.SeedSequence(args.seed).spawn(len(priors))

    frames = {}
    for (name, kw), ss in zip(priors, seeds):
        print(f"Sampling {args.n:,} systems: {name}")
        frames[name] = run_prior(args.n, ss, args.lambda_g, args.lambda_h, **kw)
        frames[name].to_csv(out / f"ensemble_{name}.csv.gz", index=False)

    base_summary_all = summarize(frames)
    base_summary_all.to_csv(out / "summary_by_prior.csv", index=False)

    base = frames[BASE]
    conv = convergence_table(base, KEY_OUTPUTS)
    conv.to_csv(out / "convergence_base.csv", index=False)
    per_output, ok = convergence_report(conv, args.tol)
    print(f"\nConvergence (base prior, tol {args.tol:.0%}): "
          f"{'PASSED' if ok else 'NOT met -> increase --n'}")
    print("Max change of tracked quantiles (fraction of 5-95% spread) over the last two size increments:")
    print(per_output.round(4).to_string())

    drivers = driver_table(base)
    drivers.to_csv(out / f"drivers_{BASE}.csv", index=False)
    write_r2_table(drivers, out, args.n)
    service_sweep(base, args.sweep_lambda_g, args.sweep_lambda_h).to_csv(out / "service_sweep_base.csv", index=False)

    base_summary = summarize({BASE: base})
    write_summary_report(out / "summary_report.md", base_summary, args.n, ok)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "priors": [name for name, _ in priors],
        "n_per_prior": args.n,
        "convergence_ok": ok,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    s = base_summary.set_index("output")
    print("\nBase ensemble, key outputs (median [5%-95%]):")
    for c in KEY_OUTPUTS:
        print(f"  {c:>14}: {s.loc[c, 'q50']:.4g}  [{s.loc[c, 'q05']:.4g} - {s.loc[c, 'q95']:.4g}]")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
