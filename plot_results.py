"""Figures from the files written by run_ensemble.py (saved to results/figures).

    fig1_distributions.png     base-ensemble distributions: N_M,M, N_M,U2, P(RO), V*, rho(Q), tau_M
    fig2_prior_comparison.png  ECDFs across the Dirichlet priors
    fig3_drivers.png           SRRC of outputs on transition probabilities
    fig4_relationships.png     persistence, terminal fate and closed-loop vs service
    fig5_service_sweeps.png    hexagon density of V_M and V* vs lambda_G (median line; V* = 1 marked)
    figS_duration_sweeps.png   hexagon density of tau_M and the second-life share of use-phase time vs lambda_H
    fig6_dirichlet_simplex.png sampled RC row (p_RC_M, p_RC_RO, p_RC_L) as points on the
                                2-simplex: density by alpha (top row), and the base ensemble
                                colored by N_M_M / B_M_RO (bottom row), tying the SRRC
                                drivers back to where they sit in probability space
    fig7_dirichlet_pdf.png     the exact analytical Dirichlet(alpha,alpha,alpha) density,
                                contoured on the 2-simplex, for the four alpha actually used
                                (0.5, 1, 2, 5) -- the theory-side companion to fig6, which
                                instead shows the sampled ensembles themselves

Usage:  python plot_results.py [--dpi 600]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from lifecycle import PARAM_NAMES

RES = Path(__file__).resolve().parent / "results"
FIG = RES / "figures"
BASE = "alpha_1"

# hexbin colorbars use a log colour scale (bins="log"); the ticks show the counts themselves
CBAR_LABEL = "Systems per hexagon"

LABELS = {
    "N_M_M": "Expected manufacturing visits, $N_{M,M}$",
    "B_M_RO": "P(absorbed in RO), $B_{M,RO}$",
    "B_M_L": "P(absorbed in L), $B_{M,L}$",
    "V_M": "Expected lifetime energy service, $V_M$ (units of $E_{U1}$)",
    "service_ratio": "Normalized cumulative service, $V^*_M = V_M/(E_{U1}+E_{U2})$",
    "rho": r"Spectral radius, $\rho(Q)$",
    "F_RC_M": r"Expected closed-loop returns, $F_{RC\to M}$",
    "tau_M": r"Expected time to absorption, $\tau_M$ (units of $\mu_{U1}$)",
    "N_M_U2": "Expected second-life visits, $N_{M,U2}$",
}


def load_ensemble(name):
    return pd.read_csv(RES / f"ensemble_{name}.csv.gz")


def hist(ax, x, key, log=False, bins=60, xlabel=None, panel=None, ref=None):
    """Histogram with the median (dashed) and the 5th/95th percentiles (dotted).
    ref = (value, text) adds a solid red reference line with `text` printed beside it
    (not in the legend)."""
    x = x[np.isfinite(x) & (x > 0)] if log else x
    b = np.geomspace(x.min(), x.max(), bins) if log else bins
    ax.hist(x, bins=b, color="#4C72B0", edgecolor="white", linewidth=0.3)
    q05, q50, q95 = np.percentile(x, [5, 50, 95])
    ax.axvline(q50, color="black", ls="--", lw=1, label=f"median: {q50:.3g}")
    ax.axvline(q05, color="0.2", ls=":", lw=1.4, label=f"5th–95th: {q05:.3g}–{q95:.3g}")
    ax.axvline(q95, color="0.2", ls=":", lw=1.4)
    if ref is not None:
        ax.axvline(ref[0], color="firebrick", ls="-", lw=1.2)
        ax.annotate(ref[1], xy=(ref[0], 0.93), xycoords=ax.get_xaxis_transform(),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="top",
                    fontsize=7, color="firebrick",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or LABELS[key], fontsize=8, fontweight="bold")
    # y counts shown in units of 10^3 on every panel, as scaled tick labels plus a
    # "(x10^3)" ylabel; matplotlib's own offset text would collide with the panel letter.
    exp = 3
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos, e=exp: f"{v / 10**e:g}"))
    ax.set_ylabel(rf"Systems ($\times 10^{{{exp}}}$)", fontsize=8, fontweight="bold")
    if panel:
        ax.set_title(f"({panel})", loc="left", fontsize=10, fontweight="bold")
    # Opaque white legend box: the percentile lines pass behind it instead of through the text.
    ax.legend(prop={"size": 7, "weight": "bold"}, frameon=True, facecolor="white",
              edgecolor="none", framealpha=1)


# Bold math (subscripted symbols in the axis titles) so it matches the bold axis-title text;
# applied to Fig. 1 only.
BOLD_MATH = {"mathtext.fontset": "custom", "mathtext.default": "it",
             "mathtext.rm": "DejaVu Sans:bold", "mathtext.it": "DejaVu Sans:italic:bold",
             "mathtext.bf": "DejaVu Sans:bold"}


@plt.rc_context(BOLD_MATH)
def fig_distributions(base, dpi):
    """Six panels, each a distinct output (no algebraic duplicates):
      (a) N_M,M   expected manufacturing visits; = 1 + F_RC->M (the initial visit counts)
      (b) N_M,U2  expected second-life visits
      (c) B_M,RO  open-loop absorption probability; P(L) = 1 - P(RO) is omitted as redundant
      (d) V*      normalized cumulative service V_M/(E_U1+E_U2); V_M itself has the same shape
      (e) rho(Q)  persistence of circulation
      (f) tau_M   expected (use-phase) time to absorption
    Panels a, b, d, f use log x-axes. lambda_G / lambda_H (V*, tau_M) and the identities
    above belong in the caption rather than on the axes."""
    vstar = base["service_ratio"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    ax = axes.ravel()
    hist(ax[0], base["N_M_M"].to_numpy(), "N_M_M", log=True, panel="a",
         xlabel=r"Expected manufacturing visits, $N_{M,M}$")
    hist(ax[1], base["N_M_U2"].to_numpy(), "N_M_U2", log=True, panel="b")
    hist(ax[2], base["B_M_RO"].to_numpy(), "B_M_RO", panel="c",
         xlabel=r"Open-loop absorption probability, $B_{M,RO}$")
    hist(ax[3], vstar, "service_ratio", log=True, panel="d",
         xlabel=r"Normalized cumulative service, $V^*_M$",
         ref=(1.0, rf"{np.mean(vstar >= 1):.0%} $\geq$ 1"))
    hist(ax[4], base["rho"].to_numpy(), "rho", panel="e")
    hist(ax[5], base["tau_M"].to_numpy(), "tau_M", log=True, panel="f")
    fig.tight_layout()   # no suptitle: the title lives in the manuscript's figure caption
    fig.savefig(FIG / "fig1_distributions.png", dpi=dpi)
    plt.close(fig)


def fig_prior_comparison(priors, dpi):
    keys = [("B_M_L", False), ("N_M_M", True), ("service_ratio", True)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (k, log) in zip(axes, keys):
        for name, df in priors.items():
            x = np.sort(df[k].to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=1.4, label=name)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(LABELS[k], fontsize=8)
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_prior_comparison.png", dpi=dpi)
    plt.close(fig)


# Outputs shown in the driver heatmap, in the order of the Fig. 1 panels. The omitted outputs are
# exact duplicates: F_RC->M == N_M,M - 1 (same ranks), B_M,L == 1 - B_M,RO (SRRC = -SRRC of RO),
# V* == V_M / (1 + lambda_G) (same ranks).
DRIVER_ROWS = [("N_M_M", r"$N_{M,M}$"), ("N_M_U2", r"$N_{M,U2}$"), ("B_M_RO", r"$B_{M,RO}$"),
               ("V_M", r"$V^*_M$"), ("rho", r"$\rho(Q)$"), ("tau_M", r"$\tau_M$")]


def fig_drivers(dpi):
    """SRRC heatmap: discrete cells, no grid lines (fig3_drivers.png)."""
    d = pd.read_csv(RES / f"drivers_{BASE}.csv").dropna(subset=["srrc"])
    M = d.pivot(index="output", columns="param", values="srrc")
    # columns in topology order (first life -> second life -> remanufacturing -> recycling)
    cols = [p for p in PARAM_NAMES if p in M.columns]
    M = M.loc[[k for k, _ in DRIVER_ROWS], cols]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    # All SRRCs are >= -0.01 once the redundant B_M,L row is dropped, so a sequential 0-1 scale
    # is enough (the printed values still show the sign). One-hue blue scale built around the
    # #4C72B0 used for the Fig. 1 histograms: near-white = no effect, deep navy = dominant driver.
    blues = LinearSegmentedColormap.from_list("srrc_blue", [(0.0, "#F4F7FB"), (0.5, "#4C72B0"), (1.0, "#1B2F52")])
    im = ax.imshow(M.to_numpy(), cmap=blues, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(M.shape[1]), [rf"$p_{{{c[2:].replace('_', chr(92) + 'to ', 1)}}}$" for c in M.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0]), [lab for _, lab in DRIVER_ROWS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=8,
                    color="white" if v >= 0.45 else "black")
    ax.set_xlabel("Transition probability", fontweight="bold")
    ax.set_ylabel("Lifecycle outcome", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, label="SRRC")
    cb.ax.yaxis.label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_drivers.png", dpi=dpi)
    plt.close(fig)


@plt.rc_context(BOLD_MATH)
def fig_relationships(base, dpi, clip=True, name="fig4_relationships.png"):
    """clip=True: panels (b), (c) truncated at the 99th percentile of each variable (fig4_relationships.png).
    clip=False: every panel shows 100% of the systems (fig4_relationships_full.png)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    hb = ax.hexbin(base["rho"], base["N_M_M"], gridsize=60, yscale="log", bins="log", cmap="viridis", mincnt=1)
    ax.set_xlabel(LABELS["rho"]); ax.set_ylabel(LABELS["N_M_M"])
    # plain "1" and "10" on the log y-axis instead of 10^0 and 10^1
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.colorbar(hb, ax=ax, label=CBAR_LABEL)
    # Note: B_M_RO + B_M_L = 1 exactly (they're the two columns of one absorption-
    # probability row), so plotting them against each other is a tautological
    # straight line, not a finding -- N_M_U2 vs V_M below is the informative pairing.
    # Panels (b) and (c) are treated alike: both axes clipped at the 99th percentile
    # (shared V_M range), lower V_M limit 0.9 since V_M >= E_U1 (M -> U1 has probability 1).
    q = 0.99 if clip else 1.0
    ymax = base["V_M"].quantile(q)
    for ax, xcol in ((axes[1], "N_M_U2"), (axes[2], "F_RC_M")):
        xmax = base[xcol].quantile(q)
        hb = ax.hexbin(base[xcol], base["V_M"], gridsize=60, bins="log", cmap="viridis", mincnt=1,
                       extent=(0, xmax, 0.9, ymax))
        ax.set_xlim(0, xmax); ax.set_ylim(0.9, ymax)
        ax.set_xlabel(LABELS[xcol]); ax.set_ylabel("$V_M$ (units of $E_{U1}$)")
        fig.colorbar(hb, ax=ax, label=CBAR_LABEL)
        shown = ((base[xcol] <= xmax) & (base["V_M"] <= ymax)).mean()
        print(f"{name} panel {xcol}: {'clipped at the 99th percentile; ' if clip else ''}"
              f"{shown:.1%} of systems shown")
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", fontsize=10, fontweight="bold")
    for a in fig.axes:   # all axis titles, including the colorbar labels
        a.xaxis.label.set_fontweight("bold")
        a.yaxis.label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=dpi)
    plt.close(fig)


@plt.rc_context(BOLD_MATH)
@plt.rc_context(BOLD_MATH)
@plt.rc_context(BOLD_MATH)
def _sweep_hexbin(ax, fig, base, lam_range, f, ylim, xlabel, ylabel, rng, ref=None):
    """Density landscape of a lambda sweep. Every system is evaluated at one lambda drawn
    uniformly from lam_range, so each system appears once and 'systems per hexagon' keeps its
    meaning (no discrete lambda columns). The black line is the exact median over all systems
    on a fine lambda grid. f(base, lam) returns the outcome (lam may be an array or scalar)."""
    lo, hi = lam_range
    lam = rng.uniform(lo, hi, size=len(base))
    y = np.asarray(f(base, lam))
    hb = ax.hexbin(lam, y, gridsize=45, extent=(lo, hi, ylim[0], ylim[1]), bins="log",
                   cmap="viridis", mincnt=1)
    grid = np.linspace(lo, hi, 31)
    ax.plot(grid, [np.median(f(base, g)) for g in grid], color="black", lw=2, label="median")
    if ref is not None:
        ax.axhline(ref[0], color="firebrick", lw=2.0, label=ref[1])
    ax.set_xlim(lo, hi); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(hb, ax=ax, label=CBAR_LABEL)
    return ((y >= ylim[0]) & (y <= ylim[1])).mean()


@plt.rc_context(BOLD_MATH)
def fig_sweeps(base, dpi):
    """Main-text candidate: 2 panels, hexagon density of the 100,000 systems (not bands).
      (a) V_M vs lambda_G     (b) V*_M vs lambda_G with the V* = 1 line."""
    rng = np.random.default_rng(2026)
    xg = r"Relative second-life service per visit, $\lambda_G = g_{U2}/g_{U1}$"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    VM = lambda b, l: b["N_M_U1"].to_numpy() + l * b["N_M_U2"].to_numpy()
    VS = lambda b, l: VM(b, l) / (1 + l)
    pool = np.concatenate([VM(base, g) for g in (0.25, 1.0)])
    top_a = np.quantile(pool, 0.99)
    top_b = np.quantile(np.concatenate([VS(base, g) for g in (0.25, 1.0)]), 0.99)
    sa = _sweep_hexbin(axes[0], fig, base, (0.25, 1.0), VM, (0.9, top_a), xg,
                       r"Lifetime energy service, $V_M$ (units of $E_{U1}$)", rng)
    sb = _sweep_hexbin(axes[1], fig, base, (0.25, 1.0), VS, (0.5, top_b), xg,
                       r"Normalized cumulative service, $V^*_M$", rng, ref=(1.0, r"$V^*_M = 1$"))
    print(f"fig5_service_sweeps: y-axes clipped at the ~99th percentile; "
          f"{sa:.1%} (a) and {sb:.1%} (b) of systems shown")
    for a, letter in zip(fig.axes, "ab"):
        a.set_title(f"({letter})", loc="left", fontsize=10, fontweight="bold")
    for a in fig.axes:
        a.xaxis.label.set_fontweight("bold"); a.yaxis.label.set_fontweight("bold")
    # one shared legend below the panels (panel (b) carries both the median and the V* = 1 line)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               prop={"size": 9, "weight": "bold"}, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIG / "fig5_service_sweeps.png", dpi=dpi)
    plt.close(fig)


@plt.rc_context(BOLD_MATH)
def fig_sweeps_duration_si(base, dpi):
    """Supplement companion: hexagon density for the lambda_H sweep.
      (a) tau_M vs lambda_H   (b) second-life share of expected use-phase time, f_U2, vs lambda_H.
    Residence times of M, RM, RC are zero in the model, so tau_M and f_U2 cover first/second life only."""
    rng = np.random.default_rng(2027)
    xh = r"Relative second-life duration, $\lambda_H = \mu_{U2}/\mu_{U1}$"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    tau = lambda b, l: b["N_M_U1"].to_numpy() + l * b["N_M_U2"].to_numpy()
    fu2 = lambda b, l: 100 * l * b["N_M_U2"].to_numpy() / tau(b, l)
    top_t = np.quantile(np.concatenate([tau(base, g) for g in (0.5, 2.0)]), 0.99)
    top_f = np.quantile(fu2(base, 2.0), 0.99)
    sa = _sweep_hexbin(axes[0], fig, base, (0.5, 2.0), tau, (0.9, top_t), xh,
                       r"Expected use-phase time, $\tau_M$ (units of $\mu_{U1}$)", rng)
    sb = _sweep_hexbin(axes[1], fig, base, (0.5, 2.0), fu2, (0, top_f), xh,
                       "Second-life share of use-phase time (%)", rng)
    print(f"figS_duration_sweeps: y-axes clipped at the ~99th percentile; "
          f"{sa:.1%} (a) and {sb:.1%} (b) of systems shown")
    for a, letter in zip(fig.axes, "ab"):
        a.set_title(f"({letter})", loc="left", fontsize=10, fontweight="bold")
    for a in fig.axes:
        a.xaxis.label.set_fontweight("bold"); a.yaxis.label.set_fontweight("bold")
    # one shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               prop={"size": 9, "weight": "bold"}, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIG / "figS_duration_sweeps.png", dpi=dpi)
    plt.close(fig)


def _simplex_xy(p_m, p_ro, p_l):
    """Barycentric (p_RC_M, p_RC_RO, p_RC_L) -> Cartesian, on an equilateral
    triangle with M at the top apex, RO at bottom-left, L at bottom-right."""
    x = p_l + 0.5 * p_m
    y = p_m * (np.sqrt(3) / 2)
    return x, y


def _draw_simplex_frame(ax, title):
    h = np.sqrt(3) / 2
    tri = plt.Polygon([(0, 0), (1, 0), (0.5, h)], closed=True, fill=False,
                       edgecolor="0.3", linewidth=1)
    ax.add_patch(tri)
    ax.text(0.5, h + 0.05, "$p_{RC,M}=1$", ha="center", fontsize=8)
    ax.text(-0.06, -0.05, "$p_{RC,RO}=1$", ha="center", fontsize=8)
    ax.text(1.06, -0.05, "$p_{RC,L}=1$", ha="center", fontsize=8)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.12, h + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def fig_dirichlet_simplex(priors, dpi):
    """The RC row is the only row with exactly 3 outgoing arcs (M, RO, L), so it is
    the one row that can be drawn directly as a point on the 2-simplex.

    Row 1: one panel per alpha, showing how the concentration parameter redistributes
    density across the triangle (Sec. 3.3); each panel is annotated with the measured
    P(a draw lands within 10% of a vertex), since the log-color hexbin understates how
    much more common near-deterministic rows are at alpha<1 (a real, large effect --
    15.4% of alpha=0.5 draws vs ~0% at alpha=5 -- that is easy to miss by eye on a log
    color scale where most of the triangle still has some fill).
    Row 2: the base ensemble (alpha=1) colored by the median of N_M_M and B_M_RO in
    each bin, showing where the two drivers identified in Sec. 3.4 (RC->M for
    circulation, RC->RO/RC->L for terminal fate) sit in this triangle.
    Row 3: the marginal density of p_RC_M alone, overlaid across all four alpha, which
    is the cleanest way to see the corner/edge effect directly -- it is the classic
    Beta(alpha, 2*alpha) marginal of a symmetric Dirichlet, and even at alpha=1 (joint
    uniform over the triangle) this marginal is not flat: Beta(1,2) has density
    2(1-p), so a single arc's probability is inherently more often small than large,
    simply because more of the triangle's area has p_RC_M small than large.
    """
    order = sorted(priors, key=lambda n: float(n.split("_", 1)[1]))
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])

    for j, name in enumerate(order):
        df = priors[name]
        x, y = _simplex_xy(df["p_RC_M"], df["p_RC_RO"], df["p_RC_L"])
        ax = fig.add_subplot(gs[0, j])
        hb = ax.hexbin(x, y, gridsize=45, bins="log", cmap="viridis", mincnt=1,
                        extent=(-0.05, 1.05, -0.05, np.sqrt(3) / 2 + 0.05))
        _draw_simplex_frame(ax, name.replace("alpha_", r"$\alpha=$"))
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label=CBAR_LABEL)
        mx = df[["p_RC_M", "p_RC_RO", "p_RC_L"]].max(axis=1)
        near_vertex = (mx > 0.9).mean()
        ax.text(0.5, -0.09, f"P(within 10% of a vertex) = {near_vertex:.1%}",
                ha="center", fontsize=7.5, transform=ax.transAxes)

    base = priors[BASE]
    x, y = _simplex_xy(base["p_RC_M"], base["p_RC_RO"], base["p_RC_L"])
    for k, (col, cmap, label) in enumerate([
        ("N_M_M", "inferno", LABELS["N_M_M"]),
        ("B_M_RO", "cividis", LABELS["B_M_RO"]),
    ]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        hb = ax.hexbin(x, y, C=base[col], reduce_C_function=np.median, gridsize=45,
                        cmap=cmap, mincnt=5,
                        extent=(-0.05, 1.05, -0.05, np.sqrt(3) / 2 + 0.05))
        _draw_simplex_frame(ax, f"Base ensemble ($\\alpha=1$), colored by median {label}")
        fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02, label=f"median {label}")

    ax = fig.add_subplot(gs[2, :])
    colors = {"alpha_0.5": "#D55E00", "alpha_1": "#000000", "alpha_2": "#0072B2", "alpha_5": "#009E73"}
    for name in order:
        df = priors[name]
        ax.hist(df["p_RC_M"], bins=100, range=(0, 1), density=True, histtype="step",
                linewidth=1.6, color=colors.get(name, None),
                label=name.replace("alpha_", r"$\alpha=$"))
    ax.set_yscale("log")
    ax.set_xlabel("$p_{RC,M}$ (marginal, single arc)")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Marginal density of a single recycling-row arc, by $\\alpha$ "
                 "(theoretical shape: Beta($\\alpha$, 2$\\alpha$))")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("Sampled recycling-row probabilities $(p_{RC,M}, p_{RC,RO}, p_{RC,L})$ on the 2-simplex")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_dirichlet_simplex.png", dpi=dpi)
    plt.close(fig)


def _symmetric_dirichlet_pdf(p_m, p_ro, p_l, alpha):
    """Exact PDF of a symmetric Dirichlet(alpha, alpha, alpha), no scipy required.
    For equal shape parameters the normalizing constant collapses to
    Gamma(3*alpha) / Gamma(alpha)**3, and every exponent is (alpha-1), so
    f(x) = C * (p_m * p_ro * p_l) ** (alpha - 1)."""
    log_c = math.lgamma(3 * alpha) - 3 * math.lgamma(alpha)
    prod = p_m * p_ro * p_l
    with np.errstate(divide="ignore"):
        log_f = log_c + (alpha - 1) * np.log(prod)
    return np.exp(log_f)


def fig_dirichlet_pdf(dpi, alphas=(0.5, 1.0, 2.0, 5.0), eps=2e-3, grid_n=400):
    """The exact analytical PDF of a symmetric Dirichlet(alpha,alpha,alpha) on the
    2-simplex, for the alpha values actually used in Sec. 2.6/3.3 -- the theoretical
    companion to fig6 (which plots the sampled ensembles rather than the formula).
    For alpha<1 the density diverges at the simplex boundary, so points within `eps`
    of any edge are masked; the divergence itself is a real feature (Sec. 3.3/fig6's
    marginal panel), not a plotting artifact, and is noted in the caption rather than
    hidden by clipping the colour scale."""
    h = np.sqrt(3) / 2
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, h, grid_n)
    X, Y = np.meshgrid(xs, ys)
    p_m = Y / h
    p_l = X - 0.5 * p_m
    p_ro = 1 - p_m - p_l
    valid = (p_m > eps) & (p_ro > eps) & (p_l > eps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, alpha in zip(axes.ravel(), alphas):
        z = np.full_like(X, np.nan)
        z[valid] = _symmetric_dirichlet_pdf(p_m[valid], p_ro[valid], p_l[valid], alpha)
        # Round away floating-point noise (~1e-13) that otherwise dominates the
        # auto-picked contour levels when the field is exactly flat (alpha=1).
        z_valid = np.round(z[valid], 6)
        levels = np.linspace(0, max(z_valid.max(), 1e-6) * 1.02, 21)
        cf = ax.tricontourf(X[valid], Y[valid], z_valid, levels=levels, cmap="Blues")
        _draw_simplex_frame(ax, rf"$\alpha=({alpha:g},\ {alpha:g},\ {alpha:g})$")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label=r"PDF $f(\mathbf{x};\alpha)$")
        # Force a tick at the actual max (matplotlib's auto-locator otherwise
        # rarely lands exactly on the true data value for an arbitrary range).
        cb.set_ticks(np.linspace(0, z_valid.max(), 6))

    fig.suptitle(r"Analytical density of a symmetric Dirichlet$(\alpha,\alpha,\alpha)$ on the 2-simplex"
                 "\n(theoretical companion to Fig. 6's sampled ensembles; boundary strip "
                 f"< {eps:g} masked where the $\\alpha<1$ density diverges)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_dirichlet_pdf.png", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    names = [p.name[len("ensemble_"):-len(".csv.gz")] for p in sorted(RES.glob("ensemble_*.csv.gz"))]
    if not names:
        raise FileNotFoundError(f"no ensemble_*.csv.gz found in {RES} -- run run_ensemble.py first")
    if BASE not in names:
        raise FileNotFoundError(f"base prior '{BASE}' not found in {RES} -- rerun run_ensemble.py")
    names.sort(key=lambda n: (n != BASE, n))  # base first
    priors = {n: load_ensemble(n) for n in names}

    fig_distributions(priors[BASE], args.dpi)
    fig_prior_comparison(priors, args.dpi)
    fig_drivers(args.dpi)
    fig_relationships(priors[BASE], args.dpi)
    fig_sweeps(priors[BASE], args.dpi)
    fig_sweeps_duration_si(priors[BASE], args.dpi)
    fig_dirichlet_simplex(priors, args.dpi)
    fig_dirichlet_pdf(args.dpi)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
