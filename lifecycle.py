"""Core of the ensemble framework: lifecycle topology, Dirichlet sampling of
transition matrices, and absorbing semi-Markov analysis.

Implements the manuscript's Methods section (Beyond a Single Electric Vehicle
Battery Life...):
    Sec. 2.2   state space S = {M, U1, U2, RM, RC, RO, L}
    Sec. 2.3   canonical form P=[[Q,R],[0,I]], fundamental matrix N=(I-Q)^-1,
               absorption probabilities B=N R, expected time to absorption tau=N*mu
    Sec. 2.5   reward vector g, expected cumulative service V=N g
    Sec. 2.6   admissible topology, Dirichlet(alpha) sampling of each row, validity
               checks (nonnegativity, unit row sums, topology, rho(Q)<1)
    Sec. 2.7   nu=N 1, expected transition flow F_ij(M)=N_Mi p_ij, spectral radius
               rho(Q), and the normalized ratios
                   lambda_H = mu_U2 / mu_U1   (relative second-life duration)
                   lambda_G = g_U2  / g_U1    (relative second-life service)
               with mu_U1 = g_U1 = 1 as the first-life reference.

State order is fixed: transient M, U1, U2, RM, RC, then absorbing RO, L.
Running this file directly performs a self-check of the analytics against a
Monte Carlo simulation of the chain (see also test_lifecycle.py for unit tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["M", "U1", "U2", "RM", "RC", "RO", "L"]
TRANSIENT = STATES[:5]
ABSORBING = STATES[5:]
IDX = {s: i for i, s in enumerate(STATES)}
N_T = len(TRANSIENT)

# Admissible transitions (Sec. 2.6). Everything else is a structural zero.
TOPOLOGY = {
    "M": ["U1"],
    "U1": ["U2", "RM", "RC", "L"],
    "U2": ["RC", "L"],
    "RM": ["U1", "U2", "RC", "L"],
    "RC": ["M", "RO", "L"],
}
SAMPLED = [s for s in TOPOLOGY if s != "M"]  # M -> U1 is fixed at 1
PARAM_NAMES = [f"p_{i}_{j}" for i in SAMPLED for j in TOPOLOGY[i]]

KEY_OUTPUTS = ["N_M_M", "N_M_U1", "N_M_U2", "nu_M", "B_M_RO", "B_M_L", "rho",
               "V_M", "tau_M", "service_ratio", "F_RC_M"]
DRIVER_OUTPUTS = ["N_M_M", "N_M_U2", "B_M_RO", "B_M_L", "V_M", "service_ratio", "rho", "F_RC_M", "tau_M"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_transition_matrices(n, rng, alpha=1.0):
    """Draw n full 7x7 transition matrices.

    Each sampled row i is p_i ~ Dirichlet(alpha_i) over its admissible transitions,
    with alpha_i = alpha * ones(K_i). alpha=1.0 samples uniformly over the simplex
    (Sec. 2.6, the base ensemble); alpha < 1 / > 1 concentrate mass toward the
    corners / centre and are used only for the sensitivity runs in run_ensemble.py.

    Raises ValueError if n<=0 or alpha<=0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    P = np.zeros((n, len(STATES), len(STATES)))
    P[:, IDX["M"], IDX["U1"]] = 1.0
    for a in ABSORBING:
        P[:, IDX[a], IDX[a]] = 1.0
    for i in SAMPLED:
        cols = [IDX[j] for j in TOPOLOGY[i]]
        P[:, IDX[i], cols] = rng.dirichlet(np.full(len(cols), alpha), size=n)
    return P


def validate_matrices(P, atol=1e-9):
    """Stochastic validity + topology consistency (Sec. 2.6). Raises ValueError
    (not a bare assert) so this remains a real check under python -O."""
    allowed = np.zeros((len(STATES), len(STATES)), dtype=bool)
    for i, js in TOPOLOGY.items():
        allowed[IDX[i], [IDX[j] for j in js]] = True
    for a in ABSORBING:
        allowed[IDX[a], IDX[a]] = True
    if not np.isfinite(P).all():
        raise ValueError("transition matrix contains NaN/Inf")
    if (P < -atol).any():
        raise ValueError("negative probabilities")
    if not np.allclose(P.sum(axis=-1), 1.0, atol=atol):
        bad = np.abs(P.sum(axis=-1) - 1.0).max()
        raise ValueError(f"rows do not sum to 1 (max deviation {bad:.2e})")
    if (P[..., ~allowed] > atol).any():
        raise ValueError("probability mass on a structurally-zero transition")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def reward_vector(lambda_G):
    """g (Sec. 2.5, 2.7): reward per visit, g_U1 = 1 (first-life reference),
    g_U2 = lambda_G = g_U2/g_U1."""
    return np.array([0.0, 1.0, lambda_G, 0.0, 0.0])


def residence_vector(lambda_H, mu_M=0.0, mu_RM=0.0, mu_RC=0.0):
    """mu (Sec. 2.3, 2.7): mean residence times, mu_U1 = 1 (first-life reference),
    mu_U2 = lambda_H = mu_U2/mu_U1. M/RM/RC neglected by default (Supplement S2.2)."""
    return np.array([mu_M, 1.0, lambda_H, mu_RM, mu_RC])


def build_frame(P, lambda_G=0.5, lambda_H=1.0, mu_other=(0.0, 0.0, 0.0)):
    """Evaluate Sec. 2.3-2.7 for every matrix in P (batched) and return a DataFrame
    with one row per matrix: sampled probabilities, N_M row, absorption
    probabilities, spectral radius, service/time metrics and expected flows F(M).

    Raises ValueError if lambda_G/lambda_H are not positive, or if any resulting
    row has rho(Q) >= 1 (fundamental matrix would not exist / N would diverge).
    """
    if lambda_G <= 0:
        raise ValueError(f"lambda_G must be positive, got {lambda_G}")
    if lambda_H <= 0:
        raise ValueError(f"lambda_H must be positive, got {lambda_H}")
    P = np.asarray(P, dtype=float)
    if P.ndim == 2:
        P = P[None]
    Q = P[:, :N_T, :N_T]
    R = P[:, :N_T, N_T:]
    rho = np.abs(np.linalg.eigvals(Q)).max(axis=1)  # spectral radius of Q
    if (rho >= 1).any():
        raise ValueError(f"{(rho >= 1).sum()} system(s) have rho(Q) >= 1; "
                          "the fundamental matrix does not exist for these rows")
    N = np.linalg.inv(np.eye(N_T) - Q)              # fundamental matrix
    B = N @ R                                       # absorption probabilities
    N_M = N[:, IDX["M"], :]                         # expected visits starting from M

    g = reward_vector(lambda_G)
    mu = residence_vector(lambda_H, *mu_other)

    cols = {}
    for i in SAMPLED:
        for j in TOPOLOGY[i]:
            cols[f"p_{i}_{j}"] = P[:, IDX[i], IDX[j]]
    for k, s in enumerate(TRANSIENT):
        cols[f"N_M_{s}"] = N_M[:, k]
    cols["nu_M"] = N_M.sum(axis=1)
    cols["B_M_RO"] = B[:, IDX["M"], 0]
    cols["B_M_L"] = B[:, IDX["M"], 1]
    cols["rho"] = rho
    cols["V_M"] = N_M @ g                           # expected lifetime energy service
    cols["tau_M"] = N_M @ mu                        # expected physical time to absorption
    cols["service_ratio"] = cols["V_M"] / g.sum()   # realised / technical potential (M->U1->U2)
    F = N_M[:, :, None] * P[:, :N_T, :]             # F_ij(M) = N_Mi * p_ij
    for i in TRANSIENT[1:]:
        for j in TOPOLOGY[i]:
            cols[f"F_{i}_{j}"] = F[:, IDX[i], IDX[j]]
    df = pd.DataFrame(cols)
    if not np.isfinite(df.to_numpy()).all():
        raise FloatingPointError("non-finite value produced in build_frame output")
    return df


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def simulate(P, n_traj, rng, max_steps=100_000):
    """Monte Carlo of one chain started in M: mean transient visits and absorption freqs."""
    cum = np.cumsum(P, axis=1)
    state = np.zeros(n_traj, dtype=int)
    visits = np.zeros(len(STATES))
    visits[0] = n_traj
    for _ in range(max_steps):
        active = np.flatnonzero(state < N_T)
        if active.size == 0:
            break
        u = rng.random(active.size)
        nxt = np.minimum((u[:, None] > cum[state[active]]).sum(axis=1), len(STATES) - 1)
        state[active] = nxt
        visits += np.bincount(nxt, minlength=len(STATES))
    return visits[:N_T] / n_traj, np.bincount(state, minlength=len(STATES))[N_T:] / n_traj


def selfcheck(n_traj=200_000, seed=0):
    rng = np.random.default_rng(seed)
    P = sample_transition_matrices(1, rng)  # one random draw, used as a concrete test case
    validate_matrices(P)
    df = build_frame(P)
    N_M = df[[f"N_M_{s}" for s in TRANSIENT]].to_numpy()[0]
    B_M = df[["B_M_RO", "B_M_L"]].to_numpy()[0]
    sim_N, sim_B = simulate(P[0], n_traj, rng)
    print("expected visits from M   analytic:", np.round(N_M, 4))
    print("                         simulated:", np.round(sim_N, 4))
    print("absorption [RO, L]       analytic:", np.round(B_M, 4))
    print("                         simulated:", np.round(sim_B, 4))
    if not np.allclose(N_M, sim_N, atol=0.02 + 0.02 * N_M):
        raise AssertionError("N mismatch vs simulation")
    if not np.allclose(B_M, sim_B, atol=0.01):
        raise AssertionError("B mismatch vs simulation")
    if not np.isclose(B_M.sum(), 1.0):
        raise AssertionError("absorption probabilities do not sum to 1")

    # random ensemble: validity + identities
    Pens = sample_transition_matrices(10_000, rng)
    validate_matrices(Pens)
    ens = build_frame(Pens)
    if not (ens["rho"] < 1).all():
        raise AssertionError("spectral radius >= 1")
    if not np.allclose(ens["B_M_RO"] + ens["B_M_L"], 1.0):
        raise AssertionError("absorption probabilities do not sum to 1")
    if not np.allclose(ens["F_RC_M"], ens["N_M_M"] - 1.0):
        raise AssertionError("F_RC_M != N_M_M - 1 identity broken")
    print("self-check passed")


if __name__ == "__main__":
    selfcheck()
