"""Unit tests for lifecycle.py. Run with:  pytest test_lifecycle.py -v

Covers what the Monte Carlo self-check (lifecycle.selfcheck) doesn't: a
hand-computed deterministic case, algebraic identities across the whole
ensemble, error handling on invalid inputs, and reproducibility.
"""
import numpy as np
import pytest

import lifecycle as lc


# ---------------------------------------------------------------------------
# A hand-computable deterministic system
# ---------------------------------------------------------------------------
def _deterministic_matrix():
    """A fully deterministic (0/1) lifecycle: M->U1->U2->RC->M->... looping
    forever except a fixed 0.1 leak from RC to L each pass. Hand-derivable:
    Q has one nonzero entry per row (a directed cycle M->U1->U2->RC->M with
    RC->M = 0.9), so N = I + Q + Q^2 + ... telescopes to a simple closed form.
    """
    P = np.zeros((len(lc.STATES), len(lc.STATES)))
    P[lc.IDX["M"], lc.IDX["U1"]] = 1.0
    P[lc.IDX["U1"], lc.IDX["U2"]] = 1.0
    P[lc.IDX["U2"], lc.IDX["RC"]] = 1.0          # U2 -> RC = 1.0, RC in {RC,L}
    P[lc.IDX["RM"], lc.IDX["RC"]] = 1.0          # RM unreachable here but must sum to 1
    P[lc.IDX["RC"], lc.IDX["M"]] = 0.9
    P[lc.IDX["RC"], lc.IDX["L"]] = 0.1
    for a in lc.ABSORBING:
        P[lc.IDX[a], lc.IDX[a]] = 1.0
    return P


def test_deterministic_case_matches_hand_derivation():
    P = _deterministic_matrix()
    lc.validate_matrices(P[None])
    df = lc.build_frame(P[None])

    # The only cycle is M->U1->U2->RC->M with loop probability 0.9, so a battery
    # starting at M is expected to complete 1/(1-0.9) = 10 full M-U1-U2-RC laps
    # before being absorbed into L (RC->L = 0.1 each pass).
    expected_N_M_M = 1 / 0.1
    assert df["N_M_M"].iloc[0] == pytest.approx(expected_N_M_M)
    assert df["N_M_U1"].iloc[0] == pytest.approx(expected_N_M_M)
    assert df["N_M_U2"].iloc[0] == pytest.approx(expected_N_M_M)
    assert df["N_M_RM"].iloc[0] == pytest.approx(0.0)
    assert df["B_M_L"].iloc[0] == pytest.approx(1.0)
    assert df["B_M_RO"].iloc[0] == pytest.approx(0.0)
    assert df["rho"].iloc[0] == pytest.approx(0.9 ** 0.25, rel=1e-6)  # (0.9)^(1/4): 4-cycle


# ---------------------------------------------------------------------------
# Algebraic identities that must hold across the whole ensemble
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_ensemble():
    rng = np.random.default_rng(42)
    P = lc.sample_transition_matrices(20_000, rng, alpha=1.0)
    return lc.build_frame(P, lambda_G=0.5, lambda_H=1.0)


def test_absorption_probabilities_sum_to_one(base_ensemble):
    assert np.allclose(base_ensemble["B_M_RO"] + base_ensemble["B_M_L"], 1.0)


def test_spectral_radius_strictly_below_one(base_ensemble):
    assert (base_ensemble["rho"] < 1).all()
    assert (base_ensemble["rho"] > 0).all()


def test_F_RC_M_equals_N_M_M_minus_one(base_ensemble):
    # The only path back to M is through RC->M, so every revisit to M consumes
    # exactly one RC->M traversal: N_M,M = 1 + F_RC->M.
    assert np.allclose(base_ensemble["F_RC_M"], base_ensemble["N_M_M"] - 1.0)


def test_visit_counts_at_least_one_for_M_and_U1(base_ensemble):
    # M is visited at least once (the start), and U1 is always reached from M.
    assert (base_ensemble["N_M_M"] >= 1.0 - 1e-9).all()
    assert (base_ensemble["N_M_U1"] >= 1.0 - 1e-9).all()


def test_nu_M_equals_sum_of_visit_counts(base_ensemble):
    visit_cols = [f"N_M_{s}" for s in lc.TRANSIENT]
    assert np.allclose(base_ensemble["nu_M"], base_ensemble[visit_cols].sum(axis=1))


def test_service_ratio_consistent_with_V_M(base_ensemble):
    assert np.allclose(base_ensemble["service_ratio"], base_ensemble["V_M"] / 1.5)


def test_sampled_probabilities_are_a_valid_topology(base_ensemble):
    for i in lc.SAMPLED:
        row_sum = sum(base_ensemble[f"p_{i}_{j}"] for j in lc.TOPOLOGY[i])
        assert np.allclose(row_sum, 1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_ensemble():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    df1 = lc.build_frame(lc.sample_transition_matrices(500, rng1))
    df2 = lc.build_frame(lc.sample_transition_matrices(500, rng2))
    pd_testing_equal = (df1.to_numpy() == df2.to_numpy()).all()
    assert pd_testing_equal


def test_different_seeds_give_different_ensembles():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    df1 = lc.build_frame(lc.sample_transition_matrices(500, rng1))
    df2 = lc.build_frame(lc.sample_transition_matrices(500, rng2))
    assert not np.allclose(df1["N_M_M"].to_numpy(), df2["N_M_M"].to_numpy())


# ---------------------------------------------------------------------------
# Error handling on invalid inputs
# ---------------------------------------------------------------------------
def test_sample_rejects_nonpositive_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        lc.sample_transition_matrices(0, rng)


def test_sample_rejects_nonpositive_alpha():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        lc.sample_transition_matrices(10, rng, alpha=0.0)


def test_build_frame_rejects_nonpositive_lambda():
    rng = np.random.default_rng(0)
    P = lc.sample_transition_matrices(5, rng)
    with pytest.raises(ValueError):
        lc.build_frame(P, lambda_G=0.0)
    with pytest.raises(ValueError):
        lc.build_frame(P, lambda_H=-1.0)


def test_validate_rejects_bad_row_sums():
    P = lc.sample_transition_matrices(1, np.random.default_rng(0))
    P[0, lc.IDX["U1"], lc.IDX["U2"]] += 0.5  # break the row sum
    with pytest.raises(ValueError, match="row"):
        lc.validate_matrices(P)


def test_validate_rejects_probability_on_structural_zero():
    # Built by hand (not sampled) so the row sums to exactly 1 with no negative
    # entries, isolating the structural-zero check from the other two checks:
    # U1's admissible transitions are {U2, RM, RC, L} -- RO is not among them.
    P = _deterministic_matrix()
    P[lc.IDX["U1"], lc.IDX["U2"]] = 0.95   # was 1.0
    P[lc.IDX["U1"], lc.IDX["RO"]] = 0.05   # structurally disallowed for U1
    with pytest.raises(ValueError, match="structurally-zero"):
        lc.validate_matrices(P[None])


def test_validate_rejects_negative_probability():
    P = lc.sample_transition_matrices(1, np.random.default_rng(0))
    P[0, lc.IDX["U1"], lc.IDX["U2"]] = -0.1
    with pytest.raises(ValueError, match="negative"):
        lc.validate_matrices(P)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
