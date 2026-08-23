"""Tests for the realistic total GSEE cost assembler (`src_PI/estimation/gsee_total_cost.py`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.estimation.gsee_total_cost import total_gsee_cost

# L=10 anchor-ish numbers (adopted π headline).
ANCHOR = dict(physical_lambda=8.797966e6, walk_T=3.27e8, walk_register_qubits=10021, eps_qpe=0.975,
              n_bos_modes=3000, N_f=4)


def test_warm_beats_cold_and_saving_tracks_reps():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=5000, **ANCHOR)
    assert r["warm"]["total_T"] < r["cold"]["total_T"]
    assert r["warmstart_saving_x"] > 1.0
    # both on the sampling side (p0 > 0.003) -> saving ≈ R_cold/R_warm ≈ p0_warm/p0_cold, but
    # slightly BELOW it because the warm start pays a (sub-dominant) prep cost the cold one doesn't.
    rep_ratio = r["cold"]["R"] / r["warm"]["R"]
    assert r["warmstart_saving_x"] <= rep_ratio
    assert abs(r["warmstart_saving_x"] - rep_ratio) / rep_ratio < 1e-3
    # saving tracks the overlap ratio (up to ⌈⌉ rounding of R=⌈ln(1/δ)/p0⌉):
    assert 10.0 < r["warmstart_saving_x"] < 25.0


def test_prep_is_subdominant():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=10000, **ANCHOR)
    assert r["prep_frac_of_window"] < 1e-3          # T_prep ≪ one QPE window
    assert r["warm"]["T_prep"] > 0                  # ...but nonzero (D-det QROAM load)


def test_total_width_is_max_not_sum():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=5000, **ANCHOR)
    walk, m, a = ANCHOR["walk_register_qubits"], r["m_qpe"], r["warm"]["a_prep"]
    assert r["warm"]["total_qubits"] == walk + max(m, a)     # sampling: no AE register
    assert r["warm"]["total_qubits"] < walk + m + a          # strictly less than the naive sum


def test_binary_branch_adds_ae_register_at_low_overlap():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=1e-4, D_warm=5000, **ANCHOR)
    assert r["cold"]["branch"] == "binary"                   # poor cold overlap -> amplification
    assert r["cold"]["ae_register_qubits"] >= 1
    assert r["warm"]["branch"] == "sampling"                 # good warm overlap stays cheap side


if __name__ == "__main__":
    for fn in (test_warm_beats_cold_and_saving_tracks_reps, test_prep_is_subdominant,
               test_total_width_is_max_not_sum, test_binary_branch_adds_ae_register_at_low_overlap):
        fn()
    print("all gsee_total_cost checks passed")
