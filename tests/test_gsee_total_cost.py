"""Tests for the realistic total GSEE cost assembler (`src_PI/estimation/gsee_total_cost.py`)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.estimation.gsee_total_cost import total_gsee_cost, total_gsee_cost_from_record

# L=10 anchor-ish numbers.
ANCHOR = dict(physical_lambda=8.797966e6, walk_T=3.27e8, walk_register_qubits=10021, eps_qpe=0.975,
              n_bos_modes=3000, N_f=4)


def test_warm_beats_cold_two_named_metrics():
    """Reports BOTH operational metrics (warmstart audit §1): the fixed-confidence exact-Bernoulli
    R ratio AND the mean-shots p0 ratio — never conflated."""
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=5000, **ANCHOR)
    assert r["warm"]["branch"] == "sampling" and r["cold"]["branch"] == "sampling"
    assert r["warm"]["total_T"] < r["cold"]["total_T"]
    # fixed-confidence: R = exact Bernoulli ⌈ln δ/ln(1−p0)⌉
    assert r["cold"]["R"] == math.ceil(math.log(0.01) / math.log1p(-0.02))   # 228
    assert r["warm"]["R"] == math.ceil(math.log(0.01) / math.log1p(-0.5))    # 7
    rep_ratio = r["cold"]["R"] / r["warm"]["R"]
    assert r["warmstart_saving_x"] <= rep_ratio                              # minus sub-dominant prep
    assert abs(r["warmstart_saving_x"] - rep_ratio) / rep_ratio < 1e-3
    # mean-shots: the DISTINCT continuous metric = p0_warm/p0_cold
    assert abs(r["warmstart_saving_expected_x"] - 0.5 / 0.02) < 1e-9


def test_prep_is_subdominant():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=10000, **ANCHOR)
    assert r["prep_frac_of_window"] < 1e-3          # T_prep ≪ one QPE window
    assert r["warm"]["T_prep"] > 0                  # ...but nonzero (D-det QROAM load)


def test_total_width_is_max_not_sum():
    r = total_gsee_cost(p0_warm=0.5, p0_cold=0.02, D_warm=5000, **ANCHOR)
    walk, m, a = ANCHOR["walk_register_qubits"], r["m_qpe"], r["warm"]["a_prep"]
    assert r["warm"]["total_qubits"] == walk + max(m, a)     # sampling: no AE register
    assert r["warm"]["total_qubits"] < walk + m + a          # strictly less than the naive sum


def test_binary_is_opt_in_not_auto():
    """Default is exact sampling even at low overlap (audit §2: no auto-binary)."""
    r = total_gsee_cost(p0_warm=0.5, p0_cold=1e-4, D_warm=5000, **ANCHOR)
    assert r["cold"]["branch"] == "sampling" and r["cold"]["ae_register_qubits"] == 0
    rb = total_gsee_cost(p0_warm=0.5, p0_cold=1e-4, D_warm=5000, branch="binary", **ANCHOR)
    assert rb["cold"]["branch"] == "binary" and rb["cold"]["ae_register_qubits"] >= 1


def test_from_record_consumes_accepted_n_walk():
    """audit §8: the assembler must consume the shard's stored QPE_Walk_Queries (√2·π), not
    silently re-derive N_walk with a different constant."""
    rec = {"Physical_Lambda": 8.797966e6, "Walk_T_Count": 3.27e8, "Logical_Qubits": 10021,
           "QPE_Walk_Queries": 4.0091e7,          # the accepted √2·π value stored in the shard
           "QPE_Budget": {"eps_qpe": 0.975}}
    r = total_gsee_cost_from_record(rec, p0_warm=0.5, p0_cold=0.02, D_warm=5000,
                                    n_bos_modes=3000, N_f=4)
    assert r["n_walk_from_record"] is True
    assert abs(r["N_walk"] - 4.0091e7) < 1.0                # used the stored value verbatim
    assert abs(r["coherent_query_T"] - 4.0091e7 * 3.27e8) < 1e6


if __name__ == "__main__":
    for fn in (test_warm_beats_cold_two_named_metrics, test_prep_is_subdominant,
               test_total_width_is_max_not_sum, test_binary_is_opt_in_not_auto,
               test_from_record_consumes_accepted_n_walk):
        fn()
    print("all gsee_total_cost checks passed")
