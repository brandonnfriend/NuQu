"""
Tests for the standalone total-QPE-cost computation (Phase E).
"""

import json
import math
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.estimation.qpe_cost import (
    DEFAULT_DELTA_E_MEV,
    WALK_QUERY_CONSTANT,
    compute_total_qpe_cost,
    qpe_phase_register_qubits,
    qpe_phase_register_qubits_from_nwalk,
    total_logical_qubits,
    total_qpe_t_count,
    walk_queries,
)


def _make_sweep_file(tmpdir, results, metadata=None):
    path = os.path.join(tmpdir, 'sweep.json')
    data = {'metadata': metadata or {'L': 2, 'dim': 3}, 'results': results}
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def test_walk_queries_formula():
    """N_walk = π·Λ / ΔE (adopted constant)."""
    lam, dE = 1.97e5, 1.0
    expected = math.pi * lam / dE
    assert abs(walk_queries(lam, dE) - expected) < 1e-6


def test_total_qpe_t_count_formula():
    t_step, lam, dE = 6.85e6, 1.97e5, 1.0
    expected = t_step * math.pi * lam / dE
    assert abs(total_qpe_t_count(t_step, lam, dE) - expected) < 1e-3


def test_compute_writes_fields_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            {'A': 1, 'Total_T_Count': 6.85e6, 'Physical_Lambda': 1.97e5},
            {'A': 2, 'Total_T_Count': 2.06e6, 'Physical_Lambda': 5.36e5},
        ]
        path = _make_sweep_file(tmp, results)
        data = compute_total_qpe_cost(path, delta_E=1.0)

        for r in data['results']:
            assert 'QPE_Walk_Queries' in r
            assert 'QPE_Total_T_Count' in r
            exp_nq = math.pi * r['Physical_Lambda'] / 1.0
            assert abs(r['QPE_Walk_Queries'] - exp_nq) < 1e-3
            assert abs(r['QPE_Total_T_Count'] - r['Total_T_Count'] * exp_nq) < 1.0
        assert data['metadata']['delta_E_MeV'] == 1.0

        # Verify it was written to disk too.
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk['results'][0]['QPE_Total_T_Count'] == data['results'][0]['QPE_Total_T_Count']


def test_idempotent():
    """Running twice converges to the same numbers (recomputed from invariants)."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [{'A': 1, 'Total_T_Count': 6.85e6, 'Physical_Lambda': 1.97e5}]
        path = _make_sweep_file(tmp, results)
        d1 = compute_total_qpe_cost(path, delta_E=1.0)
        v1 = d1['results'][0]['QPE_Total_T_Count']
        d2 = compute_total_qpe_cost(path, delta_E=1.0)
        v2 = d2['results'][0]['QPE_Total_T_Count']
        assert v1 == v2


def test_delta_e_scaling():
    """Halving ΔE doubles the total cost."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [{'A': 1, 'Total_T_Count': 1e6, 'Physical_Lambda': 1e5}]
        path = _make_sweep_file(tmp, results)
        d_full = compute_total_qpe_cost(path, delta_E=1.0, write=False)
        d_half = compute_total_qpe_cost(path, delta_E=0.5, write=False)
        ratio = d_half['results'][0]['QPE_Total_T_Count'] / d_full['results'][0]['QPE_Total_T_Count']
        assert abs(ratio - 2.0) < 1e-9


def test_skips_entries_missing_fields():
    """Entries without Total_T_Count / Physical_Lambda are skipped, not crashed on."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            {'A': 1, 'Total_T_Count': 1e6, 'Physical_Lambda': 1e5},
            {'A': 2},  # legacy / malformed entry — no T or Λ
        ]
        path = _make_sweep_file(tmp, results)
        data = compute_total_qpe_cost(path, delta_E=1.0)
        assert 'QPE_Total_T_Count' in data['results'][0]
        assert 'QPE_Total_T_Count' not in data['results'][1]


def test_default_delta_e_is_one_mev():
    assert DEFAULT_DELTA_E_MEV == 1.0


# --- QPE phase-register ancilla (Babbush 2018 Eq. 24) -------------------------- #

def test_phase_register_equals_babbush_eq24():
    """m = ⌈log₂(N_walk / 2)⌉ = ⌈log₂(π·λ / (2·ε_qpe))⌉ — Babbush Eq. 24 form, adopted π constant."""
    lam, eps_qpe = 46521.0, 0.96
    expected = math.ceil(math.log2(math.pi * lam / (2.0 * eps_qpe)))
    assert qpe_phase_register_qubits(lam, eps_qpe) == expected


def test_phase_register_ties_to_reported_nwalk():
    """m from λ,ε_qpe must equal m from the reported N_walk (m = ⌈log₂(N_walk/2)⌉)."""
    lam, eps_qpe = 46521.0, 0.96
    n_walk = walk_queries(lam, eps_qpe)
    assert qpe_phase_register_qubits(lam, eps_qpe) == \
        qpe_phase_register_qubits_from_nwalk(n_walk)


def test_phase_register_anchor_magnitudes():
    """Sanity: the committed anchor N_walk values give m = 13..25 across L=1..10."""
    # (N_walk, expected m) from results/quantum_pauli_lcu_resources.md
    cases = [(9.76e3, 13), (2.15e5, 17), (8.77e5, 19), (4.01e7, 25)]
    for n_walk, m in cases:
        assert qpe_phase_register_qubits_from_nwalk(n_walk) == m


def test_phase_register_constant_switch():
    """The historical √2·π upper bound never gives fewer phase bits than the adopted π default."""
    lam, eps_qpe = 46521.0, 0.96
    m_pi = qpe_phase_register_qubits(lam, eps_qpe)                                    # adopted π (default)
    m_upper = qpe_phase_register_qubits(lam, eps_qpe, constant=math.sqrt(2)*math.pi)  # √2·π
    assert m_pi <= m_upper  # π (fewer walks) → never more phase bits


def test_total_logical_qubits_reuse_is_max_not_sum():
    """Peak width = walk + max(m_QPE, a_prep): state prep and phase register don't coexist."""
    walk, m, a_prep = 10021, 25, 12
    assert total_logical_qubits(walk, m, a_prep) == walk + m       # m dominates
    assert total_logical_qubits(walk, m, 40) == walk + 40          # prep dominates
    assert total_logical_qubits(walk, m) == walk + m               # no prep modeled yet
    # never the sum
    assert total_logical_qubits(walk, m, a_prep) < walk + m + a_prep


def test_walk_query_constant_is_heisenberg_pi():
    """Adopted module default is π (Heisenberg), not Babbush's √2·π upper bound. The upper
    bound stays available as a named constant to reproduce the historical r3 raw shards."""
    from src_PI.estimation.qpe_cost import WALK_QUERY_CONSTANT_BABBUSH_UB
    assert abs(WALK_QUERY_CONSTANT - math.pi) < 1e-12
    assert abs(WALK_QUERY_CONSTANT - 3.14159265) < 1e-6
    assert abs(WALK_QUERY_CONSTANT_BABBUSH_UB - math.sqrt(2.0) * math.pi) < 1e-12


import pytest

from src_PI.estimation.qpe_cost import overlap_repetition_factor


def test_repetition_exact_bernoulli():
    """R = ⌈ln δ / ln(1−p0)⌉ — the EXACT 'at least one ground hit in R shots' count (warmstart
    audit §1: cold p0=0.4819 -> 8, warm p0=0.9051 -> 2, at 99% confidence)."""
    for p0, expect in [(0.4819, 8), (0.9051, 2)]:
        r = overlap_repetition_factor(p0, confidence=0.99)
        assert r["branch"] == "sampling" and r["ae_register_qubits"] == 0
        assert r["R"] == expect, (p0, r["R"])
        assert (1 - p0) ** r["R"] <= 0.01 + 1e-12          # meets the confidence...
        assert (1 - p0) ** (r["R"] - 1) > 0.01             # ...and R is minimal
        assert abs(r["expected_shots"] - 1.0 / p0) < 1e-12  # the distinct mean-shots metric


def test_old_log_over_p0_was_an_overcount():
    """The retired ⌈ln(1/δ)/p0⌉ approximation over-counts at high p0 vs the exact Bernoulli."""
    p0, delta = 0.9051, 0.01
    old_approx = math.ceil(math.log(1.0 / delta) / p0)     # 6
    exact = overlap_repetition_factor(p0)["R"]             # 2
    assert exact < old_approx


def test_binary_is_experimental_opt_in_only():
    """Binary is NEVER default/auto (audit §2): default is sampling even at low p0; binary must be
    explicitly requested and carries its own AE register; 'auto' is gone."""
    assert overlap_repetition_factor(1e-4)["branch"] == "sampling"
    rb = overlap_repetition_factor(1e-4, branch="binary")
    assert rb["branch"] == "binary" and rb["ae_register_qubits"] >= 1
    with pytest.raises(ValueError):
        overlap_repetition_factor(0.5, branch="auto")


def test_higher_overlap_fewer_repetitions():
    """The warm-start payoff: raising p0 never increases R (exact sampling)."""
    assert overlap_repetition_factor(0.9)["R"] <= overlap_repetition_factor(0.1)["R"]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll qpe_cost tests passed.")
