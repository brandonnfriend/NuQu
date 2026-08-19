"""
Tests for the analytic PauliLCU resource scaling (the L=10 publication anchor).

The load-bearing claim is that the block-encoding subnormalisation λ (and the
qubit count) scale EXACTLY as `a·S + b·N_bonds` for the translation-invariant
lattice Hamiltonian, so λ(L=10) / N_walk / logical qubits are pinned by exact
combinatorics. The walk_T / total_T are an explicitly-labelled model (biased low
by JW non-locality), not asserted exact.

Run: `python -m pytest tests/test_pauli_lcu_scaling.py -q`
"""

import pytest

from src_PI.estimation.pauli_lcu_scaling import (
    n_bonds,
    pauli_lcu_resources,
    validation_table,
)


def test_n_bonds_open_bc():
    # dim=1 open chain: L-1 bonds; dim=3: 3·L²·(L-1)
    assert n_bonds(1, 1) == 0
    assert n_bonds(10, 1) == 9
    assert n_bonds(1, 3) == 0
    assert n_bonds(2, 3) == 12
    assert n_bonds(3, 3) == 54
    assert n_bonds(10, 3) == 2700


def test_lambda_model_is_exact_on_calibration():
    """λ = a·S + b·N_bonds reproduces the compiled λ to < 0.1% at L=1,2,3."""
    for r in validation_table(3):
        rel = abs(r['lam_model'] - r['lam_actual']) / r['lam_actual']
        assert rel < 1e-3, f"L={r['L']}: λ model off by {rel:.1%}"


def test_walk_T_model_matches_at_calibration_scale():
    """The walk_T ≈ κ·total_weight model is within ~5% of compiled at L=2,3
    (the regime that anchors the extrapolation); L=1 is a tiny-system outlier."""
    for r in validation_table(3):
        if r['L'] == 1:
            continue
        rel = abs(r['walkT_model'] - r['walkT_actual']) / r['walkT_actual']
        assert rel < 0.05, f"L={r['L']}: walk_T model off by {rel:.1%}"


def test_l10_dim3_anchor_is_sane():
    r = pauli_lcu_resources(10, 3)
    assert r['sites'] == 1000 and r['bonds'] == 2700
    # exact quantities
    assert 8.0e6 < r['lambda'] < 9.5e6
    assert r['walk_queries'] > 3.5e7
    assert 8000 < r['logical_qubits'] < 9200        # ~ (4 + 3·2)·1000
    # modelled total_T is order 1e16, with an asymmetric (low-biased) band
    assert 1e16 < r['total_T_model'] < 2e16
    lo, hi = r['total_T_band']
    assert lo < r['total_T_model'] < hi
    assert hi > r['total_T_model'] * 1.5            # upward JW-nonlocality room


def test_lambda_monotone_in_L():
    prev = 0.0
    for L in (1, 2, 3, 6, 10):
        lam = pauli_lcu_resources(L, 3)['lambda']
        assert lam > prev
        prev = lam


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
