"""
P0-4 (sparse compile): the precision/error budget.

Codex's audit flagged the hard-coded `probability_epsilon=1e-3`, `kappa=8` as
not publication-grade. This checks the derived budget:

  * the total energy error splits into ε_QPE (resolution) + ε_BE (block encoding),
  * every approximation parameter (alias `probability_epsilon`, rotation
    `circuit_precision`/`kappa`) is derived from ΔE and λ, with the expected
    scaling (N_walk ∝ 1/ΔE, precision bits ∝ log(1/ΔE)), and
  * the budget flows into the actual resource estimate: tightening ΔE raises the
    walk-T and qubit count, while λ (the subnormalization) is precision-invariant.

Run: `python -m pytest tests/test_precision_budget.py -q`
"""

import math

import pytest

from src_PI.estimation.sparse_oracle.precision_budget import (
    qpe_error_budget,
    sensitivity_table,
)


def test_budget_splits_energy_and_derives_parameters():
    b = qpe_error_budget(physical_lambda=4471.0, delta_E=1.0, qpe_fraction=0.5)
    # energy split adds up to ΔE
    assert abs(b['eps_qpe'] + b['eps_be'] - 1.0) < 1e-12
    # N_walk = π·λ/ε_QPE (adopted constant)
    assert abs(b['walk_queries'] - math.pi * 4471.0 / b['eps_qpe']) < 1e-6
    # block-encoding error λ·(ε_prep + ε_rot) == ε_BE (even split)
    assert abs(4471.0 * (b['probability_epsilon'] + b['circuit_precision'])
               - b['eps_be']) < 1e-9
    # integer precisions satisfy 2^-bits ≤ ε
    assert 2.0 ** (-b['kappa']) <= b['circuit_precision']
    assert 2.0 ** (-b['alias_mu_bits']) <= b['probability_epsilon']


def test_budget_scaling_with_delta_E():
    """N_walk ∝ 1/ΔE; precision bits grow ~log2(1/ΔE) (≈ +3.3 bits/decade)."""
    rows = sensitivity_table(4471.0, delta_Es=(1.0, 0.1))
    coarse, fine = rows[0], rows[1]
    assert abs(fine['walk_queries'] / coarse['walk_queries'] - 10.0) < 1e-6
    assert 3 <= fine['kappa'] - coarse['kappa'] <= 4          # ~log2(10) ≈ 3.32


def test_tighter_lambda_needs_more_precision():
    """A larger λ needs a smaller probability_epsilon (finer coeff precision) to
    keep λ·ε_prep under the same energy budget."""
    small = qpe_error_budget(1000.0, 1.0)
    big = qpe_error_budget(100000.0, 1.0)
    assert big['probability_epsilon'] < small['probability_epsilon']
    assert big['alias_mu_bits'] > small['alias_mu_bits']


def test_budget_rejects_bad_inputs():
    with pytest.raises(ValueError):
        qpe_error_budget(4471.0, 1.0, qpe_fraction=0.0)
    with pytest.raises(ValueError):
        qpe_error_budget(-1.0, 1.0)


def test_budget_flows_into_resource_estimate():
    """Tightening ΔE raises the walk-T + qubit count via the derived precision,
    while λ (the subnormalization) is precision-invariant."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.hermitian_bundle import (
        estimate_hermitian_sparse_resources,
    )
    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    coarse = estimate_hermitian_sparse_resources(mh, 2, 2, delta_E=10.0)
    fine = estimate_hermitian_sparse_resources(mh, 2, 2, delta_E=0.1)
    assert fine['Walk_T_Count'] > coarse['Walk_T_Count']
    assert fine['Logical_Qubits'] >= coarse['Logical_Qubits']
    assert abs(fine['Physical_Lambda'] - coarse['Physical_Lambda']) < 1e-6
    # the budget is reported for reproducibility
    assert coarse['budget']['delta_E'] == 10.0
    assert fine['budget']['delta_E'] == 0.1


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
