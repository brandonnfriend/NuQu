"""
Validation for the Hermitian full-bundle assembly (C1 walk-validity rebuild).

`hermitian_bundle.build_hermitian_bundle_sim` re-groups a `MixedHamiltonian`
into Hermitian atoms and LCU-combines them with a Hermitian SELECT, so the block
encoding `U` is Hermitian (`U=U†`) and the single-reflection walk `W=(2Π−I)U`
qubitizes `H` (the property the non-Hermitian d=1 bundle lacked). Also checks the
Hermitian `α_tot` is a valid, *tighter* Λ than the per-monomial
`compute_native_lambda`.

Run: `python -m pytest tests/test_hermitian_bundle.py -q`
"""

import numpy as np
import pytest
from openfermion import BosonOperator, FermionOperator

from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian, MixedTerm
from src_PI.estimation.sparse_oracle.hermitian_bundle import (
    build_hermitian_bundle_sim,
    extract_hermitian_atoms,
    hermitian_bundle_reference,
)


def _walk_qubitizes(U, alpha, N, H):
    Pi = np.diag([1.0] * N + [0.0] * (len(U) - N))
    W = (2 * Pi - np.eye(len(U))) @ U
    wph = np.angle(np.linalg.eigvals(W))
    for e in np.linalg.eigvalsh((H + H.conj().T) / 2):
        th = np.arccos(np.clip(e / alpha, -1.0, 1.0))
        if np.min(np.abs((wph - th + np.pi) % (2 * np.pi) - np.pi)) > 1e-6:
            return False
    return True


def _assert_valid_walk(mh, n_b, m2q, w_sys):
    U, alpha, _ws = build_hermitian_bundle_sim(mh, n_b, m2q, w_sys)
    H = hermitian_bundle_reference(mh, n_b, m2q, w_sys)
    N = 1 << w_sys
    assert np.allclose(H, H.conj().T, atol=1e-9), "test Hamiltonian must be Hermitian"
    assert np.allclose(U, U.conj().T, atol=1e-9), "bundle U must be Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-9), "bundle U must be self-inverse"
    assert np.allclose(U[:N, :N] * alpha, H, atol=1e-9), "α_tot·⟨0|U|0⟩ = H"
    assert _walk_qubitizes(U, alpha, N, H), "walk must qubitize H"


def test_hermitian_bundle_boson_only_qubitizes():
    """(â+â†) + 0.5 n̂ on one mode — the toy the old bundle's xfail uses."""
    bp = (BosonOperator('0', 1.0) + BosonOperator('0^', 1.0)
          + BosonOperator('0^ 0', 0.5))
    mh = MixedHamiltonian(boson_part=bp, mode_to_qubits={0: [0, 1]})
    _assert_valid_walk(mh, 2, {0: [0, 1]}, 2)


def test_hermitian_bundle_heterogeneous_qubitizes():
    """Imaginary H_WT-style boson phase + fermion + a mixed term (F⊗B)."""
    bp = (BosonOperator('0 1', 0.7j) + BosonOperator('0^ 1^', -0.7j)
          + BosonOperator('0^ 0', 0.5))                          # Hermitian
    fp = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    mt = MixedTerm(coeff=0.4,
                   fermion_factor=FermionOperator('0^ 1') + FermionOperator('1^ 0'),
                   boson_factor=BosonOperator('0', 1.0) + BosonOperator('0^', 1.0))
    mh = MixedHamiltonian(boson_part=bp, fermion_part=fp, mixed_terms=[mt],
                          mode_to_qubits={0: [2], 1: [3]})
    _assert_valid_walk(mh, 1, {0: [2], 1: [3]}, 4)


def test_hermitian_bundle_alpha_is_valid_and_tighter():
    """On the real L=2 dim=1 bundle, α_tot = Σ atom α is ≤ compute_native_lambda
    (Hermitization + edge colouring tighten Λ)."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    atoms = extract_hermitian_atoms(mh, 2, mh.mode_to_qubits)
    alpha_tot = sum(a.alpha for a in atoms)
    lam = compute_native_lambda(mh, 2)['physical_lambda']
    assert alpha_tot > 0
    assert alpha_tot <= lam + 1e-6 * lam


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
