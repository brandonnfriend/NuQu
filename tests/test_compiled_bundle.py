"""
P0-2/P0-3 (sparse compile): the Hermitian bundle composite is a real decomposable
circuit whose extracted matrix == H and whose walk qubitizes.

Codex's audit found the bundle had no executable decomposition. This exercises
the PRODUCTION composite (`compiled_bundle.compiled_bundle_ops`) — outer
`PREP·SELECT·PREP†` with heterogeneous dispatch (compiled single-mode boson atoms
+ the fermion atom) — on small instances:

  * it decomposes to elementary gates (no `DecomposeNotImplementedError`),
  * `α_tot·⟨0|_flag U|0⟩ = Σ_l M_l = H` (the actual production circuit, not a
    parallel NumPy model),
  * `U = U†`, `U² = I`, and the walk has the qubitization spectrum, and
  * the cost is traversed from the circuit (a real T-count), not a hand
    `_t_complexity_`.

Scope: the dominant sectors (single-mode boson ~88% of Λ, fermion ~11%). Two-mode
boson atoms raise `NotImplementedError` (P0-1 remaining) — nothing is silently
dropped.
"""

import cirq
import numpy as np
import pytest
from openfermion import BosonOperator, FermionOperator

from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
from src_PI.estimation.sparse_oracle.compiled_bundle import (
    compiled_bundle_ops,
    compiled_bundle_reference,
    compiled_bundle_widths,
    extract_compiled_bundle,
)


def _walk_qubitizes(U, alpha, N, H):
    Pi = np.diag([1.0] * N + [0.0] * (len(U) - N))
    W = (2 * Pi - np.eye(len(U))) @ U
    wph = np.angle(np.linalg.eigvals(W))
    for e in np.linalg.eigvalsh((H + H.conj().T) / 2):
        th = np.arccos(np.clip(e / alpha, -1, 1))
        if np.min(np.abs((wph - th + np.pi) % (2 * np.pi) - np.pi)) > 1e-6:
            return False
    return True


def _assert_valid_composite(mh, n_b):
    U, alpha, block = extract_compiled_bundle(mh, n_b, mh.mode_to_qubits)
    H = compiled_bundle_reference(mh, n_b, mh.mode_to_qubits)
    N = block.shape[0]
    assert np.allclose(block, H, atol=1e-6), "α_tot·⟨0|U|0⟩ != H"
    assert np.allclose(U, U.conj().T, atol=1e-6), "composite U not Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-6), "composite U not self-inverse"
    assert _walk_qubitizes(U, alpha, N, H), "composite walk does not qubitize"


def test_boson_composite_qubitizes():
    """Two single-mode boson atoms (2 modes) assemble into a valid walk."""
    bp = (BosonOperator('0', 1.0) + BosonOperator('0^', 1.0)
          + BosonOperator('0^ 0', 0.5)
          + BosonOperator('1', 1.0) + BosonOperator('1^', 1.0))
    mh = MixedHamiltonian(boson_part=bp, mode_to_qubits={0: [0, 1], 1: [2, 3]})
    _assert_valid_composite(mh, 2)


def test_heterogeneous_composite_qubitizes():
    """Compiled boson atoms + the fermion atom (heterogeneous dispatch)."""
    bp = (BosonOperator('0', 1.0) + BosonOperator('0^', 1.0)
          + BosonOperator('0^ 0', 0.5)
          + BosonOperator('1', 1.0) + BosonOperator('1^', 1.0))
    fp = FermionOperator('4^ 5', 1.0) + FermionOperator('5^ 4', 1.0)
    mh = MixedHamiltonian(boson_part=bp, fermion_part=fp,
                          mode_to_qubits={0: [0, 1], 1: [2, 3]})
    _assert_valid_composite(mh, 2)


def test_composite_decomposes_with_genuine_t_count():
    """The composite fully decomposes to elementary gates (no
    DecomposeNotImplementedError) with a real, finite, circuit-traversed T-count
    — not a hand-assembled `_t_complexity_`."""
    bp = (BosonOperator('0', 1.0) + BosonOperator('0^', 1.0)
          + BosonOperator('0^ 0', 0.5)
          + BosonOperator('1', 1.0) + BosonOperator('1^', 1.0))
    mh = MixedHamiltonian(boson_part=bp, mode_to_qubits={0: [0, 1], 1: [2, 3]})
    b_out, b_inner, w_sys, atoms = compiled_bundle_widths(mh, 2, mh.mode_to_qubits)
    osel = [cirq.NamedQubit(f'o{i}') for i in range(b_out)]
    isel = [cirq.NamedQubit(f'i{i}') for i in range(b_inner)]
    b_dil = cirq.NamedQubit('b_dil')
    sysq = [cirq.NamedQubit(f's{i}') for i in range(w_sys)]
    flat = cirq.Circuit(cirq.decompose(cirq.Circuit(
        compiled_bundle_ops(osel, isel, b_dil, sysq, atoms, mh.mode_to_qubits, 2))))
    t = sum(1 for op in flat.all_operations() if op.gate in (cirq.T, cirq.T ** -1))
    assert 0 < t < 100000, f"genuine compiled T-count = {t}"


def test_two_mode_atom_raises_not_silently_dropped():
    """A bundle with a two-mode boson atom raises (P0-1 remaining) rather than
    silently dropping it."""
    bp = BosonOperator('0 1', 0.7) + BosonOperator('0^ 1^', 0.7)   # two-mode
    mh = MixedHamiltonian(boson_part=bp, mode_to_qubits={0: [0, 1], 1: [2, 3]})
    with pytest.raises(NotImplementedError, match="two-mode"):
        extract_compiled_bundle(mh, 2, mh.mode_to_qubits)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
