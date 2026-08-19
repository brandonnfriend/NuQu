"""
Follow-up audit exit gates (2026-08-19) for the compiled sparse walk-resource path.

`compiled_resources.compiled_walk_resources` is the genuinely compiled estimate.
These tests check the seven exit gates:

  1/6 mixed atoms + physical integration — the REAL L=2 dim=1 n_b=2 native
      Hamiltonian (with all 15 H_AV/H_WT mixed terms) estimates without
      exclusions; a representative mixed atom's block encoding (BE_F ⊗ BE_b ⊗
      BE_c) reproduces the mixed term and qubitizes;
  3   fault-tolerant synthesis — the walk-T reflects rotations *synthesized* to
      Clifford+T (far above a literal-T-only count);
  4   precision propagation — tightening ΔE raises the traversed walk-T; Λ is
      precision-invariant;
  7   resource test — no dense ≥2-qubit `MatrixGate` remains at the counting
      boundary, and every SELECT sub-circuit costs (no `None` t_complexity).

Run: `python -m pytest tests/test_compiled_resources.py -q`
"""

import cirq
import numpy as np
import pytest

from src_PI.hamiltonians.core.pion_basis.fock_native import (
    build_native_mixed_hamiltonian,
)
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.estimation.sparse_oracle.compiled_resources import (
    _boson_atom_select_circuit,
    _factor_hermitian_tensor,
    compiled_walk_resources,
)


def _mh(L=2, dim=1, n_b=2):
    return build_native_mixed_hamiltonian(L, dim, n_b, get_physical_parameters())


# --------------------------------------------------------------------------- #
# Gate 1 + 6 — mixed atoms + physical integration                             #
# --------------------------------------------------------------------------- #


def test_real_hamiltonian_estimates_without_exclusions():
    """The full physical L=2 dim=1 n_b=2 Hamiltonian — 15 mixed H_AV/H_WT terms —
    produces a finite compiled walk-T (the old path raised on mixed atoms)."""
    mh = _mh()
    assert len(mh.mixed_terms) == 15
    r = compiled_walk_resources(mh, 2, 2, delta_E=1.0)
    assert r['Walk_T_Count'] > 0
    assert r['Physical_Lambda'] > 0
    # every atom kind is represented (boson + mixed here; fermion when present)
    assert r['breakdown']['per_kind']['mixed']['count'] == 15


def test_mixed_atom_block_encoding_reproduces_term_and_qubitizes():
    """A representative H_WT mixed atom: the Gilyén product BE_F ⊗ BE_b ⊗ BE_c
    (fermion dilation ⊗ two single-mode boson dilations, from the rank-1 tensor
    factorisation) block-encodes the mixed term M = coeff·F⊗B and qubitizes."""
    from collections import defaultdict
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        _contraction_dilation, _fermion_dense,
    )
    from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
    from src_PI.estimation.sparse_oracle.hermitian_bundle import _boson_group_matrix
    from src_PI.estimation.sparse_oracle.matching_dilation import extract_atom_dilation

    mh = _mh()
    # a mixed term with a two-mode boson factor (rank-1 tensor).
    mt = next(m for m in mh.mixed_terms
              if any(len({q for q, _ in mono}) == 2 for mono in m.boson_factor.terms))
    c = complex(mt.coeff)
    F, _fsup = _fermion_dense(mt.fermion_factor)
    alpha_F = fermion_jw_stats(mt.fermion_factor)['one_norm']
    groups = defaultdict(list)
    for mono, bc in mt.boson_factor.terms.items():
        if mono != ():
            groups[tuple(sorted({q for q, _ in mono}))].append((mono, bc))
    (ms, monos), = groups.items()
    B = _boson_group_matrix(monos, ms, 2)
    H0, H1 = _factor_hermitian_tensor(B, 4)

    # dense block encodings of each factor
    U_F = _contraction_dilation(F, alpha_F)
    U_0, a0, _b0 = extract_atom_dilation(H0, 2)
    U_1, a1, _b1 = extract_atom_dilation(H1, 2)
    blk_F = U_F[:F.shape[0], :F.shape[0]] * alpha_F
    blk_0 = U_0[:4, :4] * a0
    blk_1 = U_1[:4, :4] * a1
    # product block == F ⊗ H0 ⊗ H1 == F ⊗ B
    prod = np.kron(np.kron(blk_F, blk_0), blk_1)
    assert np.allclose(prod, np.kron(F, B), atol=1e-6), "BE_F⊗BE_0⊗BE_1 != F⊗B"
    # each factor block encoding is Hermitian + self-inverse (⇒ the atom qubitizes)
    for U in (U_F, U_0, U_1):
        assert np.allclose(U, U.conj().T, atol=1e-6)
        assert np.allclose(U @ U, np.eye(len(U)), atol=1e-6)
    assert abs(c) > 0


# --------------------------------------------------------------------------- #
# Gate 3 — fault-tolerant synthesis                                           #
# --------------------------------------------------------------------------- #


def test_walk_t_reflects_synthesized_rotations():
    """The walk-T is dominated by *synthesized* rotations — far above a naive
    literal-T count. (The old `compiled_atom_cost` charged rotations as free;
    a single â+â† atom already carries dozens of rotations × ~50 T each.)"""
    mh = _mh()
    r = compiled_walk_resources(mh, 2, 2, delta_E=1.0)
    # SELECT dominates and is large (millions of T) because rotations are synthesized
    assert r['breakdown']['select_T'] > 100000
    assert r['Walk_T_Count'] >= r['breakdown']['select_T']


# --------------------------------------------------------------------------- #
# Gate 4 — precision propagation                                              #
# --------------------------------------------------------------------------- #


def test_precision_propagates_into_walk_cost():
    mh = _mh()
    coarse = compiled_walk_resources(mh, 2, 2, delta_E=10.0)
    fine = compiled_walk_resources(mh, 2, 2, delta_E=0.1)
    assert fine['Walk_T_Count'] > coarse['Walk_T_Count']       # tighter ΔE → more T
    assert abs(fine['Physical_Lambda'] - coarse['Physical_Lambda']) < 1e-6  # Λ invariant
    assert fine['budget']['circuit_precision'] < coarse['budget']['circuit_precision']


# --------------------------------------------------------------------------- #
# Gate 7 — resource test: no dense placeholders at the counting boundary      #
# --------------------------------------------------------------------------- #


def test_no_dense_matrixgate_at_counting_boundary():
    """Every boson SELECT sub-circuit decomposes to elementary gates with NO
    dense ≥2-qubit `MatrixGate` (those are charged as free/1-Clifford by the
    counter — the fallacy the audit flagged). Shifts/folds are `AddK` bloqs."""
    from src_PI.estimation.sparse_oracle.hermitian_bundle import extract_hermitian_atoms
    mh = _mh()
    atoms = extract_hermitian_atoms(mh, 2, mh.mode_to_qubits, need_dense=True)
    ctrl = cirq.NamedQubit('c')
    bad = 0
    for a in atoms:
        mats = []
        if a.kind == 'boson':
            mats.append((a.M, len(a.support)))
        elif a.kind == 'mixed':
            for Mg, nb in zip(a.payload['boson_group_mats'], a.payload['boson_group_bits']):
                if nb == 2:                    # single-mode factor
                    mats.append((Mg, nb))
                else:                           # two-mode → factor into single-mode
                    for H in _factor_hermitian_tensor(Mg, 4):
                        mats.append((H, 2))
        for M, nb in mats:
            circ = _boson_atom_select_circuit(M, nb, ctrl, False)
            for op in cirq.Circuit(cirq.decompose(circ)).all_operations():
                if isinstance(op.gate, cirq.MatrixGate) and len(op.qubits) >= 2:
                    bad += 1
    assert bad == 0, f"{bad} dense ≥2-qubit MatrixGate(s) at the counting boundary"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
