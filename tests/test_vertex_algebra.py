"""
Physics-oracle tests for the nucleon spin-isospin vertices (Workstream A2).

WHY THIS FILE EXISTS
--------------------
The broad unit suite checked *software consistency* (shapes, dispatch, term
counts) but never checked the *algebra* against an independent source. That is
how the vertex bug survived: the transition builders returned
`a†_α a_β + a†_β a_α` while every caller already summed over all ordered (α,β)
pairs, so symmetric channels were doubled and antisymmetric/imaginary channels
(τ_y, σ_y·τ_y, ...) cancelled to exactly zero. Hermiticity and fermion-number
conservation both still held with the bug, so those checks alone are NOT enough.

The decisive test here is `test_vertex_one_body_matrix_equals_kron`: the one-body
coefficient matrix of the assembled vertex must equal the Kronecker product
`σ_S ⊗ τ_I` built from bare Pauli matrices — an oracle that owes nothing to the
codebase's own conventions. Every spin_idx ∈ {0,1,2,3} × iso_idx ∈ {1,2,3}
channel is covered, including all imaginary ones.

Run: .venv/bin/python -m pytest -q tests/test_vertex_algebra.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openfermion import (
    FermionOperator,
    QubitOperator,
    get_sparse_operator,
    hermitian_conjugated,
    jordan_wigner,
    normal_ordered,
)

from src_PI.hamiltonians.core.Operators import Nucleon_Transition_JW
from src_PI.hamiltonians.core.pion_basis import fock
from src_PI.hamiltonians.core.pion_basis.fock_native import (
    _MODES,
    _nucleon_transition_fermion,
)
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.LatticeGeometry import (
    get_total_sites,
    site_to_nucleon_qubit,
    site_to_pion_qubit,
)
from src_PI.utils.utils import calculate_chiral_coeff, sigma_mats

# All spin factors (0 = identity, used by H_WT; 1,2,3 = σ_x,σ_y,σ_z used by H_AV)
# crossed with all isospin/pion species (1,2,3 = τ_x,τ_y,τ_z).
_SPIN_IDXS = (0, 1, 2, 3)
_ISO_IDXS = (1, 2, 3)
# Channels whose coefficient matrix is antisymmetric/imaginary — the ones the bug
# erased. Guarded explicitly so a regression to cancellation can't pass silently.
_IMAGINARY_CHANNELS = [(iso, spin) for iso in _ISO_IDXS for spin in _SPIN_IDXS
                       if 2 in (iso, spin) and not (iso == 2 and spin == 2)]


# --------------------------------------------------------------------------- #
# Independent constructions                                                    #
# --------------------------------------------------------------------------- #

def _expected_vertex_fermionop(iso_idx, spin_idx, n_b=1, site=0):
    """Independent Σ_{αβ} χ_{αβ} a†_α a_β on `site` (ordered sum, no h.c. added)."""
    op = FermionOperator()
    for a in _MODES:
        for b in _MODES:
            c = calculate_chiral_coeff(a, b, iso_idx, spin_idx)
            if abs(c) > 1e-12:
                qa = site_to_nucleon_qubit(site, a, n_b)
                qb = site_to_nucleon_qubit(site, b, n_b)
                op += FermionOperator(f'{qa}^ {qb}', c)
    return normal_ordered(op)


def _implemented_vertex_fermionop(iso_idx, spin_idx, n_b=1, site=0):
    """Vertex assembled the way the H_AV / H_WT builders do it."""
    op = FermionOperator()
    for a in _MODES:
        for b in _MODES:
            c = calculate_chiral_coeff(a, b, iso_idx, spin_idx)
            if abs(c) > 1e-12:
                op += c * _nucleon_transition_fermion(site, a, b, n_b)
    return normal_ordered(op)


def _one_body_matrix(fermion_op, n_b=1, site=0):
    """Extract the 4x4 one-body coefficient matrix in _MODES order."""
    qubit_to_pos = {site_to_nucleon_qubit(site, m, n_b): k
                    for k, m in enumerate(_MODES)}
    M = np.zeros((4, 4), dtype=complex)
    for term, coeff in normal_ordered(fermion_op).terms.items():
        if term == ():
            continue
        assert len(term) == 2 and term[0][1] == 1 and term[1][1] == 0, term
        M[qubit_to_pos[term[0][0]], qubit_to_pos[term[1][0]]] += coeff
    return M


def _qubitop_allclose(a, b, tol=1e-9):
    diff = a - b
    return all(abs(c) < tol for c in diff.terms.values())


# --------------------------------------------------------------------------- #
# Part 1 — the decisive oracle: vertex one-body matrix == kron(σ_S, τ_I)        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("iso_idx", _ISO_IDXS)
@pytest.mark.parametrize("spin_idx", _SPIN_IDXS)
def test_vertex_one_body_matrix_equals_kron(iso_idx, spin_idx):
    """Σ_{αβ} χ_{αβ} a†_α a_β must have one-body matrix σ_S ⊗ τ_I.

    Fully independent of codebase conventions (bare Pauli Kronecker product).
    _MODES is ordered (spin outer, isospin inner), so the Kronecker factor
    order is (σ_S, τ_I).
    """
    oracle = np.kron(sigma_mats[spin_idx], sigma_mats[iso_idx])
    got = _one_body_matrix(_implemented_vertex_fermionop(iso_idx, spin_idx))
    assert np.allclose(got, oracle, atol=1e-12), (
        f"(iso={iso_idx}, spin={spin_idx})\n got=\n{got}\n oracle=\n{oracle}")


@pytest.mark.parametrize("iso_idx,spin_idx", _IMAGINARY_CHANNELS)
def test_imaginary_channels_do_not_cancel(iso_idx, spin_idx):
    """The channels the bug erased must be nonzero (guard against regression)."""
    M = _one_body_matrix(_implemented_vertex_fermionop(iso_idx, spin_idx))
    assert np.linalg.norm(M) > 1e-9, f"channel (iso={iso_idx}, spin={spin_idx}) vanished"
    assert np.max(np.abs(M.imag)) > 1e-9, "expected a genuinely imaginary channel"


@pytest.mark.parametrize("iso_idx", _ISO_IDXS)
@pytest.mark.parametrize("spin_idx", _SPIN_IDXS)
def test_native_builder_matches_independent_construction(iso_idx, spin_idx):
    """fock_native vertex == independent ordered-bilinear sum (no doubling)."""
    assert (_implemented_vertex_fermionop(iso_idx, spin_idx)
            == _expected_vertex_fermionop(iso_idx, spin_idx))


@pytest.mark.parametrize("iso_idx", _ISO_IDXS)
@pytest.mark.parametrize("spin_idx", _SPIN_IDXS)
def test_jw_path_matches_native_path(iso_idx, spin_idx):
    """The JW/PauliLCU builder (Operators.py) agrees with the native builder.

    Guards against the two independent builders (used by the PauliLCU vs
    sparse/classical paths) drifting apart — e.g. different qubit indexing.
    """
    n_b = 1
    jw = QubitOperator()
    for a in _MODES:
        for b in _MODES:
            c = calculate_chiral_coeff(a, b, iso_idx, spin_idx)
            if abs(c) > 1e-12:
                jw += c * Nucleon_Transition_JW(0, a, b, n_b)
    expected = jordan_wigner(_implemented_vertex_fermionop(iso_idx, spin_idx, n_b))
    assert _qubitop_allclose(jw, expected)


@pytest.mark.parametrize("iso_idx", _ISO_IDXS)
@pytest.mark.parametrize("spin_idx", _SPIN_IDXS)
def test_vertex_is_hermitian(iso_idx, spin_idx):
    op = _implemented_vertex_fermionop(iso_idx, spin_idx)
    assert normal_ordered(op - hermitian_conjugated(op)) == FermionOperator()


# --------------------------------------------------------------------------- #
# Part 2 — full assembled H_AV / H_WT (integration; small system)              #
# --------------------------------------------------------------------------- #

def _params():
    return get_physical_parameters()


def _n_qubits(*ops):
    n = 0
    for op in ops:
        for term in op.terms:
            for q, _ in term:
                n = max(n, q + 1)
    return n


def _nucleon_number_op(L, dim, n_b):
    N = FermionOperator()
    for x in range(get_total_sites(L, dim)):
        for m in _MODES:
            q = site_to_nucleon_qubit(x, m, n_b)
            N += FermionOperator(f'{q}^ {q}')
    return jordan_wigner(N)


@pytest.mark.parametrize("builder", [fock.H_axial_vector, fock.H_WT_Logic])
def test_full_vertex_hamiltonian_is_hermitian(builder):
    L, dim, n_b = 2, 1, 1
    H = builder(L, dim, n_b, _params())
    nq = _n_qubits(H)
    m = get_sparse_operator(H, n_qubits=nq)
    assert abs((m - m.getH())).max() < 1e-9


@pytest.mark.parametrize("builder", [fock.H_axial_vector, fock.H_WT_Logic])
def test_full_vertex_conserves_fermion_number(builder):
    L, dim, n_b = 2, 1, 1
    H = builder(L, dim, n_b, _params())
    N = _nucleon_number_op(L, dim, n_b)
    nq = _n_qubits(H, N)
    Hm = get_sparse_operator(H, n_qubits=nq)
    Nm = get_sparse_operator(N, n_qubits=nq)
    comm = Hm @ Nm - Nm @ Hm
    assert abs(comm).max() < 1e-9


# --------------------------------------------------------------------------- #
# Part 3 — zero/linear coupling limits                                          #
# --------------------------------------------------------------------------- #

def test_h_av_vanishes_at_zero_axial_coupling():
    L, dim, n_b = 2, 1, 1
    p = _params()
    p['g_A'] = 0.0
    H = fock.H_axial_vector(L, dim, n_b, p)
    assert all(abs(c) < 1e-12 for c in H.terms.values()), "H_AV must vanish at g_A=0"


def test_h_av_is_linear_in_axial_coupling():
    L, dim, n_b = 2, 1, 1
    p1 = _params(); p1['g_A'] = 1.3
    p2 = _params(); p2['g_A'] = 2.6
    H1 = fock.H_axial_vector(L, dim, n_b, p1)
    H2 = fock.H_axial_vector(L, dim, n_b, p2)
    assert _qubitop_allclose(H2, 2.0 * H1)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
