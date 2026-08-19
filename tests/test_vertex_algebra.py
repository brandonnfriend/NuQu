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
from src_PI.hamiltonians.core.pion_basis.fock import _bosonop_to_qubitop
from src_PI.hamiltonians.core.pion_basis.fock_native import (
    _MODES,
    _nucleon_transition_fermion,
    build_native_mixed_hamiltonian,
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
# Representative (spin, iso) channels for the secondary cross-checks (the
# decisive per-channel guarantee is the full-breadth kron oracle + the imaginary
# guard above; these cross-checks need only a sample: real, imaginary, H_WT spin=0).
_SAMPLE_CHANNELS = [(1, 1), (2, 2), (0, 3), (3, 1)]


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

def test_vertex_one_body_matrix_equals_kron():
    """The decisive oracle over ALL 12 channels: Σ_{αβ} χ_{αβ} a†_α a_β has
    one-body matrix σ_S ⊗ τ_I (bare Pauli Kronecker product, independent of
    codebase conventions). _MODES is (spin outer, isospin inner)."""
    for spin_idx in _SPIN_IDXS:
        for iso_idx in _ISO_IDXS:
            oracle = np.kron(sigma_mats[spin_idx], sigma_mats[iso_idx])
            got = _one_body_matrix(_implemented_vertex_fermionop(iso_idx, spin_idx))
            assert np.allclose(got, oracle, atol=1e-12), (
                f"(iso={iso_idx}, spin={spin_idx})\n got=\n{got}\n oracle=\n{oracle}")


def test_imaginary_channels_do_not_cancel():
    """The channels the bug erased must be nonzero + genuinely imaginary
    (regression guard, over all such channels)."""
    for iso_idx, spin_idx in _IMAGINARY_CHANNELS:
        M = _one_body_matrix(_implemented_vertex_fermionop(iso_idx, spin_idx))
        tag = f"(iso={iso_idx}, spin={spin_idx})"
        assert np.linalg.norm(M) > 1e-9, f"channel {tag} vanished"
        assert np.max(np.abs(M.imag)) > 1e-9, f"{tag}: expected imaginary"


def test_vertex_builders_agree_and_are_hermitian():
    """Over a representative channel sample: the native builder == the independent
    ordered-bilinear sum (no doubling), the JW/PauliLCU builder agrees with it,
    and the vertex is Hermitian — guarding the two builders against drift."""
    n_b = 1
    for spin_idx, iso_idx in _SAMPLE_CHANNELS:
        native = _implemented_vertex_fermionop(iso_idx, spin_idx)
        assert native == _expected_vertex_fermionop(iso_idx, spin_idx)
        jw = QubitOperator()
        for a in _MODES:
            for b in _MODES:
                c = calculate_chiral_coeff(a, b, iso_idx, spin_idx)
                if abs(c) > 1e-12:
                    jw += c * Nucleon_Transition_JW(0, a, b, n_b)
        assert _qubitop_allclose(
            jw, jordan_wigner(_implemented_vertex_fermionop(iso_idx, spin_idx, n_b)))
        assert normal_ordered(native - hermitian_conjugated(native)) == FermionOperator()


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


def test_full_vertex_hamiltonian_is_hermitian_and_conserves_number():
    """Both assembled builders (H_AV, H_WT) are Hermitian and conserve nucleon
    number (small system)."""
    L, dim, n_b = 2, 1, 1
    N = _nucleon_number_op(L, dim, n_b)
    for builder in (fock.H_axial_vector, fock.H_WT_Logic):
        H = builder(L, dim, n_b, _params())
        nq = _n_qubits(H, N)
        Hm = get_sparse_operator(H, n_qubits=nq)
        assert abs((Hm - Hm.getH())).max() < 1e-9, f"{builder.__name__}: not Hermitian"
        Nm = get_sparse_operator(N, n_qubits=nq)
        assert abs(Hm @ Nm - Nm @ Hm).max() < 1e-9, f"{builder.__name__}: [H,N]≠0"


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


# --------------------------------------------------------------------------- #
# Part 4 — cross-builder equivalence (the shared-representation guarantee)      #
# --------------------------------------------------------------------------- #
# The classical/sparse path (fock_native.py → MixedHamiltonian) and the        #
# PauliLCU/quantum path (fock.py → QubitOperator) are independent builders.    #
# The whole "shared native representation prevents classical/quantum drift"    #
# claim rests on them being the SAME operator. Multiply the native mixed terms #
# out and assert equality with the fock.py Pauli build, sector by sector.      #

def _native_mixed_to_qubitop(mh, n_b):
    """Multiply out the native MixedHamiltonian's mixed terms into a QubitOp."""
    H = QubitOperator()
    for mt in mh.mixed_terms:
        f_q = jordan_wigner(mt.fermion_factor)
        # Do NOT normal-order the boson factor: the truncated register makes
        # that unsound (see fock._bosonop_to_qubitop).
        b_q = _bosonop_to_qubitop(mt.boson_factor, n_b, mh.mode_to_qubits)
        H += mt.coeff * (f_q * b_q)
    return H


def test_native_mixed_terms_match_fock_pauli():
    """fock_native H_AV+H_WT (multiplied out) == fock.py H_AV+H_WT (L=2 dim=1,2)."""
    p = _params()
    for L, dim, n_b in [(2, 1, 2), (2, 2, 2)]:
        mh = build_native_mixed_hamiltonian(L, dim, n_b, p)
        native = _native_mixed_to_qubitop(mh, n_b)
        pauli = fock.H_axial_vector(L, dim, n_b, p) + fock.H_WT_Logic(L, dim, n_b, p)
        assert _qubitop_allclose(native, pauli, tol=1e-8), f"L={L} dim={dim}"


def test_native_boson_part_matches_fock_free_pion():
    """fock_native H_pion_free (boson_part) == fock.py H_pion_free (L=2 dim=1,2)."""
    p = _params()
    for L, dim, n_b in [(2, 1, 2), (2, 2, 2)]:
        mh = build_native_mixed_hamiltonian(L, dim, n_b, p)
        native = _bosonop_to_qubitop(mh.boson_part, n_b, mh.mode_to_qubits)
        pauli = fock.H_pion_free(L, dim, n_b, p)
        assert _qubitop_allclose(native, pauli, tol=1e-8), f"L={L} dim={dim}"


# --------------------------------------------------------------------------- #
# Part 5 — H_WT conserves per-site total pion number (cutoff-derivation linchpin)#
# --------------------------------------------------------------------------- #
# The rigorous boson-cutoff derivation (task 25, gaussian_cutoff.py) hinges on
# H_WT being number-preserving w.r.t. the per-site TOTAL pion occupation: the
# ε-antisymmetry kills the a†a†/aa squeezing pieces, leaving ε^{abc} a^{b†}a^c.
# If this regresses, the "H_WT ∈ Tong's H_R" resolution of the Watson
# obstruction breaks. See claude/research/bosonic-encodings/05_*.md.

def test_h_wt_conserves_persite_total_pion_number():
    L, dim, n_b = 1, 1, 2
    H_WT = fock.H_WT_Logic(L, dim, n_b, _params())
    N_pion = QubitOperator()
    for a in range(3):  # sum over the three pion species at the (single) site
        N_pion += fock._number_op_register(0, a, n_b)
    nq = _n_qubits(H_WT, N_pion)
    Hm = get_sparse_operator(H_WT, n_qubits=nq)
    Nm = get_sparse_operator(N_pion, n_qubits=nq)
    comm = Hm @ Nm - Nm @ Hm
    assert abs(comm).max() < 1e-9, "H_WT must conserve total per-site pion number"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
