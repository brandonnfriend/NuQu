"""
Fock (occupation-number) basis implementation of the Dynamical Pion EFT.

The qubit register at each (site, pion-species) holds a binary encoding of
the local Fock state |n⟩, n ∈ {0, …, N_f − 1} where N_f = 2^n_b. The pion
field operators are polynomials in single-mode ladder operators:

    π̂_x  = c_π  · (â_x + â_x†)
    Π̂_x  =   i · c_Π · (â_x† − â_x)

with coefficients (derived in claude/research/bosonic-encodings/
01_two_readings_oscillator_basis.md, "Theoretical derivation" section):

    c_π  = 1 / √(2 ω_0 · a_L^d)         [MeV]
    c_Π  =   √(ω_0  / (2 a_L^d))         [MeV²]

These satisfy the lattice canonical commutator
[π̂(x), Π̂(y)] = i δ_{xy} / a_L^d.

For ω_0 = m_π, the *local* combined free-pion operator
(a_L^d/2)·(Π̂² + m_π²·π̂²) collapses **exactly** to the diagonal
harmonic-oscillator number operator m_π·(â†â + ½). The gradient (∇π)²
term remains off-diagonal because it couples neighboring sites.

Front-end: OpenFermion's BosonOperator carries the ladder algebra
symbolically. Each unique *ordered* single-mode monomial (e.g. â·â† vs
â†·â — distinct on the truncated register because [â, â†] is **not** the
identity at |N_f−1⟩) is Pauli-expanded once on local qubits [0..n_b−1]
and cached in `_TEMPLATE_CACHE`; assembly into full-register operators
is just qubit remapping plus per-term coefficient multiplication. We do
*not* call `normal_ordered` — that would apply the full-Hilbert-space
commutator and produce a truncation defect at the top register state.
This avoids the ~16^n_b Pauli-product cost the previous front-end paid
for every QubitOp multiplication while still representing the truncated
operator faithfully.

Returns a single-walk Hamiltonian bundle — no QFT between bases needed,
since π and Π live in the same Fock register.
"""

from collections import defaultdict

import numpy as np
from openfermion import BosonOperator, QubitOperator

from src_PI.hamiltonians.core.Operators import Nucleon_Transition_JW
from src_PI.utils.LatticeGeometry import (
    site_to_pion_qubit, get_total_sites, index_to_coord,
)
from src_PI.utils.utils import calculate_chiral_coeff


BASIS_NAME = 'fock'

_PION_SPECIES = (0, 1, 2)
_MODES = [(0, 0), (0, 1), (1, 0), (1, 1)]
_EPSILONS = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1),
             (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]

_IMAG_TOL = 1e-9
_TEMPLATE_CACHE = {}


# --- Low-level: |i⟩⟨j| → Pauli decomposition -------------------------------
# Only used to build the â and â† templates once per n_b; everything above
# that goes through the symbolic BosonOperator path.


def _ket_bra(i, j, n_b, qubits):
    """Build |i⟩⟨j| as a QubitOperator using the binary occupation encoding.

    |0⟩⟨0| = (I + Z)/2,  |1⟩⟨1| = (I − Z)/2,
    |0⟩⟨1| = (X + iY)/2, |1⟩⟨0| = (X − iY)/2.
    """
    op = QubitOperator((), 1.0)
    for k in range(n_b):
        i_bit = (i >> k) & 1
        j_bit = (j >> k) & 1
        q = qubits[k]
        if i_bit == 0 and j_bit == 0:
            factor = QubitOperator((), 0.5) + QubitOperator(f"Z{q}", 0.5)
        elif i_bit == 1 and j_bit == 1:
            factor = QubitOperator((), 0.5) - QubitOperator(f"Z{q}", 0.5)
        elif i_bit == 0 and j_bit == 1:
            factor = QubitOperator(f"X{q}", 0.5) + QubitOperator(f"Y{q}", 0.5j)
        else:  # i_bit == 1, j_bit == 0
            factor = QubitOperator(f"X{q}", 0.5) - QubitOperator(f"Y{q}", 0.5j)
        op = op * factor
    return op


def _drop_imag_noise(op):
    """Strip floating-point imaginary noise; raise on real non-Hermitian terms."""
    new_terms = {}
    for term, coeff in op.terms.items():
        if abs(coeff.imag) > _IMAG_TOL:
            raise ValueError(
                f"Non-Hermitian operator: term {term} has Im(coeff)={coeff.imag}"
            )
        if abs(coeff.real) > _IMAG_TOL:
            new_terms[term] = float(coeff.real)
    out = QubitOperator()
    out.terms = new_terms
    return out


# --- Single-mode templates and the symbolic-to-qubit bridge ---------------


def _a_template(n_b):
    """â on local qubits [0..n_b−1] as a QubitOperator. â = Σ √n |n−1⟩⟨n|."""
    key = ('a', n_b)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    N_f = 2 ** n_b
    local_qubits = list(range(n_b))
    op = QubitOperator()
    for n in range(1, N_f):
        op += float(np.sqrt(n)) * _ket_bra(n - 1, n, n_b, local_qubits)
    _TEMPLATE_CACHE[key] = op
    return op


def _a_dag_template(n_b):
    """â† on local qubits [0..n_b−1] as a QubitOperator. â† = Σ √n |n⟩⟨n−1|."""
    key = ('a_dag', n_b)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    N_f = 2 ** n_b
    local_qubits = list(range(n_b))
    op = QubitOperator()
    for n in range(1, N_f):
        op += float(np.sqrt(n)) * _ket_bra(n, n - 1, n_b, local_qubits)
    _TEMPLATE_CACHE[key] = op
    return op


def _monomial_template(signature, n_b):
    """Single-mode monomial built from an ordered ladder-letter signature.

    `signature` is a tuple of '+' (â†) and '-' (â) read left-to-right —
    the operator product is left-multiplied in that order so that
    `signature=('+', '-')` builds â†·â, and `signature=('-', '+')` builds
    â·â†. The two differ on the truncated register at the |N_f−1⟩ boundary,
    so ordering must be preserved in the cache key.
    """
    key = ('mono', n_b, signature)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    a = _a_template(n_b)
    a_dag = _a_dag_template(n_b)
    op = QubitOperator((), 1.0)
    for s in signature:
        if s == '+':
            op = op * a_dag
        else:
            op = op * a
    _TEMPLATE_CACHE[key] = op
    return op


def _remap_qubits(op, target_qubits):
    """Return op with local qubits [0..n−1] remapped to target_qubits[0..n−1]."""
    new_terms = {}
    for term, coeff in op.terms.items():
        new_term = tuple((target_qubits[idx], pauli) for idx, pauli in term)
        new_terms[new_term] = coeff
    out = QubitOperator()
    out.terms = new_terms
    return out


def _site_to_pion_qubits(site_id, species, n_b):
    """Contiguous qubits backing the (site, species) Fock register."""
    return [site_to_pion_qubit(site_id, species, k, n_b) for k in range(n_b)]


def _bosonop_to_qubitop(b_op, n_b, mode_qubits):
    """Convert a BosonOperator to a QubitOperator on the truncated register.

    Within each mode, the term's ladder letters are collected in their
    *original* left-to-right order — that order is meaningful on the
    truncated register. Different modes act on disjoint qubits so the
    across-mode order doesn't matter and is dropped.

    Args:
        b_op: BosonOperator. Do **not** pre-normal-order — that uses the
            full-Hilbert-space commutator and produces a truncation defect.
        n_b: bits per mode.
        mode_qubits: dict mapping mode_index → list of n_b target qubits.
    """
    result = QubitOperator()
    for term, coeff in b_op.terms.items():
        per_mode = defaultdict(list)
        for mode_idx, action in term:
            per_mode[mode_idx].append('+' if action == 1 else '-')

        term_qop = QubitOperator((), coeff)
        for mode_idx, signature in per_mode.items():
            local_qop = _monomial_template(tuple(signature), n_b)
            shifted = _remap_qubits(local_qop, mode_qubits[mode_idx])
            term_qop = term_qop * shifted
        result += term_qop
    return result


# --- HO basis coefficients ------------------------------------------------


def _get_omega_0(params):
    """Oscillator basis frequency. Defaults to m_π."""
    return params.get('m_0', params['m_pi'])


def _basis_coefficients(params, dim):
    """Returns (c_π, c_Π) for the chosen ω_0 and lattice spacing."""
    omega_0 = _get_omega_0(params)
    a_L = params['a_L']
    aL_d = a_L ** dim
    c_pi = 1.0 / np.sqrt(2.0 * omega_0 * aL_d)
    c_Pi = np.sqrt(omega_0 / (2.0 * aL_d))
    return c_pi, c_Pi


# --- Symbolic ladder building blocks --------------------------------------


def _b_x(mode):
    """Symbolic (â + â†) on `mode` as a BosonOperator."""
    return BosonOperator(f'{mode}') + BosonOperator(f'{mode}^')


def _b_p(mode):
    """Symbolic i·(â† − â) on `mode` as a BosonOperator."""
    return 1j * (BosonOperator(f'{mode}^') - BosonOperator(f'{mode}'))


def _number_op_register(site_id, species, n_b):
    """â†â on the (site, species) register, diagonal: Σ_k 2^k·(I − Z_k)/2."""
    op = QubitOperator()
    for k in range(n_b):
        q = site_to_pion_qubit(site_id, species, k, n_b)
        coeff = (2 ** k) / 2.0
        op += QubitOperator((), coeff)
        op += QubitOperator(((q, 'Z'),), -coeff)
    return op


# --- EFT Hamiltonian terms in the Fock basis ------------------------------


def H_pion_free_local(L, dim, n_b, params):
    """Local free-pion: (a_L^d/2)·(Π̂² + m_π²·π̂²) per site per species.

    With ω_0 = m_π this collapses *exactly* to m_π·(â†â + ½) per site per
    species (a_L^d factors cancel; see derivation in the bosonic-encodings
    doc). With ω_0 ≠ m_π there is a residual off-diagonal piece, which
    this function does *not* compute — set ω_0 = m_π in your config for
    the clean collapse.
    """
    omega_0 = _get_omega_0(params)
    m_pi = params['m_pi']
    if abs(omega_0 - m_pi) > 1e-6:
        import warnings
        warnings.warn(
            f"H_pion_free_local: ω_0={omega_0} differs from m_π={m_pi}; "
            "the diagonal collapse formula assumes equality. "
            "Residual off-diagonal (â² + (â†)²) terms are NOT included.",
            RuntimeWarning,
        )
    num_sites = get_total_sites(L, dim)
    H = QubitOperator()
    for x in range(num_sites):
        for I in _PION_SPECIES:
            H += m_pi * _number_op_register(x, I, n_b)
            H += QubitOperator((), m_pi / 2.0)  # zero-point energy
    return H


def H_pion_free_gradient(L, dim, n_b, params):
    """Gradient term (a_L^(d−2)/2)·Σ_d (π_y − π_x)² in the Fock basis.

    Builds the diff² symbolically as a 2-mode BosonOperator (mode 0 → x,
    mode 1 → y), normal-orders once, then expands per (x, y, species)
    against the cached single-mode monomial templates.
    """
    a_L = params['a_L']
    c_pi, _ = _basis_coefficients(params, dim)
    factor = (a_L ** (dim - 2)) / 2.0 * (c_pi ** 2)
    num_sites = get_total_sites(L, dim)

    diff = _b_x(1) - _b_x(0)
    diff_sq = diff * diff

    H = QubitOperator()
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            for I in _PION_SPECIES:
                mode_qubits = {
                    0: _site_to_pion_qubits(x, I, n_b),
                    1: _site_to_pion_qubits(site_next, I, n_b),
                }
                H += factor * _bosonop_to_qubitop(diff_sq, n_b, mode_qubits)
    return H


def H_pion_free(L, dim, n_b, params):
    """Total free-pion: local (m·n̂ + ½) + gradient."""
    return H_pion_free_local(L, dim, n_b, params) \
           + H_pion_free_gradient(L, dim, n_b, params)


def H_axial_vector(L, dim, n_b, params):
    """Axial-vector coupling H_AV in the Fock basis.

    H_AV = (g_A / (2 f_π)) · Σ_x Σ_d Σ_I N†(σ_d τ_I)N · ∇_d π_I (x)

    Pion gradient is linear in â/â† and lives on two neighbor modes
    (0 → x, 1 → y); normal-ordering is trivial but goes through the same
    BosonOperator → QubitOperator path as the gradient and WT terms for
    consistency.
    """
    a_L = params['a_L']
    c_pi, _ = _basis_coefficients(params, dim)
    g_A = params['g_A']
    f_pi = params['f_pi']
    prefactor = g_A / (2.0 * f_pi * a_L) * c_pi
    num_sites = get_total_sites(L, dim)

    pion_grad_b = _b_x(1) - _b_x(0)

    H = QubitOperator()
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            spin_idx = 3 if dim == 1 else d + 1
            for I in _PION_SPECIES:
                mode_qubits = {
                    0: _site_to_pion_qubits(x, I, n_b),
                    1: _site_to_pion_qubits(site_next, I, n_b),
                }
                pion_grad = _bosonop_to_qubitop(pion_grad_b, n_b, mode_qubits)

                nucleon_sum = QubitOperator()
                for m_alpha in _MODES:
                    for m_beta in _MODES:
                        coeff = calculate_chiral_coeff(
                            m_alpha, m_beta, iso_idx=I + 1, spin_idx=spin_idx
                        )
                        if abs(coeff) > 1e-9:
                            nucleon_sum += coeff * Nucleon_Transition_JW(
                                x, m_alpha, m_beta, n_b
                            )
                H += prefactor * (nucleon_sum * pion_grad)
    H = _drop_imag_noise(H)
    return H


def H_WT_Logic(L, dim, n_b, params):
    """Weinberg-Tomozawa term H_WT in the Fock basis.

    H_WT = (1/(4 f_π²)) · Σ_x Σ_{I1,I2,I3} ε_{I1 I2 I3} · N†τ_{I1} N · π_{I2}(x) Π_{I3}(x)

    π · Π on different species (I2 ≠ I3 from ε) becomes a 2-mode product
    that normal-ordering splits into four single-mode-tensor monomials.
    """
    c_pi, c_Pi = _basis_coefficients(params, dim)
    f_pi = params['f_pi']
    prefactor = 1.0 / (4.0 * f_pi ** 2)
    pion_coeff = c_pi * c_Pi
    num_sites = get_total_sites(L, dim)

    pi_dot_Pi_b = _b_x(0) * _b_p(1)

    H = QubitOperator()
    for x in range(num_sites):
        for I1, I2, I3, sign in _EPSILONS:
            mode_qubits = {
                0: _site_to_pion_qubits(x, I2, n_b),
                1: _site_to_pion_qubits(x, I3, n_b),
            }
            pion_block = pion_coeff * _bosonop_to_qubitop(
                pi_dot_Pi_b, n_b, mode_qubits
            )
            nucleon_block = QubitOperator()
            for m_alpha in _MODES:
                for m_beta in _MODES:
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1 + 1, 0)
                    if abs(coeff) > 1e-9:
                        nucleon_block += coeff * Nucleon_Transition_JW(
                            x, m_alpha, m_beta, n_b
                        )
            H += (prefactor * sign) * (pion_block * nucleon_block)
    H = _drop_imag_noise(H)
    return H


def Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, pi_max, params):
    """Build the dynamical-pion sector in the Fock basis.

    The `pi_max` argument is accepted (and ignored) for signature
    compatibility with the amplitude path. In Fock basis, the relevant
    parameter is ω_0 (read from params['m_0'] or defaulted to m_π).

    Returns a list of (name, QubitOperator) sub-Hamiltonians:
        [('fock', H_full)]

    A single sub-Hamiltonian → single walk → no QFT cost.
    """
    H_free = H_pion_free(L, dim, n_b, params)
    H_AV = H_axial_vector(L, dim, n_b, params)
    H_WT = H_WT_Logic(L, dim, n_b, params)
    H_full = H_free + H_AV + H_WT
    return [('fock', H_full)]
