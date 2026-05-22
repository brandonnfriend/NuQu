"""
Fock (occupation-number) basis implementation of the Dynamical Pion EFT.

The qubit register at each (site, pion-species) holds a binary encoding of
the local Fock state |n⟩, n ∈ {0, …, N_f − 1} where N_f = 2^n_b. The pion
field operators are polynomials in single-mode ladder operators:

    π̂_x  = c_π  · (â_x + â_x†)
    Π̂_x  = c_Π  · i·(â_x† − â_x)

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

Returns a single-walk Hamiltonian bundle — no QFT between bases needed,
since π and Π live in the same Fock register.
"""

import numpy as np
from openfermion import QubitOperator

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

_IMAG_TOL = 1e-9  # Tolerance for dropping floating-point imaginary noise.


# --- Low-level: |i⟩⟨j| → Pauli decomposition ------------------------------


def _ket_bra(i, j, n_b, qubits):
    """Build |i⟩⟨j| as a QubitOperator on the given n_b qubits.

    Standard binary-encoding projector decomposition:
        |0⟩⟨0| = (I + Z)/2
        |1⟩⟨1| = (I − Z)/2
        |0⟩⟨1| = (X + iY)/2
        |1⟩⟨0| = (X − iY)/2
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
    """Return a new QubitOperator with imaginary float-noise stripped.

    Raises if any coefficient has |Im| > _IMAG_TOL — that's a real bug,
    not noise. Hermitian operators should have purely real coefficients.
    """
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


# --- Template builders for single-register ladder operators ---------------
# These build operators on contiguous local qubits [0..n_b−1] once per n_b
# and cache. Per-site/-species instances are made by shifting qubit indices.


_TEMPLATE_CACHE = {}


def _build_x_ladder_template(n_b):
    """(â + â†) on local qubits [0..n_b−1]. Hermitian, real coefficients."""
    cache_key = ('x', n_b)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    N_f = 2 ** n_b
    local_qubits = list(range(n_b))
    op = QubitOperator()
    for n in range(1, N_f):
        c = float(np.sqrt(n))
        op += c * (_ket_bra(n - 1, n, n_b, local_qubits)
                   + _ket_bra(n, n - 1, n_b, local_qubits))
    op = _drop_imag_noise(op)
    _TEMPLATE_CACHE[cache_key] = op
    return op


def _build_p_ladder_template(n_b):
    """i·(â† − â) on local qubits [0..n_b−1]. Hermitian, real coefficients."""
    cache_key = ('p', n_b)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    N_f = 2 ** n_b
    local_qubits = list(range(n_b))
    op = QubitOperator()
    for n in range(1, N_f):
        c = float(np.sqrt(n))
        # i·(â† − â) connects |n−1⟩↔|n⟩ with matrix element ±i·√n.
        # Build directly so the result is manifestly real.
        # i·(|n⟩⟨n−1| − |n−1⟩⟨n|) multiplied by √n.
        op += (1j * c) * (_ket_bra(n, n - 1, n_b, local_qubits)
                          - _ket_bra(n - 1, n, n_b, local_qubits))
    op = _drop_imag_noise(op)
    _TEMPLATE_CACHE[cache_key] = op
    return op


def _shift_qubits(op, shift):
    """Return a copy of op with all qubit indices shifted by `shift`."""
    new_terms = {}
    for term, coeff in op.terms.items():
        new_term = tuple((idx + shift, pauli) for idx, pauli in term)
        new_terms[new_term] = coeff
    out = QubitOperator()
    out.terms = new_terms
    return out


def _x_ladder_register(site_id, species, n_b):
    """(â + â†) on the (site_id, species) Fock register."""
    template = _build_x_ladder_template(n_b)
    base = site_to_pion_qubit(site_id, species, 0, n_b)
    return _shift_qubits(template, base)


def _p_ladder_register(site_id, species, n_b):
    """i·(â† − â) on the (site_id, species) Fock register."""
    template = _build_p_ladder_template(n_b)
    base = site_to_pion_qubit(site_id, species, 0, n_b)
    return _shift_qubits(template, base)


def _number_op_register(site_id, species, n_b):
    """â†â on the (site_id, species) Fock register.

    Diagonal: â†â = Σ_k 2^k · (I − Z_k)/2 in the binary encoding.
    """
    op = QubitOperator()
    for k in range(n_b):
        q = site_to_pion_qubit(site_id, species, k, n_b)
        coeff = (2 ** k) / 2.0
        op += QubitOperator((), coeff)
        op += QubitOperator(((q, 'Z'),), -coeff)
    return op


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
        # The clean collapse requires ω_0 = m_π. If the user has chosen a
        # different ω_0 (e.g. lattice-effective mass), they'll need to
        # build Π² and π² explicitly. For now, warn.
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
    """Gradient term (a_L^d/2)·Σ_d (∇_d π̂)².

    Bilinear in (â_x + â_x†)(â_y + â_y†) across neighboring sites.
    Off-diagonal in the Fock basis.
    """
    a_L = params['a_L']
    c_pi, _ = _basis_coefficients(params, dim)
    # (a_L^d / 2) · (1/a_L²) = a_L^(d−2)/2 per gradient pair.
    factor = (a_L ** (dim - 2)) / 2.0 * (c_pi ** 2)
    num_sites = get_total_sites(L, dim)
    H = QubitOperator()
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            for I in _PION_SPECIES:
                # diff = (a_y + a_y†) − (a_x + a_x†)
                # (pi_y − pi_x)² = c_π² · diff²; the c_π² is in `factor`.
                diff = (_x_ladder_register(site_next, I, n_b)
                        - _x_ladder_register(x, I, n_b))
                H += factor * (diff * diff)
    return H


def H_pion_free(L, dim, n_b, params):
    """Total free-pion: local (m, Π²) + gradient."""
    return H_pion_free_local(L, dim, n_b, params) \
           + H_pion_free_gradient(L, dim, n_b, params)


def H_axial_vector(L, dim, n_b, params):
    """Axial-vector coupling H_AV in the Fock basis.

    H_AV = (g_A / (2 f_π)) · Σ_x Σ_d Σ_I N†(σ_d τ_I)N · ∇_d π_I (x)
    """
    a_L = params['a_L']
    c_pi, _ = _basis_coefficients(params, dim)
    g_A = params['g_A']
    f_pi = params['f_pi']
    prefactor = g_A / (2.0 * f_pi * a_L) * c_pi
    num_sites = get_total_sites(L, dim)
    H = QubitOperator()
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            spin_idx = 3 if dim == 1 else d + 1
            for I in _PION_SPECIES:
                # ∇_d π_I (x) ≈ (π_I(x+e_d) − π_I(x))/a_L,
                #             = c_π · (x_ladder_y − x_ladder_x)/a_L
                # the c_π / a_L is in `prefactor`.
                pion_grad = (_x_ladder_register(site_next, I, n_b)
                             - _x_ladder_register(x, I, n_b))
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

    H_WT = (1/(4 f_π²)) · Σ_x Σ_{I1,I2,I3} ε_{I1 I2 I3} · N†τ_I1 N · π_I2(x) Π_I3(x)
    """
    c_pi, c_Pi = _basis_coefficients(params, dim)
    f_pi = params['f_pi']
    prefactor = 1.0 / (4.0 * f_pi ** 2)
    pion_coeff = c_pi * c_Pi  # absorbs the c_π · c_Π that comes from π · Π
    num_sites = get_total_sites(L, dim)
    H = QubitOperator()
    for x in range(num_sites):
        for I1, I2, I3, sign in _EPSILONS:
            # π_I2(x) · Π_I3(x) = c_π · c_Π · x_ladder(I2) · p_ladder(I3)
            x_lad = _x_ladder_register(x, I2, n_b)
            p_lad = _p_ladder_register(x, I3, n_b)
            pion_block = pion_coeff * (x_lad * p_lad)
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
