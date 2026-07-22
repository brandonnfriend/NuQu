"""
Native-algebra Fock-basis builder for the Dynamical Pion EFT.

Produces a `MixedHamiltonian` carrying:
  * `fermion_part`: pure-fermion contributions. Empty here; the static
    nucleon sector (a `FermionOperator`) is folded in by `ConstructEFT`
    without applying Jordan-Wigner.
  * `boson_part`: pure-boson contributions from H_pion_free — `m_π·n̂`
    per (site, species) mode plus the gradient² term — carried as a
    single `BosonOperator` over global mode indices. Zero-point
    `m_π/2` per mode is included as an identity coefficient on the
    `BosonOperator`; `normalize_for_qpe` will extract it as the
    classical shift downstream.
  * `mixed_terms`: H_AV and H_WT as a list of `MixedTerm(coeff, F, B)`,
    where the fermion and boson factors act on disjoint registers and
    are carried **unmultiplied** (pseudocode demand for sparse / LOBE).

Used when `config.pion_basis == 'fock'` and `config.block_encoder` is
'sparse' or 'lobe'. The PauliLCU path continues to use `fock.py`
(separate pre-built Pauli operators, bit-identical to pre-refactor).

The math in this module mirrors `fock.py`; only the *output type*
changes (BosonOperator + FermionOperator + MixedTerm triples instead of
QubitOperator). Per-term coefficients and per-loop iteration order are
intentionally kept identical to `fock.py` so a future cross-check can
verify the two builders produce mathematically equivalent operators.
"""

import numpy as np
from openfermion import BosonOperator, FermionOperator

from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian, MixedTerm
from src_PI.utils.LatticeGeometry import (
    get_total_sites,
    index_to_coord,
    site_to_nucleon_qubit,
    site_to_pion_qubit,
)
from src_PI.utils.utils import calculate_chiral_coeff


BASIS_NAME = 'fock_native'

_PION_SPECIES = (0, 1, 2)
_MODES = [(0, 0), (0, 1), (1, 0), (1, 1)]
_EPSILONS = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1),
             (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]


def _global_mode(site, species):
    """Global pion-mode index: contiguous over (site, species)."""
    return site * len(_PION_SPECIES) + species


def _build_mode_to_qubits(num_sites, n_b):
    """Map global mode index -> list of n_b qubit indices for that pion mode."""
    return {
        _global_mode(x, I): [site_to_pion_qubit(x, I, k, n_b) for k in range(n_b)]
        for x in range(num_sites) for I in _PION_SPECIES
    }


def _nucleon_transition_fermion(site_id, mode_alpha, mode_beta, n_b):
    """FermionOp for (a†_α a_β + a†_β a_α) on the given site (Hermitian transition)."""
    idx_alpha = site_to_nucleon_qubit(site_id, mode_alpha, n_b)
    idx_beta = site_to_nucleon_qubit(site_id, mode_beta, n_b)
    return (FermionOperator(f'{idx_alpha}^ {idx_beta}')
            + FermionOperator(f'{idx_beta}^ {idx_alpha}'))


def _b_x_global(global_mode):
    """Symbolic (â + â†) on a global pion mode."""
    return BosonOperator(f'{global_mode}') + BosonOperator(f'{global_mode}^')


def _b_p_global(global_mode):
    """Symbolic i·(â† − â) on a global pion mode."""
    return 1j * (BosonOperator(f'{global_mode}^') - BosonOperator(f'{global_mode}'))


def _basis_coefficients(params, dim):
    """(c_π, c_Π) for ω_0 = m_0 (defaults to m_π) and lattice spacing a_L."""
    omega_0 = params.get('m_0', params['m_pi'])
    a_L = params['a_L']
    aL_d = a_L ** dim
    c_pi = 1.0 / np.sqrt(2.0 * omega_0 * aL_d)
    c_Pi = np.sqrt(omega_0 / (2.0 * aL_d))
    return c_pi, c_Pi


# --- Pure-boson sector: H_pion_free ---------------------------------------


def H_pion_free_native(L, dim, n_b, params):
    """Pure-boson H_pion_free as a `BosonOperator` over global modes.

    Same math as `fock.py`'s `H_pion_free`, expressed natively:
      * Local: m_π · â†_m â_m per global mode (assumes ω_0 = m_π for the
        Jordan-Lee-Preskill collapse; warn otherwise).
      * Zero-point: identity `BosonOperator(''') · m_π/2` per mode.
      * Gradient: (a_L^(d-2) / 2) · c_π² · (b̂_y − b̂_x)² over each
        adjacent (x, y) pair per pion species.
    """
    omega_0 = params.get('m_0', params['m_pi'])
    m_pi = params['m_pi']
    if abs(omega_0 - m_pi) > 1e-6:
        import warnings
        warnings.warn(
            f"H_pion_free_native: ω_0={omega_0} differs from m_π={m_pi}; "
            "the diagonal collapse formula assumes equality.",
            RuntimeWarning,
        )

    c_pi, _ = _basis_coefficients(params, dim)
    a_L = params['a_L']
    grad_factor = (a_L ** (dim - 2)) / 2.0 * (c_pi ** 2)
    num_sites = get_total_sites(L, dim)

    H_b = BosonOperator()
    # Local n̂_m and zero-point per (site, species).
    for x in range(num_sites):
        for I in _PION_SPECIES:
            m = _global_mode(x, I)
            H_b += m_pi * BosonOperator(f'{m}^ {m}')
            H_b += BosonOperator('', m_pi / 2.0)  # zero-point

    # Gradient: (b_y + b_y^ - b_x - b_x^)^2 over adjacent (x, y) pairs.
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            for I in _PION_SPECIES:
                mx = _global_mode(x, I)
                my = _global_mode(site_next, I)
                diff = _b_x_global(my) - _b_x_global(mx)
                H_b += grad_factor * diff * diff
    return H_b


# --- Mixed sectors: H_AV, H_WT --------------------------------------------


def H_axial_vector_native(L, dim, n_b, params):
    """H_AV mixed-terms list (each term coeff · F_op · B_op, unmultiplied).

    For each (x, d, I) triple, one MixedTerm capturing:
      * fermion_factor: Σ_{α,β} χ_{αβ}^I · (a†_α a_β + a†_β a_α) on site x
      * boson_factor:   (b̂_y + b̂†_y) − (b̂_x + b̂†_x) on global modes
                         (gradient of π_I across the (x, y) bond).
    Empty fermion-factor terms (zero chiral coefficient) are skipped.
    """
    c_pi, _ = _basis_coefficients(params, dim)
    a_L = params['a_L']
    g_A = params['g_A']
    f_pi = params['f_pi']
    prefactor = g_A / (2.0 * f_pi * a_L) * c_pi
    num_sites = get_total_sites(L, dim)

    mixed = []
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            spin_idx = 3 if dim == 1 else d + 1
            for I in _PION_SPECIES:
                mx = _global_mode(x, I)
                my = _global_mode(site_next, I)
                pion_grad_b = _b_x_global(my) - _b_x_global(mx)

                nucleon_F = FermionOperator()
                for m_alpha in _MODES:
                    for m_beta in _MODES:
                        coeff = calculate_chiral_coeff(
                            m_alpha, m_beta, iso_idx=I + 1, spin_idx=spin_idx
                        )
                        if abs(coeff) > 1e-9:
                            nucleon_F += coeff * _nucleon_transition_fermion(
                                x, m_alpha, m_beta, n_b
                            )
                if len(nucleon_F.terms) > 0:
                    mixed.append(MixedTerm(
                        coeff=prefactor,
                        fermion_factor=nucleon_F,
                        boson_factor=pion_grad_b,
                    ))
    return mixed


def H_WT_native(L, dim, n_b, params):
    """H_WT mixed-terms list (each term coeff · F_op · B_op, unmultiplied).

    For each (x, ε_{I1 I2 I3}) sign-tagged tuple, one MixedTerm capturing:
      * fermion_factor: Σ_{α,β} χ_{αβ}^{I1} · (a†_α a_β + a†_β a_α) on site x
      * boson_factor:   π_{I2}(x) · Π_{I3}(x) = c_π·c_Π · (b̂_m2)·(i·(b̂†_m3 − b̂_m3))
        where m2, m3 are the global modes for (x, I2) and (x, I3).
    """
    c_pi, c_Pi = _basis_coefficients(params, dim)
    f_pi = params['f_pi']
    prefactor = 1.0 / (4.0 * f_pi ** 2)
    pion_coeff = c_pi * c_Pi
    num_sites = get_total_sites(L, dim)

    mixed = []
    for x in range(num_sites):
        for I1, I2, I3, sign in _EPSILONS:
            m2 = _global_mode(x, I2)
            m3 = _global_mode(x, I3)
            pion_b = pion_coeff * (_b_x_global(m2) * _b_p_global(m3))

            nucleon_F = FermionOperator()
            for m_alpha in _MODES:
                for m_beta in _MODES:
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1 + 1, 0)
                    if abs(coeff) > 1e-9:
                        nucleon_F += coeff * _nucleon_transition_fermion(
                            x, m_alpha, m_beta, n_b
                        )
            if len(nucleon_F.terms) > 0:
                mixed.append(MixedTerm(
                    coeff=prefactor * sign,
                    fermion_factor=nucleon_F,
                    boson_factor=pion_b,
                ))
    return mixed


# --- Public builder --------------------------------------------------------


def build_native_mixed_hamiltonian(L, dim, n_b, params):
    """Assemble the native MixedHamiltonian for the dynamical-pion sector.

    The static-nucleon (FermionOperator) is *not* added here; the caller
    (ConstructEFT) folds it into `MixedHamiltonian.fermion_part` directly
    without applying Jordan-Wigner.
    """
    num_sites = get_total_sites(L, dim)
    return MixedHamiltonian(
        fermion_part=FermionOperator(),
        boson_part=H_pion_free_native(L, dim, n_b, params),
        mode_to_qubits=_build_mode_to_qubits(num_sites, n_b),
        mixed_terms=H_axial_vector_native(L, dim, n_b, params)
                   + H_WT_native(L, dim, n_b, params),
    )
