"""
Amplitude basis (Watson Lemma 5 / Jordan-Lee-Preskill field-eigenstate basis)
implementation of the Dynamical Pion EFT.

The qubit register holds a binary encoding of the pion field amplitude
π(x) ∈ [-π_max, +π_max] on a uniform grid of 2^n_b points per pion species
per site. π is diagonal in this register; Π is diagonal in the conjugate
(momentum) register, accessed by a QFT.

Returns a 2-walk Hamiltonian bundle: (H_pos, H_mom) — the QFT between bases
is paid in the resource estimator, not the Hamiltonian construction.

The operator-construction code here was previously in core/DynamicalPion.py
and core/Operators.py. The behavior is unchanged; the refactor moves these
to a per-basis namespace and lifts P/Q/Pp/Qp computation up one level so
they're computed once instead of three times per `Full_Dynamical_Pion_Hamiltonian`
call.
"""

from openfermion import QubitOperator

from src_PI.hamiltonians.core.Operators import (
    Nucleon_Transition_JW,
    _add, _z_key, _to_qubit_op,
)
from src_PI.utils.LatticeGeometry import (
    site_to_pion_qubit, get_total_sites, index_to_coord,
)
from src_PI.utils.utils import calculate_chiral_coeff, get_P_Q, get_Pp_Qp


BASIS_NAME = 'amplitude'


# --- Single-site / pair amplitude-basis operators --------------------------
# (Previously in src_PI/hamiltonians/core/Operators.py. Moved here because
# they encode the amplitude-basis-specific P, Q, Pp, Qp parameters that
# don't exist in the Fock basis.)


def _zz_key(idx_a, idx_b):
    """Canonical Z_a Z_b key (assumes a != b; collapse Z*Z=I if equal)."""
    if idx_a < idx_b:
        return ((idx_a, 'Z'), (idx_b, 'Z'))
    return ((idx_b, 'Z'), (idx_a, 'Z'))


def Pi_Squared_Operator(site_id, pion_species, n_b, P, Q):
    """π̂² on a single site/species. Eq. (71). Diagonal in the field basis."""
    terms = {}
    _add(terms, (), P * P)
    two_PQ = 2.0 * P * Q
    Q_sq = Q * Q

    indices = [site_to_pion_qubit(site_id, pion_species, m, n_b) for m in range(n_b)]
    for m, idx in enumerate(indices):
        _add(terms, _z_key(idx), two_PQ * (1 << m))

    for m in range(n_b):
        idx_m = indices[m]
        pow_m = 1 << m
        for mp in range(n_b):
            idx_mp = indices[mp]
            coeff = Q_sq * pow_m * (1 << mp)
            if idx_m == idx_mp:
                _add(terms, (), coeff)
            else:
                _add(terms, _zz_key(idx_m, idx_mp), coeff)
    return _to_qubit_op(terms)


def Gradient_Squared_Operator(site_x, site_y, pion_species, n_b, Q, a_L):
    """((π(y) − π(x))/a_L)² over an adjacent site pair. Eq. (72)."""
    terms = {}
    inv_aL2 = 1.0 / (a_L * a_L)
    Q_sq = Q * Q

    idx_x = [site_to_pion_qubit(site_x, pion_species, m, n_b) for m in range(n_b)]
    idx_y = [site_to_pion_qubit(site_y, pion_species, m, n_b) for m in range(n_b)]

    for m in range(n_b):
        ix_m = idx_x[m]
        iy_m = idx_y[m]
        pow_m = 1 << m
        for n in range(n_b):
            ix_n = idx_x[n]
            iy_n = idx_y[n]
            common_coeff = Q_sq * pow_m * (1 << n) * inv_aL2

            if ix_m == ix_n:
                _add(terms, (), common_coeff)
            else:
                _add(terms, _zz_key(ix_m, ix_n), common_coeff)

            if iy_m == iy_n:
                _add(terms, (), common_coeff)
            else:
                _add(terms, _zz_key(iy_m, iy_n), common_coeff)

            _add(terms, _zz_key(ix_m, iy_n), -2.0 * common_coeff)
    return _to_qubit_op(terms)


def Momentum_Squared_Operator(site_id, pion_species, n_b, Pp, Qp):
    """Π̂² (conjugate momentum) in its diagonal basis. Eq. (74)."""
    terms = {}
    _add(terms, (), Pp * Pp)
    two_PpQp = 2.0 * Pp * Qp
    Qp_sq = Qp * Qp

    indices = [site_to_pion_qubit(site_id, pion_species, m, n_b) for m in range(n_b)]
    for m, idx in enumerate(indices):
        _add(terms, _z_key(idx), two_PpQp * (1 << m))

    for m in range(n_b):
        idx_m = indices[m]
        pow_m = 1 << m
        for mp in range(n_b):
            idx_mp = indices[mp]
            coeff = Qp_sq * pow_m * (1 << mp)
            if idx_m == idx_mp:
                _add(terms, (), coeff)
            else:
                _add(terms, _zz_key(idx_m, idx_mp), coeff)
    return _to_qubit_op(terms)


# --- EFT Hamiltonian terms in the amplitude basis --------------------------


def H_pion_free(L, dim, n_b, P, Q, Pp, Qp, m_pi, a_L):
    """Free-pion sector (mass + gradient + kinetic) in two parts."""
    num_sites = get_total_sites(L, dim)
    H_pos_free = QubitOperator()
    H_mom_free = QubitOperator()
    vol_factor = (a_L ** 3) / 2.0
    mass_factor = vol_factor * (m_pi ** 2)

    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for I in range(3):
            H_pos_free += mass_factor * Pi_Squared_Operator(x, I, n_b, P, Q)
            H_mom_free += vol_factor * Momentum_Squared_Operator(x, I, n_b, Pp, Qp)

            for d in range(dim):
                if coords[d] < L - 1:
                    site_next = x + L ** d
                    H_pos_free += vol_factor * Gradient_Squared_Operator(
                        x, site_next, I, n_b, Q, a_L
                    )
    return H_pos_free, H_mom_free


def H_axial_vector(L, dim, n_b, Q, g_A, f_pi, a_L):
    """Axial-vector coupling H_AV with spin-momentum coupling."""
    num_sites = get_total_sites(L, dim)
    H_AV = QubitOperator()
    prefactor = g_A / (2.0 * f_pi * a_L)
    modes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)

        for d in range(dim):
            if coords[d] >= L - 1:
                continue
            site_next = x + L ** d
            spin_idx = 3 if dim == 1 else d + 1

            for I in range(3):
                grad_terms = {}
                for n in range(n_b):
                    pow_n = 1 << n
                    idx_next = site_to_pion_qubit(site_next, I, n, n_b)
                    idx_curr = site_to_pion_qubit(x, I, n, n_b)
                    _add(grad_terms, _z_key(idx_next), pow_n)
                    _add(grad_terms, _z_key(idx_curr), -pow_n)
                pion_grad = _to_qubit_op(grad_terms)

                nucleon_sum = QubitOperator()
                for m_alpha in modes:
                    for m_beta in modes:
                        coeff = calculate_chiral_coeff(
                            m_alpha, m_beta, iso_idx=I + 1, spin_idx=spin_idx
                        )
                        if abs(coeff) > 1e-9:
                            nucleon_sum += coeff * Nucleon_Transition_JW(
                                x, m_alpha, m_beta, n_b
                            )

                H_AV += prefactor * Q * (nucleon_sum * pion_grad)
    return H_AV


def H_WT_Logic(L, dim, n_b, P, Q, Pp, Qp, f_pi, a_L):
    """Weinberg-Tomozawa term H_WT (Eq. 70/74) with full algebraic expansion of π·Π."""
    num_sites = get_total_sites(L, dim)
    H_WT = QubitOperator()

    prefactor = 1.0 / (4.0 * f_pi ** 2)
    modes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    epsilons = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1),
                (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]

    PPp = P * Pp
    PQp = P * Qp
    PpQ = Pp * Q
    QQp = Q * Qp

    for x in range(num_sites):
        for I1, I2, I3, sign in epsilons:
            block_terms = {(): PPp}

            for m in range(n_b):
                pow_m = 1 << m
                idx_I2 = site_to_pion_qubit(x, I2, m, n_b)
                idx_I3 = site_to_pion_qubit(x, I3, m, n_b)
                _add(block_terms, _z_key(idx_I3), PQp * pow_m)
                _add(block_terms, _z_key(idx_I2), PpQ * pow_m)

                for l in range(n_b):
                    pow_l = 1 << l
                    idx_I3_l = site_to_pion_qubit(x, I3, l, n_b)
                    coeff = QQp * pow_m * pow_l
                    if idx_I2 == idx_I3_l:
                        _add(block_terms, (), coeff)
                    else:
                        if idx_I2 < idx_I3_l:
                            key = ((idx_I2, 'Z'), (idx_I3_l, 'Z'))
                        else:
                            key = ((idx_I3_l, 'Z'), (idx_I2, 'Z'))
                        _add(block_terms, key, coeff)

            pion_block = _to_qubit_op(block_terms)

            nucleon_block = QubitOperator()
            for m_alpha in modes:
                for m_beta in modes:
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1 + 1, 0)
                    if abs(coeff) > 1e-9:
                        nucleon_block += coeff * Nucleon_Transition_JW(
                            x, m_alpha, m_beta, n_b
                        )

            H_WT += (prefactor * sign) * (pion_block * nucleon_block)
    return H_WT


def Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, pi_max, params):
    """Build the dynamical-pion sector in the amplitude basis.

    Returns a list of (name, QubitOperator) sub-Hamiltonians:
        [('pos_dyn', H_pos_dyn), ('mom', H_mom)]
    The static-nucleon sector is added by the top-level ConstructEFT
    into the 'pos_dyn' bucket.
    """
    a_L = params['a_L']

    # Compute P, Q, Pp, Qp once and reuse across all sub-Hamiltonian builders.
    P, Q = get_P_Q(pi_max, n_b)
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L, dim)

    H_pos_free, H_mom = H_pion_free(
        L, dim, n_b, P, Q, Pp, Qp, params['m_pi'], a_L
    )
    H_AV = H_axial_vector(L, dim, n_b, Q, params['g_A'], params['f_pi'], a_L)
    H_WT = H_WT_Logic(L, dim, n_b, P, Q, Pp, Qp, params['f_pi'], a_L)

    H_pos_dyn = H_pos_free + H_AV + H_WT
    return [('pos_dyn', H_pos_dyn), ('mom', H_mom)]
