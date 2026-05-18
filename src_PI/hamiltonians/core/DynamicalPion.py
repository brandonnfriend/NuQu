from openfermion import QubitOperator
from src_PI.hamiltonians.core.Operators import (
    Nucleon_Transition_JW,
    Pi_Squared_Operator,
    Gradient_Squared_Operator,
    Momentum_Squared_Operator,
    _add,
    _z_key,
    _to_qubit_op,
)
from src_PI.utils.LatticeGeometry import site_to_pion_qubit, get_total_sites, index_to_coord
from src_PI.utils.utils import calculate_chiral_coeff, get_P_Q, get_Pp_Qp


def H_pion_free(L, dim, n_b, pi_max, m_pi, a_L):
    """Returns H_pos_free and H_mom_free for D-dimensions.
    L: lattice size in each dimension
    dim: number of dimensions (1,2 or 3)
    nb: number of quibits used to encode the field strength
    pi_max: upper bound for coupling strength
    m_pi: pion mass
    a_L: lattice spacing """
    num_sites = get_total_sites(L, dim)
    P, Q = get_P_Q(pi_max, n_b)
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L)

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
                    H_pos_free += vol_factor * Gradient_Squared_Operator(x, site_next, I, n_b, Q, a_L)

    return H_pos_free, H_mom_free


def H_axial_vector(L, dim, n_b, pi_max, g_A, f_pi, a_L):
    """Implements the Axial-Vector term H_AV with N-dimensional spin-momentum coupling."""
    num_sites = get_total_sites(L, dim)
    P, Q = get_P_Q(pi_max, n_b)
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
                # Build pion_grad via dict accumulator: Z_{next,n} 2^n - Z_{curr,n} 2^n.
                # Indices on (next, curr) are always distinct, so each Z lives in its own bucket.
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
                        coeff = calculate_chiral_coeff(m_alpha, m_beta, iso_idx=I + 1, spin_idx=spin_idx)
                        if abs(coeff) > 1e-9:
                            nucleon_sum += coeff * Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)

                H_AV += prefactor * Q * (nucleon_sum * pion_grad)
    return H_AV


def H_WT_Logic(L, dim, n_b, pi_max, f_pi, a_L):
    """Implements the Weinberg-Tomozawa term H_WT (Eq. 70/74) with full algebraic expansion of pi * Pi."""
    num_sites = get_total_sites(L, dim)
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L)
    P, Q = get_P_Q(pi_max, n_b)
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
            # pion_block via dict accumulator
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
                        # Canonical-key Z_a Z_b
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
                        nucleon_block += coeff * Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)

            H_WT += (prefactor * sign) * (pion_block * nucleon_block)
    return H_WT


def Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, pi_max, params):
    """Combines all position-basis and momentum-basis terms."""
    a_L = params['a_L']

    H_pos_free, H_mom = H_pion_free(L, dim, n_b, pi_max, params['m_pi'], a_L)
    H_AV = H_axial_vector(L, dim, n_b, pi_max, params['g_A'], params['f_pi'], a_L)
    H_WT = H_WT_Logic(L, dim, n_b, pi_max, params['f_pi'], a_L)

    H_pos = H_pos_free + H_AV + H_WT
    return H_pos, H_mom
