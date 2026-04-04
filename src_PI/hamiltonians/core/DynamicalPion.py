from openfermion import QubitOperator
from src_PI.hamiltonians.core.Operators import Nucleon_Transition_JW, Pi_Squared_Operator, Gradient_Squared_Operator, Momentum_Squared_Operator
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
    vol_factor = (a_L**3) / 2.0 
    
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for I in range(3):
            # Mass term (Position)
            H_pos_free += vol_factor * (m_pi**2) * Pi_Squared_Operator(x, I, n_b, P, Q)
            # Kinetic term (Momentum) 
            H_mom_free += vol_factor * Momentum_Squared_Operator(x, I, n_b, Pp, Qp)
            
            # Gradient term: Sum over forward derivatives in all available dimensions
            for d in range(dim):
                if coords[d] < L - 1: # Open Boundary Condition check
                    site_next = x + L**d
                    H_pos_free += vol_factor * Gradient_Squared_Operator(x, site_next, I, n_b, Q, a_L)
            
    return H_pos_free, H_mom_free

def H_axial_vector(L, dim, n_b, pi_max, g_A, f_pi, a_L):
    """Implements the Axial-Vecotr term H_AV with N-dimensional spin-momentum coupling.
    L: lattice size in each dimension
    dim: number of dimensions (1,2 or 3)
    nb: number of quibits used to encode the field strength
    pi_max: upper bound for coupling strength
    g_A: nuclear axial charge
    f_pi: pion decay constant
    a_L: lattice spacing 
    """
    num_sites = get_total_sites(L, dim)
    P, Q = get_P_Q(pi_max, n_b)
    H_AV = QubitOperator()
    prefactor = g_A / (2.0 * f_pi * a_L)
    
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        
        # Loop through each spatial dimension (x, y, z)
        for d in range(dim):
            if coords[d] < L - 1: # Forward derivative
                site_next = x + L**d
                
                # Assign spin matrix: If 1D, default to z-axis (3). Else map 0->1(x), 1->2(y), 2->3(z)
                spin_idx = 3 if dim == 1 else d + 1
                
                for I in range(3):
                    pion_grad = QubitOperator()
                    for n in range(n_b):
                        idx_next = site_to_pion_qubit(site_next, I, n, n_b)
                        idx_curr = site_to_pion_qubit(x, I, n, n_b)
                        pion_grad += QubitOperator(f'Z{idx_next}', 2**n)
                        pion_grad -= QubitOperator(f'Z{idx_curr}', 2**n)
                    
                    nucleon_sum = QubitOperator()
                    for m_alpha in [(0,0), (0,1), (1,0), (1,1)]:
                        for m_beta in [(0,0), (0,1), (1,0), (1,1)]:
                            coeff = calculate_chiral_coeff(m_alpha, m_beta, iso_idx=I+1, spin_idx=spin_idx) 
                            if abs(coeff) > 1e-9:
                                n_trans = Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)
                                nucleon_sum += coeff * n_trans
                    
                    H_AV += prefactor * Q * (nucleon_sum * pion_grad)
    return H_AV

def H_WT_Logic(L, dim, n_b, pi_max, f_pi, a_L):
    """Implements the Weinberg-Tomozawa term H_WT (Eq. 70/74) with full algebraic expansion of pi * Pi.
    L: lattice size in each dimension
    dim: number of dimensions (1,2 or 3)
    nb: number of quibits used to encode the field strength
    pi_max: upper bound for coupling strength
    f_pi: pion decay constant
    a_L: lattice spacing"""
    num_sites = get_total_sites(L, dim)
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L)
    P, Q = get_P_Q(pi_max, n_b)
    H_WT = QubitOperator()
    
    prefactor = 1.0 / (4.0 * f_pi**2)
    modes = [(0,0), (0,1), (1,0), (1,1)]
    
    for x in range(num_sites):
        epsilons = [(0,1,2, 1), (1,2,0, 1), (2,0,1, 1), 
                    (0,2,1, -1), (2,1,0, -1), (1,0,2, -1)]
        
        for I1, I2, I3, sign in epsilons:
            pion_block = QubitOperator()
            pion_block += QubitOperator('', P * Pp)
            
            for m in range(n_b):
                idx_I2 = site_to_pion_qubit(x, I2, m, n_b)
                idx_I3 = site_to_pion_qubit(x, I3, m, n_b)
                pion_block += QubitOperator(f'Z{idx_I3}', P * Qp * (2**m))
                pion_block += QubitOperator(f'Z{idx_I2}', Pp * Q * (2**m))
                
                for l in range(n_b):
                    idx_I3_l = site_to_pion_qubit(x, I3, l, n_b)
                    coeff = Q * Qp * (2**(m+l))
                    if idx_I2 == idx_I3_l:
                        pion_block += QubitOperator('', coeff)
                    else:
                        pion_block += QubitOperator(f'Z{idx_I2} Z{idx_I3_l}', coeff)
            
            nucleon_block = QubitOperator()
            for m_alpha in modes:
                for m_beta in modes:
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1+1, 0)
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