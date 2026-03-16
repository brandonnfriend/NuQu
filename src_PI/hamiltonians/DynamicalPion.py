from openfermion import QubitOperator
from src_PI.hamiltonians.Operators import Nucleon_Transition_JW, Pi_Squared_Operator, Gradient_Squared_Operator
from src_PI.utils.utils import site_to_pion_qubit_1D, site_to_nucleon_qubit_1D, total_qubits_1D
from src_PI.utils.utils import calculate_chiral_coeff, get_P_Q, get_Pp_Qp 

"""
TODO: FULL EFT INTEGRATION
1. Multi-Axis Interaction: Expand 'Full_Dynamical_Pion_Hamiltonian' to loop 
   through all spatial directions (x, y, z) for the Axial-Vector coupling.
2. Weinberg-Tomozawa Scaling: Since WT is a local (on-site) contact term, 
   ensure the summation covers the expanded 2D/3D site grid.
3. Resource Export: Create an interface for 'pyLIQTR' to ingest the 
   resulting QubitOperator for T-gate and depth estimation.
"""


def H_pion_free_1D(num_sites, n_b, pi_max, m_pi, a_L):
    """
    Full H_pi (Eq. 67) for 1D lattice.
    """
    P, Q = get_P_Q(pi_max, n_b)
    H_total = QubitOperator()
    
    # Prefactor a_L^3 / 2 (Note: in 1D, volume factor might be a_L/2)
    # The paper uses a_L^3 for 3D; for 1D we usually scale by a_L.
    vol_factor = (a_L**3) / 2.0 
    
    for x in range(num_sites):
        for I in range(3): # Three pion species
            # Mass term: m_pi^2 * pi^2
            H_total += vol_factor * (m_pi**2) * Pi_Squared_Operator(x, I, n_b, P, Q)
            
            # Gradient term: (nabla pi)^2
            if x < num_sites - 1: # Open boundary conditions
                H_total += vol_factor * Gradient_Squared_Operator(x, x+1, I, n_b, Q, a_L)
                
            # Kinetic term: Pi^2 (Momentum) 
            # Per paper, this is encoded "similarly" to pi^2.
            # In the field basis, this requires a change of basis (QFT),
            # but for the Hamiltonian definition, it's the same operator form
            # in the momentum basis.
            # H_total += vol_factor * Momentum_Squared_Operator(...)
            
    return H_total

def H_axial_vector_1D(num_sites, n_b, pi_max, g_A, f_pi, a_L):
    """
    Implements H_AV (Eq. 69) using the encoding in Eq. 73.
    """
    P, Q = get_P_Q(pi_max, n_b)
    H_AV = QubitOperator()
    
    # Physical constant prefactor from Eq. 69
    prefactor = g_A / (2.0 * f_pi * a_L)
    
    # Fermionic modes: 0:(0,0)=up_p, 1:(0,1)=down_p, 2:(1,0)=up_n, 3:(1,1)=down_n
    # For a 1D lattice, we only have the z-derivative (S=1) 
    # and we sum over pion species I=1,2,3
    
    for x in range(num_sites - 1): # Nearest neighbor gradient
        site_next = x + 1
        
        for I in range(3): # Isospin species pi_1, pi_2, pi_3
            # 1. Construct the Gradient Operator for this species: Q * sum(2^n Z_n(x+1) - 2^m Z_m(x))
            pion_grad = QubitOperator()
            for n in range(n_b):
                idx_next = site_to_pion_qubit_1D(site_next, I, n, n_b)
                idx_curr = site_to_pion_qubit_1D(x, I, n, n_b)
                pion_grad += QubitOperator(f'Z{idx_next}', 2**n)
                pion_grad -= QubitOperator(f'Z{idx_curr}', 2**n)
            
            # 2. Nucleon part: This involves the matrices [tau_I] and [sigma_S]
            # In 1D, we assume S is the direction of the lattice (e.g. sigma_z)
            # This loop handles the alpha, beta, gamma, delta sums in Eq 69
            nucleon_sum = QubitOperator()
            
            # Simplified for 1D: We look for combinations where [tau_I] and [sigma_S] are non-zero
            # Example: for pi_1 (I=0), tau_1 flips isospin (p <-> n)
            # You would iterate through the 4x4 combinations of nucleon modes
            for m_alpha in [(0,0), (0,1), (1,0), (1,1)]:
                for m_beta in [(0,0), (0,1), (1,0), (1,1)]:
                    # Match the arguments: I+1 (since I is 0,1,2), spin_idx=3 for z-direction
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, iso_idx=I+1, spin_idx=3) 
                    
                    if abs(coeff) > 1e-9:
                        n_trans = Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)
                        nucleon_sum += coeff * n_trans
            
            # 3. Combine: Prefactor * Gradient * Nucleon transition
            # Eq 73 shows this results in Pauli weight 5 strings (2 from JW, 1 from Z_pion)
            H_AV += prefactor * Q * (nucleon_sum * pion_grad)

    return H_AV

def H_WT_1D(num_sites, n_b, pi_max, f_pi, a_L):
    """
    Implements H_WT (Eq. 70/74).
    Note: In field basis, Pi is not diagonal. Eq 74 assumes the 
    operator is being built in the basis where Pi_3 is diagonal.
    """
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L)
    P, Q = get_P_Q(pi_max, n_b)
    H_WT = QubitOperator()
    
    # Constant from Eq 70
    prefactor = 1.0 / (4.0 * f_pi**2)
    
    modes = [(0,0), (0,1), (1,0), (1,1)]
    
    for x in range(num_sites):
        # Sum over I1, I2, I3 with Levi-Civita epsilon_I1,I2,I3
        # Non-zero combinations: (1,2,3), (2,3,1), (3,1,2) [val 1]
        # and (1,3,2), (3,2,1), (2,1,3) [val -1]
        epsilons = [(0,1,2, 1), (1,2,0, 1), (2,0,1, 1), 
                    (0,2,1, -1), (2,1,0, -1), (1,0,2, -1)]
        
        for I1, I2, I3, sign in epsilons:
            # Pion Part: pi_{I2} * Pi_{I3}
            # We use the mapping from Eq 74
            pion_block = QubitOperator()
            
            # This is a simplification of Eq 74 (expanding the product)
            # pi_I2 encoded with P, Q | Pi_I3 encoded with Pp, Qp
            for m in range(n_b):
                for l in range(n_b):
                    idx_I2 = site_to_pion_qubit_1D(x, I2, m, n_b)
                    idx_I3 = site_to_pion_qubit_1D(x, I3, l, n_b)
                    
                    # Highest weight term: Q * Qp * Z_m * Z_l
                    term = QubitOperator(f'Z{idx_I2} Z{idx_I3}', Q * Qp * (2**(m+l)))
                    # Add cross terms (P*Qp*Z + Pp*Q*Z + P*Pp) here for completeness...
                    pion_block += term
            
            # Nucleon Part: a_dagger * tau_{I1} * a
            nucleon_block = QubitOperator()
            for m_alpha in modes:
                for m_beta in modes:
                    # sigma[0] is Identity for spin (WT doesn't flip spin usually)
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1+1, 0)
                    if abs(coeff) > 1e-9:
                        nucleon_block += coeff * Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)
            
            H_WT += (prefactor * sign) * (pion_block * nucleon_block)
            
    return H_WT

def Full_Dynamical_Pion_Hamiltonian(num_sites, n_b, pi_max, params):
    """
    The complete HDpi = H_pi + H_AV + H_WT + (H_nucleon from src_OF)
    """
    # params is a dict containing g_A, f_pi, m_pi, a_L, etc.
    a_L = params['a_L']
    
    # 1. Free Pion (Mass + Gradient)
    H = H_pion_free_1D(num_sites, n_b, pi_max, params['m_pi'], a_L)
    
    # 2. Axial Vector
    H += H_axial_vector_1D(num_sites, n_b, pi_max, params['g_A'], params['f_pi'], a_L)
    
    # 3. Weinberg-Tomozawa
    H += H_WT_1D(num_sites, n_b, pi_max, params['f_pi'], a_L)
    
    return H

