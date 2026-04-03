from openfermion import QubitOperator
from src_PI.hamiltonians.Lattice1D.Operators1D import Nucleon_Transition_JW, Pi_Squared_Operator, Gradient_Squared_Operator, Momentum_Squared_Operator
from src_PI.utils.utils import site_to_pion_qubit_1D, site_to_nucleon_qubit_1D, total_qubits_1D
from src_PI.utils.utils import calculate_chiral_coeff, get_P_Q, get_Pp_Qp 

"""
FIXME: BASIS ALIGNMENT
The H_WT term combines pi (Field Basis) and Pi (Momentum Basis). 
Our QubitOperator currently represents both as Z-strings. 
1. This is valid for resource estimation (Pauli weight is correct).
2. For simulation: The Pi-qubits (I3) MUST be wrapped in QFT/IQFT unitaries
   to ensure the momentum is physically conjugate to the field.
3. Ensure the scaling factors P, Q (Field) and Pp, Qp (Momentum) 
   use the distinct definitions from Eq. 10 and Eq. 32.
"""

"""
TODO: FULL EFT INTEGRATION
1. Multi-Axis Interaction: Expand 'Full_Dynamical_Pion_Hamiltonian' to loop 
   through all spatial directions (x, y, z) for the Axial-Vector coupling.
2. Weinberg-Tomozawa Scaling: Since WT is a local (on-site) contact term, 
   ensure the summation covers the expanded 2D/3D site grid.
3. Resource Export: Create an interface for 'pyLIQTR' to ingest the 
   resulting QubitOperator for T-gate and depth estimation.
TODO: KINETIC ENERGY INTEGRATION
1. Term Integration: Add the H_kin = 0.5 * sum(Pi^2) term to the 
   'Full_Dynamical_Pion_Hamiltonian' loop.
2. Volume Scaling: Ensure the 0.5 factor and the lattice volume factor (a_L^3) 
   are correctly applied to match the EFT Lagrangian density.
3. Lambda Analysis: Observe how the large coefficients of the Pi operator 
   (proportional to 1/delta_pi) impact the total Lambda (normalization) 
   and T-gate counts—this term is likely a major contributor to resource costs.
"""


def H_pion_free_1D(num_sites, n_b, pi_max, m_pi, a_L):
    """
    Returns H_pos_free and H_mom_free separately.
    """
    P, Q = get_P_Q(pi_max, n_b)
    Pp, Qp = get_Pp_Qp(pi_max, n_b, a_L)
    
    H_pos_free = QubitOperator()
    H_mom_free = QubitOperator()
    
    vol_factor = (a_L**3) / 2.0 
    
    for x in range(num_sites):
        for I in range(3):
            # Mass term (Position)
            H_pos_free += vol_factor * (m_pi**2) * Pi_Squared_Operator(x, I, n_b, P, Q)
            
            # Gradient term (Position)
            if x < num_sites - 1:
                H_pos_free += vol_factor * Gradient_Squared_Operator(x, x+1, I, n_b, Q, a_L)
                
            # Kinetic term (Momentum) 
            H_mom_free += vol_factor * Momentum_Squared_Operator(x, I, n_b, Pp, Qp)
            
    return H_pos_free, H_mom_free

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
    Implements H_WT (Eq. 70/74) with full algebraic expansion of pi * Pi.
    """
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
            
            # FULL EXPANSION: (P + Q sum(Z)) * (Pp + Qp sum(Z))
            # 1. Constant term: P * Pp
            pion_block += QubitOperator('', P * Pp)
            
            for m in range(n_b):
                idx_I2 = site_to_pion_qubit_1D(x, I2, m, n_b)
                idx_I3 = site_to_pion_qubit_1D(x, I3, m, n_b)
                
                # 2. Cross term A: P * Qp * Z_{I3}
                pion_block += QubitOperator(f'Z{idx_I3}', P * Qp * (2**m))
                
                # 3. Cross term B: Pp * Q * Z_{I2}
                pion_block += QubitOperator(f'Z{idx_I2}', Pp * Q * (2**m))
                
                # 4. Highest weight term: Q * Qp * Z_{I2} * Z_{I3}
                for l in range(n_b):
                    idx_I3_l = site_to_pion_qubit_1D(x, I3, l, n_b)
                    coeff = Q * Qp * (2**(m+l))
                    # Handle if I2 and I3 somehow mapped to the same qubit (shouldn't happen for diff species)
                    if idx_I2 == idx_I3_l:
                        pion_block += QubitOperator('', coeff)
                    else:
                        pion_block += QubitOperator(f'Z{idx_I2} Z{idx_I3_l}', coeff)
            
            # Nucleon Part: remains the same
            nucleon_block = QubitOperator()
            for m_alpha in modes:
                for m_beta in modes:
                    coeff = calculate_chiral_coeff(m_alpha, m_beta, I1+1, 0)
                    if abs(coeff) > 1e-9:
                        nucleon_block += coeff * Nucleon_Transition_JW(x, m_alpha, m_beta, n_b)
            
            H_WT += (prefactor * sign) * (pion_block * nucleon_block)
            
    return H_WT

def Full_Dynamical_Pion_Hamiltonian_1D(num_sites, n_b, pi_max, params):
    """
    Returns H_pos and H_mom to be fed into the split-oracle resource estimator.
    """
    a_L = params['a_L']
    
    # 1. Free Pion
    H_pos_free, H_mom = H_pion_free_1D(num_sites, n_b, pi_max, params['m_pi'], a_L)
    
    # 2. Axial Vector (Position)
    H_AV = H_axial_vector_1D(num_sites, n_b, pi_max, params['g_A'], params['f_pi'], a_L)
    
    # 3. Weinberg-Tomozawa (Mixed, treated as Position for LCU coefficient purposes)
    H_WT = H_WT_1D(num_sites, n_b, pi_max, params['f_pi'], a_L)
    
    # Combine all position-basis terms
    H_pos = H_pos_free + H_AV + H_WT
    
    return H_pos, H_mom
