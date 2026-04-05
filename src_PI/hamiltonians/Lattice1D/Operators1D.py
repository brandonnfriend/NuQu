# src_PI/hamiltonians/Operators.py
from openfermion import QubitOperator, FermionOperator, jordan_wigner
from src_PI.utils.utils import site_to_nucleon_qubit_1D, site_to_pion_qubit_1D

"""
TODO: OPERATOR VECTORIZATION & OPTIMIZATION
1. Directional Gradients: Update 'Gradient_Squared_Operator' to take a 
   'direction' vector (dx, dy, dz). 2D/3D requires summing (delta_i pi)^2 
   over all spatial axes.
2. Axial Spin Coupling: Ensure 'Nucleon_Transition_JW' can handle spin-flips 
   aligned with specific spatial directions (sigma_x for d_x pi, etc.) 
   to maintain the dot product integrity of the Axial-Vector term.
3. String Compression: Implement a parser to combine redundant Pauli strings 
   during construction to prevent linear growth of the term count in 3D.
TODO: CONJUGATE MOMENTUM (Pi^2) IMPLEMENTATION
1. Define 'Delta_Pi' and 'Pi_max' constants using Eq. 32:
   Delta_Pi = (2 * pi) / (vol * delta_pi * (2 * pi_max / delta_pi + 1))
2. Implement 'Momentum_Squared_Operator': This will look nearly identical 
   to 'Pion_Squared_Operator' but must use 'Delta_Pi' and 'Pi_max' as the 
   linear coefficients for the Z-string expansion.
3. Basis Awareness: Note that this operator is diagonal in the Fourier Basis. 
   For full simulation, QFT circuits must wrap this term, but for 
   Hamiltonian construction, we represent it in its diagonal (Z-basis) form.
"""

def Pi_Squared_Operator(site_id, pion_species, n_b, P, Q):
    """Implements Eq. (71): pi^2 expressed in Pauli Z operators."""
    H = QubitOperator('', P**2)
    
    # Term 2: 2PQ * sum(2^m * Z_m)
    for m in range(n_b):
        idx = site_to_pion_qubit_1D(site_id, pion_species, m, n_b)
        H += QubitOperator(f'Z{idx}', 2 * P * Q * (2**m))
        
    # Term 3: Q^2 * sum(2^(m+m') * Z_m * Z_m')
    for m in range(n_b):
        for mp in range(n_b):
            idx_m = site_to_pion_qubit_1D(site_id, pion_species, m, n_b)
            idx_mp = site_to_pion_qubit_1D(site_id, pion_species, mp, n_b)
            coeff = Q**2 * (2**(m + mp))
            
            if idx_m == idx_mp:
                H += QubitOperator('', coeff) # Z_m * Z_m = I
            else:
                H += QubitOperator(f'Z{idx_m} Z{idx_mp}', coeff)
    return H

def Gradient_Squared_Operator(site_x, site_y, pion_species, n_b, Q, a_L):
    """Implements Eq. (72): ((pi(y) - pi(x)) / a_L)^2"""
    H = QubitOperator()
    inv_aL2 = 1.0 / (a_L**2)
    
    for m in range(n_b):
        idx_x_m = site_to_pion_qubit_1D(site_x, pion_species, m, n_b)
        idx_y_m = site_to_pion_qubit_1D(site_y, pion_species, m, n_b)
        for n in range(n_b):
            idx_x_n = site_to_pion_qubit_1D(site_x, pion_species, n, n_b)
            idx_y_n = site_to_pion_qubit_1D(site_y, pion_species, n, n_b)
            
            common_coeff = (Q**2 * 2**(m+n)) * inv_aL2
            
            # site x internal
            if idx_x_m == idx_x_n: H += QubitOperator('', common_coeff)
            else: H += QubitOperator(f'Z{idx_x_m} Z{idx_x_n}', common_coeff)
                
            # site y internal
            if idx_y_m == idx_y_n: H += QubitOperator('', common_coeff)
            else: H += QubitOperator(f'Z{idx_y_m} Z{idx_y_n}', common_coeff)
                
            # cross term -2 * pi(x) * pi(y)
            H += QubitOperator(f'Z{idx_x_m} Z{idx_y_n}', -2 * common_coeff)
    return H

def Momentum_Squared_Operator(site_id, pion_species, n_b, Pp, Qp):
    """Implements the Pi^2 (conjugate momentum) operator in its diagonal basis."""
    H = QubitOperator('', Pp**2)
    
    # Term 2: 2 * Pp * Qp * sum(2^m * Z_m)
    for m in range(n_b):
        idx = site_to_pion_qubit_1D(site_id, pion_species, m, n_b)
        H += QubitOperator(f'Z{idx}', 2 * Pp * Qp * (2**m))
        
    # Term 3: Qp^2 * sum(2^(m+m') * Z_m * Z_m')
    for m in range(n_b):
        for mp in range(n_b):
            idx_m = site_to_pion_qubit_1D(site_id, pion_species, m, n_b)
            idx_mp = site_to_pion_qubit_1D(site_id, pion_species, mp, n_b)
            coeff = Qp**2 * (2**(m + mp))
            
            if idx_m == idx_mp:
                H += QubitOperator('', coeff) # Z_m * Z_m = I
            else:
                H += QubitOperator(f'Z{idx_m} Z{idx_mp}', coeff)
    return H

def Nucleon_Transition_JW(site_id, mode_alpha, mode_beta, n_b):
    """Jordan-Wigner mapped Pauli string for (a^dagger_alpha a_beta + h.c.)"""
    idx_alpha = site_to_nucleon_qubit_1D(site_id, mode_alpha, n_b)
    idx_beta = site_to_nucleon_qubit_1D(site_id, mode_beta, n_b)
    
    f_op = FermionOperator(f'{idx_alpha}^ {idx_beta}') + FermionOperator(f'{idx_beta}^ {idx_alpha}')
    return jordan_wigner(f_op)