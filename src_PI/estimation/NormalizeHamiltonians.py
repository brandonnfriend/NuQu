from openfermion import QubitOperator

def get_hamiltonian_stats(H):
    """
    Extracts the Identity coefficient and the LCU lambda 
    (sum of absolute values of non-identity coefficients).
    """
    id_coeff = 0.0
    lcu_lambda = 0.0
    for term, coeff in H.terms.items():
        if term == ():
            id_coeff += coeff
        else:
            lcu_lambda += abs(coeff)
    return id_coeff, lcu_lambda

def normalize_for_qpe(H_pos, H_mom, safety_factor=2.5):
    """
    Performs full normalization for Qubitized Phase Estimation.
    1. Shifts: Removes Identity terms (tracked as classical offsets).
    2. Scales: Divides by Delta (safety_factor * Lambda) to fit eigenvalues in [0, 0.5].
    """
    id_pos, lam_pos = get_hamiltonian_stats(H_pos)
    id_mom, lam_mom = get_hamiltonian_stats(H_mom)
    
    total_physical_lambda = lam_pos + lam_mom
    total_identity_shift = id_pos + id_mom
    
    # Define Delta (Spectral Range). Delta ~ safety_factor * Lambda
    delta = safety_factor * total_physical_lambda
    
    # Shift and Scale: H_norm = (H - Identity) / Delta
    H_pos_norm = QubitOperator()
    for term, coeff in H_pos.terms.items():
        if term != ():
            H_pos_norm += QubitOperator(term, coeff / delta)
            
    H_mom_norm = QubitOperator()
    for term, coeff in H_mom.terms.items():
        if term != ():
            H_mom_norm += QubitOperator(term, coeff / delta)
            
    return {
        'H_pos_norm': H_pos_norm,
        'H_mom_norm': H_mom_norm,
        'delta': delta,
        'identity_shift': total_identity_shift,
        'physical_lambda': total_physical_lambda
    }