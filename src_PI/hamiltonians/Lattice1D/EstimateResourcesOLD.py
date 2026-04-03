from openfermion import jordan_wigner
from src_PI.hamiltonians.Lattice1D.DynamicalPion1D import Full_Dynamical_Pion_Hamiltonian_1D
from src_PI.hamiltonians.Lattice1D.StaticTerms1D import Static_Nucleon_Hamiltonian_1D
from src_PI.utils.utils import total_qubits_1D
from src_PI.utils.LatticeGeometry import total_qubits, get_total_sites
from src_PI.estimation.estimators import run_qubitization_analysis
from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe


def evaluate_resources(num_sites, dim, n_b, pi_max, params):
    """Calculates and prints hardware requirements for the EFT."""
    print(f"--- Resource Evaluation: {num_sites} Sites, {n_b} Bits/Species ---")
    
    q_count = total_qubits_1D(num_sites, n_b)
    print(f"Total Qubits:      {q_count}")
    
    # 1. Build Static Nucleon Sector
    print("Building Static Nucleon Sector...")
    H_static_f = Static_Nucleon_Hamiltonian_1D(
        params['h'], params['C'], params['CI'], num_sites, n_b
    )
    H_static_q = jordan_wigner(H_static_f)
    
    # 2. Build Dynamical Pion Sector
    print("Building Dynamical Pion Sector...")
    H_pos_dyn, H_mom = Full_Dynamical_Pion_Hamiltonian_1D(num_sites, n_b, pi_max, params)
    
    # Combine all position-basis terms
    H_pos = H_static_q + H_pos_dyn
    
    # 3. Normalize Hamiltonians
    print("Normalizing Hamiltonians for QPE...")
    norm_data = normalize_for_qpe(H_pos, H_mom, safety_factor=2.5)
    
    print(f"-> Extracted classical energy shift: {norm_data['identity_shift'].real:.4f}")
    print(f"-> Physical Lambda:                  {norm_data['physical_lambda']:.4f}")
    print(f"-> Spectral Delta (Scaling factor):  {norm_data['delta']:.4f}")

    H_total_norm = norm_data['H_pos_norm'] + norm_data['H_mom_norm']
    num_terms = len(H_total_norm.terms)
    weights = [len(term) for term in H_total_norm.terms]
    max_w = max(weights) if weights else 0
    w5_count = sum(1 for w in weights if w == 5)
    
    print("\n" + "="*45)
    print(f"Total Pauli Strings (Non-Identity): {num_terms}")
    print(f"Maximum Pauli Weight:               {max_w}")
    print(f"Weight-5 Strings:                   {w5_count}")
    print("="*45)

    # 4. Resource Estimation via pyLIQTR
    print("Starting pyLIQTR analysis with split-oracle...")
    run_qubitization_analysis(
        norm_data['H_pos_norm'], 
        norm_data['H_mom_norm'], 
        num_sites, 
        n_b
    )
    
    return norm_data