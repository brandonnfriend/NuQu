# test_dynamical_pions.py
from openfermion import jordan_wigner
from src_PI.hamiltonians.DynamicalPion import Full_Dynamical_Pion_Hamiltonian
from src_PI.hamiltonians.StaticTerms import Static_Nucleon_Hamiltonian_1D
from src_PI.utils.utils import total_qubits_1D
from src_PI.estimation.estimators import run_qubitization_analysis

"""
TODO: RESOURCE ESTIMATION & BENCHMARKING
1. Parameter Sweeps: Update the script to loop over n_b (1 to 4) and 
   lattice_size to generate scaling plots for Qubits vs. Pauli Strings.
2. pyLIQTR Integration: Incorporate the Block Encoding and QSP (Quantum 
   Signal Processing) estimation logic currently in the Jupyter Notebook.
3. Weight Analysis: Monitor the 'Maximum Pauli Weight' as dimensionality 
   increases; high weights (>20) may necessitate exploring Bravyi-Kitaev 
   mappings to reduce hardware requirements.
"""

def evaluate_resources(num_sites, n_b, pi_max, params):
    """Calculates and prints hardware requirements for the EFT."""
    print(f"--- Resource Evaluation: {num_sites} Sites, {n_b} Bits/Species ---")
    
    # 1. Total Qubit Count
    q_count = total_qubits_1D(num_sites, n_b)
    print(f"Total Qubits:      {q_count}")
    
    # 2. Static Nucleon Sector (H_free, H_C, H_CI)
    # Built as FermionOperators natively aware of the pion gaps
    print("Building Static Nucleon Sector...")
    H_static_f = Static_Nucleon_Hamiltonian_1D(
        params['h'], params['C'], params['CI'], num_sites, n_b
    )
    H_static_q = jordan_wigner(H_static_f)
    
    # 3. Dynamical Pion Sector (H_pi, H_AV, H_WT)
    # Built directly as QubitOperators (Pauli strings)
    print("Building Dynamical Pion Sector...")
    H_dyn_q = Full_Dynamical_Pion_Hamiltonian(num_sites, n_b, pi_max, params)
    
    # 4. Final Hamiltonian
    H_total = H_static_q + H_dyn_q
    
    # 5. Extract Metrics
    num_terms = len(H_total.terms)
    weights = [len(term) for term in H_total.terms]
    max_w = max(weights) if weights else 0
    w5_count = sum(1 for w in weights if w == 5)
    
    print("="*45)
    print(f"Total Pauli Strings:      {num_terms}")
    print(f"Maximum Pauli Weight:     {max_w}")
    print(f"Weight-5 Strings:         {w5_count}")
    print("="*45)

    # 6. Resource Estimation via pyLIQTR
    print("Starting pyLIQTR analysis...")
    run_qubitization_analysis(H_total)
    
    return H_total

if __name__ == "__main__":
    # Physical Constants (Watson et al. 2025)
    test_params = {
        'h': 13.29, 'C': -3.41, 'CI': 1.2, 
        'g_A': 1.27, 'f_pi': 92.4, 'm_pi': 138.0, 'a_L': 1.0
    }
    
    # Evaluation: 2 sites, 2-bit field digitization
    # This should yield 20 qubits and strings with max weight 5.
    try:
        evaluate_resources(num_sites=2, n_b=2, pi_max=5.0, params=test_params)
    except Exception as e:
        print(f"\nResource evaluation failed: {e}")
        import traceback
        traceback.print_exc()

