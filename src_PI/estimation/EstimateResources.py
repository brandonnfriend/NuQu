import math
from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.estimation.estimators import run_qubitization_analysis
from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe

def calculate_qft_cost(L, dim, n_b):
    """
    Estimates T-gate overhead for QFTs in the Split-Oracle.
    Each pion species on each site needs a QFT of size n_b.
    """
    num_pion_registers = 3 * (L**dim)
    
    # Proper scaling for an n-bit QFT synthesis 
    if n_b <= 1:
        t_gates_per_qft = 0  # Only Hadamards needed, which are Clifford (0 T-gates)
    else:
        t_gates_per_qft = int(8 * n_b * math.log2(n_b))
        
    return num_pion_registers * t_gates_per_qft

def evaluate_resources(L, dim, n_b, pi_max, params):
    """Calculates and prints hardware requirements for the D-dimensional EFT."""
    
    print(f"--- Resource Evaluation: {L}^{dim} Lattice, {n_b} Bits/Species ---")
    
    # 1 & 2. Build Hamiltonians via the Constructor
    print("Constructing Full EFT Hamiltonian...")
    H_pos, H_mom, q_count, num_sites = build_eft_hamiltonian(L, dim, n_b, pi_max, params)
    
    print(f"Total Qubits:      {q_count}")
    print(f"Total Sites:       {num_sites}")
    
    # 3. Normalize Hamiltonians
    print("Normalizing Hamiltonians for QPE...")
    norm_data = normalize_for_qpe(H_pos, H_mom, safety_factor=2.5)
    
    print(f"-> Extracted classical energy shift: {norm_data['identity_shift'].real:.4e}")
    print(f"-> Physical Lambda:                  {norm_data['physical_lambda']:.4e}")
    print(f"-> Spectral Delta (Scaling factor):  {norm_data['delta']:.4e}")

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
    liqtr_results = run_qubitization_analysis(
        norm_data['H_pos_norm'], 
        norm_data['H_mom_norm'], 
        num_sites, 
        n_b
    )
    
    # 5. QFT Overhead Calculation
    qft_overhead = calculate_qft_cost(L, dim, n_b)
    # Multiply by 2 because each walk step requires a QFT into momentum basis and an IQFT back
    total_qft_step_cost = 2 * qft_overhead 
    
    print("\n" + "-"*50)
    print("       SPLIT-ORACLE BASIS TRANSFORMATION COST")
    print("-"*50)
    print(f"Pion Registers per QFT:      {3 * (L**dim)}")
    print(f"T-gates per QFT/IQFT module: {qft_overhead: .4e}")
    print(f"Total T-gates per walk step: {total_qft_step_cost: .4e}")
    print("-"*50)
    
    # 6. Package Data for the Sweep Logger
    base_t_count = liqtr_results.get('T', 0) if isinstance(liqtr_results, dict) else 0
    base_clifford = liqtr_results.get('Clifford', 0) if isinstance(liqtr_results, dict) else 0
    
    # PyLIQTR combined the pos and mom walk qubits, but they happen sequentially on the same hardware.
    # Therefore, the actual logical qubits needed is just half of the combined total.
    combined_qubits = liqtr_results.get('Logicalqubits', 0) if isinstance(liqtr_results, dict) else 0
    walk_logical_qubits = combined_qubits // 2 

    # Save exactly the fields requested for the JSON output
    norm_data['Walk_T_Count'] = base_t_count
    norm_data['QFT_T_Count'] = total_qft_step_cost
    norm_data['Total_T_Count'] = base_t_count + total_qft_step_cost
    norm_data['Walk_Clifford_Count'] = base_clifford
    norm_data['Logical_Qubits_Per_Walk'] = walk_logical_qubits
    norm_data['Physical_Lambda'] = norm_data['physical_lambda'] # The non-normalized Lambda
    
    return norm_data