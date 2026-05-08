from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources
from src_PI.estimation.instances import MyCustomHamiltonian


def _ham_to_pyliqtr_instance(qubit_ham):
    """Helper to convert OpenFermion QubitOperator to MyCustomHamiltonian."""
    pauli_dict = {}
    for term, coeff in qubit_ham.terms.items():
        p_string = " ".join([f"{op}{idx}" for idx, op in term]) if term else "I"
        pauli_dict[p_string] = float(coeff.real)
    return MyCustomHamiltonian(pauli_dict)

def run_qubitization_analysis(pos_ham, mom_ham, n_sites, n_qubits_per_site):
    """
    Analyzes resources for Qubitized Phase Estimation by splitting the 
    Hamiltonian into position and momentum space walks.
    """
    # 1. Create instances for both Hamiltonians
    pos_instance = _ham_to_pyliqtr_instance(pos_ham)
    mom_instance = _ham_to_pyliqtr_instance(mom_ham)

    # 2. Generate Block Encodings
    encoding_type = VALID_ENCODINGS.PauliLCU
    pos_encoding = getEncoding(encoding_type)(pos_instance)
    mom_encoding = getEncoding(encoding_type)(mom_instance)

    # 3. Create Walk Operators
    pos_walk = QubitizedWalkOperator(pos_encoding)
    mom_walk = QubitizedWalkOperator(mom_encoding)

    # 4. Get PyLIQTR Estimates
    pos_results = estimate_resources(pos_walk)
    mom_results = estimate_resources(mom_walk)

    # 5. Combine results with the correct per-key semantics for the split-oracle.
    # Gate counts (T, Clifford) are summed: one walk-step runs both encodings sequentially.
    # LogicalQubits is the peak hardware requirement: since the two walks reuse the same
    # hardware (system register + ancillas), the peak is max(pos, mom), not pos + mom.
    pos_lq = pos_results.get('LogicalQubits', 0)
    mom_lq = mom_results.get('LogicalQubits', 0)

    combined_results = {
        'T': pos_results.get('T', 0) + mom_results.get('T', 0),
        'Clifford': pos_results.get('Clifford', 0) + mom_results.get('Clifford', 0),
        'LogicalQubits': max(pos_lq, mom_lq),
        'Pos_LogicalQubits': pos_lq,
        'Mom_LogicalQubits': mom_lq,
    }

    # Pass through any other keys (e.g. 'Rotations' when profile=True) by summing.
    for key in set(pos_results.keys()).union(mom_results.keys()):
        if key not in combined_results:
            combined_results[key] = pos_results.get(key, 0) + mom_results.get(key, 0)

    # 6. Print the summary
    print("\n" + "="*50)
    print("      SPLIT ORACLE RESOURCE ESTIMATION")
    print("="*50)
    print(f"System Qubits (Pos instance): {pos_instance.n_qubits()}")
    print(f"System Qubits (Mom instance): {mom_instance.n_qubits()}")
    print(f"Logical Qubits (Pos Walk):    {pos_lq}")
    print(f"Logical Qubits (Mom Walk):    {mom_lq}")
    print(f"Logical Qubits (peak, max):   {combined_results['LogicalQubits']}")
    print(f"Lambda (Pos Normalization):   {pos_encoding.alpha:.4f}")
    print(f"Lambda (Mom Normalization):   {mom_encoding.alpha:.4f}")
    print(f"Total Lambda (Alpha_pos + Alpha_mom): {(pos_encoding.alpha + mom_encoding.alpha):.4f}")
    print("-" * 50)

    for key, value in combined_results.items():
        label = key.replace('_', ' ').title()
        if isinstance(value, (int, float)) and value > 10000:
            print(f"{label:25}: {value:.4e}")
        else:
            print(f"{label:25}: {value}")
    print("="*50 + "\n")

    return combined_results