from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources
from src_PI.estimation.instances import MyCustomHamiltonian

def run_qubitization_analysis(qubit_ham):
    """
    Converts an OpenFermion QubitOperator to a pyLIQTR instance 
    and prints the resource requirements for Qubitized Phase Estimation.
    """
    # 1. Convert OpenFermion QubitOperator to the dict format expected by the bridge
    pauli_dict = {}
    for term, coeff in qubit_ham.terms.items():
        # Standardize "I" for identity or "Z0 X1" for sparse terms
        p_string = " ".join([f"{op}{idx}" for idx, op in term]) if term else "I"
        pauli_dict[p_string] = float(coeff.real)

    # 2. Instantiate our bridge class
    instance = MyCustomHamiltonian(pauli_dict)

    # 3. Block Encoding (LCU)
    # This uses your version's BlockEncodings path
    encoding_type = VALID_ENCODINGS.PauliLCU
    block_encoding = getEncoding(encoding_type)(instance)

    # 4. Walk Operator for Qubitization
    # This uses your version's qubitization path
    walk_operator = QubitizedWalkOperator(block_encoding)

    # 5. Resource Analysis
    print("\n" + "="*45)
    print("      pyLIQTR RESOURCE ESTIMATION")
    print("="*45)
    print(f"Total Qubits in Hamiltonian: {instance.n_qubits()}")
    print(f"Lambda (Normalization):      {block_encoding.alpha:.4f}")
    
    # Run the estimation on the walk operator
    # Note: estimate_resources returns a dictionary of counts (T-gates, Cliffords, etc.)
    results = estimate_resources(walk_operator)
    
    for key, value in results.items():
        # Clean up keys for printing (e.g., 'T' -> 'T-Gates')
        label = key.replace('_', ' ').title()
        print(f"{label:25}: {value}")
    print("="*45 + "\n")
    
    return results