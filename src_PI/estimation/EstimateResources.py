import math

from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.estimation.estimators import run_qubitization_analysis
from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe


def calculate_qft_cost(L, dim, n_b, config):
    """Estimates T-gate overhead for QFTs in the split-oracle walk step.

    Returns 0 for any basis that doesn't need a basis rotation between
    sub-Hamiltonians (e.g. Fock basis, where π and Π share one register).
    For the amplitude basis, each pion species per site needs an n_b-bit
    QFT, doubled because each walk step needs a QFT into momentum basis
    and an IQFT back.
    """
    if config.pion_basis != 'amplitude':
        return 0

    num_pion_registers = 3 * (L ** dim)
    if n_b <= 1:
        t_gates_per_qft = 0  # Hadamards only → 0 T-gates.
    else:
        t_gates_per_qft = int(8 * n_b * math.log2(n_b))
    # ×2: QFT to momentum basis + IQFT back, per walk step.
    return 2 * num_pion_registers * t_gates_per_qft


def evaluate_resources(L, dim, n_b, pi_max, params, config):
    """Calculates and prints hardware requirements for the D-dimensional EFT.

    Basis-agnostic: dispatches via `config.pion_basis` through
    `build_eft_hamiltonian`. Iterates over the resulting HamiltonianBundle's
    sub-Hamiltonians; the only basis-conditional code path is the QFT
    overhead calculation.
    """
    print(f"--- Resource Evaluation: {L}^{dim} Lattice, {n_b} Bits/Species, "
          f"basis={config.pion_basis}, walk_mode={config.walk_mode} ---")

    # 1. Build the Hamiltonian bundle via the basis-dispatching constructor.
    print("Constructing Full EFT Hamiltonian...")
    bundle, q_count, num_sites = build_eft_hamiltonian(
        L, dim, n_b, pi_max, params, config
    )

    print(f"Total Qubits:      {q_count}")
    print(f"Total Sites:       {num_sites}")
    print(f"Sub-Hamiltonians:  {bundle.names()}")

    # 2. Normalize every sub-Hamiltonian against a shared Δ.
    print("Normalizing Hamiltonians for QPE...")
    norm_data = normalize_for_qpe(bundle, safety_factor=2.5)

    print(f"-> Extracted classical energy shift: {norm_data['identity_shift'].real if hasattr(norm_data['identity_shift'], 'real') else norm_data['identity_shift']:.4e}")
    print(f"-> Physical Lambda (total):          {norm_data['physical_lambda']:.4e}")
    print(f"-> Spectral Delta (Scaling factor):  {norm_data['delta']:.4e}")
    for name, lam in norm_data['sub_lambdas']:
        share = (lam / norm_data['physical_lambda'] * 100.0) if norm_data['physical_lambda'] else 0.0
        print(f"   - sub '{name}': Λ = {lam:.4e}  ({share:.2f}% of total)")

    # Diagnostic: combined Pauli stats across all sub-Hamiltonians.
    num_terms = 0
    weights = []
    for _, H_norm in norm_data['sub_hamiltonians']:
        num_terms += len(H_norm.terms)
        weights.extend(len(t) for t in H_norm.terms)
    max_w = max(weights) if weights else 0

    print("\n" + "=" * 45)
    print(f"Total Pauli Strings (Non-Identity): {num_terms}")
    print(f"Maximum Pauli Weight:               {max_w}")
    print("=" * 45)

    # 3. pyLIQTR resource estimation across the bundle.
    print("Starting pyLIQTR analysis...")
    liqtr_results = run_qubitization_analysis(norm_data, num_sites, n_b)

    # 4. QFT overhead — basis-conditional (zero for Fock).
    total_qft_step_cost = calculate_qft_cost(L, dim, n_b, config)
    if total_qft_step_cost > 0:
        print("\n" + "-" * 50)
        print("       BASIS-ROTATION (QFT) COST PER WALK STEP")
        print("-" * 50)
        print(f"Pion Registers (QFT'd):      {3 * (L**dim)}")
        print(f"Total T-gates per walk step: {total_qft_step_cost: .4e}")
        print("-" * 50)
    else:
        print(f"\n[basis={config.pion_basis}] No QFT-between-walks cost; "
              f"all sub-Hamiltonians share one register.\n")

    # 5. Package data for the sweep logger.
    if not isinstance(liqtr_results, dict):
        liqtr_results = {}

    base_t_count = liqtr_results.get('T', 0)
    base_clifford = liqtr_results.get('Clifford', 0)
    logical_qubits = liqtr_results.get('LogicalQubits', 0)
    per_sub = liqtr_results.get('per_sub', [])

    norm_data['Walk_T_Count'] = base_t_count
    norm_data['QFT_T_Count'] = total_qft_step_cost
    norm_data['Total_T_Count'] = base_t_count + total_qft_step_cost
    norm_data['Walk_Clifford_Count'] = base_clifford
    norm_data['Logical_Qubits'] = logical_qubits
    norm_data['Physical_Lambda'] = norm_data['physical_lambda']

    # Per-sub-Hamiltonian breakouts for the JSON record. Encodes the
    # bundle structure in a backward-friendly way for plotting tools.
    norm_data['Per_Sub_Walk'] = [
        {
            'name': e['name'],
            'T': e['T'],
            'Clifford': e['Clifford'],
            'LogicalQubits': e['LogicalQubits'],
            'alpha': e['alpha'],
        }
        for e in per_sub
    ]

    return norm_data
