import math

from src_PI.estimation.block_encoders import get_block_encoder
from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian


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

    Basis-agnostic + encoder-agnostic: dispatches via `config.pion_basis`
    through `build_eft_hamiltonian` for construction, and via
    `config.block_encoder` through `get_block_encoder(...)` for normalization
    + walk-step resource estimation. The only basis-conditional code path
    in the orchestrator is the QFT overhead calculation (Fock = 0,
    amplitude > 0).
    """
    print(f"--- Resource Evaluation: {L}^{dim} Lattice, {n_b} Bits/Species, "
          f"basis={config.pion_basis}, encoder={config.block_encoder}, "
          f"walk_mode={config.walk_mode} ---")

    # 1. Build the Hamiltonian bundle via the basis-dispatching constructor.
    print("Constructing Full EFT Hamiltonian...")
    bundle, q_count, num_sites = build_eft_hamiltonian(
        L, dim, n_b, pi_max, params, config
    )

    print(f"Total Qubits:      {q_count}")
    print(f"Total Sites:       {num_sites}")
    print(f"Sub-Hamiltonians:  {bundle.names()}")

    # 2. Dispatch to the block-encoder strategy: normalize + walk estimate.
    strategy = get_block_encoder(config.block_encoder)
    norm_data = strategy.estimate(bundle, num_sites, n_b, config)

    # 3. QFT overhead — basis-conditional (zero for Fock), encoder-agnostic.
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

    # 4. Final per-step T budget = walk + QFT.
    norm_data['QFT_T_Count'] = total_qft_step_cost
    norm_data['Total_T_Count'] = norm_data.get('Walk_T_Count', 0) + total_qft_step_cost

    return norm_data
