import math

from src_PI.estimation.block_encoders import get_block_encoder
from src_PI.estimation.combined_walk import compose_combined_walk, wt_basis_change_t
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

    # 3. Momentum-branch QFT (basis change H_pos → H_mom). Zero for the Fock basis.
    momentum_qft = calculate_qft_cost(L, dim, n_b, config)

    # 4. Walk composition. For the amplitude SPLIT (≥2 sub-walks) the QPE-valid step is
    #    a single controlled-sum LCU walk (combined_walk.compose_combined_walk): one
    #    PREPARE over the branches, one SELECT with the momentum + WT species-selective
    #    branches basis-changed by QFT *inside* SELECT, one reflection. The legacy
    #    'split_sum' just sums two independent walk costs + a flat QFT charge, which is
    #    NOT a QPE algorithm for H_pos+H_mom (codex audit P0-4) — kept only for the A/B
    #    delta. The single-walk Fock/PauliLCU anchor is untouched (needs ≥2 sub-walks).
    per_sub = norm_data.get('Per_Sub_Walk', [])
    # Only the amplitude split (≥2 sub-walks) with the block-encoding pieces present
    # routes through the combined-walk composition; the single-walk Fock/sparse path
    # and any encoder that doesn't expose SELECT+PREPARE fall through unchanged.
    use_combined = (getattr(config, 'walk_composition', 'combined_lcu') == 'combined_lcu'
                    and len(per_sub) >= 2
                    and all(e.get('T_enc') for e in per_sub))

    if use_combined:
        combined = compose_combined_walk(per_sub, momentum_qft, L, dim, n_b)
        norm_data['Walk_T_Count'] = combined['Walk_T_Count']
        norm_data['Walk_Clifford_Count'] = combined['Walk_Clifford_Count']
        norm_data['Logical_Qubits'] = combined['Logical_Qubits']
        # The basis change is INSIDE the walk step now — reported as a component, not
        # added again to the total.
        norm_data['QFT_T_Count'] = (momentum_qft
                                    + wt_basis_change_t(L, dim, n_b))
        norm_data['Total_T_Count'] = combined['Walk_T_Count']
        norm_data['walk_composition'] = 'combined_lcu'
        norm_data['composition_components'] = combined['composition_components']
        norm_data['composition_label'] = combined['composition_label']
        c = combined['composition_components']
        print("\n" + "-" * 60)
        print("   QPE-VALID COMBINED WALK (controlled-sum LCU of the split oracle)")
        print("-" * 60)
        print(f"  Σ SELECT+PREPARE (sub-oracles):  {c['sum_select_prepare_T']:.4e} T")
        print(f"  momentum-branch QFT+IQFT:        {c['momentum_qft_T']:.4e} T")
        print(f"  WT species-selective basis-chg:  {c['wt_species_basis_change_T']:.4e} T")
        print(f"  k-way PREPARE + reflection:      "
              f"{c['lcu_prepare_T'] + c['reflection_T']:.4e} T")
        print(f"  -> combined walk-step T:         {combined['Walk_T_Count']:.4e}")
        print(f"  -> logical qubits (LCU+QFT wk):  {combined['Logical_Qubits']}")
        print("-" * 60 + "\n")
    else:
        # Legacy split_sum (invalid for ≥2 sub-walks) OR the single-walk Fock/sparse path.
        norm_data['QFT_T_Count'] = momentum_qft
        norm_data['Total_T_Count'] = norm_data.get('Walk_T_Count', 0) + momentum_qft
        norm_data['walk_composition'] = (
            'split_sum' if len(per_sub) >= 2 else 'single_walk')
        if momentum_qft > 0:
            print("\n" + "-" * 50)
            print("       [LEGACY split_sum] BASIS-ROTATION (QFT) PER STEP")
            print("-" * 50)
            print(f"Pion Registers (QFT'd):      {3 * (L**dim)}")
            print(f"Total T-gates per walk step: {momentum_qft: .4e}")
            print("-" * 50)
        else:
            print(f"\n[basis={config.pion_basis}] single-walk oracle; "
                  f"no basis-rotation cost.\n")

    return norm_data
