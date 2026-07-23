import time

import numpy as np

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.qpe_cost import compute_total_qpe_cost, DEFAULT_DELTA_E_MEV
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    calculate_ns_cutoffs,
    estimate_boson_cutoff,
    get_physical_parameters,
)
from src_PI.utils.Config import Config
from src_PI.utils.DataIO import save_sweep_data


def get_A_sweep_values():
    """Default A-sweep: dense 1–10, spread up to 100."""
    A_dense = np.arange(1, 11)
    A_sparse = np.linspace(20, 100, 15)
    A_values = np.concatenate([A_dense, np.round(A_sparse)]).astype(int)
    return np.unique(A_values)


def get_L_for_A(A):
    """Hook to dynamically pick L from A. Currently fixed; keep at 2 for tests."""
    # WARNING: Keep this at 2 for local testing! Crank to 10 for paper-grade HPC runs.
    return 2


def get_sweep_config(**overrides):
    """Build the run-config for a sweep.

    Centralizes all run-level parameters (lattice, A range, basis, walk
    mode, future kwargs). All fields take sensible defaults; override any
    by keyword.
    """
    defaults = {
        'L': None,                       # None ⇒ use get_L_for_A(A) per iteration
        'dim': 3,
        'A_values': get_A_sweep_values(),
        'pion_basis': 'amplitude',
        'walk_mode': 'series',
        # Amplitude-basis cutoff prescription: 'energy_bound' (Watson Lemma 5)
        # or 'ns' (Nyquist-Shannon optimal, Path B). Ignored by the Fock basis.
        'cutoff_method': 'energy_bound',
        # Per-site boson register-size method: 'heuristic' (default log2(1+A)
        # starter formula) or 'tong' (rigorous Tong-2022 polylog bound,
        # n_q=4-5). Drives the Fock basis + the NS amplitude register.
        'boson_cutoff_method': 'heuristic',
        # Block-encoder strategy: 'pauli_lcu' (default), 'sparse' (task 26),
        # or 'lobe' (task 28). See src_PI/estimation/block_encoders/.
        'block_encoder': 'pauli_lcu',
        'epsilon_cut': 0.1,
        'E_bound_per_A_MeV': 10.0,       # E_max = E_bound_per_A_MeV * A
        # QPE energy-precision target (MeV) for the total-cost computation
        # (QPE_Total_T_Count = Total_T_Count · √2·π·Λ/ΔE). Watson uses 1 MeV.
        'delta_E_MeV': DEFAULT_DELTA_E_MEV,
        # Optional override: if set, used instead of the basis-specific
        # cutoff calculation. Useful for smoke tests where you want a
        # tiny register, and for direct-comparison runs where you want
        # both bases at the same n_b.
        'n_b_override': None,
        'extras': {},
        # Optional short human tag folded into the saved run-id folder, so an
        # important run is easy to find later (e.g. 'paperfig', 'L4-highA').
        'label': None,
    }
    defaults.update(overrides)
    return defaults


def _compute_cutoffs(L, dim, A, params, run_cfg, config):
    """Dispatch cutoff calculation by basis + cutoff_method. Returns
    (n_b, pi_max, Pi_max).

    All branches return the same 3-tuple shape so the caller doesn't need
    to know which one ran:
      - amplitude + 'energy_bound': Watson Lemma 5 (calculate_dynamic_cutoffs).
      - amplitude + 'ns':           Nyquist-Shannon optimal (calculate_ns_cutoffs).
      - fock:                       estimate_boson_cutoff; pi_max/Pi_max are
                                    diagnostic-only and don't drive operators.

    The per-site boson register size follows config.boson_cutoff_method
    ('heuristic' or 'tong') for the fock and ns paths; the energy_bound path
    sets its own n_b from Lemma 5 and ignores it.
    """
    E_max = run_cfg['E_bound_per_A_MeV'] * A
    eps = run_cfg['epsilon_cut']
    bcm = config.boson_cutoff_method
    if config.pion_basis == 'amplitude':
        if config.cutoff_method == 'ns':
            n_b, pi_max, Pi_max = calculate_ns_cutoffs(
                L, dim, A, params, epsilon_cut=eps, E_bound=E_max,
                boson_cutoff_method=bcm,
            )
        else:
            # Watson Lemma 5 sets its own n_b from the energy bound; the
            # per-site boson-cutoff method does not apply here.
            n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(
                L, dim, A, params, epsilon_cut=eps, E_bound=E_max
            )
    else:
        n_b, pi_max, Pi_max = estimate_boson_cutoff(
            L, dim, A, params, epsilon_cut=eps, E_bound=E_max,
            boson_cutoff_method=bcm,
        )
    if run_cfg.get('n_b_override') is not None:
        n_b = int(run_cfg['n_b_override'])
    return n_b, pi_max, Pi_max


def run_sweep(**overrides):
    """Run a single sweep with the given run-config overrides."""
    run_cfg = get_sweep_config(**overrides)
    config = Config(
        pion_basis=run_cfg['pion_basis'],
        walk_mode=run_cfg['walk_mode'],
        cutoff_method=run_cfg['cutoff_method'],
        boson_cutoff_method=run_cfg['boson_cutoff_method'],
        block_encoder=run_cfg['block_encoder'],
        extras=run_cfg['extras'],
    )

    print("========================================================")
    print(f" INITIATING NUCLEON SWEEP (basis={config.pion_basis}, "
          f"cutoff={config.cutoff_method}, encoder={config.block_encoder}, "
          f"walk_mode={config.walk_mode})")
    print(f" A values = {list(run_cfg['A_values'])}")
    print("========================================================")

    params = get_physical_parameters()
    dim = run_cfg['dim']
    A_values = run_cfg['A_values']

    sweep_results = []
    for A in A_values:
        L = run_cfg['L'] if run_cfg['L'] is not None else get_L_for_A(A)
        print(f"\n{'-'*50}\n Starting Simulation for A = {A} (L={L} in {dim}D)\n{'-'*50}")

        try:
            n_b, pi_max, Pi_max = _compute_cutoffs(L, dim, A, params, run_cfg, config)

            start_time = time.time()
            norm_data = evaluate_resources(L, dim, n_b, pi_max, params, config)
            duration_seconds = time.time() - start_time

            if 'Total_T_Count' in norm_data:
                # Compose the result entry. Per-sub-Hamiltonian breakouts are
                # stored as a list so plotting can read them regardless of
                # how many walks the basis produced (1 for Fock, 2 for
                # amplitude, N for future hybrids).
                result_entry = {
                    'A': int(A),
                    'L': L,
                    'dim': dim,
                    'n_b': n_b,
                    'pi_max': float(pi_max) if pi_max == pi_max else None,
                    'Pi_max': float(Pi_max) if Pi_max == Pi_max else None,
                    'Runtime_Seconds': round(duration_seconds, 2),
                    'Physical_Lambda': norm_data['Physical_Lambda'],
                    'Logical_Qubits': norm_data['Logical_Qubits'],
                    'Walk_Clifford_Count': norm_data['Walk_Clifford_Count'],
                    'Walk_T_Count': norm_data['Walk_T_Count'],
                    'QFT_T_Count': norm_data['QFT_T_Count'],
                    'Total_T_Count': norm_data['Total_T_Count'],
                    'Per_Sub_Walk': norm_data.get('Per_Sub_Walk', []),
                }
                sweep_results.append(result_entry)
                print(f"Iteration completed in {duration_seconds:.2f} seconds.")
            else:
                print(f"Warning: T-count not found in norm_data for A={A}. Data not recorded.")

        except Exception as e:
            print(f"FAILED for A={A}. Error: {e}")
            import traceback; traceback.print_exc()
            continue

    if sweep_results:
        # Save using the first iteration's L (sweeps with L=L(A) are reflected
        # per-entry; metadata captures the *file-naming* L).
        first_L = sweep_results[0]['L']
        saved_filepath = save_sweep_data(first_L, dim, params, sweep_results,
                                         config=config, label=run_cfg['label'])

        # Post-process: compute and save the total QPE cost
        # (QPE_Total_T_Count = Total_T_Count · √2·π·Λ/ΔE) into the same file,
        # so downstream plotting reads it instead of recomputing. The raw
        # sweep is already safely on disk (above), so guard this step: a
        # failure here must not mask an otherwise-successful (expensive)
        # sweep — log and return the saved path regardless. The cost can be
        # recomputed later via `python -m src_PI.estimation.qpe_cost`.
        delta_E = run_cfg['delta_E_MeV']
        try:
            compute_total_qpe_cost(saved_filepath, delta_E=delta_E)
            print(f"[qpe_cost] Saved total QPE cost (ΔE={delta_E} MeV) into {saved_filepath}")
        except Exception as e:
            print(f"[qpe_cost] WARNING: total-QPE-cost post-process failed ({e}); "
                  f"sweep data is saved — recompute with "
                  f"`python -m src_PI.estimation.qpe_cost {saved_filepath}`")

        print(f"\nSweep completed successfully. Plot via plot_sweep_data.py")
        print(f"Saved data file: {saved_filepath}")
        return saved_filepath
    else:
        print("\nSweep finished, but no data was successfully recorded.")
        return None


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Run the nucleon resource-estimation sweep.")
    ap.add_argument('--label', default=None,
                    help="Optional short tag folded into the saved run-id "
                         "folder (e.g. 'paperfig', 'L4-highA').")
    args = ap.parse_args()
    run_sweep(label=args.label)
