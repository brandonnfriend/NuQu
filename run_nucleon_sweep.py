import time

import numpy as np

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    calculate_fock_cutoff,
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
        'epsilon_cut': 0.1,
        'E_bound_per_A_MeV': 10.0,       # E_max = E_bound_per_A_MeV * A
        # Optional override: if set, used instead of the basis-specific
        # cutoff calculation. Useful for smoke tests where you want a
        # tiny register, and for direct-comparison runs where you want
        # both bases at the same n_b.
        'n_b_override': None,
        'extras': {},
    }
    defaults.update(overrides)
    return defaults


def _compute_cutoffs(L, dim, A, params, run_cfg, config):
    """Dispatch cutoff calculation by basis. Returns (n_b, pi_max, Pi_max).

    Both basis branches return the same 3-tuple shape so the caller doesn't
    need to know which one ran. In the Fock branch, pi_max/Pi_max are
    diagnostic-only (computed via the amplitude formula for comparison)
    and don't drive any operator construction.
    """
    E_max = run_cfg['E_bound_per_A_MeV'] * A
    eps = run_cfg['epsilon_cut']
    if config.pion_basis == 'amplitude':
        n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(
            L, dim, A, params, epsilon_cut=eps, E_bound=E_max
        )
    else:
        n_b, pi_max, Pi_max = calculate_fock_cutoff(
            L, dim, A, params, epsilon_cut=eps, E_bound=E_max
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
        extras=run_cfg['extras'],
    )

    print("========================================================")
    print(f" INITIATING NUCLEON SWEEP (basis={config.pion_basis}, "
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
        saved_filepath = save_sweep_data(first_L, dim, params, sweep_results, config=config)
        print(f"\nSweep completed successfully. Plot via plot_sweep_data.py")
        print(f"Saved data file: {saved_filepath}")
        return saved_filepath
    else:
        print("\nSweep finished, but no data was successfully recorded.")
        return None


if __name__ == '__main__':
    run_sweep()
