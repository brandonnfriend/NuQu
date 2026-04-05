import time
import numpy as np
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs
from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.utils.DataIO import save_sweep_data

def get_A_sweep_values():
    """Generates values for A: dense from 1-10, spreading out up to 100."""
    A_dense = np.arange(1, 11)                  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    A_sparse = np.linspace(20, 100, 15)           # 15 values from 20 to 100, spaced out
    A_values = np.concatenate([A_dense, np.round(A_sparse)]).astype(int)
    return np.unique(A_values)

def get_L_for_A(A):
    """
    Hook to dynamically change L based on A. 
    Currently returns a fixed L, but perfectly set up for future L(A) functions.
    """
    # WARNING: Keep this at 2 for local testing! Crank to 10 for paper-grade HPC runs.
    return 3

def run_sweep():
    print("========================================================")
    print(" INITIATING NUCLEON SWEEP (A=1 to 100)")
    print("========================================================")
    
    params = get_physical_parameters()
    A_values = get_A_sweep_values()
    dim = 3  # Forcing 3D simulation
    
    sweep_results = []
    
    for A in A_values:
        L = get_L_for_A(A)
        print(f"\n{'-'*50}\n Starting Simulation for A = {A} (L={L} in {dim}D)\n{'-'*50}")
        E_max = A*10 # Rough estimate of max energy based on nucleon count (10 MeV per nucleon)

        try:
            # 1. Calculate dynamic bounds
            n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(
                L, dim, A, params, epsilon_cut=0.1, E_bound=E_max
            )
            
            # --- START TIMING ---
            start_time = time.time()

            # 2. Run resource estimation
            norm_data = evaluate_resources(L, dim, n_b, pi_max, params)
            
            end_time = time.time()
            duration_seconds = end_time - start_time
            # --- END TIMING ---

            # 3. Package the data for this iteration
            if 'Total_T_Count' in norm_data:
                result_entry = {
                    'A': int(A),
                    'L': L,
                    'dim': dim,
                    'n_b': n_b,
                    'Runtime_Seconds': round(duration_seconds, 2),
                    'Physical_Lambda': norm_data['Physical_Lambda'],
                    'Logical_Qubits_Per_Walk': norm_data['Logical_Qubits_Per_Walk'],
                    'Walk_Clifford_Count': norm_data['Walk_Clifford_Count'],
                    'Walk_T_Count': norm_data['Walk_T_Count'],
                    'QFT_T_Count': norm_data['QFT_T_Count'],
                    'Total_T_Count': norm_data['Total_T_Count']
                }
                sweep_results.append(result_entry)
                print(f"Iteration completed in {duration_seconds:.2f} seconds.")
            else:
                print(f"Warning: T-count not found in norm_data for A={A}. Data not recorded.")
                
        except Exception as e:
            print(f"FAILED for A={A}. Error: {e}")
            continue

    # 4. Save the gathered data to a JSON file
    if sweep_results:
        saved_filepath = save_sweep_data(L, dim, params, sweep_results)
        print(f"\nSweep completed successfully. You can now plot this data using plot_sweep_data.py")
        print(f"Saved data file: {saved_filepath}")
    else:
        print("\nSweep finished, but no data was successfully recorded.")

if __name__ == '__main__':
    run_sweep()