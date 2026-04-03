import numpy as np
import matplotlib.pyplot as plt
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs
from src_PI.estimation.EstimateResources import evaluate_resources

def get_A_sweep_values():
    """Generates 15 values for A: dense from 1-10, spreading out up to 50."""
    A_dense = np.arange(1, 11)                  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    A_sparse = np.linspace(20, 30, 5)           # [20, 27.5, 35, 42.5, 50]
    A_values = np.concatenate([A_dense, np.round(A_sparse)]).astype(int)
    return np.unique(A_values)

def get_L_for_A(A):
    """
    Hook to dynamically change L based on A. 
    Currently returns a fixed L, but perfectly set up for future L(A) functions.
    """
    # WARNING: Keep this at 2 for local testing! Crank to 10 for paper-grade runs.
    return 2 

def run_sweep():
    print("========================================================")
    print(" INITIATING NUCLEON SWEEP (A=1 to 50)")
    print("========================================================")
    
    params = get_physical_parameters()
    A_values = get_A_sweep_values()
    dim = 3  # Forcing 3D simulation
    
    t_counts = []
    actual_A_plotted = []
    
    for A in A_values:
        L = get_L_for_A(A)
        print(f"\n{'-'*50}\n Starting Simulation for A = {A} (L={L} in {dim}D)\n{'-'*50}")
        
        try:
            # 1. Calculate dynamic bounds
            n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(
                L, dim, A, params, epsilon_cut=0.1, E_bound=140.0
            )
            
            # 2. Run resource estimation
            norm_data = evaluate_resources(L, dim, n_b, pi_max, params)
            
            # 3. Extract Total T-count (Ensure EstimateResources.py is passing this!)
            if 'total_t_count' in norm_data and norm_data['total_t_count'] > 0:
                t_counts.append(norm_data['total_t_count'])
                actual_A_plotted.append(A)
            else:
                print(f"Warning: T-count not found in norm_data for A={A}. Did pyLIQTR return it?")
                
        except Exception as e:
            print(f"FAILED for A={A}. Error: {e}")
            continue

    # 4. Generate Plot
    if t_counts:
        plt.figure(figsize=(10, 6))
        plt.plot(actual_A_plotted, t_counts, marker='o', linestyle='-', color='#d62728', linewidth=2)
        plt.title('T-Gate Cost vs. Nucleon Number (A) for 3D Dynamical Pion EFT', fontsize=14)
        plt.xlabel('Number of Nucleons (A)', fontsize=12)
        plt.ylabel('T-Gates per Trotter Step', fontsize=12)
        
        # Using log scale because T-counts scale rapidly with A
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.6)
        
        plt.tight_layout()
        plot_filename = 'T_count_vs_A_Sweep.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"\n========================================================")
        print(f" Sweep Complete! Plot saved successfully as '{plot_filename}'")
        print(f"========================================================")
    else:
        print("\nSweep finished, but no T-counts were successfully recorded to plot.")

if __name__ == '__main__':
    run_sweep()