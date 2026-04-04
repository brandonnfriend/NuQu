import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

def load_and_plot(filepath):
    """Loads sweep JSON data and plots T-counts and Lambda vs Nucleon Number (A)."""
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    # 1. Load Data
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    metadata = data['metadata']
    results = data['results']
    
    # 2. Extract Arrays
    A_vals = [r['A'] for r in results]
    L_vals = [r.get('L', metadata.get('L')) for r in results]
    
    # Hardware Stats
    total_T = [r['Total_T_Count'] for r in results]
    walk_T = [r['Walk_T_Count'] for r in results]
    qft_T = [r['QFT_T_Count'] for r in results]
    
    # Physics & Performance Stats
    lambdas = [r['Physical_Lambda'] for r in results]
    runtimes = [r['Runtime_Seconds'] for r in results]
    
    # Determine L for the title
    unique_L = np.unique(L_vals)
    if len(unique_L) == 1:
        l_string = f"L={unique_L[0]}"
    else:
        l_string = f"L Range [{min(unique_L)}-{max(unique_L)}]"

 # 3. Create Plots (3 Subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Dynamical Pion EFT Resource Scaling ({l_string}, {metadata['dim']}D)", fontsize=18, fontweight='bold')

    # --- Plot 1: T-Gate Scaling ---
    ax1.plot(A_vals, total_T, marker='o', color='black', linewidth=2.5, label='Total Step Cost')
    ax1.set_title("Total T-Gate Costs per Step", fontsize=14)
    ax1.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax1.set_ylabel("T-Gates", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # --- Plot 2: Physical Lambda ---
    ax2.plot(A_vals, lambdas, marker='D', color='purple', linewidth=2)
    ax2.set_title("Physical $\Lambda$ (Energy Scale)", fontsize=14)
    ax2.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax2.set_ylabel("$\Lambda$ (MeV)", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 3: Classical Runtime ---
    ax3.plot(A_vals, runtimes, marker='o', color='green', linewidth=2)
    ax3.set_title("Estimation Runtime", fontsize=14)
    ax3.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax3.set_ylabel("Wall Clock Time (Seconds)", fontsize=12)
    ax3.grid(True, ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for the main title
    
    # Save the plot
    plot_filename = filepath.replace('.json', '_plot.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully to: {plot_filename}")
    
    plt.show()

if __name__ == "__main__":
    # You can change this path to point to whichever JSON file you want to plot!
    # Example: target_file = "data/2026-04-04/sweep_L2_3D_143000.json"
    
    target_file = "data/2026-04-04/sweep_L2_3D_172410.json" # <-- UPDATE THIS PATH TO YOUR JSON FILE
    
    if target_file == "INSERT_FILEPATH_HERE.json":
        print("Please update 'target_file' at the bottom of the script with the path to your JSON data.")
    else:
        load_and_plot(target_file)