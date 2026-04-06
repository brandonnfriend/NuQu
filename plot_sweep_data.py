import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from datetime import datetime
from src_PI.trotter_theory.TrotterCost import get_total_trotter_cost


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

def plot_total_tcost_comparison(filepath, delta_E=1.0, e=0.1, E_kin=10, Cp=1e-3):
    """
    Loads pyLIQTR Qubitization sweep data (JSON), computes Trotterization T-costs 
    over a range of A, plots the comparison, and saves to a date-stamped folder.
    
    Args:
        filepath (str): Path to the saved Qubitization JSON data.
        delta_E (float): Desired energy accuracy for QPE in MeV. Defaults to 1.0.
        e (float): Trotter error tolerance. Defaults to 0.1.
        E_kin (float): Kinetic energy parameter. 
        Cp (float): Trotter simulation time parameter.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    # 1. Load the Qubitization Data
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    metadata = data['metadata']
    results = data['results']
    
    L = metadata.get('L', results[0].get('L'))
    dim = metadata.get('dim', 3)
    
    # --- 2. Process Qubitization (QPE) Data ---
    qpe_A_vals = []
    qpe_total_costs = []
    qpe_save_data = []

    for r_entry in results:
        A = r_entry['A']
        t_step_cost = r_entry['Total_T_Count']
        lam = r_entry['Physical_Lambda']
        
        # Total T-cost = Total_T_Count * (sqrt(2*pi) * Physical_Lambda / delta_E)
        qpe_walk_queries = (np.sqrt(2) * np.pi * lam) / delta_E
        total_qpe = t_step_cost * qpe_walk_queries
        
        qpe_A_vals.append(A)
        qpe_total_costs.append(total_qpe)
        
        qpe_save_data.append({
            "A": A,
            "Physical_Lambda": lam,
            "QPE_Step_T_Count": t_step_cost,
            "QPE_Total_T_Count": total_qpe
        })

    # --- 3. Compute Trotterization Data (A from 1 to 100) ---
    trotter_A_vals = np.arange(1, 101, 1)
    trotter_total_costs = []
    trotter_save_data = []
    
    for A in trotter_A_vals:
        # We leave E_bound as None so the function calculates it as E_kin * A
        total_trotter = get_total_trotter_cost(
            A=A, L=L, e=e, E_kin=E_kin, E_bound=None, Cp=Cp, dim=dim
        )
        trotter_total_costs.append(total_trotter)
        
        trotter_save_data.append({
            "A": int(A),
            "Trotter_Total_T_Count": total_trotter
        })

    # --- 4. Setup Directories & Save JSON ---
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("data", date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    base_filename = f"total_costs_L{L}_{dim}D"
    json_save_path = os.path.join(save_dir, f"{base_filename}.json")
    plot_save_path = os.path.join(save_dir, f"{base_filename}_plot.png")

    output_data_package = {
        "metadata": {
            "L": L,
            "dim": dim,
            "delta_E": delta_E,
            "epsilon_trotter": e,
            "E_kin": E_kin,
            "Cp": Cp,
            "source_file": filepath,
            "timestamp": datetime.now().isoformat()
        },
        "qubitization_results": qpe_save_data,
        "trotter_results": trotter_save_data
    }
    
    with open(json_save_path, 'w') as f:
        json.dump(output_data_package, f, indent=4)
    print(f"Data saved successfully to: {json_save_path}")

# --- 5. Create and Save the Plot ---
    plt.figure(figsize=(10, 7))
    
    # Plot Trotterization as a smooth line first (background)
    plt.plot(trotter_A_vals, trotter_total_costs, color='red', linewidth=2, linestyle='--', 
             label=f'Trotterization ($\epsilon={e}$)')
             
    # Plot Qubitization as discrete markers (foreground)
    plt.plot(qpe_A_vals, qpe_total_costs, marker='o', color='blue', linewidth=2.5, markersize=8,
             label=f'Qubitization (QPE $\\Delta E={delta_E}$ MeV)')
    
    plt.title(f"Total T-Gate Resource Cost: Qubitization vs. Trotterization (L={L}, {dim}D)", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Nucleon Number (A)", fontsize=14)
    plt.ylabel("Total T-Gates", fontsize=14)
    
    # Set both axes to logarithmic scale for a log-log plot
    plt.xscale('log')
    plt.yscale('log') 
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved successfully to: {plot_save_path}")
    
    plt.show()

if __name__ == "__main__":
    # You can change this path to point to whichever JSON file you want to plot!
    # Example: target_file = "data/2026-04-04/sweep_L2_3D_143000.json"
    
    target_file = "data/2026-04-04/sweep_L3_3D_203524.json" # <-- UPDATE THIS PATH TO YOUR JSON FILE
    
    if target_file == "INSERT_FILEPATH_HERE.json":
        print("Please update 'target_file' at the bottom of the script with the path to your JSON data.")
    else:
        plot_total_tcost_comparison(target_file)