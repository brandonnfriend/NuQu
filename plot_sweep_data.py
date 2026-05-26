import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from datetime import datetime
from src_PI.trotter_theory.TrotterCost import get_total_trotter_cost


# Standard color per basis for overlay plots.
_BASIS_COLORS = {
    'amplitude': 'tab:blue',
    'fock': 'tab:orange',
}

# Color per "series" — basis plus, for the amplitude basis, the cutoff method.
# This keeps energy_bound vs NS amplitude runs visually distinct in overlays.
_SERIES_COLORS = {
    'amplitude/energy_bound': 'tab:blue',
    'amplitude/ns': 'tab:green',
    'fock': 'tab:orange',
}


def _get_basis_label(metadata):
    """Extract pion_basis from metadata['config'] if present; default 'amplitude'.

    Old sweep files predate the Config object and have no basis field. We treat
    them as amplitude basis for backward compatibility (which is what they were).
    """
    cfg = metadata.get('config') or {}
    return cfg.get('pion_basis', 'amplitude')


def _get_cutoff_label(metadata):
    """Extract cutoff_method from metadata['config']; default 'energy_bound'.

    Old files predate the cutoff_method axis; they used the Watson Lemma 5
    energy-bound cutoff, so that is the backward-compatible default.
    """
    cfg = metadata.get('config') or {}
    return cfg.get('cutoff_method', 'energy_bound')


def _get_series_key(metadata):
    """Composite series id: basis, plus cutoff method for the amplitude basis.

    e.g. 'amplitude/energy_bound', 'amplitude/ns', 'fock'. Lets a single
    comparison overlay distinguish the two amplitude cutoff prescriptions.
    """
    basis = _get_basis_label(metadata)
    if basis == 'amplitude':
        return f"amplitude/{_get_cutoff_label(metadata)}"
    return basis


def _series_color(series_key):
    """Color for a series key, falling back to basis color then black."""
    if series_key in _SERIES_COLORS:
        return _SERIES_COLORS[series_key]
    return _BASIS_COLORS.get(series_key.split('/')[0], 'black')


def load_and_plot(filepath):
    """Loads sweep JSON data and plots T-counts, Lambda, qubits, runtime vs A."""

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    # 1. Load Data
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    results = data['results']
    basis = _get_basis_label(metadata)
    series = _get_series_key(metadata)

    # 2. Extract Arrays
    A_vals = [r['A'] for r in results]
    L_vals = [r.get('L', metadata.get('L')) for r in results]

    total_T = [r['Total_T_Count'] for r in results]
    walk_T = [r.get('Walk_T_Count', 0) for r in results]
    qft_T = [r.get('QFT_T_Count', 0) for r in results]

    lambdas = [r['Physical_Lambda'] for r in results]
    runtimes = [r.get('Runtime_Seconds', 0) for r in results]
    qubits = [r.get('Logical_Qubits', 0) for r in results]

    # Title bits
    unique_L = np.unique(L_vals)
    if len(unique_L) == 1:
        l_string = f"L={unique_L[0]}"
    else:
        l_string = f"L Range [{min(unique_L)}-{max(unique_L)}]"

    # 3. Create Plots (4 subplots — added Logical Qubits)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    ax1, ax2, ax3, ax4 = axes
    fig.suptitle(
        f"Dynamical Pion EFT Resource Scaling ({l_string}, {metadata['dim']}D, "
        f"basis={series})",
        fontsize=18, fontweight='bold',
    )
    color = _series_color(series)

    # --- Plot 1: T-Gate Scaling ---
    ax1.plot(A_vals, total_T, marker='o', color=color, linewidth=2.5, label='Total Step Cost')
    if any(qft_T):
        ax1.plot(A_vals, qft_T, marker='s', color=color, linewidth=1.5,
                 linestyle='--', alpha=0.6, label='QFT step cost')
    ax1.set_title("Total T-Gate Costs per Step", fontsize=14)
    ax1.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax1.set_ylabel("T-Gates", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # --- Plot 2: Physical Lambda ---
    ax2.plot(A_vals, lambdas, marker='D', color=color, linewidth=2)
    ax2.set_title("Physical $\\Lambda$ (Energy Scale)", fontsize=14)
    ax2.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax2.set_ylabel("$\\Lambda$ (MeV)", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 3: Logical Qubits ---
    ax3.plot(A_vals, qubits, marker='^', color=color, linewidth=2)
    ax3.set_title("Logical Qubits (peak)", fontsize=14)
    ax3.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax3.set_ylabel("Logical Qubits", fontsize=12)
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 4: Classical Runtime ---
    ax4.plot(A_vals, runtimes, marker='o', color=color, linewidth=2)
    ax4.set_title("Estimation Runtime", fontsize=14)
    ax4.set_xlabel("Nucleon Number (A)", fontsize=12)
    ax4.set_ylabel("Wall Clock Time (Seconds)", fontsize=12)
    ax4.grid(True, ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = filepath.replace('.json', '_plot.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully to: {plot_filename}")

    plt.show()


def plot_basis_comparison(filepaths, save_dir=None, save_basename=None):
    """Overlay multiple sweeps for direct A-vs-B (basis) comparison.

    Args:
        filepaths: list of sweep JSON files. Each file's metadata['config']
            gives its basis label; overlapping bases are drawn together.
        save_dir: directory to save the comparison plot to. Defaults to
            data/<today>/.
        save_basename: basename for the output PNG (no extension). Defaults
            to 'basis_comparison_L{L}_{dim}D'.

    Side effect: writes a PNG with four subplots (T-count, Λ, qubits, runtime)
    overlaying every input sweep, color-coded by basis.

    Returns the save path.
    """
    if not filepaths:
        print("plot_basis_comparison: no files passed.")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    ax_T, ax_L, ax_Q, ax_t = axes

    L_for_title = None
    dim_for_title = None

    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"plot_basis_comparison: skipping missing file {fp}")
            continue
        with open(fp) as f:
            data = json.load(f)
        meta = data['metadata']
        results = data['results']
        series = _get_series_key(meta)
        color = _series_color(series)

        if L_for_title is None:
            L_for_title = meta.get('L', results[0].get('L'))
            dim_for_title = meta.get('dim', 3)

        A_vals = [r['A'] for r in results]
        T_vals = [r['Total_T_Count'] for r in results]
        Lam_vals = [r['Physical_Lambda'] for r in results]
        Q_vals = [r.get('Logical_Qubits', 0) for r in results]
        rt_vals = [r.get('Runtime_Seconds', 0) for r in results]

        n_b_label = ', '.join(sorted({str(r['n_b']) for r in results}))
        label = f"{series}  (n_b={n_b_label})"

        ax_T.plot(A_vals, T_vals, marker='o', color=color, linewidth=2, label=label)
        ax_L.plot(A_vals, Lam_vals, marker='D', color=color, linewidth=2, label=label)
        ax_Q.plot(A_vals, Q_vals, marker='^', color=color, linewidth=2, label=label)
        ax_t.plot(A_vals, rt_vals, marker='s', color=color, linewidth=2, label=label)

    ax_T.set_title("Total T-Gate Cost per Step"); ax_T.set_yscale('log')
    ax_L.set_title("Physical $\\Lambda$ (MeV)");    ax_L.set_yscale('log')
    ax_Q.set_title("Logical Qubits (peak)")
    ax_t.set_title("Estimation Runtime (s)")

    for ax in axes:
        ax.set_xlabel("Nucleon Number (A)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=10)
    ax_T.set_ylabel("T-Gates")
    ax_L.set_ylabel("$\\Lambda$ (MeV)")
    ax_Q.set_ylabel("Logical Qubits")
    ax_t.set_ylabel("Wall-clock (s)")

    fig.suptitle(
        f"Basis Comparison: Dynamical Pion EFT Resources "
        f"(L={L_for_title}, {dim_for_title}D)",
        fontsize=18, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir is None:
        save_dir = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(save_dir, exist_ok=True)
    if save_basename is None:
        save_basename = f"basis_comparison_L{L_for_title}_{dim_for_title}D"
    out_path = os.path.join(save_dir, f"{save_basename}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved successfully to: {out_path}")
    plt.show()
    return out_path

def plot_basis_comparison_total_qpe(filepaths, delta_E=1.0, e=0.1, E_kin=10, Cp=1e-3,
                                     trotter_A_range=(1, 100), save_dir=None,
                                     save_basename=None):
    """Overlay total QPE T-cost across bases, plus a Trotterization baseline.

    For each input sweep, computes
        T_total(A) = T_step(A) × (sqrt(2)·π · Λ(A) / ΔE)
    (per-walk-step T-count times the QPE walk-query count) and overlays the
    curves color-coded by basis. A Trotter baseline is drawn at the same
    (L, dim) over `trotter_A_range`, using the first input file's geometry.

    Args:
        filepaths: list of sweep JSONs, one per basis to overlay.
        delta_E: QPE energy precision target in MeV.
        e: Trotter error tolerance.
        E_kin, Cp: Trotter cost parameters (forwarded to get_total_trotter_cost).
        trotter_A_range: (lo, hi) inclusive range for the Trotter curve.
        save_dir: output directory (default data/<today>/).
        save_basename: PNG basename without extension (default
            'basis_comparison_total_qpe_L{L}_{dim}D').

    Returns the saved PNG path. Also writes a parallel JSON with the
    computed cost arrays for reproducibility.
    """
    if not filepaths:
        print("plot_basis_comparison_total_qpe: no files passed.")
        return None

    plt.figure(figsize=(11, 7.5))
    L_for_title = None
    dim_for_title = None

    save_pkg = {
        "metadata": {
            "delta_E": delta_E,
            "epsilon_trotter": e,
            "E_kin": E_kin,
            "Cp": Cp,
            "trotter_A_range": list(trotter_A_range),
            "source_files": filepaths,
            "timestamp": datetime.now().isoformat(),
        },
        "qubitization_by_basis": {},
    }

    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"plot_basis_comparison_total_qpe: skipping missing file {fp}")
            continue
        with open(fp) as f:
            data = json.load(f)
        meta = data['metadata']
        results = data['results']
        series = _get_series_key(meta)
        color = _series_color(series)

        if L_for_title is None:
            L_for_title = meta.get('L', results[0].get('L'))
            dim_for_title = meta.get('dim', 3)

        A_vals, total_qpe, per_A_records = [], [], []
        for r in results:
            A = r['A']
            t_step = r['Total_T_Count']
            lam = r['Physical_Lambda']
            queries = (np.sqrt(2) * np.pi * lam) / delta_E
            total = t_step * queries
            A_vals.append(A)
            total_qpe.append(total)
            per_A_records.append({
                "A": int(A),
                "Physical_Lambda": float(lam),
                "QPE_Step_T_Count": int(t_step),
                "Walk_Queries": float(queries),
                "QPE_Total_T_Count": float(total),
            })

        n_b_label = ', '.join(sorted({str(r['n_b']) for r in results}))
        label = f"Qubitization {series} (n_b={n_b_label}, $\\Delta E$={delta_E} MeV)"

        plt.plot(A_vals, total_qpe, marker='o', color=color, linewidth=2.5,
                 markersize=7, label=label)
        save_pkg["qubitization_by_basis"][series] = per_A_records

    # Trotter baseline at (L, dim) from the first valid file.
    lo, hi = trotter_A_range
    trotter_A_vals = np.arange(lo, hi + 1)
    trotter_costs = [
        get_total_trotter_cost(A=int(A), L=L_for_title, e=e, E_kin=E_kin,
                               E_bound=None, Cp=Cp, dim=dim_for_title)
        for A in trotter_A_vals
    ]
    plt.plot(trotter_A_vals, trotter_costs, color='tab:red', linewidth=2,
             linestyle='--',
             label=f"Trotterization ($\\epsilon$={e}, L={L_for_title}, "
                   f"{dim_for_title}D)")
    save_pkg["trotter"] = [
        {"A": int(A), "Trotter_Total_T_Count": float(c)}
        for A, c in zip(trotter_A_vals, trotter_costs)
    ]

    plt.title(
        f"Total QPE T-Cost vs A: Basis Comparison + Trotter "
        f"(L={L_for_title}, {dim_for_title}D)",
        fontsize=15, fontweight='bold',
    )
    plt.xlabel("Nucleon Number (A)", fontsize=13)
    plt.ylabel("Total T-Gates", fontsize=13)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(save_dir, exist_ok=True)
    if save_basename is None:
        save_basename = f"basis_comparison_total_qpe_L{L_for_title}_{dim_for_title}D"
    plot_path = os.path.join(save_dir, f"{save_basename}.png")
    json_path = os.path.join(save_dir, f"{save_basename}.json")

    plt.savefig(plot_path, dpi=300)
    with open(json_path, 'w') as f:
        json.dump(save_pkg, f, indent=4)
    print(f"Plot saved to: {plot_path}")
    print(f"Data saved to: {json_path}")
    plt.show()
    return plot_path


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

def plot_multi_L_total_tcost_comparison(filepaths, delta_E=1.0, e=0.1, E_kin=10, Cp=1e-3):
    """
    Loads pyLIQTR Qubitization sweep data (JSON) from multiple files, computes Trotterization 
    T-costs over a range of A for each L, plots the comparison on a log-log scale, 
    and saves to a date-stamped folder.
    
    Args:
        filepaths (list of str): Paths to the saved Qubitization JSON data files.
        delta_E (float): Desired energy accuracy for QPE in MeV. Defaults to 1.0.
        e (float): Trotter error tolerance. Defaults to 0.1.
        E_kin (float): Kinetic energy parameter. 
        Cp (float): Trotter simulation time parameter.
    """
    plt.figure(figsize=(12, 8))
    
    # Distinct colors for different L values (add more if needed)
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'cyan']
    
    all_saved_data = {
        "metadata": {
            "delta_E": delta_E,
            "epsilon_trotter": e,
            "E_kin": E_kin,
            "Cp": Cp,
            "source_files": filepaths,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }
    
    dim_for_title = 3 # Default, will be updated from files if available

    for idx, filepath in enumerate(filepaths):
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found. Skipping...")
            continue

        # 1. Load the Qubitization Data
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        metadata = data['metadata']
        results = data['results']
        
        L = metadata.get('L', results[0].get('L'))
        dim = metadata.get('dim', 3)
        dim_for_title = dim
        
        color = colors[idx % len(colors)]
        
        # --- 2. Process Qubitization (QPE) Data ---
        qpe_A_vals = []
        qpe_total_costs = []
        qpe_save_data = []

        for r_entry in results:
            A = r_entry['A']
            t_step_cost = r_entry['Total_T_Count']
            lam = r_entry['Physical_Lambda']
            
            # Total T-cost = Total_T_Count * (sqrt(2) * pi * Physical_Lambda / delta_E)
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
            total_trotter = get_total_trotter_cost(
                A=A, L=L, e=e, E_kin=E_kin, E_bound=None, Cp=Cp, dim=dim
            )
            trotter_total_costs.append(total_trotter)
            
            trotter_save_data.append({
                "A": int(A),
                "Trotter_Total_T_Count": total_trotter
            })
            
        # Store data for JSON export
        all_saved_data["results"][f"L_{L}"] = {
            "qubitization": qpe_save_data,
            "trotter": trotter_save_data
        }

        # --- 4. Plot this L's data ---
        # Plot Trotterization (dashed line, background)
        plt.plot(trotter_A_vals, trotter_total_costs, color=color, linewidth=2, linestyle='--', 
                 label=f'Trotter ($\epsilon={e}$, L={L})')
                 
        # Plot Qubitization (solid line with markers, foreground)
        plt.plot(qpe_A_vals, qpe_total_costs, marker='o', color=color, linewidth=2.5, markersize=6,
                 label=f'Qubitization (QPE $\\Delta E={delta_E}$, L={L})')

    # --- 5. Setup Directories & Save JSON ---
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("data", date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    base_filename = f"multi_L_total_costs_{dim_for_title}D"
    json_save_path = os.path.join(save_dir, f"{base_filename}.json")
    plot_save_path = os.path.join(save_dir, f"{base_filename}_plot.png")
    
    with open(json_save_path, 'w') as f:
        json.dump(all_saved_data, f, indent=4)
    print(f"Data saved successfully to: {json_save_path}")

    # --- 6. Format and Save the Plot ---
    plt.title(f"Total T-Gate Resource Cost: Qubitization vs. Trotterization ({dim_for_title}D)", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Nucleon Number (A)", fontsize=14)
    plt.ylabel("Total T-Gates", fontsize=14)
    
    plt.xscale('log')
    plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Put legend outside the plot if it gets too crowded, or keep it inside
    plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved successfully to: {plot_save_path}")
    
    plt.show()

def plot_tcost_vs_L_for_chosen_A(filepaths, target_A_vals=[1, 10, 60, 100], delta_E=1.0, e=0.1, E_kin=10, Cp=1e-3):
    """
    Loads pyLIQTR Qubitization sweep data from multiple files, extracts data for 
    specific values of A, computes corresponding Trotterization T-costs, and 
    plots Total T-cost vs L.
    
    Args:
        filepaths (list of str): Paths to the saved Qubitization JSON data files (varying L).
        target_A_vals (list of int): Specific nucleon numbers to plot curves for.
        delta_E (float): Desired energy accuracy for QPE in MeV.
        e (float): Trotter error tolerance.
        E_kin (float): Kinetic energy parameter. 
        Cp (float): Trotter simulation time parameter.
    """
    # 1. Initialize Data Structures grouped by A
    # format: { A: [(L, qpe_cost, lam, t_step_cost), ...] }
    qpe_data_by_A = {A: [] for A in target_A_vals}
    trotter_data_by_A = {A: [] for A in target_A_vals}
    
    dim_for_title = 3 
    
    # 2. Extract Data from Files
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found. Skipping...")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)
            
        metadata = data['metadata']
        results = data['results']
        
        L = metadata.get('L', results[0].get('L'))
        dim = metadata.get('dim', 3)
        dim_for_title = dim
        
        # Look for the target A values in this L's file
        for r_entry in results:
            A = r_entry['A']
            if A in target_A_vals:
                t_step_cost = r_entry['Total_T_Count']
                lam = r_entry['Physical_Lambda']
                
                # Total QPE T-cost = Total_T_Count * (sqrt(2*pi) * Physical_Lambda / delta_E)
                qpe_walk_queries = (np.sqrt(2) * np.pi * lam) / delta_E
                total_qpe = t_step_cost * qpe_walk_queries
                
                qpe_data_by_A[A].append((L, total_qpe, lam, t_step_cost))
                
                # Compute Trotterization Data for this exact A and L
                total_trotter = get_total_trotter_cost(
                    A=A, L=L, e=e, E_kin=E_kin, E_bound=None, Cp=Cp, dim=dim
                )
                trotter_data_by_A[A].append((L, total_trotter))

    # 3. Setup JSON Export Package
    all_saved_data = {
        "metadata": {
            "delta_E": delta_E,
            "epsilon_trotter": e,
            "E_kin": E_kin,
            "Cp": Cp,
            "source_files": filepaths,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }

    # 4. Sort by L and Setup Plotting
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'cyan']
    
    for idx, A in enumerate(target_A_vals):
        # Skip if we didn't find data for this A
        if not qpe_data_by_A[A]:
            continue
            
        # Sort the (L, cost) tuples by L to ensure lines draw left-to-right
        qpe_data_by_A[A].sort(key=lambda x: x[0])
        trotter_data_by_A[A].sort(key=lambda x: x[0])
        
        # Unpack sorted arrays
        L_vals_qpe = [item[0] for item in qpe_data_by_A[A]]
        costs_qpe = [item[1] for item in qpe_data_by_A[A]]
        
        L_vals_trotter = [item[0] for item in trotter_data_by_A[A]]
        costs_trotter = [item[1] for item in trotter_data_by_A[A]]
        
        color = colors[idx % len(colors)]
        
        # Plot Trotterization (dashed line, background)
        plt.plot(L_vals_trotter, costs_trotter, color=color, linewidth=2, linestyle='--', 
                 label=f'Trotter ($\epsilon={e}$, A={A})')
                 
        # Plot Qubitization (solid line with markers, foreground)
        plt.plot(L_vals_qpe, costs_qpe, marker='o', color=color, linewidth=2.5, markersize=8,
                 label=f'Qubitization (QPE $\\Delta E={delta_E}$, A={A})')
                 
        # Save to JSON structure
        all_saved_data["results"][f"A_{A}"] = {
            "qubitization": [{"L": L, "Total_T_Count": c} for L, c in zip(L_vals_qpe, costs_qpe)],
            "trotter": [{"L": L, "Total_T_Count": c} for L, c in zip(L_vals_trotter, costs_trotter)]
        }

    # 5. Export JSON and Save Plot
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("data", date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    base_filename = f"tcost_vs_L_comparison_{dim_for_title}D"
    json_save_path = os.path.join(save_dir, f"{base_filename}.json")
    plot_save_path = os.path.join(save_dir, f"{base_filename}_plot.png")
    
    with open(json_save_path, 'w') as f:
        json.dump(all_saved_data, f, indent=4)
    print(f"Data saved successfully to: {json_save_path}")

    # 6. Formatting
    plt.title(f"Total T-Gate Resource Cost vs Lattice Size ({dim_for_title}D)", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Lattice Spatial Extent (L)", fontsize=14)
    plt.ylabel("Total T-Gates", fontsize=14)
    
    plt.xscale('log')
    plt.yscale('log') 
    
    # Since L values are likely integers (e.g., 2, 3, 4), force integer ticks on the X-axis
    all_L = list(set([L for A in target_A_vals for L, _, _, _ in qpe_data_by_A[A]]))
    if all_L:
        plt.xticks(all_L, all_L)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved successfully to: {plot_save_path}")
    
    plt.show()

if __name__ == "__main__":
    file_list = [
        "data/2026-05-22/sweep_L2_3D_amplitude_165018.json",
        "data/2026-05-22/sweep_L2_3D_fock_165517.json",
    ]
    plot_basis_comparison_total_qpe(file_list)