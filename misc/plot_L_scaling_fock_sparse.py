"""
Plot total QPE T-cost vs L from the fock/sparse L-scaling data, overlaid
against the Watson Trotter resource estimate at the same fixed A.

Reads `data/quantum/2026-06-03/L_scaling_fock_sparse_A10.json` (consolidated driver
output). Writes the plot + a JSON of the curve data into
`data/<today>/L_scaling_fock_sparse_A10_with_trotter.{png,json}`.

Run from project root:
    source .venv/bin/activate
    python misc/plot_L_scaling_fock_sparse.py
"""

import json
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt
import numpy as np

from src_PI.trotter_theory.TrotterCost import get_total_trotter_cost


SRC = 'data/quantum/2026-06-03/L_scaling_fock_sparse_A10.json'
DELTA_E = 1.0   # MeV — matches the post-processing ΔE saved in the consolidated file
TROTTER_E = 0.1
TROTTER_EKIN = 10
TROTTER_CP = 1e-3


def main():
    with open(SRC) as f:
        d = json.load(f)
    meta = d['metadata']
    rows = d['results']
    A = int(meta['A'])
    dim = int(meta['dim'])

    L = np.array([r['L'] for r in rows], dtype=int)
    qpe_total = np.array([r['QPE_Total_T_Count'] for r in rows], dtype=float)
    tstep = np.array([r['Total_T_Count'] for r in rows], dtype=float)
    lam = np.array([r['Physical_Lambda'] for r in rows], dtype=float)
    qubits = np.array([r['Logical_Qubits'] for r in rows], dtype=float)

    # Watson Trotter big-O T-cost at the SAME (L, A) grid.
    trotter = np.array([
        get_total_trotter_cost(A=A, L=int(Li), e=TROTTER_E, E_kin=TROTTER_EKIN,
                               E_bound=None, Cp=TROTTER_CP, dim=dim)
        for Li in L
    ], dtype=float)

    print(f'A = {A}, dim = {dim}D, fock/sparse')
    print()
    print(f"{'L':>3}  {'T_step':>10}  {'Lambda':>10}  {'QPE_total':>12}  {'Trotter':>12}  {'qubits':>7}")
    for i in range(len(L)):
        print(f'{L[i]:>3d}  {tstep[i]:>10.3e}  {lam[i]:>10.3e}  '
              f'{qpe_total[i]:>12.3e}  {trotter[i]:>12.3e}  {int(qubits[i]):>7d}')

    # Plot: 2 panels — total QPE T-cost vs L (with Trotter) and qubits vs L.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1, ax2 = axes

    ax1.loglog(L, trotter, marker='s', color='tab:red', linewidth=2,
               linestyle='--',
               label=f'Trotterization Big-O ($\\epsilon$={TROTTER_E}, $E_{{kin}}$={TROTTER_EKIN})')
    ax1.loglog(L, qpe_total, marker='o', color='tab:orange', linewidth=2.5,
               markersize=8,
               label=f'Qubitization fock/sparse ($\\Delta E$={DELTA_E} MeV)')
    ax1.set_xlabel('Lattice extent L', fontsize=13)
    ax1.set_ylabel('Total T-gates (QPE)', fontsize=13)
    ax1.set_title(f'Total QPE T-Cost vs L  (A={A}, {dim}D)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    ax1.set_xticks(L); ax1.set_xticklabels([str(int(x)) for x in L])
    ax1.legend(fontsize=10, loc='upper left')

    ax2.loglog(L, qubits, marker='^', color='tab:blue', linewidth=2.5,
               markersize=8, label='Logical qubits (peak)')
    ax2.set_xlabel('Lattice extent L', fontsize=13)
    ax2.set_ylabel('Logical qubits', fontsize=13)
    ax2.set_title(f'Logical Qubits vs L  (A={A}, fock/sparse, {dim}D)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.set_xticks(L); ax2.set_xticklabels([str(int(x)) for x in L])
    ax2.legend(fontsize=10, loc='lower right')

    plt.tight_layout()

    date_str = datetime.now().strftime('%Y-%m-%d')
    out_dir = os.path.join('data', date_str)
    os.makedirs(out_dir, exist_ok=True)
    base = f'L_scaling_fock_sparse_A{A}_with_trotter'
    png = os.path.join(out_dir, f'{base}.png')
    js = os.path.join(out_dir, f'{base}.json')
    plt.savefig(png, dpi=200)
    print(f'\nSaved plot: {png}')

    # Curve data for reproducibility / downstream tooling.
    save_pkg = {
        'metadata': {
            'source': SRC,
            'A': A, 'dim': dim,
            'delta_E_MeV': DELTA_E,
            'trotter_params': {
                'epsilon': TROTTER_E, 'E_kin': TROTTER_EKIN, 'Cp': TROTTER_CP,
            },
            'timestamp': datetime.now().isoformat(),
        },
        'curves': [
            {
                'L': int(L[i]),
                'QPE_Step_T_Count': float(tstep[i]),
                'Physical_Lambda': float(lam[i]),
                'QPE_Total_T_Count_fock_sparse': float(qpe_total[i]),
                'Trotter_Total_T_Count': float(trotter[i]),
                'Logical_Qubits': int(qubits[i]),
            }
            for i in range(len(L))
        ],
    }
    with open(js, 'w') as f:
        json.dump(save_pkg, f, indent=4)
    print(f'Saved data: {js}')


if __name__ == '__main__':
    main()
