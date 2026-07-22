"""
C4 comparison plot: sparse-oracle vs PauliLCU on the Fock-basis EFT.

Loads the small C4 sweeps + the Phase 0 PauliLCU baseline and renders
a two-panel figure:

  (a) Walk_T_Count vs n_b  — per-walk-step T-count comparison
  (b) Walk_T × Λ vs n_b    — total QPE cost proxy (∝ N_walk · T_per_step)

PauliLCU is pinned to n_b=3 (anything higher times out at L=2 dim=3).
Sparse scales freely; we show it both at fixed n_b=3 (direct overlap
with PauliLCU on the same Hamiltonian) and at heuristic n_b (realistic
scaling).

Output: `data/2026-05-27/sparse_vs_paulilcu_C4.png`.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt
import numpy as np


_DATA_DIR = os.path.join(_ROOT, 'data')

# C4 sweep + Phase 0 baseline file paths.
PAULI_FOCK_NB3 = os.path.join(_DATA_DIR, '2026-05-26', 'sweep_L2_3D_fock_180031.json')
SPARSE_FOCK_HEURISTIC = os.path.join(_DATA_DIR, '2026-05-27', 'sweep_L2_3D_fock_165130.json')
SPARSE_FOCK_NB3 = os.path.join(_DATA_DIR, '2026-05-27', 'sweep_L2_3D_fock_165213.json')
OUT_FIG = os.path.join(_DATA_DIR, '2026-05-27', 'sparse_vs_paulilcu_C4.png')


def _load(path):
    with open(path) as f:
        return json.load(f)


def _series(data):
    """Extract (n_b, A, Walk_T, Λ, Walk_T·Λ, LQ) per result row."""
    rows = []
    for r in data['results']:
        rows.append({
            'A': r['A'],
            'n_b': r['n_b'],
            'walk_T': r['Walk_T_Count'],
            'lambda': r['Physical_Lambda'],
            'cost': r['Walk_T_Count'] * r['Physical_Lambda'],
            'LQ': r['Logical_Qubits'],
        })
    return rows


def main():
    pauli_rows = _series(_load(PAULI_FOCK_NB3))
    sparse_h_rows = _series(_load(SPARSE_FOCK_HEURISTIC))
    sparse_nb3_rows = _series(_load(SPARSE_FOCK_NB3))

    # Summary on stdout.
    print('=== C4: sparse-oracle vs PauliLCU on Fock EFT (L=2, dim=3) ===\n')
    print('NOTE: sparse Walk_T uses an analytical aggregate (C3d.1 + C3d.2). The boson')
    print('      contribution is an upper bound (Gilyén product over ``(â+â†)`` per-mode');
    print('      cost); the fermion contribution is a lower bound (``4·weight`` T per JW')
    print('      Pauli, no PauliLCU PREP/SELECT overhead). Net direction at our sizes:')
    print('      sparse Walk_T_Count is ~conservative-upper because fermion is < 1% of')
    print('      total. C3d.3 polish would tighten both ends. See execution log §10.\n')
    print(f'{"encoder":<22} | {"n_b":>3} | {"A":>3} | {"Walk_T":>10} | {"Λ":>10} | {"Walk_T·Λ":>10}')
    print('-' * 80)
    for r in pauli_rows:
        print(f'{"PauliLCU @ n_b=3":<22} | {r["n_b"]:>3} | {r["A"]:>3} | '
              f'{r["walk_T"]:>10.2e} | {r["lambda"]:>10.2e} | {r["cost"]:>10.2e}')
    for r in sparse_nb3_rows:
        print(f'{"Sparse @ n_b=3":<22} | {r["n_b"]:>3} | {r["A"]:>3} | '
              f'{r["walk_T"]:>10.2e} | {r["lambda"]:>10.2e} | {r["cost"]:>10.2e}')
    for r in sparse_h_rows:
        print(f'{"Sparse @ heuristic n_b":<22} | {r["n_b"]:>3} | {r["A"]:>3} | '
              f'{r["walk_T"]:>10.2e} | {r["lambda"]:>10.2e} | {r["cost"]:>10.2e}')

    # Compute headline ratios at n_b=3 (head-to-head on same Hamiltonian).
    p = pauli_rows[0]
    s = sparse_nb3_rows[0]
    print('\n--- Head-to-head at n_b=3 (same Fock Hamiltonian) ---')
    print(f'Walk_T ratio (sparse / PauliLCU): {s["walk_T"] / p["walk_T"]:.3f}')
    print(f'Λ ratio (sparse / PauliLCU):      {s["lambda"] / p["lambda"]:.3f}')
    print(f'Walk_T · Λ ratio (sparse / PauliLCU): {s["cost"] / p["cost"]:.3f}')

    # Now the plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # PauliLCU: just a single (n_b=3) data point — pin a horizontal marker so
    # the n_b=5..7 sparse points are visually anchored against it.
    pauli_nb = [r['n_b'] for r in pauli_rows]
    pauli_T = [r['walk_T'] for r in pauli_rows]
    pauli_cost = [r['cost'] for r in pauli_rows]

    sparse_nb3_nb = [r['n_b'] for r in sparse_nb3_rows]
    sparse_nb3_T = [r['walk_T'] for r in sparse_nb3_rows]
    sparse_nb3_cost = [r['cost'] for r in sparse_nb3_rows]

    sparse_h_nb = [r['n_b'] for r in sparse_h_rows]
    sparse_h_T = [r['walk_T'] for r in sparse_h_rows]
    sparse_h_cost = [r['cost'] for r in sparse_h_rows]

    # Panel (a): Walk_T vs n_b.
    ax1.semilogy(pauli_nb, pauli_T, marker='s', linestyle='none',
                 markersize=12, color='C3', label='PauliLCU @ n_b=3 (baseline)')
    ax1.semilogy(sparse_nb3_nb, sparse_nb3_T, marker='o', linestyle='none',
                 markersize=10, color='C0', label='Sparse @ n_b=3 (same Hamiltonian)')
    ax1.semilogy(sparse_h_nb, sparse_h_T, marker='^', linestyle='-',
                 color='C0', label='Sparse @ heuristic n_b (scaling)')
    ax1.set_xlabel('n_b (bits per pion mode)')
    ax1.set_ylabel('Walk T-count per step')
    ax1.set_title('Per-walk-step T-count')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='lower right')

    # Panel (b): Walk_T · Λ vs n_b (∝ total QPE cost).
    ax2.semilogy(pauli_nb, pauli_cost, marker='s', linestyle='none',
                 markersize=12, color='C3', label='PauliLCU @ n_b=3 (baseline)')
    ax2.semilogy(sparse_nb3_nb, sparse_nb3_cost, marker='o', linestyle='none',
                 markersize=10, color='C0', label='Sparse @ n_b=3 (same Hamiltonian)')
    ax2.semilogy(sparse_h_nb, sparse_h_cost, marker='^', linestyle='-',
                 color='C0', label='Sparse @ heuristic n_b (scaling)')
    ax2.set_xlabel('n_b (bits per pion mode)')
    ax2.set_ylabel('Walk T · Λ  (∝ total QPE cost)')
    ax2.set_title('Total QPE cost proxy')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='lower right')

    # Annotate the PauliLCU timeout boundary.
    for ax in (ax1, ax2):
        ax.axvspan(4.5, ax.get_xlim()[1], alpha=0.10, color='C3', zorder=-1)
        ax.text(0.97, 0.05, 'PauliLCU\ntimes out\nat n_b ≥ 5',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color='C3', alpha=0.7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='C3'))

    fig.suptitle('C4 — Sparse-oracle vs PauliLCU on Fock-basis EFT (L=2, dim=3)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f'\nWrote {OUT_FIG}')


if __name__ == '__main__':
    main()
