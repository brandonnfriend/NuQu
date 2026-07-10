"""
One-shot driver: run the four sweeps needed for the L={2,3,4} basis-comparison
plot (amplitude/ns + fock/sparse at each L). Amplitude/energy_bound legacy
data is reused from the data/ tree — not recomputed here.

L=2 fock/sparse was already saved earlier this morning; we skip it.

Run from project root:
    source .venv/bin/activate
    python misc/run_basis_comparison_sweeps.py
"""

import os
import sys
import time

# Allow `python misc/run_basis_comparison_sweeps.py` from the project root by
# putting the project root (parent of this file's directory) on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from run_nucleon_sweep import run_sweep


# A-range schedule per L. Match the legacy amplitude/energy_bound A-grid so
# the overlay is apples-to-apples per L.
A_DEFAULT = np.concatenate([
    np.arange(1, 11),
    np.round(np.linspace(20, 100, 15)).astype(int),
])
A_DEFAULT = np.unique(A_DEFAULT)

A_L4_LEGACY = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


SWEEPS = [
    # (label, L, A_values, basis, encoder, cutoff_method)
    ('L=3 amplitude/ns',     3, A_DEFAULT,   'amplitude', 'pauli_lcu', 'ns'),
    ('L=3 fock/sparse',      3, A_DEFAULT,   'fock',      'sparse',    'energy_bound'),
    ('L=4 amplitude/ns',     4, A_L4_LEGACY, 'amplitude', 'pauli_lcu', 'ns'),
    ('L=4 fock/sparse',      4, A_L4_LEGACY, 'fock',      'sparse',    'energy_bound'),
]


def main():
    overall_t0 = time.time()
    out_files = []
    for label, L, A_vals, basis, encoder, cutoff in SWEEPS:
        print('=' * 80)
        print(f'>>> {label} ({len(A_vals)} A values: {list(A_vals)})')
        print('=' * 80)
        t0 = time.time()
        try:
            fp = run_sweep(
                L=L, dim=3, A_values=A_vals,
                pion_basis=basis, block_encoder=encoder, cutoff_method=cutoff,
            )
            wall = time.time() - t0
            print(f'>>> {label} DONE in {wall:.1f}s -> {fp}')
            out_files.append((label, fp, wall))
        except Exception as e:
            wall = time.time() - t0
            print(f'>>> {label} FAILED after {wall:.1f}s: {e}')
            import traceback; traceback.print_exc()
            out_files.append((label, None, wall))

    print('=' * 80)
    print(f'ALL SWEEPS DONE in {time.time() - overall_t0:.1f}s')
    for label, fp, wall in out_files:
        print(f'  {label}: {wall:.1f}s -> {fp}')


if __name__ == '__main__':
    main()
