"""
Scan fock/sparse runtime vs lattice extent L at fixed A.

Loop L = 2, 3, 4, ... at A = 10 (a value already covered by the L=2/3/4
basis-comparison sweeps, so the first few L's are sanity-cross-checks
against existing data). Stop after the first iteration whose wall time
exceeds RUNTIME_CAP_SEC (default 600 s = 10 min).

Each L creates a normal per-A sweep file via run_sweep; this driver also
writes a consolidated JSON with the runtime + key resource numbers per L
so the L-scaling can be read off in one place. Incremental write per L
guards against losing data if the process is killed mid-sweep.

Run from project root:
    source .venv/bin/activate
    python misc/run_L_scaling_fock_sparse.py
"""

import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from run_nucleon_sweep import run_sweep


A_FIXED = 10
RUNTIME_CAP_SEC = 600.0   # stop after the first L whose wall time exceeds this
DIM = 3
L_START = 2
L_HARD_CAP = 12           # absolute safety stop — should never be reached
                          # before the runtime cap


def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    out_dir = os.path.join('data', date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'L_scaling_fock_sparse_A{A_FIXED}.json')

    metadata = {
        'A': A_FIXED,
        'dim': DIM,
        'pion_basis': 'fock',
        'block_encoder': 'sparse',
        'cutoff_method': 'energy_bound',
        'runtime_cap_sec': RUNTIME_CAP_SEC,
        'L_start': L_START,
        'L_hard_cap': L_HARD_CAP,
        'started': datetime.now().isoformat(),
        'completed': None,
        'stop_reason': None,
    }
    results = []

    def flush():
        with open(out_path, 'w') as f:
            json.dump({'metadata': metadata, 'results': results}, f, indent=4)

    L = L_START
    while L <= L_HARD_CAP:
        print('=' * 80)
        print(f'>>> L={L}  A={A_FIXED}  fock/sparse')
        print('=' * 80)
        t0 = time.time()
        try:
            fp = run_sweep(
                L=L, dim=DIM, A_values=np.array([A_FIXED]),
                pion_basis='fock', block_encoder='sparse',
                cutoff_method='energy_bound',
            )
            wall_total = time.time() - t0
        except Exception as e:
            wall_total = time.time() - t0
            print(f'>>> L={L} FAILED after {wall_total:.1f}s: {e}')
            import traceback; traceback.print_exc()
            metadata['stop_reason'] = f'exception at L={L}: {e}'
            metadata['completed'] = datetime.now().isoformat()
            flush()
            return

        with open(fp) as f:
            sweep = json.load(f)
        r = sweep['results'][0]
        results.append({
            'L': int(L),
            'A': int(A_FIXED),
            'dim': DIM,
            'n_b': r.get('n_b'),
            'Physical_Lambda': r['Physical_Lambda'],
            'Logical_Qubits': r.get('Logical_Qubits'),
            'Walk_Clifford_Count': r.get('Walk_Clifford_Count'),
            'Walk_T_Count': r.get('Walk_T_Count'),
            'QFT_T_Count': r.get('QFT_T_Count'),
            'Total_T_Count': r['Total_T_Count'],
            'QPE_Walk_Queries': r.get('QPE_Walk_Queries'),
            'QPE_Total_T_Count': r.get('QPE_Total_T_Count'),
            'Runtime_Seconds_inner': r.get('Runtime_Seconds'),
            'wall_seconds_driver': round(wall_total, 2),
            'source_file': fp,
        })
        flush()
        print(f'>>> L={L} DONE: inner Runtime_Seconds={r.get("Runtime_Seconds")}s, '
              f'driver wall={wall_total:.2f}s, n_b={r.get("n_b")}, '
              f'LogicalQubits={r.get("Logical_Qubits")}')

        if wall_total > RUNTIME_CAP_SEC:
            metadata['stop_reason'] = (
                f'wall time {wall_total:.1f}s at L={L} exceeded cap '
                f'{RUNTIME_CAP_SEC}s'
            )
            print(f'>>> STOP: {metadata["stop_reason"]}')
            break

        L += 1
    else:
        metadata['stop_reason'] = f'reached L_HARD_CAP={L_HARD_CAP} without exceeding runtime cap'
        print(f'>>> STOP: {metadata["stop_reason"]}')

    metadata['completed'] = datetime.now().isoformat()
    flush()
    print()
    print(f'Wrote consolidated: {out_path}')
    print(f'Final L: {results[-1]["L"]}  ({len(results)} points)')


if __name__ == '__main__':
    main()
