"""One quantum resource-estimation SHARD — (L, series) with an A-sweep (task 34, I4).

The HPC-facing counterpart of `run_nucleon_sweep.run_sweep`: instead of auto-saving
into `data/quantum/<date>/`, it writes ONE self-describing JSON to an explicit
`--out` path and RE-SAVES after every A (atomic tmp+rename), so a shard that OOMs
or times out on a deep (L, A) still keeps everything it finished — the same
robustness pattern as the classical `run_frame_shard.py`.

A "series" is one design-axis column of the campaign (basis + encoder + cutoff):

    watson  = amplitude / energy_bound / pauli_lcu   (Watson Lemma-5 baseline)
    ns      = amplitude / ns          / pauli_lcu     (Nyquist-Shannon, tong register)
    sparse  = fock      / sparse      / tong          (deep-L workhorse; Tier-1 realistic)
    sparse_heuristic = fock / sparse  / heuristic     (comparison against tong)

Smoke test (run FIRST on a qis node before any real campaign):

    python -m misc.run_quantum_shard --test

imports pyLIQTR through a real L=2, A=1 sparse estimate — this is what surfaces the
`juliacall`/`juliapkg` Julia-runtime auto-provision and `gmpy2`/GMP issues that the
heavier quantum dependency tree can hit inside a fresh sandbox.

    python -m misc.run_quantum_shard --L 6 --series sparse --A-values 1,2,4 \
        --out campaign_X/shards/L6_sparse.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_nucleon_sweep import _compute_cutoffs, get_sweep_config
from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.qpe_cost import (
    DEFAULT_DELTA_E_MEV, total_qpe_t_count, walk_queries,
)
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.Config import Config
from src_PI.utils.manifest import build_manifest


# Campaign design-axis columns (basis + encoder + cutoff prescription).
SERIES = {
    'watson': dict(pion_basis='amplitude', cutoff_method='energy_bound',
                   block_encoder='pauli_lcu', boson_cutoff_method='heuristic'),
    'ns': dict(pion_basis='amplitude', cutoff_method='ns',
               block_encoder='pauli_lcu', boson_cutoff_method='tong'),
    'sparse': dict(pion_basis='fock', cutoff_method='energy_bound',
                   block_encoder='sparse', boson_cutoff_method='tong'),
    'sparse_heuristic': dict(pion_basis='fock', cutoff_method='energy_bound',
                             block_encoder='sparse', boson_cutoff_method='heuristic'),
}


def _config_from_series(series):
    s = SERIES[series]
    return Config(pion_basis=s['pion_basis'], walk_mode='series',
                  cutoff_method=s['cutoff_method'],
                  boson_cutoff_method=s['boson_cutoff_method'],
                  block_encoder=s['block_encoder'])


def run_shard(L, series, A_values, dim=3, frame_occupation=None,
              delta_E=DEFAULT_DELTA_E_MEV, out=None, extra_manifest=None):
    s = SERIES[series]
    config = _config_from_series(series)
    run_cfg = get_sweep_config(L=L, dim=dim, frame_occupation=frame_occupation, **s)
    params = get_physical_parameters()

    out_data = {
        'metadata': {
            'kind': 'quantum_shard', 'L': L, 'dim': dim, 'series': series,
            'series_config': s, 'delta_E_MeV': delta_E,
            'frame_occupation': frame_occupation,
            'params': params,
            'config': config.to_dict(),
            'manifest': build_manifest(extra=extra_manifest),
        },
        'results': [], 'done': False,
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    def save():
        if not out:
            return
        tmp = out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out_data, f, indent=2)
        os.replace(tmp, out)          # atomic — a reader never sees a half file

    save()                            # header immediately
    t0 = time.time()
    for A in A_values:
        n_b, pi_max, Pi_max = _compute_cutoffs(L, dim, A, params, run_cfg, config)
        t_pt = time.time()
        norm = evaluate_resources(L, dim, n_b, pi_max, params, config)
        dt = time.time() - t_pt
        lam = norm['Physical_Lambda']
        t_step = norm['Total_T_Count']
        entry = {
            'A': int(A), 'L': L, 'dim': dim, 'n_b': n_b,
            'pi_max': float(pi_max) if pi_max == pi_max else None,
            'Pi_max': float(Pi_max) if Pi_max == Pi_max else None,
            'Runtime_Seconds': round(dt, 3),
            'Physical_Lambda': lam,
            'Logical_Qubits': norm['Logical_Qubits'],
            'Walk_Clifford_Count': norm['Walk_Clifford_Count'],
            'Walk_T_Count': norm['Walk_T_Count'],
            'QFT_T_Count': norm['QFT_T_Count'],
            'Total_T_Count': t_step,
            'Per_Sub_Walk': norm.get('Per_Sub_Walk', []),
            # fold the QPE totals in per-entry so incremental saves carry them
            'QPE_Walk_Queries': walk_queries(lam, delta_E),
            'QPE_Total_T_Count': total_qpe_t_count(t_step, lam, delta_E),
        }
        out_data['results'].append(entry)
        out_data['wall_s'] = time.time() - t0
        save()                        # INCREMENTAL: survive an OOM on the next A
        print(f"[qshard] L={L} series={series} A={A} n_b={n_b} "
              f"Lambda={lam:.3e} QPE_T={entry['QPE_Total_T_Count']:.3e} ({dt:.1f}s)")

    out_data['done'] = True
    out_data['wall_s'] = time.time() - t0
    save()
    return out_data


def smoke_test():
    """Import pyLIQTR through a real tiny estimate — surfaces Julia/gmpy2 issues."""
    print("[qshard:test] importing pyLIQTR via a real L=2 A=1 sparse estimate ...")
    import pyLIQTR
    print(f"[qshard:test] pyLIQTR {getattr(pyLIQTR, '__version__', '?')}")
    data = run_shard(L=2, series='sparse', A_values=[1], out=None)
    r = data['results'][0]
    print(f"[qshard:test] OK  Lambda={r['Physical_Lambda']:.3e}  "
          f"logical_qubits={r['Logical_Qubits']}  QPE_T={r['QPE_Total_T_Count']:.3e}")
    print(f"[qshard:test] manifest: {data['metadata']['manifest']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="One quantum resource-estimation shard.")
    ap.add_argument('--test', action='store_true',
                    help="pyLIQTR/Julia/gmpy2 smoke test (L=2 A=1 sparse); no --out needed.")
    ap.add_argument('--L', type=int)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--series', choices=sorted(SERIES), default='sparse')
    ap.add_argument('--A-values', default='1,2,4',
                    help="comma-separated nucleon counts (default 1,2,4)")
    ap.add_argument('--frame-occupation', type=float, default=None,
                    help="per-mode <n> -> frame-reduced Fock register (task 34 seam a)")
    ap.add_argument('--delta-E', type=float, default=DEFAULT_DELTA_E_MEV)
    ap.add_argument('--out', default=None, help="output JSON path (per-shard)")
    args = ap.parse_args()

    if args.test:
        sys.exit(smoke_test())
    if args.L is None or args.out is None:
        ap.error("--L and --out are required unless --test")
    A_values = [int(x) for x in args.A_values.split(',') if x.strip()]
    data = run_shard(args.L, args.series, A_values, dim=args.dim,
                     frame_occupation=args.frame_occupation, delta_E=args.delta_E,
                     out=args.out, extra_manifest={'run_args': vars(args)})
    n = len(data['results'])
    print(f"[qshard] done: {n} points, wall={data.get('wall_s', 0):.1f}s -> {args.out}")


if __name__ == '__main__':
    main()
