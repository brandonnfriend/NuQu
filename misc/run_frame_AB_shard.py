"""Architecture A-vs-B resource shard — bare walk vs Gaussian-squeeze walk (task 34).

The quantum half of the frame-architecture decision (`docs/frame_on_quantum_side.md`):
does qubitizing the SQUEEZED Hamiltonian (Architecture B) beat qubitizing the bare H
(Architecture A, whose only frame lever is the warm-start p0)? This shard produces the
walk-operator resource estimates for BOTH frames so they can be compared.

Key simplification: `evaluate_resources` has no `A` argument — in the Fock basis the
walk operator (hence Λ, T/step, logical qubits) depends only on `(L, dim, n_b, frame)`,
NOT on the nucleon number A. A enters the decision only through the accuracy→n_b choice,
which is a CLASSICAL determination (framed vs bare ⟨n⟩ / truncation-error-vs-N_f, done
off-cluster). So this shard sweeps `(frame × n_b)` at fixed L — the resource-vs-n_b
curve — and the verdict is read by overlaying each frame's accuracy-required n_b:

    B wins iff its smaller required n_b (compaction) outweighs its larger Λ at fixed n_b.

The squeeze is `c_π→e^{r}c_π`, `c_Π→e^{−r}c_Π` per mode (canonical, exactly isospectral —
QPE returns spec(H)); `r` is the classical `analytic_squeeze` optimum, computed
OFF-CLUSTER and passed via `--squeeze-r` so this job needs no `classical.trimci` import
(only src_PI + pyLIQTR). Both frames use the SAME encoder for a fair comparison.

HPC pattern (mirrors `run_quantum_shard.py`): one self-describing JSON to `--out`,
re-saved atomically after every (frame, n_b) so an OOM/timeout on a deep n_b keeps
everything finished.

Smoke (run on a qis node before a campaign):
    python -m misc.run_frame_AB_shard --test

Real:
    python -m misc.run_frame_AB_shard --L 2 --n-b-values 2,3,4 --squeeze-r 0.2109 \
        --out campaign_X/shards/L2_AB.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.qpe_cost import (
    DEFAULT_DELTA_E_MEV, total_qpe_t_count, walk_queries,
)
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.Config import Config
from src_PI.utils.manifest import build_manifest


# frame name -> (pion_basis, needs squeeze_r). Same encoder/cutoff for a fair A-vs-B.
FRAMES = {
    'bare': dict(pion_basis='fock', squeeze=False),
    'squeeze': dict(pion_basis='fock_squeezed', squeeze=True),
}


def _config(basis, encoder):
    return Config(pion_basis=basis, walk_mode='series',
                  cutoff_method='energy_bound', boson_cutoff_method='heuristic',
                  block_encoder=encoder)


def run_ab_shard(L, n_b_values, squeeze_r, dim=3, frames=('bare', 'squeeze'),
                 encoder='pauli_lcu', delta_E=DEFAULT_DELTA_E_MEV, out=None,
                 extra_manifest=None):
    params = get_physical_parameters()
    out_data = {
        'metadata': {
            'kind': 'frame_AB_shard', 'L': L, 'dim': dim,
            'frames': list(frames), 'encoder': encoder,
            'n_b_values': list(n_b_values), 'squeeze_r': squeeze_r,
            'delta_E_MeV': delta_E, 'params': params,
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
        os.replace(tmp, out)          # atomic

    save()                            # header immediately
    t0 = time.time()
    for frame in frames:
        spec = FRAMES[frame]
        cfg = _config(spec['pion_basis'], encoder)
        p = dict(params)
        if spec['squeeze']:
            if squeeze_r is None:
                raise ValueError("squeeze frame needs --squeeze-r (classical r*)")
            p['squeeze_r'] = float(squeeze_r)
        for n_b in n_b_values:
            t_pt = time.time()
            norm = evaluate_resources(L, dim, n_b, 0.0, p, cfg)
            dt = time.time() - t_pt
            lam = norm['Physical_Lambda']
            t_step = norm['Total_T_Count']
            entry = {
                'frame': frame, 'L': L, 'dim': dim, 'n_b': n_b, 'encoder': encoder,
                'squeeze_r': (float(squeeze_r) if spec['squeeze'] else 0.0),
                'Physical_Lambda': lam,
                'Logical_Qubits': norm['Logical_Qubits'],
                'Walk_T_Count': norm['Walk_T_Count'],
                'Total_T_Count': t_step,
                'QPE_Walk_Queries': walk_queries(lam, delta_E),
                'QPE_Total_T_Count': total_qpe_t_count(t_step, lam, delta_E),
                'Runtime_Seconds': round(dt, 2),
            }
            out_data['results'].append(entry)
            out_data['wall_s'] = time.time() - t0
            save()                    # INCREMENTAL: survive an OOM on the next n_b
            print(f"[AB] L={L} {frame:8} n_b={n_b} Λ={lam:.3e} "
                  f"T_step={t_step:.3e} qubits={norm['Logical_Qubits']} "
                  f"QPE_T={entry['QPE_Total_T_Count']:.3e} ({dt:.1f}s)")

    out_data['done'] = True
    out_data['wall_s'] = time.time() - t0
    save()
    _print_verdict(out_data['results'])
    return out_data


def _print_verdict(rows):
    """Per-n_b squeeze-vs-bare ratios (at fixed n_b, isolating the walk-operator cost;
    the n_b-reduction win is the classical accuracy overlay, off-cluster)."""
    by = {}
    for r in rows:
        by.setdefault(r['n_b'], {})[r['frame']] = r
    print("\n[AB] squeeze-vs-bare at fixed n_b (Λ,QPE-T ratio; qubits identical at fixed n_b):")
    for n_b in sorted(by):
        b, s = by[n_b].get('bare'), by[n_b].get('squeeze')
        if b and s:
            print(f"[AB]   n_b={n_b}: Λ {s['Physical_Lambda']/b['Physical_Lambda']:.3f}× "
                  f"QPE_T {s['QPE_Total_T_Count']/b['QPE_Total_T_Count']:.3f}×  "
                  f"(bare Λ={b['Physical_Lambda']:.2e}, sq Λ={s['Physical_Lambda']:.2e})")
    print("[AB] NOTE: at fixed n_b the squeeze is a Λ COST; B's win is a SMALLER required "
          "n_b (classical ⟨n⟩/accuracy overlay) beating that cost.")


def smoke_test():
    print("[AB:test] importing pyLIQTR via L=2 d=3 n_b=2 bare+squeeze estimate ...")
    import pyLIQTR
    print(f"[AB:test] pyLIQTR {getattr(pyLIQTR, '__version__', '?')}")
    data = run_ab_shard(L=2, n_b_values=[2], squeeze_r=0.2109, out=None)
    print(f"[AB:test] OK — {len(data['results'])} estimates")
    print(f"[AB:test] manifest: {data['metadata']['manifest']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Architecture A-vs-B resource shard "
                                             "(bare fock vs squeezed fock walk).")
    ap.add_argument('--test', action='store_true',
                    help="pyLIQTR smoke (L=2 d=3 n_b=2 bare+squeeze); no --out needed.")
    ap.add_argument('--L', type=int)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--n-b-values', default='2,3,4',
                    help="comma- (or '+'-) separated n_b sweep (default 2,3,4)")
    ap.add_argument('--squeeze-r', type=float, default=None,
                    help="classical analytic_squeeze r* (required for the squeeze frame; "
                         "compute off-cluster). L=2 d=3=0.2109, L=3 d=3=0.2543.")
    ap.add_argument('--frames', default='bare,squeeze',
                    help="comma-separated subset of {bare,squeeze} (default both)")
    ap.add_argument('--encoder', default='pauli_lcu',
                    help="block encoder for BOTH frames (default pauli_lcu; fair A-vs-B)")
    ap.add_argument('--delta-E', type=float, default=DEFAULT_DELTA_E_MEV)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.test:
        sys.exit(smoke_test())
    if args.L is None or args.out is None:
        ap.error("--L and --out are required unless --test")
    n_b_values = [int(x) for x in args.n_b_values.replace('+', ',').split(',') if x.strip()]
    frames = tuple(f.strip() for f in args.frames.split(',') if f.strip())
    if 'squeeze' in frames and args.squeeze_r is None:
        ap.error("--squeeze-r is required when the squeeze frame is included")
    data = run_ab_shard(args.L, n_b_values, args.squeeze_r, dim=args.dim,
                        frames=frames, encoder=args.encoder, delta_E=args.delta_E,
                        out=args.out, extra_manifest={'run_args': vars(args)})
    print(f"\n[AB] done: {len(data['results'])} estimates, "
          f"wall={data.get('wall_s', 0):.1f}s -> {args.out}")


if __name__ == '__main__':
    main()
