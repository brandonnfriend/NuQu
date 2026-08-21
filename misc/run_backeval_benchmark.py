"""LF back-evaluation BENCHMARK — one (L, dim, n_b, frame) over geometric cores.

Answers the codex audit's open questions (05_classical_frames): does original-H
back-evaluation stay tractable at production-shaped selected-CI cores, and does the frame
help the PHYSICAL (original-H) energy at matched core? For each core it solves the framed
Hamiltonian, maps the solved state back with the EXACT composed unitary (squeeze∘LF), scores
E_orig = <psi|H_bare|psi> (Ritz-valid), and records the map-back cost (support growth, wall,
peak Python memory, Taylor convergence). Run one shard per frame in {bare, gaussian, lf,
gaussian+lf}; compare E_orig(frame) vs E_var(bare) at matched core/wall in analysis.

Incremental per-core save (a deep core that OOMs/times-out keeps every core below it).

    python -m misc.run_backeval_benchmark --L 2 --dim 3 --n_b 1 --frame gaussian+lf \
        --cores 250,1000,4000,16000 --A 2 --out bench_L2_gaussian+lf.json
"""
import argparse
import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.hamiltonian import build_from_eft
from classical.trimci.frame_workflow import initial_frame_state, apply_frame, _solver, new_frame_state
from classical.trimci.back_evaluate import back_evaluate_frame
from src_PI.utils.manifest import build_manifest


def run_benchmark(L, dim, n_b, frame, cores, A=1, seed=0, num_runs=16,
                  fit_core=None, support_cap=None, out=None):
    H_bare = build_from_eft(L, dim, n_b)
    has_g, has_lf, has_coo = ('gaussian' in frame), ('lf' in frame), ('coo' in frame)
    solve = _solver(True)
    fit_core = fit_core or min(cores)

    # Fit the frame ONCE (discovery), then reuse its state for every core (matches production:
    # the frame is fit, then the core is grown). 'bare' has an empty state and H_frame == H_bare.
    if frame == 'bare':
        state, H_frame = new_frame_state(H_bare.n_bos_modes), H_bare
    else:
        state, _res, H_frame, _info = initial_frame_state(
            H_bare, A, has_gaussian=has_g, has_lf=has_lf, has_coo=has_coo,
            core=fit_core, num_runs=num_runs, seed=seed)

    out_data = {
        'metadata': {
            'kind': 'backeval_benchmark', 'L': L, 'dim': dim, 'n_b': n_b, 'A': A,
            'frame': frame, 'seed': seed, 'num_runs': num_runs, 'fit_core': fit_core,
            'support_cap': support_cap,
            'n_ferm_modes': H_bare.n_ferm_modes, 'n_bos_modes': H_bare.n_bos_modes,
            'N_f': H_bare.N_f, 'manifest': build_manifest(),
            'frame_params': {'disp_scale': state.get('disp_scale'),
                             'has_r': state.get('r') is not None},
        },
        'results': [], 'done': False,
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)

    def save():
        if not out:
            return
        tmp = out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out_data, f, indent=2, default=str)
        os.replace(tmp, out)

    save()
    for core in cores:
        t0 = time.time()
        res = solve(H_frame, n_elec=A, n_dets=core, n_runs=num_runs, seed=seed)
        solve_s = time.time() - t0
        row = {'core': core, 'n_dets': res.n_dets, 'E_frame': float(res.energy),
               'solve_s': round(solve_s, 2)}
        if frame == 'bare':
            # bare frame: no map-back; E_orig == E_frame is the variational baseline.
            row.update({'E_orig': float(res.energy), 'residual': None, 'support_in': res.n_dets,
                        'support_out': res.n_dets, 'max_support': res.n_dets,
                        'backeval_s': 0.0, 'backeval_peak_mb': 0.0, 'converged': True})
        else:
            try:
                tracemalloc.start()
                t1 = time.time()
                be = back_evaluate_frame(H_bare, state, res, support_cap=support_cap)
                be_s = time.time() - t1
                peak = tracemalloc.get_traced_memory()[1] / 1e6
                tracemalloc.stop()
                row.update({'E_orig': be['E_orig'], 'residual': be['residual'],
                            'support_in': be['support_in'], 'support_out': be['support_out'],
                            'max_support': be['max_support'], 'map_steps': be['map_steps'],
                            'converged': be['converged'], 'dropped_weight': be['dropped_weight'],
                            'backeval_s': round(be_s, 2), 'backeval_peak_mb': round(peak, 1),
                            'E_frame_shift': float(res.energy) - be['E_orig']})
            except Exception as e:                       # noqa: BLE001 — never kill the shard
                tracemalloc.stop()
                row.update({'E_orig': None, 'backeval_error': f"{type(e).__name__}: {e}"})
        out_data['results'].append(row)
        save()
        eo = row.get('E_orig')
        print(f"[bench] {frame} core={core} n_dets={res.n_dets} E_frame={res.energy:.4f} "
              f"E_orig={eo if eo is None else round(eo,4)} "
              f"solve={solve_s:.1f}s backeval={row.get('backeval_s',0)}s "
              f"support={row.get('support_in')}->{row.get('support_out')} "
              f"peak={row.get('backeval_peak_mb',0)}MB conv={row.get('converged')}")
    out_data['done'] = True
    save()
    return out_data


def main():
    ap = argparse.ArgumentParser(description="LF back-evaluation benchmark (one frame, geometric cores).")
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--n_b', type=int, default=1)
    ap.add_argument('--frame', default='gaussian+lf',
                    help="bare | gaussian | lf | gaussian+lf")
    ap.add_argument('--cores', default='250,1000,4000,16000',
                    help="comma-separated geometric core sizes")
    ap.add_argument('--A', type=int, default=1)
    ap.add_argument('--filling', type=float, default=None,
                    help="if set, A = round(filling * sites) (overrides --A)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-runs', type=int, default=16)
    ap.add_argument('--support-cap', type=int, default=None,
                    help="bound the map-back fan-out to this many determinants (audit "
                         "tractability fallback for dense filling; still variational, "
                         "records dropped_weight for cap convergence-testing)")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    cores = [int(c) for c in args.cores.split(',') if c.strip()]
    data = run_benchmark(args.L, args.dim, args.n_b, args.frame, cores, A=A,
                         seed=args.seed, num_runs=args.num_runs,
                         support_cap=args.support_cap, out=args.out)
    print(f"[bench] done: {len(data['results'])} cores -> {args.out}")


if __name__ == '__main__':
    main()
