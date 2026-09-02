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
from classical.trimci.back_evaluate import (back_evaluate_frame, kato_temple_lower,
                                            rayleigh, state_dict_from_result)
from classical.trimci.observables import occupation_tail, occupation_histogram
from src_PI.utils.manifest import build_manifest


def run_benchmark(L, dim, n_b, frame, cores, A=1, seed=0, num_runs=16,
                  fit_core=None, support_cap=None, back_eval_nf=None,
                  exact_ref=False, exact_max_mem_gb=24.0, out=None):
    H_bare = build_from_eft(L, dim, n_b)
    has_g, has_lf, has_coo = ('gaussian' in frame), ('lf' in frame), ('coo' in frame)
    solve = _solver(True)
    fit_core = fit_core or min(cores)

    # Map-back REFERENCE H. The composed unitary raises boson occupation, so scoring the mapped
    # state against a bare H at the SOLVE cutoff silently drops the raised weight at the Fock
    # ceiling (theory note b). Build the reference at a LARGER cutoff so E_orig is a TIGHT bare-H
    # energy; the per-core eps_leak (1 − ‖P_Nf U|ψ̃⟩‖²) verifies the ceiling is high enough (→0).
    # Validity of E_orig never depends on this — only tightness. Only LF frames map back.
    H_ref = H_bare
    if back_eval_nf and int(back_eval_nf) > n_b:
        H_ref = build_from_eft(L, dim, int(back_eval_nf))

    # Frames whose transformed H is an EXACT operator identity (squeeze: degree-≤2 Bogoliubov
    # substitution; COO: finite fermionic rotation) are Ritz projections of a genuinely
    # isospectral operator, so E_frame is ALREADY a variational upper bound — no map-back needed
    # (and COO's boson-less rotation has no Fock leak). Only the truncated leading-order LF frame
    # is non-isospectral and REQUIRES back-evaluation. See docs/lf_backevaluation.md.
    needs_backeval = has_lf
    variational_directly = (frame == 'bare') or has_coo or (has_g and not has_lf)

    # Fit the frame ONCE (discovery), then reuse its state for every core (matches production:
    # the frame is fit, then the core is grown). 'bare' has an empty state and H_frame == H_bare.
    if frame == 'bare':
        state, H_frame = new_frame_state(H_bare.n_bos_modes), H_bare
    else:
        state, _res, H_frame, _info = initial_frame_state(
            H_bare, A, has_gaussian=has_g, has_lf=has_lf, has_coo=has_coo,
            core=fit_core, num_runs=num_runs, seed=seed)

    # Exact anchor (small ED systems only, guarded): true E_0 → gap_orig = E_orig − E_exact ≥ 0,
    # and E_1 → β for the Kato–Temple certified LOWER bound. Computed on H_ref (the tight cutoff).
    E_exact, beta_kt = None, None
    if exact_ref:
        try:
            from classical.trimci.lanczos import lanczos_ground_state
            e0, linfo = lanczos_ground_state(H_ref, n_elec=A, k=12,
                                             max_mem_gb=exact_max_mem_gb)
            E_exact = float(e0)
            evs = [float(e) for e in (linfo.get('all_evals') or [])]
            # Kato–Temple β = the first level STRICTLY above E_0 (skip the degenerate ground
            # multiplet — a degenerate partner is not a valid β). None if k didn't reach a gap.
            above = [e for e in evs if e > E_exact + 1e-6]
            beta_kt = above[0] if above else None
        except Exception:
            E_exact, beta_kt = None, None

    out_data = {
        'metadata': {
            'kind': 'backeval_benchmark', 'L': L, 'dim': dim, 'n_b': n_b, 'A': A,
            'frame': frame, 'seed': seed, 'num_runs': num_runs, 'fit_core': fit_core,
            'support_cap': support_cap, 'needs_backeval': needs_backeval,
            'variational_directly': variational_directly,
            'back_eval_nf': (int(back_eval_nf) if back_eval_nf else None),
            'N_f_ref': H_ref.N_f, 'exact_ref': bool(exact_ref),
            'E_exact': E_exact, 'beta_kt': beta_kt,
            'n_ferm_modes': H_bare.n_ferm_modes, 'n_bos_modes': H_bare.n_bos_modes,
            'N_f': H_bare.N_f, 'manifest': build_manifest(),
            'frame_params': {'disp_scale': state.get('disp_scale'),
                             'has_r': state.get('r') is not None,
                             'has_R': state.get('R') is not None},
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
    # gaussian/LF frames map back through the EXACT unitary; COO/bare are operator-identity
    # (E_frame already variational) so they skip the map-back (COO's R would raise anyway).
    do_backeval = has_lf or (has_g and not has_coo)
    for core in cores:
        t0 = time.time()
        res = solve(H_frame, n_elec=A, n_dets=core, n_runs=num_runs, seed=seed)
        solve_s = time.time() - t0
        row = {'core': core, 'n_dets': res.n_dets, 'E_frame': float(res.energy),
               'solve_s': round(solve_s, 2)}
        # Framed-basis boson-cutoff diagnostics: leaked-weight tail a Fock box of size N_f would
        # drop, and the per-mode occupation histogram p(n) (near-vacuum in a good frame -> the
        # "similar-enough n_b" evidence). Best-effort — never kills the rung.
        try:
            row['occ_tail'] = occupation_tail(res, [2, 3, 4, 5, 6, 8])
            row['occ_hist'] = occupation_histogram(res).tolist()
        except Exception:
            row['occ_tail'], row['occ_hist'] = None, None

        if do_backeval:
            try:
                tracemalloc.start()
                t1 = time.time()
                be = back_evaluate_frame(H_ref, state, res, support_cap=support_cap)
                be_s = time.time() - t1
                peak = tracemalloc.get_traced_memory()[1] / 1e6
                tracemalloc.stop()
                row.update({'E_orig': be['E_orig'], 'residual': be['residual'],
                            'eps_leak': be['eps_leak'], 'norm_ratio': be['norm_ratio'],
                            'support_in': be['support_in'], 'support_out': be['support_out'],
                            'max_support': be['max_support'], 'map_steps': be['map_steps'],
                            'converged': be['converged'], 'dropped_weight': be['dropped_weight'],
                            'backeval_s': round(be_s, 2), 'backeval_peak_mb': round(peak, 1),
                            'E_frame_shift': float(res.energy) - be['E_orig']})
            except Exception as e:                       # noqa: BLE001 — never kill the shard
                tracemalloc.stop()
                row.update({'E_orig': None, 'backeval_error': f"{type(e).__name__}: {e}"})
        else:
            # Operator-identity (bare / COO / gaussian+COO): E_frame IS the variational upper
            # bound. Residual (for Kato-Temple) w.r.t. the exact H it is a Ritz value of: H_bare
            # for 'bare', the (isospectral) framed H for COO. eps_leak = 0 (no boson map-back).
            resid = None
            try:
                sd = state_dict_from_result(res)
                _, resid = rayleigh(H_bare if frame == 'bare' else H_frame, sd)
            except Exception:
                resid = None
            row.update({'E_orig': float(res.energy), 'residual': resid, 'eps_leak': 0.0,
                        'support_in': res.n_dets, 'support_out': res.n_dets,
                        'max_support': res.n_dets, 'backeval_s': 0.0, 'backeval_peak_mb': 0.0,
                        'converged': True, 'variational_directly': True})

        eo = row.get('E_orig')
        if eo is not None and E_exact is not None:
            row['gap_orig'] = eo - E_exact                       # honest variational penalty (>=0)
        row['kato_temple_lower'] = kato_temple_lower(eo, row.get('residual'), beta_kt)
        out_data['results'].append(row)
        save()
        print(f"[bench] {frame} core={core} n_dets={res.n_dets} E_frame={res.energy:.4f} "
              f"E_orig={eo if eo is None else round(eo,4)} "
              f"eps_leak={row.get('eps_leak')} solve={solve_s:.1f}s "
              f"backeval={row.get('backeval_s',0)}s "
              f"support={row.get('support_in')}->{row.get('support_out')} "
              f"conv={row.get('converged')}")
    out_data['done'] = True
    save()
    return out_data


def main():
    ap = argparse.ArgumentParser(description="LF back-evaluation benchmark (one frame, geometric cores).")
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--n_b', type=int, default=1)
    ap.add_argument('--frame', default='gaussian+lf',
                    help="bare | gaussian | lf | gaussian+lf | coo | gaussian+coo "
                         "(coo/bare/gaussian+coo are operator-identity -> E_frame is variational, "
                         "no map-back; only lf frames back-evaluate)")
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
    ap.add_argument('--back-eval-nf', type=int, default=None,
                    help="build the map-back REFERENCE H at this n_b (N_f=2^n_b), LARGER than "
                         "the solve n_b, so the composed unitary's raised occupation isn't "
                         "silently truncated at the Fock ceiling. eps_leak (reported) verifies "
                         "it's high enough (->0). Only affects lf/gaussian map-back frames.")
    ap.add_argument('--exact-ref', action='store_true',
                    help="compute the exact E_0 (and E_1 for the Kato-Temple beta) via guarded "
                         "Lanczos on the reference H -> gap_orig and a certified interval. "
                         "Small ED systems only (records None if the sector is too large).")
    ap.add_argument('--exact-max-mem-gb', type=float, default=24.0,
                    help="memory ceiling for the exact-ref Lanczos (refuses cleanly above)")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    cores = [int(c) for c in args.cores.split(',') if c.strip()]
    data = run_benchmark(args.L, args.dim, args.n_b, args.frame, cores, A=A,
                         seed=args.seed, num_runs=args.num_runs,
                         support_cap=args.support_cap, back_eval_nf=args.back_eval_nf,
                         exact_ref=args.exact_ref, exact_max_mem_gb=args.exact_max_mem_gb,
                         out=args.out)
    print(f"[bench] done: {len(data['results'])} cores -> {args.out}")


if __name__ == '__main__':
    main()
