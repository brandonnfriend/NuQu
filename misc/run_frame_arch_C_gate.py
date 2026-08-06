"""Architecture-C admissibility gate driver (task 34, Stage 1).

Measures the leading-order LF transition-vertex residual ‖R_trans‖(λ) on the
smallest ED-able real EFT system (L=2 d=1 — the smallest with a σ⊗τ transition
vertex; L=1 has none), squeeze-referenced via Lanczos at a converged Fock cutoff,
fits the λ-scaling, and reads it off at the production coupling to decide whether
Architecture C (walking the squeeze+LF frame) is admissible at the production
geometry. See `classical/trimci/frame_arch_cgate.py` for the full rationale/caveats.

Run:
  python -m misc.run_frame_arch_C_gate                     # N_f=4, A=1,2; L=2/L=3 targets
  python -m misc.run_frame_arch_C_gate --n-b 3 --A 2       # push N_f (heavier Lanczos)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci import frame_arch_cgate as cg

# Production targets: (label, sites). Sites = L^dim for the real 3D systems.
PRODUCTION_TARGETS = [('L=2 d=3', 8), ('L=3 d=3', 27), ('L=4 d=3', 64)]


def main():
    ap = argparse.ArgumentParser(description="Architecture-C ‖R_trans‖ admissibility gate.")
    ap.add_argument('--L', type=int, default=2, help="ED lattice length (default 2)")
    ap.add_argument('--dim', type=int, default=1, help="ED dimension (default 1 = smallest)")
    ap.add_argument('--n-b', type=int, default=2,
                    help="ED boson bits -> N_f=2^n_b (default 2 => N_f=4, converged squeeze)")
    ap.add_argument('--A', type=int, nargs='*', default=[1, 2],
                    help="nucleon fillings to test (‖R_trans‖ grows with filling)")
    ap.add_argument('--lambdas', type=float, nargs='*',
                    default=[0.05, 0.1, 0.14, 0.2, 0.28, 0.4],
                    help="effective LF amplitudes to sweep")
    ap.add_argument('--production-lambda', type=float,
                    default=cg.DEFAULT_PRODUCTION_LAMBDA,
                    help=f"production λ to read off (default {cg.DEFAULT_PRODUCTION_LAMBDA})")
    ap.add_argument('--budget', type=float, default=cg.DEFAULT_BUDGET_MEV,
                    help=f"ε budget in MeV (default {cg.DEFAULT_BUDGET_MEV})")
    ap.add_argument('--max-states', type=int, default=200_000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f"[cgate] ED system L={args.L} d={args.dim} N_f={2**args.n_b}; "
          f"production λ={args.production_lambda}, budget={args.budget} MeV\n")
    t0 = time.time()
    sweeps, verdicts = [], []
    for A in args.A:
        sweep = cg.rtrans_lambda_sweep(args.L, args.dim, args.n_b, A, args.lambdas,
                                       max_states=args.max_states, verbose=False)
        sweeps.append(sweep)
        print(f"A={A}: N_f={sweep['N_f']} sites={sweep['sites']} "
              f"seed λ={sweep['seed_lambda']:.5f}  "
              f"squeeze_iso(trunc floor)={sweep['squeeze_iso_vs_bare']:.2e} MeV")
        fit = sweep['fit']
        if fit:
            print(f"     fit ‖R_trans‖ = {fit['c']:.3e}·λ^{fit['p']:.2f} (r²={fit['r2']:.3f})")
        print(f"     {'λ':>6} {'‖R_trans‖(MeV)':>15} {'/site':>10}")
        for p in sweep['points']:
            print(f"     {p['lam']:>6.3f} {p['R_trans']:>15.4e} {p['R_per_site']:>10.4e}")
        # verdict per production target
        for label, sites in PRODUCTION_TARGETS:
            v = cg.c_gate_verdict(sweep, production_lambda=args.production_lambda,
                                  production_sites=sites, budget_mev=args.budget,
                                  production_label=label)
            v['ed_A'] = A
            verdicts.append(v)
            flag = 'OK ' if v['admissible'] else 'XX '
            print(f"     -> {flag}{label}: ‖R_trans‖≈{v['rtrans_total_mev']:.2f} MeV "
                  f"({v['rtrans_per_site_mev']:.3e}/site × {sites}) "
                  f"{'<' if v['admissible'] else '>='} {args.budget} -> {v['verdict'].split(':')[0]}")
        print()

    # headline: is C admissible anywhere at production λ?
    adm = [v for v in verdicts if v['admissible']]
    print(f"[verdict] Architecture C admissible in {len(adm)}/{len(verdicts)} "
          f"(target × filling) cases at λ={args.production_lambda}.")
    inadm_L3 = [v for v in verdicts if v['production_label'] == 'L=3 d=3'
                and not v['admissible']]
    if inadm_L3:
        worst = max(inadm_L3, key=lambda v: v['rtrans_total_mev'])
        print(f"  L=3 d=3: INADMISSIBLE (‖R_trans‖ up to {worst['rtrans_total_mev']:.2f} "
              f"MeV > {args.budget} MeV) -> fall back to B (squeeze walk) + A (LF warm-start).")
    print(f"  CAVEAT: {verdicts[0]['caveat']}")

    outdir = 'data/quantum/2026-08-05'
    os.makedirs(outdir, exist_ok=True)
    outpath = args.out or os.path.join(outdir, 'frame_arch_C_gate.json')
    with open(outpath, 'w') as f:
        json.dump({'kind': 'architecture_C_gate',
                   'ed_system': {'L': args.L, 'dim': args.dim, 'N_f': 2**args.n_b},
                   'production_lambda': args.production_lambda, 'budget_mev': args.budget,
                   'sweeps': sweeps, 'verdicts': verdicts,
                   'wall_s': time.time() - t0}, f, indent=2)
    print(f"\n[cgate] wrote sweep + verdicts -> {outpath}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
