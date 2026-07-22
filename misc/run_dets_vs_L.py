"""
Phase C driver — the dets-vs-L exponent at fixed per-site accuracy.

Measures N*(L; eps) = the determinant count to reach a FIXED per-site accuracy eps
of the L-converged ground energy E_inf(L), for L = 2,3,4 (dim=3), and fits N* vs
system volume to read the scaling exponent gamma (the paper's central number). Every
step rests on the Phase-B per-L reference and is reported honestly: an L whose
reference is not pinned, or whose N* is only bounded, is logged as a BOUND, not fit.

DILUTE first (A=1 fixed) per the plan; pass --filling <f> for the fixed-filling
(A = round(f*sites)) companion curve. Laptop guards keep each L bounded: a geometric
core ladder that stops growing once a rung's wall-clock exceeds --max-rung-seconds or
the next core would exceed --max-core, so large L self-limit instead of running for
hours. Outputs: data/classical/<label>.{json,png}.

Examples:
    python -m misc.run_dets_vs_L                       # dilute A=1, L=2,3,4, eps 1 & 0.1
    python -m misc.run_dets_vs_L --L 2 3 --max-core 16000 --max-rung-seconds 300
    python -m misc.run_dets_vs_L --filling 1.0         # fixed-filling companion
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.run_cpp import dets_vs_L_at_fixed_accuracy


def main():
    ap = argparse.ArgumentParser(description="Phase C: dets-vs-L at fixed per-site accuracy")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--L", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--A", type=int, default=1, help="dilute nucleon count (filling=None)")
    ap.add_argument("--n_b", type=int, default=2)
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.1],
                    help="per-site accuracy targets (MeV/site)")
    ap.add_argument("--filling", type=float, default=None,
                    help="fixed-filling A=round(filling*sites); omit for dilute A")
    ap.add_argument("--ladder-start", type=int, default=500)
    ap.add_argument("--n-rungs", type=int, default=6)
    ap.add_argument("--max-core", type=int, default=8000)
    ap.add_argument("--max-rung-seconds", type=float, default=120.0)
    ap.add_argument("--n_runs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    res = dets_vs_L_at_fixed_accuracy(
        dim=args.dim, L_values=tuple(args.L), A=args.A, n_b=args.n_b,
        eps_persite_targets=tuple(args.eps), filling=args.filling,
        ladder_start=args.ladder_start, n_rungs=args.n_rungs, max_core=args.max_core,
        max_rung_seconds=args.max_rung_seconds, n_runs=args.n_runs, seed=args.seed,
        label=args.label)
    # compact result read-out
    print(f"\n  total wall: {(time.time() - t0) / 60:.1f} min")
    for eps in args.eps:
        f = res["fits"][str(eps)]
        n_pts = f.get("n_points", 0)
        if f.get("ok"):
            ex = f["exponential_in_V"]["slope"]
            po = f["polynomial_in_V"]["slope"]
            print(f"  eps={eps:g}: {n_pts} fit points -> exp gamma={ex:.4g}/site, "
                  f"poly gamma={po:.3g}")
        else:
            print(f"  eps={eps:g}: {f['reason']}")


if __name__ == "__main__":
    main()
