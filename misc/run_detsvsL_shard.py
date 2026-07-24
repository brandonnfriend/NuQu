"""
One (L, seed) SHARD of the parallel dets-vs-L campaign.

The n_runs ensemble is parallelised across Condor jobs: each shard runs a single
seed (n_runs=1) of one L's independent core ladder, in the compacting per-mode
squeeze frame (transform="gaussian"), and dumps its per-rung energies. A separate
combine step (misc/combine_detsvsL.py) takes the min over seeds per (L, core) to
reconstruct the ensemble, extrapolates the per-L reference, and fits the exponent.

Running the ensemble as parallel shards (not n_runs sequentially in one job) turns
a ~50 h serial run into a few-hour wall clock, and keeps the independent-random-init
requirement (each seed is its own job -- no warm starts).

    python -m misc.run_detsvsL_shard --L 3 --seed 7 --out /path/L3_s7.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.run_cpp import converged_reference


def main():
    ap = argparse.ArgumentParser(description="One (L, seed) dets-vs-L shard")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--A", type=int, default=1, help="dilute nucleon count")
    ap.add_argument("--n_b", type=int, default=2)
    ap.add_argument("--transform", default="gaussian",
                    help="frame axis; 'gaussian' = auto per-mode analytic squeeze (compacting)")
    # In the compacting frame N* is small, so bracket it with a low-start ladder
    # (250->128k, 10 rungs) rather than the bare basis's 8k-start large-core ladder:
    # low rungs bracket N* (they're instant in the frame), high rungs pin E_inf.
    ap.add_argument("--ladder-start", type=int, default=250)
    ap.add_argument("--n-rungs", type=int, default=10)
    ap.add_argument("--max-core", type=int, default=128000)
    ap.add_argument("--max-rung-seconds", type=float, default=7200.0)
    ap.add_argument("--out", required=True, help="output JSON path for this shard")
    args = ap.parse_args()

    t0 = time.time()
    # n_runs=1: this shard IS one ensemble member; the ensemble min is done in combine.
    ref = converged_reference(
        L=args.L, dim=args.dim, A=args.A, n_b=args.n_b, transform=args.transform,
        n_runs=1, seed=args.seed, ladder_start=args.ladder_start, n_rungs=args.n_rungs,
        max_core=args.max_core, max_rung_seconds=args.max_rung_seconds,
        eps_persite_targets=(1.0, 0.1), verbose=True)

    out = {
        "kind": "detsvsL_shard", "L": args.L, "seed": args.seed, "dim": args.dim,
        "A": args.A, "n_b": args.n_b, "N_f": ref["N_f"], "transform": args.transform,
        "sites": args.L ** args.dim, "n_terms": ref["n_terms"], "sector": ref["sector"],
        "rungs": [{k: r[k] for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext")
                   if k in r} for r in ref["rungs"]],
        "wall_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    cores = [r["core"] for r in out["rungs"]]
    print(f"[shard] L={args.L} seed={args.seed} transform={args.transform} "
          f"cores={cores} wall={out['wall_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
