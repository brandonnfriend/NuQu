"""
One (frame, L, A/filling, seed) SHARD — the unified deep-run + frame-comparison worker.

Builds the bare EFT Hamiltonian, applies a FRAME (bare | gaussian squeeze | COO
fermionic orbital optimization | gaussian+COO | bogoliubov) via
frame_workflow._build_frame, then runs an independent core ladder for one seed
(n_runs=1). Writes its per-rung energies to JSON and RE-SAVES after every rung, so a
shard that OOMs or times out at a deep rung still keeps everything it finished (needed
for the 1M-state push where big-L shards won't reach the top).

Two campaigns use it:
  * deep dilute (Task 1): --frame gaussian --A 1 --max-core 1024000   (certainly converge)
  * frame x filling (Task 2): --filling 0.5 --frame {bare,gaussian,coo,gaussian+coo}
    -- how fermionic (COO) vs boson (squeeze) frames help as nucleon interaction grows.

    python -m misc.run_frame_shard --L 3 --seed 0 --frame coo --filling 1.0 --out x.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci import build_from_eft, frame_workflow, frame
from classical.trimci.run_cpp import _adaptive_ladder_solve, _pick_solver


def _mean_occupation(res, n_bos):
    """Mean boson occupation per mode of the (framed) ground state -- a near-vacuum
    diagnostic. Basis-dependent: ~0 in the squeeze frame (by design), physical in the
    bare frame. Best-effort (returns None if the result lacks arrays)."""
    try:
        c2 = np.abs(np.asarray(res.coeffs, dtype=complex)) ** 2
        tot = c2.sum()
        if tot <= 0:
            return None
        nbos = np.asarray(res.bos_arr).sum(axis=1)
        return float((c2 * nbos).sum() / (tot * n_bos))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="One frame/L/seed dets-vs-L shard")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n_b", type=int, default=2)
    ap.add_argument("--frame", default="gaussian",
                    help="bare | gaussian | coo | gaussian+coo | bogoliubov")
    ap.add_argument("--A", type=int, default=1, help="nucleon count (dilute)")
    ap.add_argument("--filling", type=float, default=None,
                    help="if set, A = round(filling * sites) (overrides --A)")
    ap.add_argument("--ladder-start", type=int, default=250)
    ap.add_argument("--n-rungs", type=int, default=13, help="250 x2^12 -> 1,024,000")
    ap.add_argument("--max-core", type=int, default=1024000)
    ap.add_argument("--max-rung-seconds", type=float, default=14400.0, help="4h/rung cap")
    # COO orbital-optimization (Phase-0) knobs — ignored by the analytic boson frames
    ap.add_argument("--phase0-core", type=int, default=2000)
    ap.add_argument("--phase0-runs", type=int, default=8)
    ap.add_argument("--orbopt-cycles", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    t0 = time.time()

    # bare H, then the frame
    Hbare = build_from_eft(args.L, args.dim, args.n_b, transform="bare")
    solver, pt2_diag, _ = _pick_solver(arrays=True)
    if args.frame in ("lf", "gaussian+lf"):
        # projector-conditioned Lang-Firsov: the polaron displacement that removes the
        # LINEAR fermion-boson coupling (H_AV), variationally optimized via selected-CI
        # (scales past ED). "gaussian+lf" squeezes the bosons FIRST, then applies LF.
        Hb = Hbare
        combined = (args.frame == "gaussian+lf")
        if combined:
            r, phi = frame.analytic_squeeze(Hbare)
            Hb = frame.squeeze_terms(Hbare, -r, phi)
        best = frame.optimize_displacement(Hb, A, core=args.phase0_core, seed=args.seed)
        H_frame = frame.displace_terms(Hb, lambdas=best["scale"], gen=best["gen"])
        finfo = {"method": "projector-LF" + (" (squeeze+polaron)" if combined else " (polaron)"),
                 "lf_scale": best["scale"], "lf_opt_energy": best["energy"]}
    else:
        H_frame, finfo = frame_workflow._build_frame(
            Hbare, A, args.frame, phase0_core=args.phase0_core, phase0_runs=args.phase0_runs,
            orbopt_cycles=args.orbopt_cycles, seed=args.seed, verbose=True, solve=solver)

    out = {
        "kind": "frame_shard", "L": args.L, "dim": args.dim, "A": A,
        "filling": args.filling, "frame": args.frame, "seed": args.seed,
        "n_b": args.n_b, "N_f": H_frame.N_f, "sites": sites,
        "n_terms": len(H_frame.terms),
        "frame_info": {k: v for k, v in finfo.items()
                       if isinstance(v, (str, int, float, bool))},
        "rungs": [], "done": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, args.out)   # atomic — a reader never sees a half-written file

    save()  # header immediately: even a shard that dies in Phase-0 leaves a record

    def on_rung(rung, res):
        r = {k: rung[k] for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext", "wall_s")
             if k in rung}
        r["mean_occ"] = _mean_occupation(res, H_frame.n_bos_modes)
        out["rungs"].append(r)
        out["wall_s"] = time.time() - t0
        save()   # INCREMENTAL: survive an OOM/timeout on the next (deeper) rung

    _adaptive_ladder_solve(
        H_frame, A, args.ladder_start, args.n_rungs, solver, pt2_diag,
        max_core=args.max_core, max_rung_seconds=args.max_rung_seconds,
        n_runs=1, seed=args.seed, verbose=True, on_rung=on_rung)

    out["done"] = True
    out["wall_s"] = time.time() - t0
    save()
    print(f"[frameshard] L={args.L} A={A} frame={args.frame} seed={args.seed} "
          f"rungs={[r['core'] for r in out['rungs']]} wall={out['wall_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
