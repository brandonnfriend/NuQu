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
from classical.trimci.run_cpp import _adaptive_ladder_solve, _pick_solver, growing_ladder


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
                    help="bare | gaussian | coo | gaussian+coo | lf | gaussian+lf | "
                         "gaussian+coo+lf | bogoliubov")
    ap.add_argument("--A", type=int, default=1, help="nucleon count (dilute)")
    ap.add_argument("--filling", type=float, default=None,
                    help="if set, A = round(filling * sites) (overrides --A)")
    ap.add_argument("--ladder-mode", default="grow", choices=["grow", "independent"],
                    help="'grow' = Phase-0 heavy ensemble then warm-start growth (default); "
                         "'independent' = the old from-scratch solve per rung")
    ap.add_argument("--ladder-start", type=int, default=1000,
                    help="smallest core = the Phase-0 ensemble core (grow mode)")
    ap.add_argument("--n-rungs", type=int, default=11, help="1000 x2^10 -> 1,024,000")
    ap.add_argument("--max-core", type=int, default=1024000)
    ap.add_argument("--max-rung-seconds", type=float, default=14400.0, help="4h/rung cap")
    ap.add_argument("--phase0-runs", type=int, default=64,
                    help="Phase-0 ensemble seeds (heavy small-core search; grow mode)")
    # frame-optimization (COO orbopt / LF displacement) knobs
    ap.add_argument("--phase0-core", type=int, default=2000, help="frame-opt core")
    ap.add_argument("--frame-runs", type=int, default=16, help="COO orbopt num_runs")
    ap.add_argument("--orbopt-cycles", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    t0 = time.time()

    # bare H, then the frame
    Hbare = build_from_eft(args.L, args.dim, args.n_b, transform="bare")
    solver, pt2_diag, _ = _pick_solver(arrays=True)
    fr = args.frame
    if fr in ("lf", "gaussian+lf", "gaussian+coo+lf"):
        # Lang-Firsov (polaron) frames: projector-conditioned displacement that removes
        # the LINEAR fermion-boson coupling H_AV, amplitude optimized via selected-CI.
        # squeeze the bosons FIRST (all but pure "lf"); "+coo" then rotates the fermion
        # orbitals on the squeezed+displaced H -> all three transforms stacked.
        Hb = Hbare
        if fr != "lf":
            r, phi = frame.analytic_squeeze(Hbare)
            Hb = frame.squeeze_terms(Hbare, -r, phi)
        best = frame.optimize_displacement(Hb, A, core=args.phase0_core, seed=args.seed)
        Hlf = frame.displace_terms(Hb, lambdas=best["scale"], gen=best["gen"])
        if fr == "gaussian+coo+lf":
            oo = frame_workflow.coo_orbopt(Hlf, A, core=args.phase0_core,
                                           num_runs=args.frame_runs, cycles=args.orbopt_cycles,
                                           seed=args.seed, solve=solver)
            H_frame = oo["H_frame"]
            method = "squeeze + projector-LF + COO (all three)"
        else:
            H_frame = Hlf
            method = "projector-LF" + (" (squeeze+polaron)" if fr == "gaussian+lf" else " (polaron)")
        finfo = {"method": method, "lf_scale": best["scale"]}
    else:
        H_frame, finfo = frame_workflow._build_frame(
            Hbare, A, fr, phase0_core=args.phase0_core, phase0_runs=args.frame_runs,
            orbopt_cycles=args.orbopt_cycles, seed=args.seed, verbose=True, solve=solver)

    out = {
        "kind": "frame_shard", "L": args.L, "dim": args.dim, "A": A,
        "filling": args.filling, "frame": args.frame, "seed": args.seed,
        "n_b": args.n_b, "N_f": H_frame.N_f, "sites": sites,
        "n_terms": len(H_frame.terms), "ladder_mode": args.ladder_mode,
        "phase0_runs": args.phase0_runs,
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
        r = {k: rung[k] for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext", "wall_s", "phase")
             if k in rung}
        r["mean_occ"] = _mean_occupation(res, H_frame.n_bos_modes)
        out["rungs"].append(r)
        out["wall_s"] = time.time() - t0
        save()   # INCREMENTAL: survive an OOM/timeout on the next (deeper) rung

    if args.ladder_mode == "grow":
        rungs, r = [], args.ladder_start
        while r <= args.max_core:
            rungs.append(r)
            r *= 2
        growing_ladder(H_frame, A, rungs, phase0_runs=args.phase0_runs, seed=args.seed,
                       pt2_diag=pt2_diag, verbose=True, on_rung=on_rung,
                       max_rung_seconds=args.max_rung_seconds)
    else:
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
