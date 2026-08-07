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
from classical.trimci.frame_qpe import warmstart_overlap
from classical.trimci.observables import occupation_tail, occupation_histogram
from classical.trimci.lf import compactness
from classical.trimci.run_cpp import (_adaptive_ladder_solve, _pick_solver,
                                      three_phase_growing_run)


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
    ap.add_argument("--ladder-n-runs", type=int, default=1,
                    help="ensemble runs PER LADDER RUNG (independent mode); >1 gives robust "
                         "convergence, needed when the boson init is unbiased")
    ap.add_argument("--boson-init-mean", default=None,
                    help="boson init: a float (truncated-geometric mean ~vacuum) or 'none' "
                         "(UNIFORM over [0,N_f), no vacuum anchor = the unbiased control). "
                         "Default keeps the solver's 0.5.")
    ap.add_argument("--pt2-max-core", type=int, default=None,
                    help="skip EN-PT2 once the core exceeds this (its external space ~223x "
                         "core -> ~150GB at 1M, OOMs before the E_var solve). Deep runs set "
                         "this low (e.g. 64000) to reach 1M+ on E_var; PT2 stays on shallow.")
    ap.add_argument("--exact-ref", action="store_true",
                    help="also compute the EXACT ground energy (Lanczos, guarded) as the "
                         "true E_inf for the Tier-1 cost-to-fixed-accuracy anchor. Only "
                         "feasible on small ED systems; records E_exact=None if the guard "
                         "refuses (sector too large).")
    ap.add_argument("--exact-max-mem-gb", type=float, default=24.0,
                    help="memory ceiling for the exact-ref Lanczos (refuses cleanly above)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    bim = 0.5   # solver default (near-vacuum truncated-geometric)
    if args.boson_init_mean is not None:
        bim = None if str(args.boson_init_mean).lower() == "none" else float(args.boson_init_mean)
    t0 = time.time()

    # bare H. The FRAME (any mix of squeeze/LF/COO) is a SINGLE optimization operation fit
    # INSIDE the 3-phase run (Phase 0 discovery + Phase 1 co-evolution); nothing is
    # pre-applied here. A run is just defined by WHICH transforms its frame contains.
    Hbare = build_from_eft(args.L, args.dim, args.n_b, transform="bare")
    solver, pt2_diag, _ = _pick_solver(arrays=True)
    fr = args.frame
    has_gaussian, has_lf, has_coo = "gaussian" in fr, "lf" in fr, "coo" in fr
    finfo = {"has_gaussian": has_gaussian, "has_lf": has_lf, "has_coo": has_coo,
             "method": "bare" if fr == "bare" else " + ".join(
                 x for x in ("squeeze" if has_gaussian else "",
                             "projector-LF" if has_lf else "", "COO" if has_coo else "") if x)}

    # EXACT reference (Tier-1): the true E_inf via guarded Lanczos on the bare H (the
    # spectrum is frame-invariant, so this is the target EVERY frame's E_var must reach).
    # core*(dE to E_exact) is the RIGOROUS cost-to-fixed-accuracy; E_exact=None when the
    # sector is too large for ED (the guard refuses cleanly), i.e. not a Tier-1 anchor.
    E_exact, exact_support = None, None
    if args.exact_ref:
        try:
            from classical.trimci.lanczos import lanczos_ground_state
            lz = lanczos_ground_state(Hbare, n_elec=A, return_vec=True,
                                      max_mem_gb=args.exact_max_mem_gb)
            E_exact = float(getattr(lz, "energy", lz[0]))
            vec = getattr(lz, "coeffs", lz[1] if isinstance(lz, tuple) and len(lz) > 1 else None)
            if vec is not None:
                vals = list(vec.values()) if hasattr(vec, "values") else vec
                exact_support = compactness(np.asarray(vals), fracs=(0.9, 0.99, 0.999, 0.9999))
        except MemoryError:
            E_exact = None
        except Exception:
            E_exact = None

    out = {
        "E_exact": E_exact, "exact_support": exact_support,
        "kind": "frame_shard", "L": args.L, "dim": args.dim, "A": A,
        "filling": args.filling, "frame": args.frame, "seed": args.seed,
        "n_b": args.n_b, "N_f": Hbare.N_f, "sites": sites,
        "n_terms": len(Hbare.terms), "ladder_mode": args.ladder_mode,
        "phase0_runs": args.phase0_runs, "ladder_n_runs": args.ladder_n_runs,
        "boson_init_mean": ("none" if bim is None else bim),
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
        r["mean_occ"] = _mean_occupation(res, Hbare.n_bos_modes)
        # QPE warm-start overlap p0 = |c_dominant|^2 of this (framed) core — the
        # quantity the frame->QPE bridge (task 34, I1) needs to fold ~1/p0 QPE
        # repetitions into the cost. Emitting it per-rung means future cluster
        # frame runs carry the warm-start data natively (the core itself is
        # discarded once the shard finishes). Best-effort.
        try:
            r["p0"] = warmstart_overlap(res.coeffs)
        except Exception:
            r["p0"] = None
        # Boson-cutoff TAIL: leaked GS weight a per-mode Fock cutoff N_f would DROP
        # (n_b=1 -> N_f=2 cut, n_b=2 -> N_f=4 cut), plus the per-mode occupation
        # histogram p(n). Quantifies how much probability our small n_b truncates --
        # the n_b=2-vs-1 justification. NOTE: only resolves the tail beyond level
        # (N_f-1) when the run itself uses a LARGER cutoff, so drive with --n_b>=3.
        try:
            r["occ_tail"] = occupation_tail(res, [2, 3, 4, 5, 6, 8])
            r["occ_hist"] = occupation_histogram(res).tolist()
        except Exception:
            r["occ_tail"] = None
            r["occ_hist"] = None
        # SUPPORT metrics (Tier-2 classical-cost measure): the effective # of
        # determinants the ground state lives on. participation_ratio = 1/Sum|c|^4,
        # and n{90,99,99.9,99.99} = # dets holding that cumulative weight. Tracked vs
        # core, these say whether the support has CONVERGED (plateaued -> that's the
        # classical cost) or is still growing (state not captured); their growth rate +
        # the sorted-coeff decay extrapolate the core needed at a given weight cutoff --
        # the cost claim that does NOT need the (unreachable) energy E_inf. A more
        # compact frame shrinks these (Goal 1 -> Goal 3).
        try:
            c = compactness(res.coeffs, fracs=(0.9, 0.99, 0.999, 0.9999))
            r["support"] = {k: c[k] for k in c}
        except Exception:
            r["support"] = None
        out["rungs"].append(r)
        out["wall_s"] = time.time() - t0
        save()   # INCREMENTAL: survive an OOM/timeout on the next (deeper) rung

    if args.ladder_mode == "grow":
        # Phase 2 rungs = 16k up to max_core (Phase 0 ~1k + Phase 1 2k/4k/8k are internal).
        p2, r = [], 16000
        while r <= args.max_core:
            p2.append(r)
            r *= 2
        three_phase_growing_run(
            Hbare, A, has_gaussian, has_lf, has_coo, phase0_core=1000,
            phase0_runs=args.phase0_runs, phase0_cycles=args.orbopt_cycles,
            phase1_cores=(2000, 4000, 8000), phase1_runs=args.frame_runs, phase1_cycles=3,
            phase2_rungs=p2, pt2_diag=pt2_diag, seed=args.seed, verbose=True, on_rung=on_rung,
            max_rung_seconds=args.max_rung_seconds)
    else:
        # comparison switch: fit the frame ONCE (Phase-0 only) then independent solves
        Hind, _ = frame_workflow.optimize_frame(
            Hbare, A, args.phase0_core, has_gaussian=has_gaussian, has_lf=has_lf,
            has_coo=has_coo, num_runs=args.frame_runs, cycles=args.orbopt_cycles, seed=args.seed)
        _adaptive_ladder_solve(
            Hind, A, args.ladder_start, args.n_rungs, solver, pt2_diag,
            max_core=args.max_core, max_rung_seconds=args.max_rung_seconds,
            n_runs=args.ladder_n_runs, seed=args.seed, verbose=True, on_rung=on_rung,
            boson_init_mean=bim, pt2_max_core=args.pt2_max_core)

    out["done"] = True
    out["wall_s"] = time.time() - t0
    save()
    print(f"[frameshard] L={args.L} A={A} frame={args.frame} seed={args.seed} "
          f"rungs={[r['core'] for r in out['rungs']]} wall={out['wall_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
