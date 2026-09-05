"""One (L, A, seed) NESTED boson-cutoff shard — the P0-3 methodology upgrade (task 35, T3).

THE PROBLEM. The volume-scaling arm that carries the L=10 cutoff conditional measures
Delta34 = E(n_b=3) - E(n_b=4) by running the two cutoffs as SEPARATE solves. The 2026-09-05
audit lists two objections to that, and they are the same objection twice: the two solves
select their determinant spaces INDEPENDENTLY, so the paired difference still contains a
selection residual -- visible as Delta34 OSCILLATING IN SIGN down the core ladder at the
~0.001-0.003 MeV/site level, which is the size of the signal itself.

THE FIX -- NESTING, not sharing. Warm-start the n_b=4 solve FROM the converged n_b=3 core.
The n_b=4 space is then a strict superset explored from the n_b=3 solution, so

    Delta34 = E_3 - E_4  >=  0   by construction   (sign-definite)

and the selection residual is gone: the n_b=4 solve cannot be in a *different* basin, only
in a *larger* one.

WHY NOT A SHARED (identical) BASIS. Comparing the two cutoffs on the SAME determinant set is
trivially null. For a core whose boson occupations are all < N_f-1, raising the cutoff adds no
reachable state inside the core, so P.H4.P == P.H3.P exactly and the difference is zero by
construction. The cutoff effect lives in the SPACE, not in the matrix elements -- which is
precisely why the comparison has to be nested (superset) rather than shared (identical).
The embedding energy `E4_embed` is computed anyway and reported: it MUST equal E_3, and a
gap would mean the core has population at the n_b=3 boundary (informative in itself).

ONE JOB PER (L, A, seed) runs the whole comparison, so the two cutoffs also share the job,
the host and the phase-0 ensemble -- removing cross-job variability on top of the nesting.

Comparison switch (CLAUDE.md): `--also-independent` additionally runs the OLD independently
selected n_b=4 solve at each rung, so nested-vs-independent is measured, not asserted.

    python -m misc.run_nb_nested_shard --L 2 --A 8 --seed 0 --max-core 128000 --out x.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.utils.manifest import build_manifest
from classical.trimci import build_from_eft
from classical.trimci.backend import cpp_diagonalize_matfree_arrays
from classical.trimci.graph_arrays import (ground_state_arrays,
                                           ground_state_ensemble_arrays)


def _core_energy(H, core):
    """Rayleigh energy of an EXPLICIT determinant set -- the quantity the nesting
    argument needs.

    NOT the same as `GroundStateResult.energy`. `global_trim_arrays` diagonalizes the
    SURVIVOR POOL and then keeps only the top-k rows as the core, so `res.energy` is the
    pool's energy while `res.ferm_arr/bos_arr` is a strictly smaller set (the same
    mismatch `pt2.epstein_nesbet_pt2` re-diagonalizes around). Both are valid Ritz upper
    bounds -- of DIFFERENT spaces -- so a difference of the two is meaningless. Every
    energy entering delta is computed here, over a determinant set we can name.

    The arrays transfer between cutoffs unchanged: an n_b=3 occupation (0..7) is a valid
    n_b=4 occupation (0..15), and the two term lists are identical, so evaluating the
    low-cutoff core under H_hi returns the low-cutoff energy exactly (asserted per rung
    as `embed_gap`)."""
    ferm = np.ascontiguousarray(core[0], dtype=np.uint64)
    bos = np.ascontiguousarray(core[1], dtype=np.uint16)
    energy, _ = cpp_diagonalize_matfree_arrays(H, ferm, bos)
    return float(energy)


def main():
    ap = argparse.ArgumentParser(description="Nested boson-cutoff (n_b lo -> hi) shard")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--A", type=int, required=True, help="nucleon count (explicit, not filling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-b-lo", type=int, default=3)
    ap.add_argument("--n-b-hi", type=int, default=4)
    ap.add_argument("--ladder-start", type=int, default=1000)
    ap.add_argument("--n-rungs", type=int, default=11)
    ap.add_argument("--max-core", type=int, default=262144)
    ap.add_argument("--phase0-runs", type=int, default=32)
    ap.add_argument("--max-rung-seconds", type=float, default=14400.0)
    ap.add_argument("--also-independent", action="store_true",
                    help="ALSO run the legacy independently-selected n_b=hi solve per rung "
                         "(the comparison switch: nested vs independent, measured)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sites = args.L ** args.dim
    t0 = time.time()
    H_lo = build_from_eft(args.L, args.dim, args.n_b_lo, transform="bare")
    H_hi = build_from_eft(args.L, args.dim, args.n_b_hi, transform="bare")

    out = {
        "kind": "nb_nested_shard", "L": args.L, "dim": args.dim, "A": args.A,
        "sites": sites, "seed": args.seed,
        "n_b_lo": args.n_b_lo, "n_b_hi": args.n_b_hi,
        "N_f_lo": H_lo.N_f, "N_f_hi": H_hi.N_f,
        "n_terms": len(H_lo.terms), "also_independent": bool(args.also_independent),
        "method": "NESTED: the high-cutoff solve is warm-started from the low-cutoff core, so "
                  "delta = E_lo - E_hi >= 0 by construction (sign-definite) and carries no "
                  "independent-selection residual.",
        "rungs": [], "done": False,
        "manifest": build_manifest(extra={
            "run": "misc.run_nb_nested_shard", "argv": sys.argv[1:],
            "physical": {"L": args.L, "dim": args.dim, "A": args.A, "sites": sites,
                         "n_b_lo": args.n_b_lo, "n_b_hi": args.n_b_hi,
                         "N_f_lo": H_lo.N_f, "N_f_hi": H_hi.N_f,
                         "n_terms": len(H_lo.terms)},
            "solver": {"ladder_start": args.ladder_start, "n_rungs": args.n_rungs,
                       "max_core": args.max_core, "phase0_runs": args.phase0_runs,
                       "seed": args.seed, "max_rung_seconds": args.max_rung_seconds,
                       "also_independent": bool(args.also_independent)},
            "condor": {k: os.environ.get(k) for k in
                       ("_CONDOR_SLOT", "_CONDOR_REQUEST_CPUS", "_CONDOR_REQUEST_MEMORY")},
        }),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    def save():
        tmp = args.out + ".tmp"
        json.dump(out, open(tmp, "w"), indent=2)
        os.replace(tmp, args.out)

    save()   # header first: a shard that dies in phase 0 still leaves its provenance

    rungs = [args.ladder_start * 2 ** k for k in range(args.n_rungs)
             if args.ladder_start * 2 ** k <= args.max_core]
    core_lo, core_ind = None, None
    for i, n in enumerate(rungs):
        t = time.time()
        # --- low cutoff: phase-0 ensemble at rung 0, then warm-grow (same as the baseline)
        if i == 0:
            r_lo = ground_state_ensemble_arrays(H_lo, n_elec=args.A, n_runs=args.phase0_runs,
                                                n_dets=n, seed=args.seed)
        else:
            r_lo = ground_state_arrays(H_lo, n_elec=args.A, n_dets=n,
                                       initial_core=core_lo, seed=args.seed + i)
        core_lo = (r_lo.ferm_arr, r_lo.bos_arr)
        # E over the SAVED CORE (not r_lo.energy, which is the larger survivor pool's) --
        # every energy in delta must be over a determinant set we can name. See _core_energy.
        E_lo = _core_energy(H_lo, core_lo)

        # --- the embedding check: H_hi on the SAME core must give exactly E_lo
        E_embed = _core_energy(H_hi, core_lo)

        # --- high cutoff, NESTED: same core budget, superset space, warm-started
        r_hi = ground_state_arrays(H_hi, n_elec=args.A, n_dets=n,
                                   initial_core=core_lo, seed=args.seed + i)
        core_hi = (r_hi.ferm_arr, r_hi.bos_arr)
        # the embedded core is itself a variational state of H_hi, so the better of the two
        # is a valid high-cutoff variational energy -- and makes delta >= 0 EXACT rather
        # than merely expected (greedy re-selection is not monotone round-to-round).
        E_hi = min(E_embed, _core_energy(H_hi, core_hi))

        rung = {
            "core": int(r_lo.n_dets), "E_lo": E_lo, "E_hi_nested": E_hi,
            "delta_nested": E_lo - E_hi,
            "delta_nested_per_site": (E_lo - E_hi) / sites,
            "E_hi_embed": E_embed,
            # MUST be ~0: identical term lists, and the core transfers between cutoffs
            # unchanged. A nonzero value means the two Hamiltonians differ on this core.
            "embed_gap": E_embed - E_lo,
            # the pool-convention energy, recorded ONLY for comparability with the legacy
            # volume-scaling shards (which store it); never used in delta.
            "E_lo_pool": float(r_lo.energy),
            "pool_minus_core": float(r_lo.energy) - E_lo,
            "core_hi": int(r_hi.n_dets), "wall_s": time.time() - t,
        }
        if abs(rung["embed_gap"]) > 1e-6 * max(1.0, abs(E_lo)):
            raise AssertionError(
                f"embed_gap={rung['embed_gap']:.3e} at core {rung['core']}: H(n_b={args.n_b_hi}) "
                f"and H(n_b={args.n_b_lo}) disagree on the SAME determinant set. The nesting "
                f"argument does not hold -- do not report this shard.")
        if args.also_independent:
            # The LEGACY arm: an n_b=hi warm-grow ladder of its own, selecting independently
            # of the low-cutoff one -- i.e. exactly what the current volume-scaling data does,
            # kept so nested-vs-independent is a measured number.
            ti = time.time()
            if i == 0:
                r_ind = ground_state_ensemble_arrays(H_hi, n_elec=args.A,
                                                     n_runs=args.phase0_runs, n_dets=n,
                                                     seed=args.seed)
            else:
                r_ind = ground_state_arrays(H_hi, n_elec=args.A, n_dets=n,
                                            initial_core=core_ind, seed=args.seed + i)
            core_ind = (r_ind.ferm_arr, r_ind.bos_arr)
            E_ind = _core_energy(H_hi, (r_ind.ferm_arr, r_ind.bos_arr))
            rung["E_hi_independent"] = E_ind
            rung["delta_independent"] = E_lo - E_ind
            rung["delta_independent_per_site"] = (E_lo - E_ind) / sites
            rung["wall_independent_s"] = time.time() - ti
        out["rungs"].append(rung)
        save()
        print(f"  core={rung['core']:>7}  E_lo={E_lo:12.5f}  E_hi={E_hi:12.5f}  "
              f"delta={rung['delta_nested']:+.5f} ({rung['delta_nested_per_site']:+.6f}/site)  "
              f"embed_gap={rung['embed_gap']:+.2e}  [{rung['wall_s']:.0f}s]", flush=True)
        if rung["wall_s"] > args.max_rung_seconds:
            print(f"  (rung wall {rung['wall_s']:.0f}s > {args.max_rung_seconds:.0f}s — stop)")
            break

    out["done"] = True
    out["wall_s"] = time.time() - t0
    save()
    d = [r["delta_nested_per_site"] for r in out["rungs"]]
    print(f"[nbnested] L={args.L} A={args.A} seed={args.seed} rungs={len(out['rungs'])} "
          f"delta/site: last={d[-1]:+.6f} max={max(d):+.6f} "
          f"wall={out['wall_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
