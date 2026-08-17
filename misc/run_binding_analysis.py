"""Aggregate the fixed-A box-convergence campaign into BINDING ENERGIES + the physics plot.

Reads the sector shards (`<frame>_L<L>d<dim>_A<A>_s<seed>.json` from
submit_box_convergence.sh), extracts each sector's converged energy E(A,L) ± sigma
(extrapolated from its warm-grow curve), forms

    BE(A,L) = A*E(1,L) - (A-1)*E(0,L) - E(A,L)     (the +202.5*sites vacuum cancels)

for A=2 (deuteron) and A=4 (alpha), finite-volume extrapolates BE vs box side, and plots
BE vs box size (fm) with experimental reference lines. Pure analysis -- safe anywhere.

    python -m misc.run_binding_analysis --shard-dir data/.../shards --out-fig results/figures/binding.png
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from classical.trimci.binding import (A_L_FM, EXPT_BE, box_convergence,
                                       finite_volume_extrapolate, vacuum_constant)
from classical.trimci.cost import extrapolate_uncertainty


def sector_energy(shard):
    """Converged E ± sigma for one sector from its warm-grow rungs (extrapolate the tail;
    fall back to the deepest E_var with sigma = last monotone drop)."""
    rs = [r for r in shard["rungs"] if r.get("E_var") is not None]
    E = np.array([r["E_var"] for r in rs], float)
    # These dilute curves are near a PLATEAU (unlike the far-from-converged filling=1.0
    # runs), so the deepest E_var is the estimate and the last monotone drop is the
    # convergence-residual sigma. A power-law extrapolation is inappropriate here and
    # manufactures a spurious far-off E_inf with a huge band.
    sig = abs(E[-1] - E[-2]) if len(E) >= 2 else 1.0
    return float(E[-1]), float(sig)


def load(shard_dir, frame="gaussian", dim=3):
    """{L: {A: (E, sigma)}} from the campaign shards."""
    se = {}
    for f in glob.glob(os.path.join(shard_dir, f"{frame}_L*d{dim}_A*_s*.json")):
        m = re.search(r"_L(\d+)d\d+_A(\d+)_", os.path.basename(f))
        if not m:
            continue
        L, A = int(m.group(1)), int(m.group(2))
        d = json.load(open(f))
        if not [r for r in d["rungs"] if r.get("E_var") is not None]:
            continue
        se.setdefault(L, {})[A] = sector_energy(d)
    return se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--frame", default="gaussian")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--targets", default="2,4", help="comma A values (need A=0,1 present)")
    ap.add_argument("--out-fig", default="results/figures/binding_convergence.png")
    args = ap.parse_args()

    se = load(args.shard_dir, args.frame, args.dim)
    targets = [int(x) for x in args.targets.split(",")]
    print(f"loaded sectors: {{L: A's}} = { {L: sorted(se[L]) for L in sorted(se)} }")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5))
    colors = {2: "#0072B2", 4: "#D55E00"}
    summary = {}
    for A in targets:
        rows = box_convergence(se, A, dim=args.dim, a_L=A_L_FM)
        if not rows:
            print(f"A={A}: no complete boxes (need A=0,1,{A})")
            continue
        box = [r["box_fm"] for r in rows]
        be = [r["BE"] for r in rows]
        err = [r["BE_sigma"] for r in rows]
        name = {2: "deuteron", 4: "alpha", 3: "triton"}.get(A, f"A={A}")
        ax.errorbar(box, be, yerr=err, fmt="o-", color=colors.get(A, "#333"), capsize=4,
                    label=f"{name} (A={A})")
        fv = finite_volume_extrapolate(rows)
        summary[A] = {"rows": rows, "fv": fv}
        print(f"\n=== A={A} ({name}) ===")
        for r in rows:
            print(f"  L={r['L']} box={r['box_fm']:.1f} fm  BE={r['BE']:.2f} ± {r['BE_sigma']:.2f} MeV")
        if fv:
            print(f"  finite-volume BE_inf = {fv['BE_inf']:.2f} MeV (kappa={fv['kappa']:.2f}/fm, r2={fv['r2']:.3f})")
            ax.axhline(fv["BE_inf"], color=colors.get(A, "#333"), ls="--", lw=1, alpha=.6)
        if A in EXPT_BE:
            ax.axhline(EXPT_BE[A], color=colors.get(A, "#333"), ls=":", lw=1.2, alpha=.9)
            print(f"  experiment: {EXPT_BE[A]} MeV (lattice a=2.2fm + unfit LECs => not expected to match)")

    ax.set_xlabel("box side  L·a$_L$  (fm)")
    ax.set_ylabel("binding energy  BE(A)  (MeV)")
    ax.set_title("Fixed-A box convergence (dynamical-pion EFT, Watson params)\n"
                 "dashed = finite-volume extrap · dotted = experiment", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_fig)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=130)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
