"""
Frame-optimization showcase — quantify HOW MUCH the per-mode squeeze frame helps the
TrimCI solver, along four independent axes, in one combined figure.

  A. amplitude scan  — E at fixed core vs squeeze fraction (0=bare, 1=analytic r*):
                       a well with its minimum at r* proves the frame works AND that
                       the closed-form analytic squeeze is (near-)optimal.
  B. advantage vs L  — bare−squeeze energy gap at a fixed core, across L: does the
                       benefit grow with system size?
  C. compaction vs L — participation ratio, mean boson ⟨n⟩, warm-start overlap p0
                       (bare vs squeeze): the physical reason the frame helps.
  D. convergence     — reuse sweep 1 (E vs core, bare vs squeeze) for the core-to-
                       accuracy / speedup number.

All runs are TrimCI at realistic size (L=2/3/4 d=3, n_b=3), thread-capped, ~13 min.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci import frame, build_from_eft
from classical.trimci.run_cpp import _solver
from classical.trimci.lf import compactness
from classical.trimci.frame_qpe import warmstart_overlap, mean_boson_number

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "frame_showcase_summary.json")
SWEEP1 = os.path.join(OUT_DIR, "L2_convergence_frame_summary.json")

CORE_B = 3000        # fixed core for parts B/C
CORE_A = 2500        # fixed core for the amplitude scan


def mr(c):
    return max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)


def metrics(res):
    """Compaction metrics from a solved core."""
    c = compactness(res.coeffs)
    return {"E": float(res.energy), "core": int(res.n_dets),
            "PR": c["participation_ratio"], "n99": c["n99"],
            "mean_n": mean_boson_number(res.coeffs, res.bos_arr),
            "p0": warmstart_overlap(res.coeffs)}


def part_A(solve):
    """Amplitude scan at L=2: E(fixed core) vs squeeze fraction of the analytic r*."""
    print("PART A: squeeze-amplitude scan (L=2 d=3, core=%d)" % CORE_A, flush=True)
    H = build_from_eft(2, 3, 3)
    r, phi = frame.analytic_squeeze(H)
    fracs = np.linspace(-0.4, 1.8, 12)
    rows = []
    for f in fracs:
        Hf = frame.squeeze_terms(H, -f * r, phi)     # f=0 bare, f=1 analytic r* (compacting sign)
        res = solve(Hf, n_elec=1, n_dets=CORE_A, seed=0, n_runs=3, max_rounds=mr(CORE_A))
        rows.append({"frac": float(f), "E": float(res.energy)})
        print(f"  frac={f:+.2f}  E0={res.energy:.3f}", flush=True)
    return rows


def part_BC(solve):
    """Bare vs squeeze at fixed core across L: energy gap + compaction metrics."""
    print("PART B/C: bare vs squeeze at fixed core=%d, L=1..4" % CORE_B, flush=True)
    rows = []
    for L in (1, 2, 3, 4):
        H = build_from_eft(L, 3, 3)
        r, phi = frame.analytic_squeeze(H)
        Hg = frame.squeeze_terms(H, -r, phi)
        mb = metrics(solve(H,  n_elec=1, n_dets=CORE_B, seed=0, n_runs=3, max_rounds=mr(CORE_B)))
        ms = metrics(solve(Hg, n_elec=1, n_dets=CORE_B, seed=0, n_runs=3, max_rounds=mr(CORE_B)))
        rows.append({"L": L, "sites": L ** 3, "bare": mb, "squeeze": ms,
                     "gap_MeV": mb["E"] - ms["E"]})
        print(f"  L={L}: bare E={mb['E']:.2f} PR={mb['PR']:.1f} <n>={mb['mean_n']:.2f} p0={mb['p0']:.2f} | "
              f"squeeze E={ms['E']:.2f} PR={ms['PR']:.1f} <n>={ms['mean_n']:.2f} p0={ms['p0']:.2f} | "
              f"gap={mb['E']-ms['E']:.2f} MeV", flush=True)
    return rows


def core_to_accuracy(rows, tol=1.0):
    """Smallest core within `tol` MeV of that series' best (largest-core) energy."""
    best = rows[-1]["E"]
    for r in rows:
        if abs(r["E"] - best) < tol:
            return r["core"]
    return rows[-1]["core"]


def run():
    solve = _solver(True)
    t0 = time.time()
    A = part_A(solve)
    BC = part_BC(solve)
    sweep1 = json.load(open(SWEEP1)) if os.path.exists(SWEEP1) else None
    out = {"amplitude_scan": A, "vs_L": BC, "core_A": CORE_A, "core_B": CORE_B,
           "wall_s": time.time() - t0}
    if sweep1:
        out["conv_core_to_1MeV"] = {
            "bare": core_to_accuracy(sweep1["bare"]),
            "squeeze": core_to_accuracy(sweep1["squeeze"])}
    json.dump(out, open(SUMMARY, "w"), indent=2)
    _plot(out, sweep1)
    _analyze(out)


def _plot(out, sweep1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # A: amplitude scan
    fr = [r["frac"] for r in out["amplitude_scan"]]
    Ea = [r["E"] for r in out["amplitude_scan"]]
    ax[0, 0].plot(fr, Ea, "o-")
    ax[0, 0].axvline(1.0, ls=":", color="C2", label="analytic r*")
    ax[0, 0].axvline(0.0, ls=":", color="gray", label="bare")
    ax[0, 0].set_xlabel("squeeze fraction of r*"); ax[0, 0].set_ylabel("E0 @ fixed core (MeV)")
    ax[0, 0].set_title(f"A. amplitude scan (L=2, core={out['core_A']})"); ax[0, 0].legend(fontsize=8)

    rows = out["vs_L"]; Ls = [r["L"] for r in rows]
    # B: energy gap vs L
    ax[0, 1].plot(Ls, [r["gap_MeV"] for r in rows], "s-", color="C3")
    ax[0, 1].set_xlabel("L"); ax[0, 1].set_ylabel("bare − squeeze  E0 (MeV)")
    ax[0, 1].set_title(f"B. frame advantage vs L (core={out['core_B']})")

    # C1: participation ratio vs L
    ax[0, 2].semilogy(Ls, [r["bare"]["PR"] for r in rows], "o-", label="bare")
    ax[0, 2].semilogy(Ls, [r["squeeze"]["PR"] for r in rows], "s-", label="squeeze")
    ax[0, 2].set_xlabel("L"); ax[0, 2].set_ylabel("participation ratio")
    ax[0, 2].set_title("C. state spread (eff. #dets)"); ax[0, 2].legend(fontsize=8)

    # C2: mean boson number vs L
    ax[1, 0].plot(Ls, [r["bare"]["mean_n"] for r in rows], "o-", label="bare")
    ax[1, 0].plot(Ls, [r["squeeze"]["mean_n"] for r in rows], "s-", label="squeeze")
    ax[1, 0].set_xlabel("L"); ax[1, 0].set_ylabel("mean boson ⟨n⟩")
    ax[1, 0].set_title("C. boson occupation"); ax[1, 0].legend(fontsize=8)

    # C3: warm-start overlap p0 vs L
    ax[1, 1].plot(Ls, [r["bare"]["p0"] for r in rows], "o-", label="bare")
    ax[1, 1].plot(Ls, [r["squeeze"]["p0"] for r in rows], "s-", label="squeeze")
    ax[1, 1].set_xlabel("L"); ax[1, 1].set_ylabel("warm-start overlap p0")
    ax[1, 1].set_title("C. QPE p0 (dominant weight)"); ax[1, 1].legend(fontsize=8)

    # D: convergence (sweep 1)
    if sweep1:
        for name, mk in (("bare", "o-"), ("squeeze", "s-")):
            cs = [x["core"] for x in sweep1[name]]
            ax[1, 2].plot(cs, [x["E"] for x in sweep1[name]], mk, label=name)
        ax[1, 2].set_xscale("log")
        ax[1, 2].set_xlabel("core (determinants)"); ax[1, 2].set_ylabel("E0 (MeV)")
        ax[1, 2].set_title("D. convergence vs core (L=2)"); ax[1, 2].legend(fontsize=8)

    for a in ax.flat:
        a.grid(True, which="both", alpha=0.25)
    fig.suptitle("Per-mode squeeze frame: how much it helps the TrimCI solver", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT_DIR, "frame_showcase.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


def _analyze(out):
    print("\n=== FRAME SHOWCASE ANALYSIS ===", flush=True)
    A = out["amplitude_scan"]
    imin = int(np.argmin([r["E"] for r in A]))
    print(f"  A. amplitude: E minimised at frac={A[imin]['frac']:+.2f} "
          f"(analytic r* is frac=1) -> analytic seed near-optimal; "
          f"bare(frac=0) E={next(r['E'] for r in A if abs(r['frac'])<1e-9):.1f} vs min {A[imin]['E']:.1f}",
          flush=True)
    for r in out["vs_L"]:
        print(f"  B/C L={r['L']}: gap={r['gap_MeV']:.2f} MeV; PR {r['bare']['PR']:.1f}->{r['squeeze']['PR']:.1f}; "
              f"<n> {r['bare']['mean_n']:.2f}->{r['squeeze']['mean_n']:.2f}; "
              f"p0 {r['bare']['p0']:.2f}->{r['squeeze']['p0']:.2f}", flush=True)
    if "conv_core_to_1MeV" in out:
        cb, cs = out["conv_core_to_1MeV"]["bare"], out["conv_core_to_1MeV"]["squeeze"]
        print(f"  D. core-to-1MeV (L=2): bare>={cb}, squeeze={cs} -> "
              f"~{cb/max(cs,1):.0f}x fewer determinants", flush=True)
    print(f"  total wall: {out['wall_s']/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
