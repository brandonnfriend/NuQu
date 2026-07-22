"""
Higher-A frame comparison, TrimCI-ALIGNED (supersedes run_frame_comparison_highA.py).

The earlier version compared frames with a single EXPENSIVE fixed-1500-det solve and a
ONE-SHOT COO — variance-limited and unreliable at high A. This one follows TrimCI's
recommended Phase-0 method (TrimCI_skill.py):
  * frame SELECTION via CHEAP stochastic probes — best-of-`num_runs` at a SMALL core
    (`probe_frame`); a more compact frame reaches a lower best-energy at the same tiny
    budget, and best-of-N cuts through the single-run ensemble variance.
  * COO via the ITERATIVE orbopt loop (`coo_orbopt`), not one-shot — the natural-orbital
    co-optimization that turned COO from a liability (−5 MeV at A=6) into a gain (+46 MeV).

Frames: bare, per-mode squeeze (analytic), COO (iterative orbopt). Metric per A: probe
best-energy and the gap bare−frame (positive = the frame compacts). Bogoliubov (122k terms)
and projector-LF (term-explosive, NO-GO) are documented elsewhere, not swept here.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci import build_from_eft, frame
from classical.trimci import frame_workflow as fw
from classical.trimci.run_cpp import _solver

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "frame_selection_highA_summary.json")

# Laptop-feasible range: at A>=8-10 the per-solve floor (connection-building over the
# sector) dominates, so even best-of-N probes cost minutes each -> HPC territory. The
# aligned method itself is A-independent; only the wall-clock is the limit.
A_LADDER = (2, 4, 6, 8)                            # squeeze/bare probes (cheap) at all A
# COO orbopt is BUDGET-SENSITIVE: with too few dets/runs the 1-RDM is noisy and the
# accept-if-better gate can't see COO's gain through the best-of-N variance -> 0 cycles
# (measured: core=250,num_runs=12 gave 0 cycles at every A; core=500,num_runs=32 gave
# +46 MeV at A=6). TrimCI recommends max_final_dets=1000-10000, num_runs=64-256 for orbopt.
# So run COO with a TrimCI-scale budget on a SMALL A subset (it's HPC-expensive at A>=8).
COO_A = (4, 6)
PROBE = {"n_probe": 100, "num_runs": 16}          # Phase-0 selection budget
COO = {"core": 500, "num_runs": 32, "cycles": 5}  # TrimCI-scale orbopt budget


def run():
    solve = _solver(True)
    H = build_from_eft(2, 3, 3)
    r, phi = frame.analytic_squeeze(H)
    Hsq = frame.squeeze_terms(H, -r, phi)
    print(f"L=2 d=3 n_b=3: bare={len(H.terms)} squeeze={len(Hsq.terms)} terms; "
          f"probe={PROBE}, coo={COO}", flush=True)

    rows = []
    for A in A_LADDER:
        t = time.time()
        pb = fw.probe_frame(H, A, solve=solve, **PROBE)
        ps = fw.probe_frame(Hsq, A, solve=solve, **PROBE)
        row = {"A": A, "best_bare": pb["best"], "best_squeeze": ps["best"],
               "spread_bare": pb["spread"], "gap_squeeze": pb["best"] - ps["best"]}
        msg = (f"A={A:>2}: bare {pb['best']:8.2f} | squeeze {ps['best']:8.2f} "
               f"(gap {row['gap_squeeze']:+6.2f})")
        if A in COO_A:                                  # COO: TrimCI-scale orbopt, subset A
            oo = fw.coo_orbopt(H, A, solve=solve, **COO)
            pc = fw.probe_frame(oo["H_frame"], A, solve=solve, **PROBE)
            row.update({"best_coo": pc["best"], "gap_coo": pb["best"] - pc["best"],
                        "coo_cycles": oo["cycles_run"], "coo_terms": len(oo["H_frame"].terms)})
            msg += f" | COO {pc['best']:8.2f} (gap {row['gap_coo']:+6.2f}, {oo['cycles_run']}cyc)"
        row["t"] = time.time() - t
        rows.append(row)
        print(msg + f"  [{row['t']:.0f}s]", flush=True)

    out = {"L": 2, "dim": 3, "n_b": 3, "probe": PROBE, "coo": COO,
           "A_ladder": list(A_LADDER), "rows": rows}
    json.dump(out, open(SUMMARY, "w"), indent=2)
    _plot(out)
    print("\n=== ALIGNED VERDICT (Phase-0 probe best-energy; gap = bare - frame, MeV) ===",
          flush=True)
    print(f"  {'A':>3} {'gap_squeeze':>12} {'gap_COO':>9} {'winner':>10}", flush=True)
    for r in rows:
        cands = [("bare", r["best_bare"]), ("squeeze", r["best_squeeze"])]
        if "best_coo" in r:
            cands.append(("COO", r["best_coo"]))
        w = min(cands, key=lambda kv: kv[1])[0]
        gc = f"{r['gap_coo']:+.2f}" if "gap_coo" in r else "--"
        print(f"  {r['A']:>3} {r['gap_squeeze']:>+12.2f} {gc:>9} {w:>10}", flush=True)
    return out


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    A = [r["A"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(A, [r["best_bare"] for r in rows], "o-", color="C0", ms=8, label="bare")
    ax[0].plot(A, [r["best_squeeze"] for r in rows], "s-", color="C1", ms=8, label="squeeze")
    ax[0].plot(A, [r["best_coo"] for r in rows], "^-", color="C2", ms=9, label="COO (iterative)")
    ax[0].set_xlabel("A (nucleons)"); ax[0].set_ylabel("Phase-0 probe best-E (MeV)")
    ax[0].set_title("Frame selection (best-of-N @ small core)\n"
                    "lower = more compact frame  [L=2 d=3]")
    ax[0].legend(); ax[0].grid(alpha=0.25)
    ax[1].plot(A, [r["gap_squeeze"] for r in rows], "s-", color="C1", ms=8, label="squeeze")
    ac = [r["A"] for r in rows if "gap_coo" in r]
    gc = [r["gap_coo"] for r in rows if "gap_coo" in r]
    if ac:
        ax[1].plot(ac, gc, "^", color="C2", ms=12, label="COO (TrimCI-budget orbopt)")
    ax[1].axhline(0, color="gray", lw=1, ls=":")
    ax[1].set_xlabel("A (nucleons)"); ax[1].set_ylabel("gap = bare - frame (MeV)")
    ax[1].set_title("Compaction gain vs A (aligned method)\n"
                    "positive = frame helps; robust best-of-N")
    ax[1].legend(); ax[1].grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "frame_selection_highA.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


if __name__ == "__main__":
    run()
