"""
Sweep 2 — L-scaling (L=1..5, d=3, n_b=3) of the per-mode squeeze frame at a LARGE
core, now that the optimized solver reaches ~20k determinants.

Fixed 20k is only feasible up to L=3 on the laptop (L=4 ~56 min, L=5 ~3.3 hr at 20k),
so the core is ADAPTIVE: 20k for L≤2, then the largest core that fits a ~5 min/L cap
for the exponentially larger L=3,4,5 systems. These high-L points do NOT converge (by
design) — the accuracy proxy is the in-run last-doubling relative energy drop |ΔE/E|
(smaller = closer to converged). bare anchors at L=1,2 give a frame reference.

Reports, vs L: ground energy, runtime, core reached, and the convergence drop.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci import frame, build_from_eft
from classical.trimci.run_cpp import _solver
from classical.trimci.graph import halving_drop

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "L_scaling_frame_summary.json")

# adaptive core per L (20k where it fits ~5 min; shrinks for the huge high-L systems)
CORES = {1: 2000, 2: 20000, 3: 12000, 4: 3000, 5: 1500}
BARE_L = (1, 2)     # cheap frame-reference anchors


def mr(c):
    return max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)


def solve_one(solve, K, c):
    t = time.time()
    res = solve(K, n_elec=1, n_dets=c, n_runs=3, seed=0, max_rounds=mr(c))
    dt = time.time() - t
    drop = halving_drop(res.history) if res.history else None
    return {"core": int(res.n_dets), "E": float(res.energy), "t": dt,
            "drop_rel": (float(drop) if drop is not None else None)}


def run():
    solve = _solver(True)
    rows = []
    t0 = time.time()
    for L in (1, 2, 3, 4, 5):
        H = build_from_eft(L, 3, 3)
        r, phi = frame.analytic_squeeze(H)
        Hg = frame.squeeze_terms(H, -r, phi)
        c = CORES[L]
        sq = solve_one(solve, Hg, c)
        rec = {"L": L, "sites": L ** 3, "n_ferm": H.n_ferm_modes, "n_bos": H.n_bos_modes,
               "n_terms": len(Hg.terms), "squeeze": sq}
        if L in BARE_L:
            rec["bare"] = solve_one(solve, H, c)
        rows.append(rec)
        d = f"{sq['drop_rel']:.1e}" if sq["drop_rel"] is not None else "--"
        print(f"  L={L} sites={L**3:>3} {H.n_ferm_modes:>3}F+{H.n_bos_modes:>3}B  "
              f"core={sq['core']:>6} E0={sq['E']:12.3f} MeV  drop={d}  ({sq['t']:.1f}s)",
              flush=True)
    out = {"rows": rows, "wall_s": time.time() - t0, "cores": CORES}
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=2)
    _plot(out)
    _analyze(out)
    return out


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    Ls = [r["L"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(Ls, [r["squeeze"]["E"] for r in rows], "s-")
    ax[0, 0].set_xlabel("L"); ax[0, 0].set_ylabel("E0 (MeV)"); ax[0, 0].set_title("energy vs L (squeeze)")
    ax[0, 1].semilogy(Ls, [r["squeeze"]["t"] for r in rows], "s-")
    ax[0, 1].set_xlabel("L"); ax[0, 1].set_ylabel("runtime (s)"); ax[0, 1].set_title("runtime vs L")
    ax[1, 0].plot(Ls, [r["squeeze"]["core"] for r in rows], "s-")
    ax[1, 0].set_xlabel("L"); ax[1, 0].set_ylabel("core reached"); ax[1, 0].set_title("adaptive core vs L")
    drops = [r["squeeze"]["drop_rel"] for r in rows]
    ax[1, 1].semilogy(Ls, [d if d else 1e-12 for d in drops], "s-")
    ax[1, 1].axhline(1e-2, ls=":", color="gray", label="1% (reasonable)")
    ax[1, 1].set_xlabel("L"); ax[1, 1].set_ylabel("last-doubling |dE/E|")
    ax[1, 1].set_title("convergence proxy vs L"); ax[1, 1].legend()
    for a in ax.flat:
        a.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "L_scaling_frame.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


def _analyze(out):
    print("\n=== ANALYSIS: L-scaling (squeeze frame, adaptive core) ===", flush=True)
    for r in out["rows"]:
        sq = r["squeeze"]
        acc = ("converged" if (sq["drop_rel"] or 1) < 1e-3 else
               "reasonable(<1%)" if (sq["drop_rel"] or 1) < 1e-2 else "unconverged")
        ref = ""
        if "bare" in r:
            ref = f"  [bare@{r['bare']['core']}={r['bare']['E']:.1f} -> squeeze lower by {r['bare']['E']-sq['E']:.1f} MeV]"
        print(f"  L={r['L']}: E0={sq['E']:.2f} @ core={sq['core']} ({r['n_terms']} terms); "
              f"drop={sq['drop_rel']:.1e} -> {acc}{ref}", flush=True)
    print(f"  total wall: {out['wall_s']/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
