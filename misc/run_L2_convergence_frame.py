"""
Sweep 1 — L=2 d=3 energy-vs-core convergence, bare vs per-mode squeeze frame,
now that the optimized solver makes 100k-determinant cores feasible on the laptop.

Squeeze converges by ~10k; bare is the slow reference that actually needs the big
cores. Budget-sized to ~18 min: squeeze to 100k (confirm convergence), bare to 60k.
Saves each solve into the data pipeline (transform-tagged) and writes a combined
E-vs-core / runtime-vs-core plot + a summary JSON for analysis.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci import frame, build_from_eft
from classical.trimci.run_cpp import _solver

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "L2_convergence_frame_summary.json")


def mr(c):
    return max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)


def run():
    solve = _solver(True)
    H = build_from_eft(2, 3, 3)                       # L=2 d=3, N_f=8 (frame-valid)
    r, phi = frame.analytic_squeeze(H)
    Hg = frame.squeeze_terms(H, -r, phi)             # per-mode squeeze (−r*: the compacting sign)
    plans = [("bare", H, (1000, 3000, 10000, 30000, 60000)),
             ("squeeze", Hg, (1000, 3000, 10000, 30000, 100000))]
    out = {}
    t0 = time.time()
    for name, K, cores in plans:
        rows = []
        for c in cores:
            t = time.time()
            res = solve(K, n_elec=1, n_dets=c, n_runs=3, seed=0, max_rounds=mr(c))
            dt = time.time() - t
            rows.append({"core": int(res.n_dets), "E": float(res.energy), "t": dt})
            print(f"  {name:8} core={res.n_dets:>7} E0={res.energy:12.4f} MeV  ({dt:6.1f}s)",
                  flush=True)
        out[name] = rows
    out["wall_s"] = time.time() - t0
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=2)
    _plot(out)
    _analyze(out)
    return out


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, mk in (("bare", "o-"), ("squeeze", "s-")):
        rows = out[name]
        c = [r["core"] for r in rows]
        ax1.plot(c, [r["E"] for r in rows], mk, label=name)
        ax2.plot(c, [r["t"] for r in rows], mk, label=name)
    ax1.set_xscale("log"); ax1.set_xlabel("core (determinants)")
    ax1.set_ylabel("ground energy E0 (MeV)"); ax1.set_title("L=2 d=3 energy convergence")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.25)
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("core (determinants)")
    ax2.set_ylabel("runtime (s)"); ax2.set_title("runtime vs core"); ax2.legend()
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "L2_convergence_frame.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


def _analyze(out):
    print("\n=== ANALYSIS: L=2 d=3 convergence (bare vs squeeze) ===", flush=True)
    for name in ("bare", "squeeze"):
        rows = out[name]
        Ef = rows[-1]["E"]
        # core-to-1MeV: smallest core within 1 MeV of the best (largest-core) energy
        c1 = next((r["core"] for r in rows if abs(r["E"] - Ef) < 1.0), rows[-1]["core"])
        last_drop = rows[-1]["E"] - rows[-2]["E"]
        print(f"  {name:8}: best E0={Ef:.3f} @ {rows[-1]['core']} dets; last-doubling drop "
              f"{last_drop:+.3f} MeV; within 1 MeV of best by core={c1}", flush=True)
    eb, es = out["bare"][-1]["E"], out["squeeze"][-1]["E"]
    print(f"  frame gap at top cores: bare({out['bare'][-1]['core']})={eb:.2f} vs "
          f"squeeze({out['squeeze'][-1]['core']})={es:.2f}  -> squeeze lower by {eb-es:.2f} MeV",
          flush=True)
    print(f"  total wall: {out['wall_s']/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
