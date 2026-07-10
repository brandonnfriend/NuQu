"""
Sweep-2 accuracy supplement — the RELIABLE convergence metric.

`run_L_scaling_frame.py` reported the in-run last-doubling drop, but that reads a
single growing run's ramp, which (per the solver docstring) "plateaus deceptively".
The reliable signal is INDEPENDENT solves at separated core sizes. This does one extra
independent solve at c/2 per L and reports |E(c) − E(c/2)| / |E(c)| — the honest
"how far from converged" proxy — updating the summary + re-analysing.
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
SUMMARY = os.path.join(OUT_DIR, "L_scaling_frame_summary.json")


def mr(c):
    return max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)


def run():
    solve = _solver(True)
    out = json.load(open(SUMMARY))
    print("=== RELIABLE convergence: independent solve at c/2 vs the sweep-2 c ===", flush=True)
    print(f"  {'L':>2} {'core c':>7} {'E(c)':>12} {'c/2':>6} {'E(c/2)':>12} "
          f"{'|dE/E| indep':>12} {'accuracy':>16}", flush=True)
    t0 = time.time()
    for r in out["rows"]:
        L = r["L"]
        c = r["squeeze"]["core"]
        Ec = r["squeeze"]["E"]
        c2 = max(20, c // 2)
        if c2 >= c:                       # L=1 solved exactly already
            r["squeeze"]["reliable_drop"] = 0.0
            print(f"  {L:>2} {c:>7} {Ec:>12.3f} {'--':>6} {'(exact)':>12} "
                  f"{0.0:>12.1e} {'converged':>16}", flush=True)
            continue
        H = build_from_eft(L, 3, 3)
        rr, phi = frame.analytic_squeeze(H)
        Hg = frame.squeeze_terms(H, -rr, phi)
        res = solve(Hg, n_elec=1, n_dets=c2, seed=0, n_runs=3, max_rounds=mr(c2))
        E2 = float(res.energy)
        rel = abs(Ec - E2) / max(abs(Ec), 1e-12)
        acc = ("converged (<0.1%)" if rel < 1e-3 else
               "reasonable (<1%)" if rel < 1e-2 else
               "coarse (<5%)" if rel < 5e-2 else "unconverged")
        r["squeeze"]["c_half"] = int(res.n_dets)
        r["squeeze"]["E_c_half"] = E2
        r["squeeze"]["reliable_drop"] = rel
        print(f"  {L:>2} {c:>7} {Ec:>12.3f} {int(res.n_dets):>6} {E2:>12.3f} "
              f"{rel:>12.1e} {acc:>16}", flush=True)
    out["accuracy_metric"] = "independent |E(c)-E(c/2)|/|E(c)|"
    json.dump(out, open(SUMMARY, "w"), indent=2)
    print(f"  supplement wall: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
