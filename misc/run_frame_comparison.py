"""
Head-to-head of the task-33 transformations on the real L=2 d=3 (A=1) system.

All ISOSPECTRAL frames converge to the same E∞, so the winner is whoever reaches E∞
with the fewest determinants AND the least wall-clock. We compare:
  * bare                — reference
  * per-mode squeeze    — Gaussian, cheap (fewer terms)
  * multi-mode Bogoliubov — Gaussian, most compact per-det but ~100x terms
  * Lang-Firsov (λ=0.1) — DEMONSTRATION: our transition vertex makes the LF substitution
                          leading-order only (non-isospectral) → converges to E ≠ E∞.
(COO is idle at A=1 — one nucleon, no orbital correlation — so it is noted, not run.)

Two metrics per frame: determinants-to-1MeV and wall-clock-to-1MeV of E∞.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import curve_fit

from classical.trimci import frame, build_from_eft
from classical.trimci.run_cpp import _solver

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "frame_comparison_summary.json")


def mr(c):
    return max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)


def sweep(solve, K, cores):
    rows = []
    for c in cores:
        t = time.time()
        res = solve(K, n_elec=1, n_dets=c, seed=0, n_runs=3, max_rounds=mr(c))
        rows.append({"core": int(res.n_dets), "E": float(res.energy), "t": time.time() - t})
        print(f"    core={res.n_dets:>6} E0={res.energy:11.3f}  ({rows[-1]['t']:6.1f}s)", flush=True)
    return rows


def cost_to_acc(rows, Einf, tol=1.0):
    """(determinants, cumulative wall-clock) to first reach within `tol` MeV of E∞."""
    cum = 0.0
    for r in rows:
        cum += r["t"]
        if abs(r["E"] - Einf) < tol:
            return r["core"], cum
    return None, cum


def run():
    solve = _solver(True)
    H = build_from_eft(2, 3, 3)
    r, phi = frame.analytic_squeeze(H)
    Hsq = frame.squeeze_terms(H, -r, phi)
    al, be = frame.analytic_bogoliubov(H)
    Hbg = frame.bogoliubov_terms(H, al, be)
    Hlf = build_from_eft(2, 3, 3, transform="LF", frame_params={"lambdas": 0.1})

    frames = {
        "bare":       (H,   (500, 1500, 5000, 15000, 40000)),
        "squeeze":    (Hsq, (500, 1500, 5000, 15000)),
        "bogoliubov": (Hbg, (300, 800, 1500)),
        "LF(λ=0.1)":  (Hlf, (1000, 3000, 10000)),
    }
    out = {}
    for name, (K, cores) in frames.items():
        print(f"  {name}  ({len(K.terms)} terms):", flush=True)
        out[name] = {"terms": len(K.terms), "rows": sweep(solve, K, cores)}

    # E∞ from the squeeze ladder (well-converged, isospectral to bare)
    Ns = np.array([x["core"] for x in out["squeeze"]["rows"]], float)
    Es = np.array([x["E"] for x in out["squeeze"]["rows"]], float)
    Einf = float(curve_fit(lambda N, E, a, b: E + a * N ** (-b), Ns, Es,
                           p0=[Es[-1] - 0.1, 1e3, 0.7],
                           bounds=([Es.min() - 50, 0, 0.2], [Es[-1], 1e9, 3]),
                           maxfev=100000)[0][0])
    out["E_inf"] = Einf

    print(f"\n=== VERDICT (E∞ = {Einf:.3f} MeV) ===", flush=True)
    print(f"  {'frame':13} {'terms':>7} {'best E':>10} {'err@best':>9} "
          f"{'dets->1MeV':>10} {'wall->1MeV':>11}", flush=True)
    for name in frames:
        rows = out[name]["rows"]
        errbest = rows[-1]["E"] - Einf     # rows are core-ascending; last = best (largest core)
        dets, wall = cost_to_acc(rows, Einf)
        out[name]["err_at_best"] = errbest
        out[name]["dets_to_1MeV"] = dets
        out[name]["wall_to_1MeV"] = wall if dets else None
        ds = f"{dets}" if dets else ">max"
        ws = f"{wall:.1f}s" if dets else "--"
        print(f"  {name:13} {out[name]['terms']:>7} {rows[-1]['E']:>10.2f} "
              f"{errbest:>+9.2f} {ds:>10} {ws:>11}", flush=True)
    json.dump(out, open(SUMMARY, "w"), indent=2)
    _plot(out, Einf)
    return out


def _plot(out, Einf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"bare": ("o-", "C0"), "squeeze": ("s-", "C1"),
              "bogoliubov": ("^-", "C2"), "LF(λ=0.1)": ("x--", "C3")}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for name, (mk, col) in styles.items():
        rows = out[name]["rows"]
        N = [x["core"] for x in rows]
        err = [abs(x["E"] - Einf) for x in rows]
        ax.loglog(N, err, mk, color=col, ms=8,
                  label=f"{name} ({out[name]['terms']} terms)")
    ax.axhline(1.0, ls=":", color="gray", lw=1, label="1 MeV target")
    ax.set_xlabel("core size (determinants)")
    ax.set_ylabel(r"$|E(\mathrm{core}) - E_\infty|$  (MeV)")
    ax.set_title(f"Transformation comparison — L=2 d=3, A=1  (E∞={Einf:.2f} MeV)\n"
                 "isospectral frames collapse to E∞; LF (transition vertex) does not")
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "frame_comparison.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


if __name__ == "__main__":
    run()
