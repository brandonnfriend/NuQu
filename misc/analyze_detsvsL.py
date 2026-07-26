"""
Analyze a combined dets-vs-L result (from misc/combine_detsvsL.py): per-L convergence,
an eps-scan for the scaling exponent, and two figures.

    python -m misc.analyze_detsvsL --json data/classical/detsvsL_hpc_<id>.json

Figures (-> data/classical/hpc/analysis/ by default):
  * <label>_convergence.png : per-site gap |E_var+PT2 - E_inf|/site vs #dets, one line
    per L (log-log). How converged each L is, and where it crosses each eps.
  * <label>_scaling.png     : N*(eps) vs volume V=sites (log-log) at a chosen eps, with
    the polynomial (N~V^g) and exponential (N~e^{gV}) fits overlaid -- which model wins.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classical.trimci.run_cpp import _extract_nstar, _nstar_repr, _fit_exponent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--eps-plot", type=float, default=0.3, help="eps for the scaling figure")
    ap.add_argument("--out-dir", default="data/classical/hpc/analysis")
    args = ap.parse_args()

    d = json.load(open(args.json))
    label = os.path.splitext(os.path.basename(args.json))[0]
    perL = sorted(d["per_L"], key=lambda p: p["L"])
    sites = {p["L"]: p["sites"] for p in perL}
    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- per-L convergence ----
    print(f"frame={d.get('transform')}  seeds/L={d.get('n_seeds_by_L')}")
    print("=== per-L convergence ===")
    for p in perL:
        print(f"  L={p['L']} ({p['sites']} sites): E_inf/site={p['E_inf_per_site']:.3f} "
              f"+/- {p['sigma_per_site']:.3f}  (gap/site @top {(p['rungs'][-1]['E_pt2']-p['E_inf'])/p['sites']:.3f})")

    # ---- eps scan ----
    print("=== N*(eps) scan (bracketed points -> exponent) ===")
    best = None
    for eps in args.eps_grid:
        pts = []
        stat = {}
        for p in perL:
            b = _extract_nstar(p["rungs"], p["E_inf"], p["sites"], eps, "E_pt2")
            stat[p["L"]] = (b["status"], _nstar_repr(b))
            if b["status"] == "bracketed":
                pts.append((p["sites"], _nstar_repr(b)))
        line = f"  eps={eps:>4}: " + " ".join(f"L{L}={stat[L][0][:5]}({stat[L][1]})" for L in sorted(stat))
        if len(pts) >= 2:
            f = _fit_exponent([s for s, _ in pts], [n for _, n in pts])
            ex, po = f["exponential_in_V"], f["polynomial_in_V"]
            better = "POLY" if (po["r2"] or -9) > (ex["r2"] or -9) else "EXP"
            line += (f"  [{len(pts)}pts] poly V^{po['slope']:.2g}(R2={po['r2']:.2f}) "
                     f"exp e^{ex['slope']:.3g}V(R2={ex['r2']:.2f}) -> {better}")
            if best is None or len(pts) > best[1]:
                best = (eps, len(pts), po, ex)
        print(line)

    # ---- Figure 1: convergence ----
    plt.figure(figsize=(7, 5))
    for p in perL:
        s, Ei = p["sites"], p["E_inf"]
        c = [r["core"] for r in p["rungs"]]
        g = [max((r["E_pt2"] - Ei) / s, 1e-4) for r in p["rungs"]]
        plt.plot(c, g, "o-", label=f"L={p['L']} ({s} sites)")
    for eps in (args.eps_plot, 0.1):
        plt.axhline(eps, ls="--", c="gray", lw=0.8)
        plt.text(c[0], eps * 1.08, f"eps={eps}", color="gray", fontsize=8)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("determinants (selected-CI core)")
    plt.ylabel(r"$|E_{var+PT2}-E_\infty|/\mathrm{site}$  (MeV)")
    plt.title(f"Frame convergence per L  ({label})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p1 = os.path.join(args.out_dir, f"{label}_convergence.png")
    plt.savefig(p1, dpi=130); plt.close()

    # ---- Figure 2: N* vs V at eps_plot ----
    eps = args.eps_plot
    V, N = [], []
    for p in perL:
        b = _extract_nstar(p["rungs"], p["E_inf"], p["sites"], eps, "E_pt2")
        if b["status"] == "bracketed":
            V.append(p["sites"]); N.append(_nstar_repr(b))
    p2 = None
    if len(V) >= 2:
        V, N = np.array(V, float), np.array(N, float)
        f = _fit_exponent(list(V), list(N))
        ex, po = f["exponential_in_V"], f["polynomial_in_V"]
        plt.figure(figsize=(7, 5))
        plt.plot(V, N, "ks", ms=10, label=f"N* (eps={eps}/site)")
        Vs = np.linspace(V.min(), V.max(), 60)
        plt.plot(Vs, np.exp(po["intercept"] + po["slope"] * np.log(Vs)), "-", c="C4",
                 label=f"poly  N~V^{po['slope']:.2g}  (R^2={po['r2']:.2f})")
        plt.plot(Vs, np.exp(ex["intercept"] + ex["slope"] * Vs), "--", c="C3",
                 label=f"exp   N~e^({ex['slope']:.3g}V)  (R^2={ex['r2']:.2f})")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("volume  V = #sites"); plt.ylabel("N*  (dets for eps/site)")
        plt.title(f"Classical cost vs volume, frame, eps={eps}  ({label})")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        p2 = os.path.join(args.out_dir, f"{label}_scaling.png")
        plt.savefig(p2, dpi=130); plt.close()

    print(f"\nwrote:\n  {p1}" + (f"\n  {p2}" if p2 else "  (no scaling fig: <2 bracketed pts at eps_plot)"))
    if best:
        eps, npts, po, ex = best
        print(f"headline (eps={eps}, {npts} pts): poly N~V^{po['slope']:.2g} (R2={po['r2']:.2f}) "
              f"vs exp (R2={ex['r2']:.2f}) -> {'POLYNOMIAL' if po['r2']>ex['r2'] else 'exp'} favored")


if __name__ == "__main__":
    main()
