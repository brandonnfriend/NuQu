"""
Log-log convergence of the L=2 squeeze-frame sweep: energy error |E(core) − E∞| vs
core size. E∞ (the converged ground energy, shared by bare & squeeze since the frame is
isospectral) is fit from the well-converged squeeze ladder as E(N) = E∞ + a·N^(−b).
The squeeze error falls as a clean power law (slope −b); bare is shown for contrast.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import curve_fit

OUT_DIR = os.path.join("data", "classical")
SUMMARY = os.path.join(OUT_DIR, "L2_convergence_frame_summary.json")


def main():
    d = json.load(open(SUMMARY))
    sq = sorted(d["squeeze"], key=lambda r: r["core"])
    ba = sorted(d["bare"], key=lambda r: r["core"])
    Ns = np.array([r["core"] for r in sq], float)
    Es = np.array([r["E"] for r in sq], float)

    # fit E∞ from the squeeze ladder: E(N) = E∞ + a N^(-b)
    def model(N, Einf, a, b):
        return Einf + a * N ** (-b)
    (Einf, a, b), _ = curve_fit(
        model, Ns, Es, p0=[Es[-1] - 0.1, 1e3, 0.8],
        bounds=([Es.min() - 50, 0, 0.2], [Es[-1], 1e9, 3.0]), maxfev=100000)
    print(f"fit: E_inf = {Einf:.4f} MeV, a = {a:.3g}, convergence exponent b = {b:.3f}", flush=True)

    err_s = np.abs(Es - Einf)
    Nb = np.array([r["core"] for r in ba], float)
    err_b = np.abs(np.array([r["E"] for r in ba], float) - Einf)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.loglog(Ns, err_s, "s-", color="C1", ms=8, label="squeeze (data)")
    xs = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 100)
    ax.loglog(xs, a * xs ** (-b), "--", color="C1", alpha=0.6,
              label=f"power-law fit  ∝ N$^{{-{b:.2f}}}$")
    ax.loglog(Nb, err_b, "o:", color="C0", ms=7, alpha=0.8, label="bare (same E∞)")
    for N, e in zip(Ns, err_s):
        ax.annotate(f"{e:.2f}", (N, e), textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xlabel("core size (determinants)")
    ax.set_ylabel(r"energy error  $|E(\mathrm{core}) - E_\infty|$  (MeV)")
    ax.set_title(f"L=2 d=3 squeeze-frame convergence (log-log)\n"
                 f"$E_\\infty$={Einf:.3f} MeV, rate $\\sim N^{{-{b:.2f}}}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "L2_convergence_loglog.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)

    print("\nsqueeze error vs core (E∞ = %.3f):" % Einf, flush=True)
    for N, e in zip(Ns, err_s):
        print(f"  core={int(N):>7}  |E-E∞|={e:.4f} MeV", flush=True)


if __name__ == "__main__":
    main()
