"""Volume scaling of the boson-cutoff shift (re-audit P0-4): does n_b=3 hold to L=10 in the TOTAL
energy? Measure the PAIRED shift Δ34(L) = E(n_b=4) - E(n_b=3) at the deepest COMMON core for L=2,3,4,
convert to a PER-SITE shift, and extrapolate to L=10 (1000 sites) with a band. If the projected
total Δ34(L=10) < the 1 MeV target, the large-volume cutoff conditional is DISCHARGED; else it is
bounded and stays conditional.

The shift is a difference of two large energies at the same core, so the core-incompleteness (the
extensivity trap) largely CANCELS — it is measurable even where the absolute energies are not
converged. Uses A=1 (dilute, best convergence) by default; A=0 vacuum as a cross-check.

Reads bare_L*.json across the energy-gate + volume-scaling campaigns (n_b in the dir name).

    python -m misc.make_nb_volscaling --src <dir1> <dir2> ...
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, CRIT, GREEN, MUTED = "#2a78d6", "#d03b3b", "#3a9b6a", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DEFAULT_SRC = [
    "data/classical/nb_energy_gate",       # L=2 (n_b=3,4)
    "data/classical/nb_energy_gate_L3",    # L=3 n_b=3
    "data/classical/nb_volscaling",        # L=3 n_b=4 + L=4 n_b={3,4} (pulled from 291048)
]


def load(srcs):
    """{(L, n_b, A): {core: E_min_over_seeds}} + sites[L]."""
    lad, sites = {}, {}
    for d in srcs:
        for f in glob.glob(f"{d}/nb*/bare_L*.json"):
            nb = int(re.search(r"/nb(\d+)/", f).group(1))
            m = re.search(r"bare_L(\d+)d\d+_A(\d+)_s(\d+)", os.path.basename(f))
            if not m:
                continue
            L, A = int(m.group(1)), int(m.group(2))
            j = json.load(open(f)); sites[L] = j["sites"]
            for r in j["rungs"]:
                if r.get("E_var") is None:
                    continue
                c = int(r["core"]); dd = lad.setdefault((L, nb, A), {})
                dd[c] = min(dd.get(c, r["E_var"]), r["E_var"])
    return lad, sites


def delta34(lad, L, A):
    """E(n_b=4) - E(n_b=3) at the deepest common core, + the ladder of (core, Δ) for stability."""
    a, b = lad.get((L, 3, A)), lad.get((L, 4, A))
    if not a or not b:
        return None, None, []
    cs = sorted(set(a) & set(b))
    if not cs:
        return None, None, []
    lad34 = [(c, b[c] - a[c]) for c in cs]
    return cs[-1], b[cs[-1]] - a[cs[-1]], lad34


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=DEFAULT_SRC)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--out-dir", default="data/classical/nb_volscaling")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lad, sites = load(args.src)

    # per-L total + per-site shift at the deepest common core
    rows = []
    for L in (2, 3, 4):
        core, d, lad34 = delta34(lad, L, args.A)
        if d is None:
            continue
        n = L ** args.dim
        # stability of the shift over the upper half of the common ladder (spread = an honest error)
        upper = [v for _, v in lad34[len(lad34) // 2:]] or [d]
        rows.append(dict(L=L, sites=n, core=core, d_tot=d, d_site=d / n,
                         spread=max(upper) - min(upper)))
    assert rows, "no paired (n_b=3, n_b=4) shifts found yet — is the volume-scaling data pulled?"

    # per-site fit (mean + spread across the sampled L) -> project to L=10 (1000 sites)
    ps = [r["d_site"] for r in rows]
    ps_mean, ps_lo, ps_hi = np.mean(ps), min(ps), max(ps)
    N10 = 10 ** args.dim
    proj = ps_mean * N10
    proj_band = (ps_lo * N10, ps_hi * N10)
    verdict = ("DISCHARGED" if abs(proj_band[0]) < 1.0 and abs(proj_band[1]) < 1.0
               else "CONDITIONAL")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    fig.patch.set_facecolor(SURFACE)
    # (a) per-site shift vs L — is it flat?
    axA.errorbar([r["L"] for r in rows], [abs(r["d_site"]) for r in rows],
                 yerr=[abs(r["spread"]) / r["sites"] for r in rows], fmt="-o", color=BLUE, lw=1.8,
                 ms=7, mec=SURFACE, mew=1.2, capsize=4, zorder=4, label=f"$A$={args.A}")
    axA.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axA.set_ylabel("|Δ$_{34}$| per site  (MeV/site)", color=INK2, fontsize=9.5)
    axA.set_title("a  Per-site cutoff shift $n_b$3→4 vs $L$", color=INK, fontsize=10.3, loc="left",
                  weight="bold")
    axA.set_xticks([r["L"] for r in rows]); axA.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(axA)
    # (b) projection to L=10 total
    axB.axhline(1.0, ls="--", color=CRIT, lw=1.3, zorder=2)
    axB.annotate("1 MeV target", (10, 1.0), color=CRIT, fontsize=8, va="bottom", ha="right")
    axB.bar([10], [abs(proj)], width=1.2, color=(GREEN if verdict == "DISCHARGED" else CRIT),
            alpha=0.5, zorder=3)
    axB.errorbar([10], [abs(proj)], yerr=[[abs(proj) - abs(proj_band[0])], [abs(proj_band[1]) - abs(proj)]],
                 fmt="o", color=INK, ms=7, capsize=5, zorder=4)
    axB.annotate(f"projected total\n|Δ$_{{34}}$|(L=10)\n≈ {abs(proj):.2f} MeV\n[{abs(proj_band[0]):.2f}, "
                 f"{abs(proj_band[1]):.2f}]", (10, abs(proj)), textcoords="offset points",
                 xytext=(14, 0), fontsize=8.5, color=INK2, va="center")
    axB.set_xlim(8, 13); axB.set_xticks([10]); axB.set_xticklabels(["L=10\n(1000 sites)"])
    axB.set_ylabel("total |Δ$_{34}$|  (MeV)", color=INK2, fontsize=9.5)
    axB.set_title(f"b  Projected n_b=3 cutoff error at L=10 → {verdict}", color=INK, fontsize=10.3,
                  loc="left", weight="bold")
    _style(axB)
    fig.suptitle(f"Volume scaling of the n_b=3 cutoff shift — L=10 projection {verdict} "
                 f"(per-site Δ34 {ps_mean:+.4f} MeV/site)", fontsize=10.6, color=INK, y=1.02, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb_volscaling.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb_volscaling.pdf / .png")

    md = ["# Volume scaling of the n_b=3 boson-cutoff shift (re-audit P0-4)\n",
          f"_Paired shift Δ34(L)=E(n_b=4)−E(n_b=3) at the deepest common core, A={args.A}, dim="
          f"{args.dim}. The shift cancels most core-incompleteness (measurable where absolute E is "
          f"not converged). Per-site shift extrapolated to L=10 (1000 sites)._\n",
          "| L | sites | deepest common core | Δ34 total (MeV) | Δ34/site (MeV) |",
          "|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['L']} | {r['sites']} | {r['core']:,} | {r['d_tot']:+.4f} | {r['d_site']:+.5f} |")
    md.append(f"\n**Projection:** per-site Δ34 = {ps_mean:+.5f} MeV/site (range [{ps_lo:+.5f}, "
              f"{ps_hi:+.5f}]) → total at L=10 (1000 sites) ≈ **{proj:+.2f} MeV** (band "
              f"[{proj_band[0]:+.2f}, {proj_band[1]:+.2f}]). **VERDICT: {verdict}** — the n_b=3→4 "
              f"cutoff error at L=10 is {'below' if verdict=='DISCHARGED' else 'NOT clearly below'} "
              f"the 1 MeV target, so the large-volume cutoff conditional is "
              f"{'DISCHARGED (n_b=3 adequate through L=10)' if verdict=='DISCHARGED' else 'retained'}. "
              f"Caveat: linear-in-sites extrapolation from L=2,3,4; the per-site shift's flatness "
              f"across L is the supporting evidence (panel a).\n")
    open(f"{args.out_dir}/nb_volscaling_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_volscaling_table.md")
    print(f"[done] per-site Δ34 {ps_mean:+.5f}; L=10 proj {proj:+.2f} MeV [{proj_band[0]:+.2f},"
          f"{proj_band[1]:+.2f}] -> {verdict}")


if __name__ == "__main__":
    main()
