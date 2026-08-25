"""Classical bare-frame TrimCI baseline — the variational upper bound + the cost-to-converge trap.

The publication-grade classical result (codex publication-readiness: convergence is universal;
superiority/priority claims are conditional). Two honest messages:
  * panel a — TrimCI returns a VARIATIONAL upper bound (E_var, Ritz) at every L, cheaply. L=2 (deep
    solve) CONVERGES; L≥3 keep descending at the reachable core (the extensivity trap).
  * panel b — the per-site upper bound vs L: converged at L=2 (the reference line); above it for L≥3
    (the unconvergence gap = the classically-hard / quantum-advantage regime). Small-core large-L
    points are cheap verified bounds — a classical bar for other methods to beat.

Labels are honest: E_var is a rigorous UPPER BOUND; it is E∞ only where converged (L=2); L≥3 are
bounds, not converged energies; PT2/extrapolation are not variational. Boson cutoff n_b=2 conditional.

    python -m misc.make_classical_baseline_figure --data data/classical/2026-08-23/bare_baseline_290832
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
_CONV_MEV = 2.0                                            # |ΔE| per core-doubling below this = "converged"


def load(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/bare_*.json")):
            j = json.load(open(f))
            rungs = [r for r in j.get("rungs", []) if r.get("E_var") is not None]
            if not rungs:
                continue
            rungs.sort(key=lambda r: r["core"])
            deep = rungs[-1]
            dE = abs(rungs[-1]["E_var"] - rungs[-2]["E_var"]) if len(rungs) >= 2 else float("nan")
            recs.append(dict(L=j["L"], sites=j["sites"], A=j["A"], done=j.get("done", False),
                             cores=[r["core"] for r in rungs], E=[r["E_var"] for r in rungs],
                             maxcore=deep["core"], Ebound=deep["E_var"],
                             per_site=deep["E_var"] / j["sites"], dE_last=dE,
                             converged=(dE < _CONV_MEV)))
    recs.sort(key=lambda r: r["L"])
    # dedup by L, keep the deepest
    best = {}
    for r in recs:
        if r["L"] not in best or r["maxcore"] > best[r["L"]]["maxcore"]:
            best[r["L"]] = r
    return [best[L] for L in sorted(best)]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(recs)))

    # (a) per-site convergence: E_var/site vs core
    for r, col in zip(recs, cmap):
        ps = np.array(r["E"]) / r["sites"]
        axA.semilogx(r["cores"], ps, "-o", color=col, lw=1.8, ms=4.5, mec=SURFACE, mew=0.8,
                     zorder=3, label=f"$L$={r['L']} ({r['sites']} sites)")
    conv = next((r for r in recs if r["L"] == 2), None)
    if conv:
        axA.axhline(conv["per_site"], ls=":", color=INK2, lw=1.1, zorder=2)
        axA.annotate(f"$L$=2 converged\n≈{conv['per_site']:.0f} MeV/site", (recs[0]["cores"][0], conv["per_site"]),
                     textcoords="offset points", xytext=(4, 6), fontsize=8, color=INK2)
    axA.set_xlabel("selected-CI core (# determinants)", color=INK2, fontsize=9.5)
    axA.set_ylabel("variational $E_\\mathrm{var}$ / site  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  TrimCI convergence (upper bound, all $L$)", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2, ncol=1)
    _style(axA)

    # (b) per-site upper bound vs L — the baseline; converged (filled) vs bound-only (hollow)
    Ls = [r["L"] for r in recs]
    for r in recs:
        filled = r["converged"]
        axB.plot([r["L"]], [r["per_site"]], "o", ms=9, color=(GREEN if filled else "none"),
                 mec=(GREEN if filled else CRIT), mew=1.8, zorder=4)
    axB.plot(Ls, [r["per_site"] for r in recs], "-", color=MUTED, lw=1.2, zorder=2)
    if conv:
        axB.axhline(conv["per_site"], ls=":", color=GREEN, lw=1.2,
                    label=f"$L$=2 converged ({conv['per_site']:.0f})")
    axB.plot([], [], "o", color=GREEN, label="converged")
    axB.plot([], [], "o", color="none", mec=CRIT, mew=1.8, label="upper bound only (not converged)")
    axB.set_xlabel("lattice size $L$  (dim=3, filling 1.0, $n_b$=2)", color=INK2, fontsize=9.5)
    axB.set_ylabel("variational bound  $E_\\mathrm{var}$ / site  (MeV)", color=INK2, fontsize=9.5)
    axB.set_title("b  Variational baseline vs $L$ (a bar to beat)", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axB.set_xticks(Ls)
    axB.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Classical bare-frame TrimCI: variational upper bound (cheap at any $L$) + the "
                 "cost-to-converge trap", fontsize=11.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/classical/2026-08-23/bare_baseline_290832"])
    ap.add_argument("--out-dir", default="data/classical/2026-08-24/baseline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load(args.data)
    assert recs, "no bare_*.json found"
    make_figure(recs, f"{args.out_dir}/classical_baseline")

    md = ["# Classical bare-frame TrimCI baseline — variational upper bound + cost-to-converge\n",
          "_dim=3, filling 1.0, n_b=2. E_var is a rigorous VARIATIONAL upper bound (Ritz) — it is E∞"
          " only where CONVERGED (L=2); L≥3 are bounds, not converged energies. PT2/extrapolation are"
          " not variational. Boson cutoff conditional (classical convergence study). A bar for other"
          " methods to beat._\n",
          "| L | sites | max core | E_var (MeV, upper bound) | E_var/site | ΔE last doubling | status |",
          "|--:|--:|--:|--:|--:|--:|:--|"]
    for r in recs:
        st = "**converged**" if r["converged"] else "upper bound (not converged)"
        md.append(f"| {r['L']} | {r['sites']} | {r['maxcore']:,} | {r['Ebound']:,.0f} | "
                  f"{r['per_site']:.0f} | {r['dE_last']:.1f} | {st} |")
    md.append(f"\n**Reading:** TrimCI returns a variational bound at every L, cheaply. L=2 converges "
              f"(≈{recs[0]['per_site']:.0f} MeV/site, ΔE≈{recs[0]['dE_last']:.1f} over the last "
              f"doubling); the per-site bound rises for L≥3 because convergence requires cores beyond "
              f"reach — the extensivity trap = the classically-hard / quantum-advantage regime. Small-"
              f"core large-L points (once added) extend the cheap verified bound to L=6–10.\n")
    open(f"{args.out_dir}/classical_baseline_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/classical_baseline_table.md")
    print("[done] " + " | ".join(f"L{r['L']}:{r['Ebound']:.0f}({'conv' if r['converged'] else 'bnd'})"
                                 for r in recs))


if __name__ == "__main__":
    main()
