"""Selected-CI occupation-tail DIAGNOSTIC vs boson cutoff n_b (post-vertex-fix).

SCOPE (per the 2026-09-02 cutoff audit): this figure reports the |c|²-weighted occupation tail —
the projected weight above a cutoff — OF THE SELECTED-CI REFERENCE STATE. It is a diagnostic, NOT an
energy/observable error bound: a small tail for an approximate state does not bound the energy of
P_N·H·P_N (it ignores the omitted sector's energy scale + the boundary coupling). It strongly REJECTS
n_b=1 and MOTIVATES n_b=2 at the SAMPLED (L,A) points; it does not certify n_b=2. The energy
truncation gate (E_0(N_f) convergence at core-converged states, with uncertainty) is a separate,
L=2-feasible study. Points, not interpolating lines. No derived accuracy threshold is claimed.

    python -m misc.make_nb_figure --data data/classical/nb_convergence
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ONE_PCT = 0.01                                               # a 1% reference line (NOT a derived budget)


def load(d):
    # dedup by (L,A): the deep-reference studyHist_* (N_f=16) and the studyBdenseCheap_* (N_f=8) both
    # carry the same config — keep the one with the DEEPEST reference (largest Nf_deepest).
    best = {}
    for f in sorted(glob.glob(f"{d}/study*.json")):
        j = json.load(open(f))
        lw = j.get("leaked_weight_vs_cutoff") or {}
        if not lw:
            continue
        dim = j.get("dim", 3)
        fill = j["A"] / (j["L"] ** dim)
        nf = j.get("Nf_deepest", j.get("Nf_ref", 0)) or 0
        rec = dict(L=j["L"], A=j["A"], dim=dim, Nf=nf, fill=fill,
                   leak1=lw.get("2"), leak2=lw.get("4"),
                   dense=(abs(fill - round(fill)) < 1e-9 and j["A"] >= j["L"] ** dim))
        key = (j["L"], j["A"])
        if key not in best or nf > best[key]["Nf"]:
            best[key] = rec
    return sorted(best.values(), key=lambda r: (r["L"], r["A"]))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(recs)))

    # (a) leaked weight vs n_b — POINTS per (L,A); all fall from >9% (n_b=1) to <1% (n_b=2).
    # A faint 1% reference line only (NOT a derived accuracy budget — the tail is not an energy bound).
    for r, col in zip(recs, cmap):
        axA.semilogy([1, 2], [r["leak1"], r["leak2"]], "o", color=col, ms=6, mec=SURFACE, mew=0.7,
                     zorder=3, label=f"$L$={r['L']}, $A$={r['A']}" + (" (dense)" if r["dense"] else ""))
    axA.axhline(ONE_PCT, ls=":", color=MUTED, lw=1.0, zorder=2)
    axA.annotate("1% (reference line, not a derived budget)", (1.02, ONE_PCT), color=MUTED,
                 fontsize=7.5, va="bottom")
    axA.set_xticks([1, 2]); axA.set_xticklabels(["$n_b$=1 (N_f=2)", "$n_b$=2 (N_f=4)"])
    axA.set_xlim(0.85, 2.35)
    axA.set_ylabel("selected-CI weight leaked above cutoff", color=INK2, fontsize=9.5)
    axA.set_title("a  n_b=1 rejected; n_b=2 tail small (selected-CI reference states)", color=INK,
                  fontsize=10.0, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=6.8, loc="lower left", labelcolor=INK2, ncol=2)
    _style(axA)

    # (b) n_b=2 leak vs density (L=2) + larger-volume points — POINTS ONLY (behavior is nonmonotone;
    # a connecting line would falsely imply interpolation).
    l2 = sorted([r for r in recs if r["L"] == 2], key=lambda r: r["A"])
    axB.plot([r["A"] for r in l2], [r["leak2"] for r in l2], "o", color=BLUE, ms=7.5, mec=SURFACE,
             mew=1.2, zorder=4, label="$L$=2, sampled $A$")
    for r in recs:
        if r["L"] != 2:
            axB.plot([r["A"]], [r["leak2"]], "s", ms=9, color=CRIT, mec=SURFACE, mew=1.0, zorder=5)
            axB.annotate(f"$L$={r['L']}" + (",dense" if r["dense"] else ""), (r["A"], r["leak2"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=CRIT)
    axB.axhline(ONE_PCT, ls=":", color=MUTED, lw=1.0, zorder=2)
    axB.annotate("1% reference", (1, ONE_PCT), color=MUTED, fontsize=7.5, va="bottom")
    axB.set_ylim(0, max(ONE_PCT * 1.15, max(r["leak2"] for r in recs) * 1.25))
    axB.set_xlabel("nucleon number $A$  (density; $A$=8 is filling 1.0 at $L$=2)", color=INK2, fontsize=9.5)
    axB.set_ylabel("$n_b$=2 selected-CI leaked weight", color=INK2, fontsize=9.5)
    axB.set_title("b  n_b=2 tail at the sampled densities / volumes", color=INK,
                  fontsize=10.5, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Selected-CI occupation-tail diagnostic — rejects n_b=1, motivates n_b=2 (NOT an "
                 "energy bound; sampled points, corrected H)", fontsize=10.8, color=INK, y=1.02,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/classical/nb_convergence")
    ap.add_argument("--out-dir", default="data/classical/nb_convergence")
    args = ap.parse_args()
    recs = load(args.data)
    assert recs, "no study*.json with leaked_weight found"
    make_figure(recs, f"{args.out_dir}/nb_occupation_tail_diagnostic")

    md = ["# Selected-CI occupation-tail diagnostic vs boson cutoff n_b — post-vertex-fix\n",
          "_The |c|²-weighted occupation tail (whole-lattice P(∃ mode ≥ N)) OF THE SELECTED-CI "
          "REFERENCE STATE — a DIAGNOSTIC, not an energy/observable error bound. A small tail for an "
          "approximate state does not bound the energy of P_N·H·P_N (it ignores the omitted sector's "
          "energy scale + the boundary coupling P_N·H·(1−P_N)). At the SAMPLED (L,A) points it rejects "
          "n_b=1 and motivates n_b=2; it does not certify n_b=2._\n",
          "| L | A | filling | leak @ n_b=1 (N_f=2) | leak @ n_b=2 (N_f=4) |",
          "|--:|--:|:--|--:|--:|"]
    for r in recs:
        md.append(f"| {r['L']} | {r['A']} | {r['fill']:.2f}{' (dense)' if r['dense'] else ''} | "
                  f"{r['leak1']*100:.1f}% | {r['leak2']*100:.2f}% |")
    md.append(f"\n**Reading (diagnostic only):** at the sampled points n_b=1 leaks "
              f"{min(r['leak1'] for r in recs)*100:.0f}–{max(r['leak1'] for r in recs)*100:.0f}% "
              f"(strongly rejected); the n_b=2 selected-CI tail is "
              f"{min(r['leak2'] for r in recs)*100:.2f}–{max(r['leak2'] for r in recs)*100:.2f}%. This "
              f"MOTIVATES n_b=2 but does NOT bound the truncation error: E_var at fixed core is "
              f"confounded (rises with N_f from core-incompleteness), and the deep-reference (N_f=16) "
              f"reruns show only that the same low-occupation selected basin was found — not resolved "
              f"solver convergence. The ENERGY gate — E_0(N_f) convergence at core-converged states "
              f"with seed uncertainty — is a separate L=2-feasible study (pending; larger L hits the "
              f"extensivity/H-build wall). The quantum anchor stays CONDITIONAL on n_b=2 until that "
              f"gate passes; report n_b=3 resource sensitivity alongside it. No per-seed uncertainty is "
              f"shown here (best-of-ensemble only — a solver-output upgrade is in progress).\n")
    open(f"{args.out_dir}/nb_occupation_tail_diagnostic_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_occupation_tail_diagnostic_table.md")
    print("[done] " + " | ".join(f"L{r['L']}A{r['A']}:{r['leak2']*100:.2f}%" for r in recs))


if __name__ == "__main__":
    main()
