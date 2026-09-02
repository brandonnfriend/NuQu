"""Boson occupation POPULATION HISTOGRAM of the selected-CI reference state (DIAGNOSTIC).

x-axis = boson occupation level n; y-axis = PER-MODE |c|²-weighted population p(n) at the deep
reference (N_f=16 = n_b=4). The n_b=2 cut (N_f=4) keeps levels n≤3; the tail beyond it is small and
the N_f=8→16 reruns agree — but this is a SELECTED-STATE diagnostic (each state holds only ~1.5–3k
determinants), NOT a solver-convergence proof of the true tail, and NOT an energy/observable error
bound (per the 2026-09-02 cutoff audit). It motivates n_b=2; the energy gate is the separate L=2
study. NB: bars are per-mode p(n); the quoted percentages are whole-lattice P(∃ mode ≥ N).

    python -m misc.make_nb_histogram --data data/classical/nb_convergence
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN, PURP = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#7b5cd6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NB2_KEEP = 4                                                  # n_b=2 -> N_f=4 -> keep levels 0..3


def load(d):
    recs = []
    for f in sorted(glob.glob(f"{d}/studyHist_*.json")):
        j = json.load(open(f))
        p = np.asarray(j.get("occ_histogram") or [], float)
        if p.size == 0:
            continue
        lw = j.get("leaked_weight_vs_cutoff") or {}          # P(∃ mode ≥ N): the conservative,
        gp = lambda k: float(lw.get(str(k), lw.get(k, 0.0)))  # whole-lattice truncation error
        recs.append(dict(L=j["L"], A=j["A"], dim=j.get("dim", 3), Nf=j.get("Nf_deepest"),
                         fill=j["A"] / (j["L"] ** j.get("dim", 3)), p=p,
                         # per-mode histogram tail (Σ p(n≥N)):
                         leak_nb1=float(p[2:].sum()), leak_nb2=float(p[NB2_KEEP:].sum()),
                         leak_nb3=float(p[8:].sum()) if p.size > 8 else 0.0,
                         # whole-lattice leaked weight P(∃ mode ≥ N) (matches topic-04 adequacy fig):
                         px_nb1=gp(2), px_nb2=gp(4), px_nb3=gp(8)))
    recs.sort(key=lambda r: (r["L"], r["A"]))
    return recs


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, axis="y", color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(recs, out_base):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    cols = [BLUE, CRIT, GREEN, PURP, ORANGE]
    nmax = max(r["p"].size for r in recs)

    # shade the n_b=2 CUT region (levels ≥4) — the tail n_b=2 drops
    ax.axvspan(NB2_KEEP - 0.5, nmax - 0.5, color=CRIT, alpha=0.06, zorder=0)
    ax.axvline(NB2_KEEP - 0.5, ls="--", color=CRIT, lw=1.4, zorder=2)
    ax.annotate("n_b=2 keeps levels 0–3 →|← cut (tail)", (NB2_KEEP - 0.5, 0.5),
                color=CRIT, fontsize=8.5, ha="center", va="bottom", rotation=0,
                textcoords="offset points", xytext=(0, 2))

    for r, c in zip(recs, cols):
        n = np.arange(r["p"].size)
        # bars are PER-MODE p(n); the quoted % is the WHOLE-LATTICE P(∃ mode≥4) (different metric)
        lbl = (f"$L$={r['L']}, $A$={r['A']}" + (" (dense)" if r["A"] >= r["L"] ** r["dim"] else "")
               + f"  —  P(∃ mode≥4) = {r['px_nb2']*100:.2f}%")
        ax.step(n, np.clip(r["p"], 1e-9, None), where="mid", color=c, lw=1.9, zorder=4, label=lbl)
        ax.plot(n, np.clip(r["p"], 1e-9, None), "o", color=c, ms=3.5, mec=SURFACE, mew=0.5, zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlim(-0.5, min(nmax - 0.5, 11.5))
    ax.set_xticks(range(0, min(nmax, 12)))
    ax.set_xlabel("boson occupation level $n$  (a mode holds $n$ pions)", color=INK2, fontsize=10)
    ax.set_ylabel("PER-MODE population  $p(n)$  (|c|²-weighted fraction of modes)", color=INK2, fontsize=10)
    ax.set_title("Selected-CI occupation histogram — no population near the $N_f$=16 boundary "
                 "(bars = per-mode $p(n)$; legend = whole-lattice P)", color=INK, fontsize=10.3,
                 loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    _style(ax)
    # diagnostic caveat: agreement with N_f=8 shows the same low-occupation SELECTED basin, not that
    # the true tail is solver-converged (each state has only ~1.5-3k determinants).
    ax.annotate("no selected population beyond $n$≈6 → same low-occupation basin\nas $N_f$=8 (a "
                "selected-state diagnostic, NOT solver-convergence of the true tail)",
                (5.4, 4e-6), color=MUTED, fontsize=7.5, ha="left", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/classical/nb_convergence")
    ap.add_argument("--out-dir", default="data/classical/nb_convergence")
    args = ap.parse_args()
    recs = load(args.data)
    assert recs, "no studyHist_*.json with occ_histogram found — run submit_nb_hist.sh first"
    make_figure(recs, f"{args.out_dir}/nb_occupation_histogram")

    md = ["# Selected-CI occupation histogram vs boson level (deep N_f=16 reference)\n",
          "_Per-level population p(n) of the SELECTED-CI reference state at N_f=16 (n_b=4). Two "
          "metrics: the per-mode histogram tail Σ_{n≥4} p(n), and the whole-lattice P(∃ mode ≥ 4) "
          "(they differ by ~the mode count). DIAGNOSTIC, not an energy bound: agreement with the N_f=8 "
          "reference shows the same low-occupation SELECTED basin was found — with only ~1.5–3k "
          "determinants in an astronomical space — NOT that the true ground-state tail is "
          "solver-converged._\n",
          "| L | A | filling | n_b=2 per-mode Σp(n≥4) | **n_b=2 P(∃≥4)** | n_b=1 P(∃≥2) |",
          "|--:|--:|:--|--:|--:|--:|"]
    for r in recs:
        md.append(f"| {r['L']} | {r['A']} | {r['fill']:.2f}"
                  f"{' (dense)' if r['A'] >= r['L']**r['dim'] else ''} | {r['leak_nb2']*100:.3f}% | "
                  f"**{r['px_nb2']*100:.2f}%** | {r['px_nb1']*100:.1f}% |")
    worst = max(r["px_nb2"] for r in recs)
    md.append(f"\n**Reading (diagnostic):** the selected state has no population beyond n≈6, and the "
              f"N_f=8→16 reruns give the same tail — so the found basin is low-occupation. n_b=1 P(∃≥2) "
              f"is {min(r['px_nb1'] for r in recs)*100:.0f}–{max(r['px_nb1'] for r in recs)*100:.0f}% "
              f"(strongly rejected); n_b=2 P(∃≥4) is at most {worst*100:.2f}%. This MOTIVATES n_b=2 but "
              f"does NOT bound the energy truncation error, nor prove solver-convergence of the true "
              f"tail (fixed ~few-k-determinant support). The energy gate (E_0(N_f) at core-converged "
              f"states, with seed uncertainty) is the separate L=2 study; the anchor stays conditional "
              f"on n_b=2 until it passes.\n")
    open(f"{args.out_dir}/nb_occupation_histogram_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_occupation_histogram_table.md")
    print("[done] " + " | ".join(f"L{r['L']}A{r['A']}: n_b2 tail {r['leak_nb2']*100:.2f}%" for r in recs))


if __name__ == "__main__":
    main()
