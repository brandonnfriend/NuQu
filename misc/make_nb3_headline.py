"""Corrected quantum headline at the CONVERGED cutoff n_b=3 (the energy gate showed n_b=2 is
under-converged: E_0 off 4-7 MeV, binding energy ~91 MeV). Uses the EXACT n_b=3 compile where it
fits (L=1..6, +L=7 when it lands) and scales L=7/8..10 from the n_b=2 anchor by the L-stable ratios
measured on L=4..6 (λ ×3.79, coherent-T ×~33, qubits ×~1.30 — all verified L-independent).

    python -m misc.make_nb3_headline
"""
import argparse
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src_PI.estimation.qpe_cost import walk_queries, WALK_QUERY_CONSTANT_HEISENBERG as PI
from misc.nb3_padding_model import _load as _pad_load, project as _pad_project

BLUE, ORANGE, CRIT, GREEN, MUTED = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(d, patt):
    """Full per-L schema (lam, walkT, q, terms, eps, a, b) + the composed QPE T. Shared by the
    headline, the padding projection, and the Trotter comparison."""
    o = _pad_load(d, patt)
    for r in o.values():
        nwalk = walk_queries(r["lam"], r["eps"], PI) if r.get("eps") else None
        r["T"] = (nwalk * r["walkT"]) if nwalk else None
    return o


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def build(nb2, nb3):
    """Per-L n_b=3 T + qubits: COMPILED where the n_b=3 shard exists (L=1..7), else PADDING-MODEL
    projected (L=8..10). walk_T is power-of-2 quantized (pyLIQTR pads PREPARE to 2^ceil(log2 terms)),
    so a smooth per-step total-T ratio JITTERS at bin boundaries — the L=7 ratio (16.3) is a genuine
    bin mis-alignment vs the smooth ~33 at L=3..6, NOT representative. So walk_T is projected by the
    quantization-aware `nb3_padding_model` (back-tested <1.1% on L=1..7); λ keeps its ×3.75 ratio and
    qubits their ×1.30 ratio. The projected T band spans the bins n_terms×(1±10%) can occupy. Display
    ratios (sc) are taken from the SMOOTH regime L=4..6 (L=7 excluded). Same (rows, sc) schema as before."""
    rows_p, meta = _pad_project(nb2, nb3)
    ls46 = meta["ls46"]                                    # [4,5,6] — smooth, bin-aligned regime
    Trs = [rows_p[L]["T"] / nb2[L]["T"] for L in ls46]
    qrs = [nb3[L]["q"] / nb2[L]["q"] for L in ls46]
    lrs = [nb3[L]["lam"] / nb2[L]["lam"] for L in ls46]
    Tr, qr, lamr = st.mean(Trs), st.mean(qrs), st.mean(lrs)
    rows = {}
    for L in sorted(rows_p):
        r = rows_p[L]
        if r["exact"]:                                     # direct compiled (L=1..7)
            rows[L] = dict(T=r["T"], q=r["q"], lam=r["lam"], exact=True,
                           Tlo=r["T"], Thi=r["T"], qlo=r["q"], qhi=r["q"], P=r["P"], terms=r["terms"])
        else:                                              # padding-model projection (L=8..10)
            rows[L] = dict(T=r["T"], q=r["q"], lam=r["lam"], exact=False,
                           Tlo=r["Tlo"], Thi=r["Thi"],
                           qlo=nb2[L]["q"] * min(qrs), qhi=nb2[L]["q"] * max(qrs), P=r["P"], terms=r["terms"])
    sc = dict(Tr=Tr, qr=qr, lamr=lamr, Trange=(min(Trs), max(Trs)), qrange=(min(qrs), max(qrs)),
              lamrange=(min(lrs), max(lrs)), fitL=ls46, model=meta["mp"])
    return rows, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb2", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--nb3", default="data/quantum/nb3_anchor")
    ap.add_argument("--out-dir", default="data/quantum/nb3_anchor")
    args = ap.parse_args()
    nb2 = load(args.nb2, "*fock_pauli*nb2*.json")
    nb3 = load(args.nb3, "*fock_pauli_nb3*.json")
    assert nb3, "no n_b=3 anchor data"
    rows, sc = build(nb2, nb3)
    Ls = sorted(rows)

    fig, (axT, axQ) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    # T-count
    axT.semilogy(Ls, [nb2[L]["T"] for L in Ls], "--s", color=MUTED, lw=1.6, ms=5, mec=SURFACE, mew=1.0,
                 zorder=3, label="n_b=2 (under-converged)")
    ex = [L for L in Ls if rows[L]["exact"]]; sc_ = [L for L in Ls if not rows[L]["exact"]]
    axT.semilogy(ex, [rows[L]["T"] for L in ex], "-o", color=CRIT, lw=2.2, ms=7, mec=SURFACE, mew=1.2,
                 zorder=5, label=f"n_b=3 COMPILED (L=1..{max(ex)})")
    if sc_:
        yerr = [[rows[L]["T"] - rows[L]["Tlo"] for L in sc_], [rows[L]["Thi"] - rows[L]["T"] for L in sc_]]
        axT.errorbar(sc_, [rows[L]["T"] for L in sc_], yerr=yerr, fmt="o", color=CRIT, ms=8, mfc="none",
                     mew=1.8, capsize=4, elinewidth=1.4, zorder=5, label="n_b=3 PROJECTED (padding model)")
    axT.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axT.set_ylabel("QPE coherent-query $T$-count ($\\pi$, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    axT.set_title(f"a  T-count — n_b=3 is ×{sc['Tr']:.0f} the n_b=2 value (smooth regime L=4–6)",
                  color=INK, fontsize=10.0, loc="left", weight="bold")
    axT.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _style(axT)
    # qubits
    axQ.plot(Ls, [nb2[L]["q"] for L in Ls], "--s", color=MUTED, lw=1.6, ms=5, mec=SURFACE, mew=1.0,
             zorder=3, label="n_b=2 (historical low cutoff)")
    axQ.plot(ex, [rows[L]["q"] for L in ex], "-o", color=BLUE, lw=2.2, ms=7, mec=SURFACE, mew=1.2,
             zorder=5, label="n_b=3 compiled")
    if sc_:
        qerr = [[rows[L]["q"] - rows[L]["qlo"] for L in sc_], [rows[L]["qhi"] - rows[L]["q"] for L in sc_]]
        axQ.errorbar(sc_, [rows[L]["q"] for L in sc_], yerr=qerr, fmt="o", color=BLUE, ms=8, mfc="none",
                     mew=1.8, capsize=4, elinewidth=1.4, zorder=5, label="n_b=3 projected")
    axQ.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axQ.set_ylabel("total logical qubits", color=INK2, fontsize=9.5)
    axQ.set_title(f"b  Logical qubits — n_b=3 is ×{sc['qr']:.2f}", color=INK, fontsize=10.3,
                  loc="left", weight="bold")
    axQ.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _style(axQ)
    fig.suptitle("Quantum headline at the L=2-selected cutoff n_b=3 — COMPILED L=1..%d + PROJECTED "
                 "L=%d..10 (padding model)" % (max(ex), max(ex) + 1), fontsize=10.6, color=INK,
                 y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb3_headline.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb3_headline.pdf / .png")

    exmax = max(L for L in rows if rows[L]["exact"])
    md = ["# Quantum headline at the L=2-selected cutoff n_b=3 — compiled L=1..%d + projected L=%d..10\n"
          % (exmax, exmax + 1),
          f"_The L=2 energy gate rejected n_b=2 and found n_b=3≈n_b=4 (at L=2); n_b=3 is the selected "
          f"cutoff scenario. **Direct compiled** n_b=3 for L=1..{exmax}; **PADDING-MODEL projected** "
          f"L={exmax+1}..10 (not compiled). walk_T is power-of-2 quantized (pyLIQTR pads PREPARE to "
          f"2^⌈log₂ terms⌉), so a smooth per-step ratio jitters at bin boundaries — the L=7 total-T "
          f"ratio (16.3) is a genuine bin mis-alignment vs the ~33 at L=3..6, NOT representative. So "
          f"walk_T is projected by the quantization-aware model (n_terms/L³={sc['model']['c0']:.0f}"
          f"{sc['model']['c1']:+.0f}/L → padded bin → walk_T; back-tested <1.1% on L=1..{exmax}); λ keeps "
          f"its ×{sc['lamr']:.2f} ratio, qubits ×{sc['qr']:.2f}. The projected T band spans the bins "
          f"n_terms×(1±10%) can occupy (widest at L=9, just under the 2²² edge). π walk constant, "
          f"ΔE=1 MeV. Large-volume cutoff adequacy is CONDITIONAL (P0-4)._\n",
          "| L | T(n_b=2) | **T(n_b=3)** | band | ×T | qubits(n_b=3) | source |",
          "|--:|--:|--:|--:|--:|--:|:--|"]
    for L in Ls:
        r = rows[L]
        band = "—" if r["exact"] else f"[{r['Tlo']:.1e}, {r['Thi']:.1e}]"
        md.append(f"| {L} | {nb2[L]['T']:.2e} | **{r['T']:.2e}** | {band} | {r['T']/nb2[L]['T']:.0f} | "
                  f"**{r['q']:.0f}** | {'compiled' if r['exact'] else 'projected'} |")
    L10 = rows[10]
    md.append(f"\n**Headline (L=10, A-independent, PROJECTED):** QPE coherent-query T ≈ **{L10['T']:.2e}** "
              f"(band [{L10['Tlo']:.1e}, {L10['Thi']:.1e}]; n_b=2 was {nb2[10]['T']:.2e}), total logical "
              f"qubits ≈ **{L10['q']:.0f}**. The ×{sc['Tr']:.0f} T rise = λ ×{sc['lamr']:.2f} × per-step "
              f"walk_T ×{sc['Tr']/sc['lamr']:.1f} (smooth-regime L=4..6). NOTE: L=1..{exmax} are direct "
              f"compiled estimates; L={exmax+1}..10 are padding-model projections (walk_T quantized in "
              f"powers of two; back-tested <1.1%). Cutoff selected at L=2 (n_b=3≈n_b=4); its total-"
              f"energy adequacy through L=10 is a separate volume-scaling test (P0-4).\n")
    open(f"{args.out_dir}/nb3_headline_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb3_headline_table.md")
    print(f"[done] T-range x{sc['Trange'][0]:.1f}-{sc['Trange'][1]:.1f}; compiled to L={exmax}; "
          f"L=10 T {rows[10]['T']:.2e} band [{rows[10]['Tlo']:.1e},{rows[10]['Thi']:.1e}]")


if __name__ == "__main__":
    main()
