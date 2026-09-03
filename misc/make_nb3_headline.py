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

BLUE, ORANGE, CRIT, GREEN, MUTED = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(d, patt):
    o = {}
    for f in glob.glob(f"{d}/{patt}"):
        j = json.load(open(f))
        if not (j.get("done") and j.get("results")) or "rep2" in os.path.basename(f):
            continue
        r = j["results"][0]
        eps = (r.get("QPE_Budget") or {}).get("eps_qpe")
        nwalk = walk_queries(r["Physical_Lambda"], eps, PI) if eps else None
        o[r["L"]] = dict(lam=r["Physical_Lambda"], walkT=r["Walk_T_Count"], q=r["Logical_Qubits"],
                         T=(nwalk * r["Walk_T_Count"]) if nwalk else None)
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
    """Per-L dict with n_b=3 T + qubits: exact where the n_b=3 shard exists, else PROJECTED from the
    n_b=2 anchor by the L≥4 ratios — with a BAND from the observed ratio VARIATION (not just the mean,
    per re-audit P0-5). Missing L (no n_b=2 either) are skipped."""
    big = [L for L in sorted(nb3) if L >= 4 and L in nb2]
    if not big:                                            # fall back to any common L if <4 overlap
        big = [L for L in sorted(nb3) if L in nb2]
    Trs = [nb3[L]["T"] / nb2[L]["T"] for L in big]
    qrs = [nb3[L]["q"] / nb2[L]["q"] for L in big]
    lrs = [nb3[L]["lam"] / nb2[L]["lam"] for L in big]
    Tr, qr, lamr = st.mean(Trs), st.mean(qrs), st.mean(lrs)
    rows = {}
    for L in sorted(nb2):
        if L in nb3:                                       # direct compiled
            rows[L] = dict(T=nb3[L]["T"], q=nb3[L]["q"], lam=nb3[L]["lam"], exact=True,
                           Tlo=nb3[L]["T"], Thi=nb3[L]["T"], qlo=nb3[L]["q"], qhi=nb3[L]["q"])
        else:                                              # projected + band from ratio spread
            rows[L] = dict(T=nb2[L]["T"] * Tr, q=nb2[L]["q"] * qr, lam=nb2[L]["lam"] * lamr, exact=False,
                           Tlo=nb2[L]["T"] * min(Trs), Thi=nb2[L]["T"] * max(Trs),
                           qlo=nb2[L]["q"] * min(qrs), qhi=nb2[L]["q"] * max(qrs))
    sc = dict(Tr=Tr, qr=qr, lamr=lamr, Trange=(min(Trs), max(Trs)), qrange=(min(qrs), max(qrs)),
              lamrange=(min(lrs), max(lrs)), fitL=big)
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
                     mew=1.8, capsize=4, elinewidth=1.4, zorder=5, label="n_b=3 PROJECTED (ratio band)")
    axT.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axT.set_ylabel("QPE coherent-query $T$-count ($\\pi$, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    axT.set_title(f"a  T-count — n_b=3 is ×{sc['Trange'][0]:.0f}–{sc['Trange'][1]:.0f} the n_b=2 value",
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
                 "L=%d..10 (ratio band)" % (max(ex), max(ex) + 1), fontsize=10.6, color=INK,
                 y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb3_headline.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb3_headline.pdf / .png")

    exmax = max(L for L in rows if rows[L]["exact"])
    md = ["# Quantum headline at the L=2-selected cutoff n_b=3 — compiled L=1..%d + projected L=%d..10\n"
          % (exmax, exmax + 1),
          f"_The L=2 energy gate rejected n_b=2 and found n_b=3≈n_b=4 (at L=2); n_b=3 is the selected "
          f"cutoff scenario. **Direct compiled** n_b=3 for L=1..{exmax}; **ratio-PROJECTED** L={exmax+1}"
          f"..10 (not compiled) — projection uses the L≥4 ratios with a BAND from their variation: "
          f"λ ×{sc['lamrange'][0]:.2f}–{sc['lamrange'][1]:.2f}, coherent-T ×{sc['Trange'][0]:.1f}–"
          f"{sc['Trange'][1]:.1f}, qubits ×{sc['qrange'][0]:.3f}–{sc['qrange'][1]:.3f} (fit on L="
          f"{sc['fitL']}). π walk constant, ΔE=1 MeV. Large-volume cutoff adequacy is CONDITIONAL "
          f"(P0-4)._\n",
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
              f"walk_T ×{sc['Tr']/sc['lamr']:.1f}. NOTE: L=1..{exmax} are direct compiled estimates; "
              f"L={exmax+1}..10 are ratio projections. Cutoff selected at L=2 (n_b=3≈n_b=4); its total-"
              f"energy adequacy through L=10 is a separate volume-scaling test (P0-4).\n")
    open(f"{args.out_dir}/nb3_headline_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb3_headline_table.md")
    print(f"[done] T-range x{sc['Trange'][0]:.1f}-{sc['Trange'][1]:.1f}; compiled to L={exmax}; "
          f"L=10 T {rows[10]['T']:.2e} band [{rows[10]['Tlo']:.1e},{rows[10]['Thi']:.1e}]")


if __name__ == "__main__":
    main()
