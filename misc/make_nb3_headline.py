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
    """Return per-L dict with n_b=3 T + qubits: exact where available, else scaled from n_b=2."""
    exactL = sorted(nb3)
    # L-stable scaling factors from the exact overlap (use L>=4 asymptote)
    big = [L for L in exactL if L >= 4]
    Tr = st.mean(nb3[L]["T"] / nb2[L]["T"] for L in big)
    qr = st.mean(nb3[L]["q"] / nb2[L]["q"] for L in big)
    lamr = st.mean(nb3[L]["lam"] / nb2[L]["lam"] for L in big)
    rows = {}
    for L in sorted(nb2):
        if L in nb3:
            rows[L] = dict(T=nb3[L]["T"], q=nb3[L]["q"], lam=nb3[L]["lam"], exact=True)
        else:
            rows[L] = dict(T=nb2[L]["T"] * Tr, q=nb2[L]["q"] * qr, lam=nb2[L]["lam"] * lamr, exact=False)
    return rows, dict(Tr=Tr, qr=qr, lamr=lamr)


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
                 zorder=5, label="n_b=3 EXACT (converged)")
    axT.semilogy(sc_, [rows[L]["T"] for L in sc_], "o", color=CRIT, ms=8, mfc="none", mew=1.8,
                 zorder=5, label=f"n_b=3 scaled (×{sc['Tr']:.0f})")
    axT.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axT.set_ylabel("QPE coherent-query $T$-count ($\\pi$, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    axT.set_title(f"a  Corrected T-count — n_b=3 is ×{sc['Tr']:.0f} the n_b=2 headline", color=INK,
                  fontsize=10.3, loc="left", weight="bold")
    axT.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _style(axT)
    # qubits
    axQ.plot(Ls, [nb2[L]["q"] for L in Ls], "--s", color=MUTED, lw=1.6, ms=5, mec=SURFACE, mew=1.0,
             zorder=3, label="n_b=2")
    axQ.plot(ex, [rows[L]["q"] for L in ex], "-o", color=BLUE, lw=2.2, ms=7, mec=SURFACE, mew=1.2,
             zorder=5, label="n_b=3 exact")
    axQ.plot(sc_, [rows[L]["q"] for L in sc_], "o", color=BLUE, ms=8, mfc="none", mew=1.8, zorder=5,
             label=f"n_b=3 scaled (×{sc['qr']:.2f})")
    axQ.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axQ.set_ylabel("total logical qubits", color=INK2, fontsize=9.5)
    axQ.set_title(f"b  Logical qubits — n_b=3 is ×{sc['qr']:.2f}", color=INK, fontsize=10.3,
                  loc="left", weight="bold")
    axQ.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _style(axQ)
    fig.suptitle("Quantum headline at the CONVERGED cutoff n_b=3 (energy gate: n_b=2 under-converged)",
                 fontsize=11.0, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb3_headline.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb3_headline.pdf / .png")

    md = ["# Corrected quantum headline — converged cutoff n_b=3\n",
          f"_The energy gate showed n_b=2 is under-converged; n_b=3 is the converged cutoff. Exact "
          f"n_b=3 compile for L=1..{max(nb3)}; L>{max(nb3)} scaled from the n_b=2 anchor by the "
          f"L-stable (L≥4) ratios: λ ×{sc['lamr']:.2f}, coherent-T ×{sc['Tr']:.1f}, qubits "
          f"×{sc['qr']:.3f}. π walk constant, ΔE=1 MeV._\n",
          "| L | T(n_b=2) | **T(n_b=3)** | ×T | qubits(n_b=2) | **qubits(n_b=3)** | source |",
          "|--:|--:|--:|--:|--:|--:|:--|"]
    for L in Ls:
        r = rows[L]
        md.append(f"| {L} | {nb2[L]['T']:.2e} | **{r['T']:.2e}** | {r['T']/nb2[L]['T']:.0f} | "
                  f"{nb2[L]['q']} | **{r['q']:.0f}** | {'exact' if r['exact'] else 'scaled'} |")
    L10 = rows[10]
    md.append(f"\n**Headline (L=10, A-independent):** corrected QPE coherent-query T ≈ **{L10['T']:.2e}** "
              f"(n_b=2 was {nb2[10]['T']:.2e}, ×{L10['T']/nb2[10]['T']:.0f} low), total logical qubits "
              f"≈ **{L10['q']:.0f}**. The ×{sc['Tr']:.0f} T increase = λ ×{sc['lamr']:.2f} × per-step "
              f"walk_T ×{sc['Tr']/sc['lamr']:.1f} (bigger Fock register + more Pauli terms). Conditional "
              f"note discharged: n_b=3 is the converged cutoff (energy gate); n_b=4 changes E_0 by "
              f"0.01 MeV. Caveat: gate is L=2 (per-mode cutoff ~L-independent); L=3 trap-limited.\n")
    open(f"{args.out_dir}/nb3_headline_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb3_headline_table.md")
    print(f"[done] scaling: T×{sc['Tr']:.1f} q×{sc['qr']:.3f} lam×{sc['lamr']:.2f}; "
          f"L=10 T {nb2[10]['T']:.2e}->{rows[10]['T']:.2e}")


if __name__ == "__main__":
    main()
