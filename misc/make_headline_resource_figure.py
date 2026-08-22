"""Headline quantum-resource figure + table from the regenerated PauliLCU anchor (290818).

Reads the manifest-versioned quantum round-2 shards (Fock/PauliLCU, n_b=2 anchor L=1..10 +
n_b sweep L=2,3), and writes:
  * headline_resources.pdf / .png  — 3-panel figure (qubits vs L, QPE-T vs L, QPE-T vs n_b)
  * headline_resources_table.md    — the anchor + n_b-sweep tables (review)
  * headline_resources_table.tex   — LaTeX booktabs versions (manuscript)

Reproducible: point --data at the pulled shard dir. Colors are the validated dataviz
reference slots (blue slot-1, orange slot-2; critical status for the pruning-flagged point).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- dataviz reference palette (light; slots 1,2 pass all-pairs) + ink tokens ------------- #
BLUE, ORANGE, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load(data_dir):
    anchor, sweep = [], []
    for f in sorted(glob.glob(f"{data_dir}/*fock_pauli*.json")):
        j = json.load(open(f))
        if not j.get("done") or not j["results"]:
            continue
        r = j["results"][0]
        b = r.get("QPE_Budget") or {}
        rec = dict(L=r["L"], n_b=r["n_b"], lam=r["Physical_Lambda"], q=r["Logical_Qubits"],
                   nwalk=r["QPE_Walk_Queries"], walkT=r["Walk_T_Count"],
                   qpeT=r["QPE_Total_T_Count"], clean=b.get("prune_within_budget", None),
                   rep="rep2" in os.path.basename(f))
        if rec["n_b"] == 2 and not rec["rep"]:
            anchor.append(rec)
        if rec["L"] in (2, 3) and not rec["rep"]:
            sweep.append(rec)
    anchor.sort(key=lambda x: x["L"])
    return anchor, sweep


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0)
    ax.set_axisbelow(True)


def make_figure(anchor, sweep, out_base):
    clean = [r for r in anchor if r["clean"]]
    flag = [r for r in anchor if r["clean"] is False]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 4.1))
    fig.patch.set_facecolor(SURFACE)

    # (A) logical qubits vs L — hardware/space feasibility (n_b=2, exact (4+3n_b)*S)
    Ls = [r["L"] for r in anchor]; qs = [r["q"] for r in anchor]
    axA.plot(Ls, qs, "-", color=BLUE, lw=2.0, zorder=2)
    axA.plot([r["L"] for r in clean], [r["q"] for r in clean], "o", color=BLUE, ms=7,
             mec=SURFACE, mew=1.5, zorder=3)
    if flag:
        axA.plot([r["L"] for r in flag], [r["q"] for r in flag], "D", color=SURFACE, ms=8,
                 mec=CRIT, mew=1.8, zorder=4)
    axA.annotate(f"{qs[-1]:,}\nqubits", (Ls[-1], qs[-1]), textcoords="offset points",
                 xytext=(-6, -2), ha="right", va="top", fontsize=9, color=INK2)
    axA.set_xlabel("lattice size $L$  (dim=3, $L^3$ sites)", color=INK2, fontsize=9.5)
    axA.set_ylabel("logical qubits (one walk register)", color=INK2, fontsize=9.5)
    axA.set_title("a  Compiled register size", color=INK, fontsize=11, loc="left", weight="bold")
    _style(axA)

    # (B) total QPE T-count vs L — time cost (log y), the compiled anchor
    axB.semilogy(Ls, [r["qpeT"] for r in anchor], "-", color=BLUE, lw=2.0, zorder=2)
    axB.semilogy([r["L"] for r in clean], [r["qpeT"] for r in clean], "o", color=BLUE, ms=7,
                 mec=SURFACE, mew=1.5, zorder=3, label="compiled, pruning-clean")
    if flag:
        axB.semilogy([r["L"] for r in flag], [r["qpeT"] for r in flag], "D", color=SURFACE,
                     ms=8, mec=CRIT, mew=1.8, zorder=4, label="pruning-flagged ($L$=10)")
    axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("total QPE $T$-count (coherent-query)", color=INK2, fontsize=9.5)
    axB.set_title("b  Compiled QPE cost vs volume", color=INK, fontsize=11, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    _style(axB)

    # (C) QPE T-count vs n_b at fixed L — the classically-informed reduction lever
    for Lx, col, mk in ((2, BLUE, "o"), (3, ORANGE, "s")):
        pts = sorted([r for r in sweep if r["L"] == Lx], key=lambda x: x["n_b"])
        nb = [r["n_b"] for r in pts]; qpe = [r["qpeT"] for r in pts]
        axC.semilogy(nb, qpe, "-", color=col, lw=2.0, marker=mk, ms=7, mec=SURFACE, mew=1.5,
                     label=f"$L$={Lx}", zorder=3)
        axC.annotate(f"$L$={Lx}", (nb[-1], qpe[-1]), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=9, color=col, weight="bold")
    axC.set_xlabel("boson register $n_b$  (bits/mode)", color=INK2, fontsize=9.5)
    axC.set_ylabel("total QPE $T$-count", color=INK2, fontsize=9.5)
    axC.set_title("c  Reduction lever: cost vs cutoff", color=INK, fontsize=11, loc="left", weight="bold")
    axC.set_xticks([1, 2, 3, 4])
    axC.annotate("smaller $n_b$\n= the reduction", (1.05, 0.06), xycoords="axes fraction",
                 fontsize=8.5, color=MUTED, style="italic")
    _style(axC)

    fig.suptitle("Compiled PauliLCU quantum resources for QPE GSEE — dynamical-pion χEFT "
                 "(corrected $H$, budget-optimal precision)", fontsize=12, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def _fmt(x, sci=False):
    return f"{x:.3e}" if sci else f"{x:,.0f}"


def make_tables(anchor, sweep, out_base):
    # --- markdown ---
    md = ["# Headline quantum-resource anchor (compiled Fock/PauliLCU, corrected H, budget-optimal)\n",
          "**Anchor — n_b=2, dim=3** (λ, N_walk, qubits are exact combinatorics; walk-T/QPE-T at the total-T-optimal precision):\n",
          "| L | sites | λ (MeV) | logical qubits | N_walk | walk-T | QPE-T | pruning |",
          "|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in anchor:
        tag = "clean" if r["clean"] else ("**flagged**" if r["clean"] is False else "—")
        md.append(f"| {r['L']} | {r['L']**3} | {r['lam']:,.0f} | {r['q']:,} | {r['nwalk']:.2e} "
                  f"| {r['walkT']:.2e} | {r['qpeT']:.2e} | {tag} |")
    md.append("\n**Reduction lever — QPE-T vs boson register n_b** (L=2, L=3):\n")
    md.append("| L | n_b | λ (MeV) | logical qubits | QPE-T |")
    md.append("|--:|--:|--:|--:|--:|")
    for Lx in (2, 3):
        for r in sorted([x for x in sweep if x["L"] == Lx], key=lambda x: x["n_b"]):
            md.append(f"| {r['L']} | {r['n_b']} | {r['lam']:,.0f} | {r['q']:,} | {r['qpeT']:.2e} |")
    open(f"{out_base}.md", "w").write("\n".join(md) + "\n")

    # --- LaTeX (booktabs) ---
    tex = [r"% Compiled PauliLCU anchor (corrected H, budget-optimal precision). Requires booktabs, siunitx.",
           r"\begin{table}[t]\centering\small",
           r"\caption{Compiled Fock/PauliLCU quantum resources for QPE ground-state energy estimation "
           r"of the dynamical-pion $\chi$EFT (corrected Hamiltonian, $n_b{=}2$, total-$T$-optimal "
           r"precision). $\lambda$, $N_\mathrm{walk}$ and the logical-qubit count are exact lattice "
           r"combinatorics; $L{=}10$ is flagged for coefficient-pruning above budget.}",
           r"\label{tab:pauli_lcu_anchor}",
           r"\begin{tabular}{r r r r r r r l}\toprule",
           r"$L$ & sites & $\lambda$ (MeV) & qubits & $N_\mathrm{walk}$ & walk-$T$ & QPE-$T$ & pruning\\\midrule"]
    for r in anchor:
        tag = "clean" if r["clean"] else ("flagged" if r["clean"] is False else "--")
        tex.append(f"{r['L']} & {r['L']**3} & {r['lam']:,.0f} & {r['q']:,} & "
                   f"\\num{{{r['nwalk']:.2e}}} & \\num{{{r['walkT']:.2e}}} & "
                   f"\\num{{{r['qpeT']:.2e}}} & {tag}\\\\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{out_base}.tex", "w").write("\n".join(tex) + "\n")
    print(f"[tbl] wrote {out_base}.md / .tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quantum/2026-08-20/vertexfix_r2_290818")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-20/headline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    anchor, sweep = load(args.data)
    print(f"[load] anchor n_b=2: L={[r['L'] for r in anchor]}; sweep points: {len(sweep)}")
    make_figure(anchor, sweep, f"{args.out_dir}/headline_resources")
    make_tables(anchor, sweep, f"{args.out_dir}/headline_resources_table")


if __name__ == "__main__":
    main()
