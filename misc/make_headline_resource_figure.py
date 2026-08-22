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
                   terms=r.get("Pauli_Term_Count"), rot=r.get("Rotation_Count"),
                   fit_resid=(b.get("walk_T_fit") or {}).get("resid"),
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
    # All r3 points are pruning-clean; assert it so an invalid point can never be plotted
    # silently as accepted data (codex plot-audit requirement).
    bad = [(r["L"], r["n_b"]) for r in anchor + sweep if r["clean"] is not True]
    assert not bad, f"refusing to plot pruning-flagged points: {bad}"
    # L=1 is a degenerate anchor (no ordinary intersite bonds) — shown but SEPARATED from the
    # bulk line and excluded from any scaling read (audit item 8).
    bulk = [r for r in anchor if r["L"] >= 2]
    deg = [r for r in anchor if r["L"] == 1]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 4.1))
    fig.patch.set_facecolor(SURFACE)

    def _deg(ax, y, log=False):                          # L=1 as a hollow degenerate marker
        if deg:
            plot = ax.semilogy if log else ax.plot
            plot([1], [deg[0][y]], "o", color=SURFACE, ms=7, mec=MUTED, mew=1.5, zorder=3)

    # (A) logical qubits vs L — one walk/block-encoding register (exact (4+3 n_b)·S)
    axA.plot([r["L"] for r in bulk], [r["q"] for r in bulk], "-o", color=BLUE, lw=2.0, ms=7,
             mec=SURFACE, mew=1.5, zorder=3)
    _deg(axA, "q")
    axA.annotate(f"{bulk[-1]['q']:,}\nqubits", (bulk[-1]["L"], bulk[-1]["q"]),
                 textcoords="offset points", xytext=(-6, -2), ha="right", va="top",
                 fontsize=9, color=INK2)
    axA.set_xlabel("lattice size $L$  (dim=3, $L^3$ sites)", color=INK2, fontsize=9.5)
    axA.set_ylabel("logical qubits (one walk register)", color=INK2, fontsize=9.5)
    axA.set_title("a  Compiled register size", color=INK, fontsize=11, loc="left", weight="bold")
    _style(axA)

    # (B) coherent walk-query T-count vs L (log y) — the compiled anchor, all pruning-clean
    axB.semilogy([r["L"] for r in bulk], [r["qpeT"] for r in bulk], "-o", color=BLUE, lw=2.0,
                 ms=7, mec=SURFACE, mew=1.5, zorder=3, label="compiled, pruning-clean")
    _deg(axB, "qpeT", log=True)
    axB.plot([], [], "o", color=SURFACE, mec=MUTED, mew=1.5, label="$L$=1 (degenerate anchor)")
    axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("coherent walk-query $T$-count", color=INK2, fontsize=9.5)
    axB.set_title("b  Compiled query cost vs volume", color=INK, fontsize=11, loc="left", weight="bold")
    axB.annotate(r"$\Delta E_\mathrm{total}=1$ MeV", (0.04, 0.93), xycoords="axes fraction",
                 fontsize=8.5, color=MUTED)
    axB.legend(frameon=False, fontsize=8.0, loc="lower right", labelcolor=INK2)
    _style(axB)

    # (C) coherent walk-query T-count vs n_b — resource sensitivity CONDITIONAL on the cutoff
    for Lx, col, mk in ((2, BLUE, "o"), (3, ORANGE, "s")):
        pts = sorted([r for r in sweep if r["L"] == Lx], key=lambda x: x["n_b"])
        nb = [r["n_b"] for r in pts]; qpe = [r["qpeT"] for r in pts]
        axC.semilogy(nb, qpe, "-", color=col, lw=2.0, marker=mk, ms=7, mec=SURFACE, mew=1.5,
                     zorder=3)
        axC.annotate(f"$L$={Lx}", (nb[-1], qpe[-1]), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=9, color=col, weight="bold")
    axC.set_xlabel("boson register $n_b$  (bits/mode)", color=INK2, fontsize=9.5)
    axC.set_ylabel("coherent walk-query $T$-count", color=INK2, fontsize=9.5)
    axC.set_title("c  Cost sensitivity to cutoff $n_b$", color=INK, fontsize=11, loc="left", weight="bold")
    axC.set_xticks([1, 2, 3, 4])
    axC.annotate("conditional on $n_b$\n(physical adequacy set\nby classical study)",
                 (1.05, 0.05), xycoords="axes fraction", fontsize=7.8, color=MUTED, style="italic")
    _style(axC)

    fig.suptitle("Compiled PauliLCU resources for QPE ground-state energy estimation",
                 fontsize=12.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def _fmt(x, sci=False):
    return f"{x:.3e}" if sci else f"{x:,.0f}"


CAPTION = ("Logical fault-tolerant COHERENT WALK-QUERY resources for qubitized phase estimation "
           "of the corrected dynamical-pion chiral-EFT Hamiltonian (Fock encoding, dim=3, one "
           "PauliLCU walk). Total algorithmic error DeltaE_total = 1 MeV, split by a total-T "
           "optimization between phase resolution and block-encoding synthesis; every walk-T is at "
           "the resulting circuit precision (log-linear in log2(1/cp), verified with a 3rd interior "
           "sample, residual 0). All points are pruning-clean (discarded coefficient one-norm below "
           "the allocated budget): the estimate is of the EXACT target Hamiltonian. lambda, "
           "N_walk, and the logical-qubit count are exact lattice combinatorics. 'logical qubits' "
           "is ONE walk/block-encoding register and EXCLUDES the QPE phase register, input-state "
           "preparation, distillation factories, and routing. Costs are CONDITIONAL on n_b=2 "
           "(boson cutoff; physical adequacy to be set by the classical cutoff study) and on "
           "input-state preparation / success probability (repetition ~1/p0 reported separately); "
           "they are NOT a complete ground-state-energy algorithm cost. L varies at fixed lattice "
           "spacing (finite-volume, not continuum). L=1 has no ordinary intersite bonds (degenerate "
           "smoke anchor; excluded from bulk scaling).")


def prune_tag(r):
    return "clean" if r["clean"] is True else ("FLAGGED" if r["clean"] is False else "?")


def make_tables(anchor, sweep, out_base):
    # --- markdown ---
    md = ["# Headline quantum-resource anchor (compiled Fock/PauliLCU, corrected H)\n",
          "_" + CAPTION + "_\n",
          "**Anchor — n_b=2, dim=3:**\n",
          "| L | sites | λ (MeV) | logical qubits | Pauli terms | rotations | N_walk | walk-T | coherent-query T | pruning |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in anchor:
        note = " *(degenerate)*" if r["L"] == 1 else ""
        md.append(f"| {r['L']}{note} | {r['L']**3} | {r['lam']:,.0f} | {r['q']:,} | {r['terms']:,} "
                  f"| {r['rot']:,} | {r['nwalk']:.2e} | {r['walkT']:.2e} | {r['qpeT']:.2e} | {prune_tag(r)} |")
    md.append("\n**Cost sensitivity to the boson cutoff n_b** (conditional; L=2, L=3):\n")
    md.append("| L | n_b | λ (MeV) | logical qubits | Pauli terms | coherent-query T | pruning |")
    md.append("|--:|--:|--:|--:|--:|--:|:--|")
    for Lx in (2, 3):
        for r in sorted([x for x in sweep if x["L"] == Lx], key=lambda x: x["n_b"]):
            md.append(f"| {r['L']} | {r['n_b']} | {r['lam']:,.0f} | {r['q']:,} | {r['terms']:,} "
                      f"| {r['qpeT']:.2e} | {prune_tag(r)} |")
    open(f"{out_base}.md", "w").write("\n".join(md) + "\n")

    # --- LaTeX (booktabs) ---
    cap_tex = CAPTION.replace("~1/p0", r"$\sim 1/p_0$").replace("DeltaE_total", r"$\Delta E_\mathrm{total}$") \
                     .replace("lambda,", r"$\lambda$,").replace("n_b=2", r"$n_b{=}2$") \
                     .replace("N_walk", r"$N_\mathrm{walk}$").replace("log2(1/cp)", r"$\log_2(1/\mathrm{cp})$")
    tex = [r"% Compiled PauliLCU anchor (corrected H). Requires booktabs, siunitx.",
           r"\begin{table}[t]\centering\small",
           r"\caption{" + cap_tex + r"}",
           r"\label{tab:pauli_lcu_anchor}",
           r"\begin{tabular}{r r r r r r r l}\toprule",
           r"$L$ & sites & $\lambda$ (MeV) & qubits & Pauli terms & $N_\mathrm{walk}$ & "
           r"coh.\ query $T$ & pruning\\\midrule"]
    for r in anchor:
        note = r"\,$^{\dagger}$" if r["L"] == 1 else ""
        tex.append(f"{r['L']}{note} & {r['L']**3} & {r['lam']:,.0f} & {r['q']:,} & "
                   f"{r['terms']:,} & \\num{{{r['nwalk']:.2e}}} & \\num{{{r['qpeT']:.2e}}} & "
                   f"{prune_tag(r)}\\\\")
    tex += [r"\bottomrule\end{tabular}",
            r"\par\footnotesize$^{\dagger}$ $L{=}1$ degenerate anchor (no intersite bonds); "
            r"excluded from bulk scaling.",
            r"\end{table}"]
    open(f"{out_base}.tex", "w").write("\n".join(tex) + "\n")
    print(f"[tbl] wrote {out_base}.md / .tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-21/headline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    anchor, sweep = load(args.data)
    print(f"[load] anchor n_b=2: L={[r['L'] for r in anchor]}; sweep points: {len(sweep)}")
    make_figure(anchor, sweep, f"{args.out_dir}/headline_resources")
    make_tables(anchor, sweep, f"{args.out_dir}/headline_resources_table")


if __name__ == "__main__":
    main()
