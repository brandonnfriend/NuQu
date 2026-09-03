"""Old-vs-New comparison at the SELECTED cutoff n_b=3 (re-audit C4 regen).

The earlier comparison used n_b=2 (now the historical low cutoff). The L=2 energy gate selected n_b=3,
where qubitization's coherent-query T is ×33 higher (λ ×3.75 × per-step ×8.9) while Trotter's cost
grows only ~×5 — so the crossover shifts LEFT (from L≈4.3 to ~L=2). Qubitization uses the compiled
n_b=3 anchor (exact L=1..6 + projected L=7..10, from make_nb3_headline); Trotter is Watson's analytic
upper bound at n_b=3. A=1, ΔE=1 MeV. NOT a superiority claim — a scenario map.

    python -m misc.make_trotter_comparison_nb3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc.make_nb3_headline import load as load_q, build
from src_PI.trotter_theory.trotter_exact import qpe_cost

BLUE, ORANGE, CRIT, GREEN, MUTED = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LS = list(range(2, 11))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def _cross(Ls, ratio):
    """Interpolate (log-log) the L where Trotter/qubit ratio crosses 1."""
    for i in range(len(Ls) - 1):
        r0, r1 = ratio[i], ratio[i + 1]
        if (r0 - 1) * (r1 - 1) <= 0 and r0 != r1:
            f = (0 - np.log(r0)) / (np.log(r1) - np.log(r0))
            return Ls[i] + f * (Ls[i + 1] - Ls[i])
    return None


def qubit_T(rows, nb2, n_b):
    """Qubitization coherent-T per L at cutoff n_b (n_b=2 from the anchor, n_b=3 from build rows)."""
    if n_b == 2:
        return {L: nb2[L]["T"] for L in LS}
    return {L: rows[L]["T"] for L in LS if L in rows}


def trot_T(n_b):
    return {L: qpe_cost(L, A=1, dE=1.0, n_b_override=n_b)["total_T"] for L in LS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb2", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--nb3", default="data/quantum/nb3_anchor")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-24/trotter_comparison")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nb2 = load_q(args.nb2, "*fock_pauli*nb2*.json")
    nb3 = load_q(args.nb3, "*fock_pauli_nb3*.json")
    rows, sc = build(nb2, nb3)

    Q3, Q2 = qubit_T(rows, nb2, 3), qubit_T(rows, nb2, 2)
    T3, T2 = trot_T(3), trot_T(2)
    xc3 = _cross(LS, [T3[L] / Q3[L] for L in LS])
    xc2 = _cross(LS, [T2[L] / Q2[L] for L in LS])
    exmax = max(L for L in rows if rows[L]["exact"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    # (a) n_b=3 crossover: qubit (compiled + projected) vs Trotter, with n_b=2 as the historical ref
    exL = [L for L in LS if L <= exmax]; prL = [L for L in LS if L > exmax]
    axA.semilogy(exL, [Q3[L] for L in exL], "-o", color=BLUE, lw=2.2, ms=6, mec=SURFACE, mew=1.2,
                 zorder=5, label="qubitization n_b=3 (compiled)")
    axA.semilogy(prL, [Q3[L] for L in prL], "o", color=BLUE, ms=7, mfc="none", mew=1.6, zorder=5,
                 label="qubitization n_b=3 (projected)")
    axA.semilogy(LS, [T3[L] for L in LS], "--s", color=ORANGE, lw=2.0, ms=5.5, mec=SURFACE, mew=1.2,
                 zorder=4, label="Trotter n_b=3 (Watson UB)")
    axA.semilogy(LS, [Q2[L] for L in LS], ":", color=MUTED, lw=1.6, zorder=3,
                 label="qubitization n_b=2 (historical)")
    if xc3:
        axA.axvline(xc3, ls=":", color=CRIT, lw=1.3)
        axA.annotate(f"n_b=3 crossover\n$L$≈{xc3:.1f}", (xc3, Q3[2]), color=CRIT, fontsize=8.5,
                     textcoords="offset points", xytext=(6, 0), va="center")
    axA.set_xlabel("lattice size $L$  ($A$=1, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    axA.set_ylabel("QPE $T$-count", color=INK2, fontsize=9.5)
    axA.set_title("a  Old vs New at the selected cutoff n_b=3", color=INK, fontsize=10.3, loc="left",
                  weight="bold")
    axA.legend(frameon=False, fontsize=7.6, loc="lower right", labelcolor=INK2)
    _style(axA)
    # (b) crossover position vs cutoff — moves LEFT with n_b
    nbs, xcs = [2, 3], [xc2, xc3]
    axB.plot(nbs, xcs, "-o", color=CRIT, lw=2.0, ms=9, mec=SURFACE, mew=1.2, zorder=4)
    for nb, xc in zip(nbs, xcs):
        if xc:
            axB.annotate(f"$L$≈{xc:.1f}", (nb, xc), textcoords="offset points", xytext=(8, 0),
                         fontsize=9, color=INK2, va="center")
    axB.set_xticks([2, 3]); axB.set_xlim(1.7, 3.6); axB.set_xlabel("boson cutoff $n_b$", color=INK2, fontsize=9.5)
    axB.set_ylabel("crossover $L$ (qubit↔Trotter, $A$=1)", color=INK2, fontsize=9.5)
    axB.set_title("b  Crossover moves LEFT with the cutoff", color=INK, fontsize=10.3, loc="left",
                  weight="bold")
    axB.annotate("higher cutoff → qubitization\nloses its small-$L$ edge", (2.5, (xc2 + xc3) / 2),
                 color=MUTED, fontsize=8, ha="center", style="italic")
    _style(axB)
    fig.suptitle("Old vs New at n_b=3 (selected) — crossover shifts %.1f→%.1f; n_b=2 kept as historical"
                 % (xc2 or 0, xc3 or 0), fontsize=10.6, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/trotter_comparison_nb3.{e}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/trotter_comparison_nb3.pdf / .png")

    md = ["# Old vs New at the SELECTED cutoff n_b=3 (re-audit C4 regen)\n",
          f"_The comparison is regenerated at n_b=3 (the L=2-gate-selected cutoff); n_b=2 is the "
          f"historical low cutoff. Qubitization = compiled n_b=3 (L=1..{exmax}) + projected L>{exmax}; "
          f"Trotter = Watson analytic UB at n_b=3. A=1, ΔE=1 MeV. NOT a superiority claim._\n",
          "| L | qubit n_b=3 T | Trotter n_b=3 T | ratio T/Q | cheaper |",
          "|--:|--:|--:|--:|:--|"]
    for L in LS:
        r = T3[L] / Q3[L]
        md.append(f"| {L} | {Q3[L]:.2e} | {T3[L]:.2e} | {r:.2f} | "
                  f"{'qubitization' if r > 1 else 'Trotter'} |")
    md.append(f"\n**Reading:** the crossover shifts LEFT from **L≈{xc2:.1f} (n_b=2)** to "
              f"**L≈{xc3:.1f} (n_b=3)** — at the selected cutoff, qubitization wins only at the "
              f"smallest L, because moving n_b=2→3 raises qubitization's T ×{sc['Tr']:.0f} (λ ×"
              f"{sc['lamr']:.2f} × per-step ×{sc['Tr']/sc['lamr']:.1f}) while Trotter grows only ~×5. "
              f"Trotter stays a loose upper bound (tightening it moves the crossover further left). "
              f"n_b=2 retained as the historical comparison.\n")
    open(f"{args.out_dir}/trotter_comparison_nb3.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/trotter_comparison_nb3.md")
    print(f"[done] crossover n_b=2:{xc2:.2f} -> n_b=3:{xc3:.2f}")


if __name__ == "__main__":
    main()
