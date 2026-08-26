"""Old (Watson analytic Trotter upper bound) vs New (compiled PauliLCU qubitization) — the honest
tiered/relative resource comparison (codex trotter_comparison_status_2026-08-24).

NOT a compiled Trotter estimate and NOT a clean algorithm-only comparison: it contrasts two
ARCHITECTURES at a matched physical scenario (n_b=2, 1 MeV, A) —
  * Trotter  = Watson's analytic product-formula UPPER BOUND (`trotter_exact.qpe_cost`), amplitude
               basis + locality-preserving fermion map, n_b=2 OVERRIDE (conditional low-cutoff);
  * Qubit.   = the DIRECT compiled Fock/PauliLCU coherent-query T (r3 shards), JW map, n_b=2.
Both at ΔE=1 MeV, A matched. Trotter is a LOOSE worst-case bound (tightening it moves the curve DOWN
→ crossover LEFT), so "Trotter wins at large L" is robust and "qubitization wins at small L" is the
fragile edge. The qubitization number uses the ADOPTED π constant (N_walk = π·λ/ε_qpe), recomputed
from each shard's own λ and ε_qpe — consistent with the headline (the raw shards carry √2·π; the
ε-split is C-independent, so this is a clean ×1/√2 tightening).

    python -m misc.make_trotter_comparison_figure
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src_PI.trotter_theory.trotter_exact import qpe_cost                     # noqa: E402
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters   # noqa: E402
from src_PI.estimation.qpe_cost import (                                     # noqa: E402
    walk_queries, WALK_QUERY_CONSTANT_HEISENBERG, WALK_QUERY_CONSTANT_BABBUSH_UB)


def qubit_total_T_pi(r):
    """Compiled qubitization coherent-query T with the ADOPTED π constant: recompute
    N_walk = π·λ/ε_qpe from the shard's own λ and ε_qpe, × per-step walk_T (not the raw √2·π
    QPE_Total_T_Count). Falls back to rescaling the raw aggregate ×π/√2π if ε_qpe is absent."""
    lam, walkT = r["Physical_Lambda"], r["Walk_T_Count"]
    eps = (r.get("QPE_Budget") or {}).get("eps_qpe")
    if eps:
        return walk_queries(lam, eps, constant=WALK_QUERY_CONSTANT_HEISENBERG) * walkT
    return float(r["QPE_Total_T_Count"]) * (WALK_QUERY_CONSTANT_HEISENBERG / WALK_QUERY_CONSTANT_BABBUSH_UB)

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load_qubitization(data_dir):
    """r3 compiled PauliLCU coherent-query T (π·λ/ε_qpe · walk_T) vs L at n_b=2 (L≥2)."""
    q = {}
    for f in sorted(glob.glob(f"{data_dir}/*fock_pauli*.json")):
        j = json.load(open(f))
        if not j.get("done") or not j["results"] or "rep2" in os.path.basename(f):
            continue
        r = j["results"][0]
        if r["n_b"] == 2 and r["L"] >= 2:
            q[r["L"]] = qubit_total_T_pi(r)
    return dict(sorted(q.items()))


def trotter_curve(Ls, A, dE=1.0, n_b=2):
    p = get_physical_parameters()
    return {L: qpe_cost(L, A=A, dE=dE, n_b_override=n_b, params=p)["total_T"] for L in Ls}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def _crossL(Ls, ratio):
    """Interpolate (log-log) the L where Trotter/Qubit ratio crosses 1."""
    for i in range(len(Ls) - 1):
        r0, r1 = ratio[i], ratio[i + 1]
        if (r0 - 1) * (r1 - 1) <= 0 and r0 != r1:
            f = (0 - np.log(r0)) / (np.log(r1) - np.log(r0))
            return Ls[i] + f * (Ls[i + 1] - Ls[i])
    return None


def make_figure(q, out_base):
    Ls = [L for L in q]                                     # L values with a compiled qubit point
    Q = np.array([q[L] for L in Ls])
    T1 = np.array([trotter_curve(Ls, 1)[L] for L in Ls])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.patch.set_facecolor(SURFACE)

    # (a) absolute T vs L, A=1, n_b=2
    axA.semilogy(Ls, Q, "-o", color=BLUE, lw=2.2, ms=7, mec=SURFACE, mew=1.5, zorder=4,
                 label="qubitization (PauliLCU), compiled")
    axA.semilogy(Ls, T1, "--s", color=ORANGE, lw=2.0, ms=6, mec=SURFACE, mew=1.5, zorder=3,
                 label="Trotter (Watson), analytic upper bound")
    xc = _crossL(Ls, T1 / Q)
    if xc:
        axA.axvline(xc, ls=":", color=MUTED, lw=1.2)
        axA.annotate(f"crossover $L\\approx${xc:.1f}", (xc, Q[0]), textcoords="offset points",
                     xytext=(6, 0), fontsize=8.5, color=INK2, va="bottom")
    axA.set_xlabel("lattice size $L$  (dim=3, $A$=1, $n_b$=2, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    axA.set_ylabel("QPE $T$-count", color=INK2, fontsize=9.5)
    axA.set_title("a  Old vs New, matched scenario", color=INK, fontsize=11, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=8.2, loc="upper left", labelcolor=INK2)
    _style(axA)

    # (b) ratio Trotter/Qubit vs L, for several A — the crossover shifts right with A
    for A, col in ((1, ORANGE), (8, CRIT), (40, "#7a3fb0")):
        T = np.array([trotter_curve(Ls, A)[L] for L in Ls])
        ratio = T / Q
        axB.semilogy(Ls, ratio, "-o", color=col, lw=1.9, ms=5.5, mec=SURFACE, mew=1.2, zorder=3,
                     label=f"$A$={A}")
        xcA = _crossL(Ls, ratio)                            # per-A crossover (own name; don't clobber xc)
        if xcA:
            axB.plot([xcA], [1.0], "v", color=col, ms=9, mec=SURFACE, mew=1.0, zorder=4)
    axB.axhline(1.0, ls="-", color=INK2, lw=1.1, zorder=2)
    # ratio = Trotter/qubitization: ABOVE 1 → Trotter costs more → qubitization wins; BELOW → Trotter wins
    axB.annotate("qubitization cheaper ↑", (Ls[0], 1.0), textcoords="offset points",
                 xytext=(2, 5), fontsize=8, color=INK2)
    axB.annotate("Trotter cheaper ↓", (Ls[0], 1.0), textcoords="offset points",
                 xytext=(2, -13), fontsize=8, color=INK2)
    axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("Trotter / qubitization $T$ ratio", color=INK2, fontsize=9.5)
    axB.set_title("b  Crossover shifts with nucleon number $A$", color=INK, fontsize=11,
                  loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2, title="markers = crossover")
    _style(axB)

    fig.suptitle("Watson-Trotter (analytic bound) vs compiled PauliLCU qubitization — architecture comparison, $n_b$=2",
                 fontsize=11.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")
    return Ls, Q, T1, xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-24/trotter_comparison")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    q = load_qubitization(args.anchor)
    assert q, "no qubitization anchor data"
    Ls, Q, T1, xc = make_figure(q, f"{args.out_dir}/trotter_comparison")

    md = ["# Old (Watson Trotter, analytic bound) vs New (compiled PauliLCU) — n_b=2, A=1, 1 MeV\n",
          "_Architecture comparison (amplitude/local-map Watson vs Fock/JW PauliLCU), matched n_b=2 /"
          " ΔE=1 MeV. Trotter = analytic product-formula UPPER BOUND (loose); qubitization = direct"
          " compiled coherent-query T with the adopted π constant (N_walk = π·λ/ε_qpe). See codex"
          " trotter_comparison_status 2026-08-24._\n",
          "| L | qubitization T (compiled) | Trotter T (A=1, bound) | ratio T/Q |",
          "|--:|--:|--:|--:|"]
    for i, L in enumerate(Ls):
        md.append(f"| {L} | {Q[i]:.2e} | {T1[i]:.2e} | {T1[i]/Q[i]:.2f} |")
    md.append(f"\n**Crossover (A=1): L ≈ {xc:.1f}.** Qubitization wins at small L (λ² penalty small);"
              " Trotter wins at large L (λ² ∝ sites² blows up while Trotter ∝ A·L³). Ratio ∝ A/L^3.5,"
              " so the crossover moves RIGHT with A (panel b). Trotter is a LOOSE upper bound —"
              " tightening it moves the crossover LEFT, shrinking the small-L qubitization region.\n")
    open(f"{args.out_dir}/trotter_comparison_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/trotter_comparison_table.md")
    print(f"[done] crossover A=1 at L≈{xc:.2f}; qubit L2..10: {Q[0]:.2e}..{Q[-1]:.2e}")


if __name__ == "__main__":
    main()
