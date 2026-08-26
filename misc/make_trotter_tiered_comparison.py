"""#6 — the tiered (H/M/L) Old-vs-New comparison (codex trotter_comparison_status_2026-08-24 §"A fair
three-tier comparison" + "Recommended paper figures"). NOT a superiority claim — a scenario map: the
algorithm ranking DEPENDS on the cutoff assumption, so the three tiers are distinct physical/cutoff
scenarios, deliberately NOT connected as a convergence curve.

  Tier L  empirical ground-state cutoff   n_b=2   qubitization COMPILED (L=2..10) + Trotter UB
  Tier M  Gaussian/reference cutoff        n_b=4   qubitization COMPILED (L=2,3)   + Trotter UB
  Tier H  Watson energy-bound cutoff       n_b≈33-40 (Lemma 5)   Trotter UB only (qubit compile-infeasible)

Trotter = Watson analytic product-formula UPPER BOUND (loose worst-case). Qubitization = direct
compiled Fock/PauliLCU coherent-query T (r3), N_walk with the adopted π constant (= π·λ/ε_qpe,
recomputed from each shard's λ / ε_qpe). Architecture comparison (amplitude/local-map vs Fock/JW),
A=1, ΔE=1 MeV. Writes trotter_tiered.{pdf,png} + trotter_tiered_table.md.

    python -m misc.make_trotter_tiered_comparison
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


def _qubit_total_T_pi(r):
    """Compiled qubitization coherent-query T with the ADOPTED π constant: N_walk = π·λ/ε_qpe
    (from the shard's own λ / ε_qpe) × walk_T — not the raw √2·π QPE_Total_T_Count."""
    lam, walkT = r["Physical_Lambda"], r["Walk_T_Count"]
    eps = (r.get("QPE_Budget") or {}).get("eps_qpe")
    if eps:
        return walk_queries(lam, eps, constant=WALK_QUERY_CONSTANT_HEISENBERG) * walkT
    return float(r["QPE_Total_T_Count"]) * (WALK_QUERY_CONSTANT_HEISENBERG / WALK_QUERY_CONSTANT_BABBUSH_UB)

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
_P = get_physical_parameters()
LS = list(range(2, 11))


def load_qubit_by_nb(data_dir):
    q = {}
    for f in sorted(glob.glob(f"{data_dir}/*fock_pauli*.json")):
        j = json.load(open(f))
        if not j.get("done") or not j["results"] or "rep2" in os.path.basename(f):
            continue
        r = j["results"][0]
        q[(r["L"], r["n_b"])] = _qubit_total_T_pi(r)
    return q


def trotter(nb, A=1):
    """Trotter T vs L; nb=None -> native Watson Lemma-5 cutoff. Returns {L: (T, n_b)}."""
    out = {}
    for L in LS:
        o = qpe_cost(L, A=A, dE=1.0, n_b_override=nb)
        out[L] = (o["total_T"], o["n_b"])
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(q, out_base):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    tiers = [
        ("L", 2, "Tier L  ·  empirical  ·  $n_b$=2", True),
        ("M", 4, "Tier M  ·  reference  ·  $n_b$=4", True),
        ("H", None, "Tier H  ·  Watson Lemma-5  ·  $n_b$≈33–40", False),
    ]
    for ax, (tier, nb, title, qubit_compiled) in zip(axes, tiers):
        tro = trotter(nb)
        ax.semilogy(LS, [tro[L][0] for L in LS], "--s", color=ORANGE, lw=2.0, ms=5.5, mec=SURFACE,
                    mew=1.3, zorder=3, label="Trotter (Watson UB)")
        if qubit_compiled:
            qL = [L for L in LS if (L, nb) in q]
            ax.semilogy(qL, [q[(L, nb)] for L in qL], "-o", color=BLUE, lw=2.2, ms=6.5, mec=SURFACE,
                        mew=1.5, zorder=4, label="qubitization (compiled)")
        else:
            ax.annotate("qubitization:\ncompile-infeasible\nat $n_b$≈33–40", (0.5, 0.18),
                        xycoords="axes fraction", ha="center", fontsize=8.5, color=MUTED, style="italic")
        ax.set_title(title, color=INK, fontsize=10, loc="left", weight="bold")
        ax.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
        _style(ax)
    axes[0].set_ylabel("QPE $T$-count  ($A$=1, $\\Delta E$=1 MeV)", color=INK2, fontsize=9.5)
    fig.suptitle("Tiered Old-vs-New comparison — the ranking depends on the cutoff scenario "
                 "(NOT a convergence curve)", fontsize=11.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def make_table(q, out_path):
    def _rank(nb):
        """which is cheaper at L=2 and L=3 (compiled tiers only)."""
        r = {}
        for L in (2, 3):
            if (L, nb) in q:
                T = qpe_cost(L, 1, dE=1.0, n_b_override=nb)["total_T"]
                r[L] = "qubitization" if q[(L, nb)] < T else "Trotter"
        return r
    md = [
        "# Tiered Old (Watson Trotter) vs New (compiled PauliLCU) comparison — #6\n",
        "_Three distinct cutoff SCENARIOS, not a convergence curve. Trotter = analytic product-formula"
        " UPPER BOUND (loose); qubitization = direct compiled coherent-query T (adopted π: N_walk ="
        " π·λ/ε_qpe). A=1, ΔE=1 MeV."
        " Architecture comparison: Trotter uses an amplitude/discretized-field basis + locality-"
        "preserving fermion map; PauliLCU uses truncated-Fock + Jordan–Wigner. Boson cutoff is"
        " CONDITIONAL for all tiers (physical adequacy from the classical convergence study)._\n",
        "| tier | cutoff $n_b$ + error status | Trotter T (UB) L=2 / L=3 | PauliLCU T (compiled) L=2 / L=3 "
        "| direct vs extrapolated | cheaper at L=2 / L=3 |",
        "|---|---|--:|--:|---|---|",
    ]
    # Tier L
    tL = {L: qpe_cost(L, 1, dE=1.0, n_b_override=2)["total_T"] for L in (2, 3)}
    rL = _rank(2)
    md.append(f"| **L** empirical | n_b=2; conditional (classical tail/energy conv.) "
              f"| {tL[2]:.2e} / {tL[3]:.2e} | {q[(2,2)]:.2e} / {q[(3,2)]:.2e} "
              f"| both direct (Trotter analytic) | {rL[2]} / {rL[3]} |")
    # Tier M
    tM = {L: qpe_cost(L, 1, dE=1.0, n_b_override=4)["total_T"] for L in (2, 3)}
    rM = _rank(4)
    md.append(f"| **M** reference | n_b=4; reference-informed, NOT a spectral guarantee "
              f"| {tM[2]:.2e} / {tM[3]:.2e} | {q[(2,4)]:.2e} / {q[(3,4)]:.2e} "
              f"| both direct (qubit L≤3 only) | {rM[2]} / {rM[3]} |")
    # Tier H
    hH = {L: qpe_cost(L, 1, dE=1.0) for L in (2, 3)}
    md.append(f"| **H** Watson Lemma-5 | n_b≈33–40; Watson's published energy-bound (conservative) "
              f"| {hH[2]['total_T']:.2e} / {hH[3]['total_T']:.2e} (n_b={hH[2]['n_b']}/{hH[3]['n_b']}) "
              f"| — (compile-infeasible; needs validated analytic continuation) "
              f"| Trotter only / — |")
    md.append("\n**Reading:** the ranking flips with the cutoff scenario. At the low empirical cutoff "
              "(L) qubitization wins at small L (crossover L≈4.3, A=1); at the higher reference cutoff "
              "(M) the Trotter *upper bound* already sits below compiled qubitization at L=2,3, because "
              "qubitization's λ grows far faster with n_b (×~510 over n_b=2→4) than Trotter (×~21); at "
              "Watson's own cutoff (H) qubitization is not compilable. All Trotter numbers are loose "
              "upper bounds — tightening them only shifts the ranking further toward Trotter. Crossover "
              "also moves with A (Trotter's number-restricted Ξ ∝ A); see trotter_comparison.{png,md}.\n")
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-24/trotter_comparison")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    q = load_qubit_by_nb(args.anchor)
    make_figure(q, f"{args.out_dir}/trotter_tiered")
    make_table(q, f"{args.out_dir}/trotter_tiered_table.md")


if __name__ == "__main__":
    main()
