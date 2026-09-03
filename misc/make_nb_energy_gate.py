"""Boson-cutoff ENERGY GATE (L=2) — cutoff SHIFT vs core (re-audit-corrected: P0-1/P0-2).

The absolute selected-CI energies are NOT converged (they keep dropping ~0.4-1.7 MeV per component
and ~35-51 MeV in binding energy per core-doubling). So we do NOT report "core-converged E_0" or a
symmetric error bar from the last doubling. Instead we report the CUTOFF SHIFT — Δ23(core)=E(n_b=2)-
E(n_b=3) and Δ34(core)=E(n_b=3)-E(n_b=4) at each COMMON core — which is a small, STABLE difference
across the whole ladder even though the absolute energy is not converged. The stability of the shift
is the empirical evidence: n_b=2→3 is large (fails the 1 MeV target), n_b=3→4 is ~0 (unresolved).

Seeds: at L=2 the workflow is seed-insensitive (5 seed labels give identical energies) — this is a
determinism/reproducibility check, NOT statistical uncertainty.

    python -m misc.make_nb_energy_gate --data data/classical/nb_energy_gate
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN, PURP = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#7b5cd6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NF = {2: 4, 3: 8, 4: 16}


def load(d):
    """{(n_b, A): {core: E_min_over_seeds}} + sites. Seed-insensitive at L=2, so min == any seed."""
    lad, sites = {}, {}
    for f in sorted(glob.glob(f"{d}/nb*/bare_L2*.json")):
        nb = int(re.search(r"/nb(\d+)/", f).group(1))
        A = int(re.search(r"_A(\d+)_s", f).group(1))
        j = json.load(open(f)); sites[A] = j["sites"]
        for r in j["rungs"]:
            if r.get("E_var") is None:
                continue
            c = int(r["core"]); d0 = lad.setdefault((nb, A), {})
            d0[c] = min(d0.get(c, r["E_var"]), r["E_var"])
    return lad, sites


def shift(lad, A, lo, hi):
    """E(n_b=lo) - E(n_b=hi) at each common core -> [(core, Δ)]."""
    a, b = lad.get((lo, A), {}), lad.get((hi, A), {})
    cs = sorted(set(a) & set(b))
    return [(c, a[c] - b[c]) for c in cs]


def be_shift(lad, A, lo, hi):
    """ΔBE = BE(n_b=lo) - BE(n_b=hi) at each common core (needs A=0,1,A)."""
    need = [(lo, 0), (lo, 1), (lo, A), (hi, 0), (hi, 1), (hi, A)]
    if not all(k in lad for k in need):
        return []
    cs = sorted(set.intersection(*[set(lad[k]) for k in need]))
    def BE(nb, c):
        return A * lad[(nb, 1)][c] - (A - 1) * lad[(nb, 0)][c] - lad[(nb, A)][c]
    return [(c, BE(lo, c) - BE(hi, c)) for c in cs]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(lad, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    fig.patch.set_facecolor(SURFACE)

    # (a) component-energy shift vs core — Δ23 large+stable, Δ34 ~0+stable
    axA.axhspan(-1, 1, color=GREEN, alpha=0.12, zorder=1)
    axA.axhline(0, ls="-", color=INK2, lw=0.9, zorder=2)
    for A, col in ((1, BLUE), (32, CRIT)):
        for pair, ls, mk in (((2, 3), "-", "o"), ((3, 4), "--", "s")):
            pts = shift(lad, A, *pair)
            if pts:
                c, y = zip(*pts)
                axA.semilogx(c, y, ls + mk, color=col, lw=1.7, ms=5, mec=SURFACE, mew=0.8, zorder=4,
                             label=f"$A$={A}: Δ$_{{{pair[0]}{pair[1]}}}$")
    axA.annotate("±1 MeV GSEE target", (2e5, 1), color=GREEN, fontsize=7.5, va="bottom", ha="right")
    axA.set_xlabel("selected-CI core (common ladder)", color=INK2, fontsize=9.5)
    axA.set_ylabel("cutoff shift  $E(n_b)-E(n_b{+}1)$  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  Component-energy shift vs core — Δ$_{23}$ ≫ 1 MeV, Δ$_{34}$ ~0 (both stable)",
                  color=INK, fontsize=9.6, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=7.5, loc="center left", labelcolor=INK2, ncol=2)
    _style(axA)

    # (b) binding-energy shift vs core (A=32)
    axB.axhline(0, ls="-", color=INK2, lw=0.9, zorder=2)
    for pair, col, mk in (((2, 3), PURP, "o"), ((3, 4), GREEN, "s")):
        pts = be_shift(lad, 32, *pair)
        if pts:
            c, y = zip(*pts)
            axB.semilogx(c, y, "-" + mk, color=col, lw=2.0, ms=6, mec=SURFACE, mew=1.0, zorder=4,
                         label=f"ΔBE$_{{{pair[0]}{pair[1]}}}$ (A=32)")
    axB.set_xlabel("selected-CI core (common ladder)", color=INK2, fontsize=9.5)
    axB.set_ylabel("binding-energy shift  (MeV)", color=INK2, fontsize=9.5)
    axB.set_title("b  Binding-energy shift — ΔBE$_{23}$≈+90 (stable), ΔBE$_{34}$≈0", color=INK,
                  fontsize=9.8, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8.5, loc="center left", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Energy gate (L=2): the cutoff SHIFT is stable across the ladder — n_b=2 rejected, "
                 "n_b=3≈n_b=4 (absolute energies NOT converged)", fontsize=10.2, color=INK, y=1.02,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{out_base}.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/classical/nb_energy_gate")
    ap.add_argument("--out-dir", default="data/classical/nb_energy_gate")
    args = ap.parse_args()
    lad, sites = load(args.data)
    assert lad, "no energy-gate shards found"
    make_figure(lad, f"{args.out_dir}/nb_energy_gate")

    # deepest-common-core shift values for the table
    def last(A, lo, hi):
        p = shift(lad, A, lo, hi); return p[-1] if p else (None, None)
    def blast(A, lo, hi):
        p = be_shift(lad, A, lo, hi); return p[-1] if p else (None, None)
    md = ["# Boson-cutoff energy gate (L=2) — cutoff shift vs core\n",
          "_The absolute selected-CI energies are NOT converged (they keep falling with core). We "
          "report the CUTOFF SHIFT Δ(core) at common cores — a small, STABLE difference across the "
          "ladder — NOT the absolute energy or a symmetric error bar. Seeds are seed-insensitive at "
          "L=2 (a determinism check, not statistical uncertainty). Absolute binding energy is "
          "UNCONVERGED and not quoted as a physical value._\n",
          "| shift (deepest common core) | A=1 | A=32 |",
          "|---|--:|--:|"]
    md.append(f"| component Δ23 = E(n_b2)−E(n_b3) | {last(1,2,3)[1]:+.2f} | {last(32,2,3)[1]:+.2f} |")
    md.append(f"| component Δ34 = E(n_b3)−E(n_b4) | {last(1,3,4)[1]:+.3f} | {last(32,3,4)[1]:+.3f} |")
    md.append(f"| binding ΔBE23 (A=32) | — | {blast(32,2,3)[1]:+.1f} |")
    md.append(f"| binding ΔBE34 (A=32) | — | {blast(32,3,4)[1]:+.2f} |")
    md.append("\n**Reading:** Δ23 is several MeV (component) / ~+90 MeV (binding) and STABLE across the "
              "ladder → n_b=2 fails the 1 MeV target (rejected). Δ34 is <0.01 MeV (component) / ~0 "
              "(binding) and stable → no n_b=3→4 cutoff effect is resolved within the explored "
              "selected-CI sequence. Both are cutoff DIFFERENCES; the absolute energies remain "
              "unconverged (last-doubling change is a diagnostic, not an error bound). L=2 only; the "
              "volume scaling of Δ34 to L=10 is a separate run (P0-4).\n")
    open(f"{args.out_dir}/nb_energy_gate_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_energy_gate_table.md")
    print(f"[done] Δ23(A=1)={last(1,2,3)[1]:.2f} Δ34(A=1)={last(1,3,4)[1]:.3f} "
          f"ΔBE23(A32)={blast(32,2,3)[1]:.0f} ΔBE34(A32)={blast(32,3,4)[1]:.1f}")


if __name__ == "__main__":
    main()
