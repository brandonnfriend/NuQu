"""Boson-cutoff ENERGY GATE (the 2026-09-02 audit's required experiment), L=2.

Unlike the occupation-tail diagnostic (a probability, not an energy bound), this measures the
quantity the cutoff claim needs: the CORE-CONVERGED ground energy E_0 and the binding energy
BE(A)=A*E(1)-(A-1)*E(0)-E(A) vs the boson cutoff N_f=2^n_b, with the solver uncertainty. Result: the
n_b=2 (N_f=4) energy is NOT converged — the n_b=2->3 change (E_0 4-7 MeV, BE ~91 MeV) far exceeds the
1 MeV target — while n_b=3 IS (n_b=3->4 negligible). So the tail's <1% probability hid a multi-MeV
energy error (rare high-occupation states carry large energy).

    python -m misc.make_nb_energy_gate --data data/classical/nb_energy_gate
"""
import argparse
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN, PURP = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#7b5cd6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NF = {2: 4, 3: 8, 4: 16}


def load(d):
    """{(n_b, A): {'E': min-seed total E_0, 'resid': core-convergence residual (median dE_last),
    'seeds': n, 'spread': seed spread, 'sites': sites}}."""
    raw = {}
    for f in sorted(glob.glob(f"{d}/nb*/bare_L2*.json")):
        nb = int(re.search(r"/nb(\d+)/", f).group(1))
        m = re.search(r"_A(\d+)_s(\d+)\.json", f)
        A, seed = int(m.group(1)), int(m.group(2))
        j = json.load(open(f))
        rungs = sorted((r for r in j["rungs"] if r.get("E_var") is not None), key=lambda r: r["core"])
        if not rungs:
            continue
        dE = abs(rungs[-1]["E_var"] - rungs[-2]["E_var"]) if len(rungs) >= 2 else float("nan")
        raw.setdefault((nb, A), {})[seed] = (rungs[-1]["E_var"], dE, j["sites"], rungs[-1].get("mean_occ"))
    out = {}
    for (nb, A), seeds in raw.items():
        Es = [v[0] for v in seeds.values()]
        out[(nb, A)] = dict(E=min(Es), spread=max(Es) - min(Es), seeds=len(seeds),
                            resid=st.median(v[1] for v in seeds.values()),
                            sites=next(iter(seeds.values()))[2],
                            occ=min(v[3] for v in seeds.values() if v[3] is not None))
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def binding(g, A, nb):
    e0, e1, eA = g.get((nb, 0)), g.get((nb, 1)), g.get((nb, A))
    if not (e0 and e1 and eA):
        return None, None
    BE = A * e1["E"] - (A - 1) * e0["E"] - eA["E"]
    # residual propagated through the extensive cancellation (correlated worst case)
    sig = A * e1["resid"] + (A - 1) * e0["resid"] + eA["resid"]
    return BE, sig


def make_figure(g, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    nbs = [2, 3, 4]
    xs = [NF[n] for n in nbs]

    # (a) DEVIATION of E_0 from the converged (n_b=4) value, per A — the absolute E_0 spans 300 MeV
    # so the MeV-scale cutoff error is invisible on an absolute axis; plot the deviation vs the 1 MeV
    # GSEE target (a REAL budget, unlike the tail's arbitrary 1%).
    axA.axhspan(-1, 1, color=GREEN, alpha=0.13, zorder=1)
    axA.axhline(0, ls="-", color=INK2, lw=1.0, zorder=2)
    axA.annotate("±1 MeV GSEE target", (16, 1), color=GREEN, fontsize=8, va="bottom", ha="right")
    for A, col in ((0, MUTED), (1, BLUE), (32, CRIT)):
        if (4, A) not in g:
            continue
        ref = g[(4, A)]["E"]
        pts = [(NF[n], g[(n, A)]["E"] - ref, g[(n, A)]["resid"]) for n in nbs if (n, A) in g]
        X, Y, S = zip(*pts)
        axA.errorbar(X, Y, yerr=S, fmt="-o", color=col, lw=1.8, ms=6, mec=SURFACE, mew=1.0,
                     capsize=3, zorder=4, label=f"$A$={A}" + (" (vacuum)" if A == 0 else ""))
    axA.set_xscale("log", base=2); axA.set_xticks(xs); axA.set_xticklabels([f"$N_f$={x}\n$n_b$={n}" for x, n in zip(xs, nbs)])
    axA.set_xlabel("boson cutoff", color=INK2, fontsize=9.5)
    axA.set_ylabel("$E_0(n_b) - E_0(n_b{=}4)$  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  $E_0$ deviation from converged — $n_b$=2 off by 4–7 MeV, $n_b$=3 ~0", color=INK,
                  fontsize=10.0, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    _style(axA)

    # (b) binding energy vs N_f — the physically meaningful observable
    A = 32
    pts = [(NF[n],) + binding(g, A, n) for n in nbs if binding(g, A, n)[0] is not None]
    X, Y, S = zip(*pts)
    axB.errorbar(X, Y, yerr=S, fmt="-o", color=PURP, lw=2.0, ms=8, mec=SURFACE, mew=1.2, capsize=4,
                 zorder=4, label=f"$BE(A={A})$")
    for x, y, s in pts:
        axB.annotate(f"{y:.0f}±{s:.0f}", (x, y), textcoords="offset points", xytext=(8, -2),
                     fontsize=8, color=INK2)
    d23 = Y[1] - Y[0]; d34 = Y[2] - Y[1]
    axB.set_xscale("log", base=2); axB.set_xticks(xs); axB.set_xticklabels([f"$N_f$={x}\n$n_b$={n}" for x, n in zip(xs, nbs)])
    axB.set_xlabel("boson cutoff", color=INK2, fontsize=9.5)
    axB.set_ylabel(f"binding energy $BE(A={A})$  (MeV)", color=INK2, fontsize=9.5)
    axB.set_title(f"b  Binding energy — $n_b$2→3 shifts {d23:+.0f} MeV, 3→4 {d34:+.1f}", color=INK,
                  fontsize=10.3, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Boson-cutoff ENERGY GATE (L=2): n_b=2 is NOT converged (tail <1% hides a multi-MeV "
                 "error); n_b=3 is", fontsize=10.8, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/classical/nb_energy_gate")
    ap.add_argument("--out-dir", default="data/classical/nb_energy_gate")
    args = ap.parse_args()
    g = load(args.data)
    assert g, "no energy-gate shards found"
    make_figure(g, f"{args.out_dir}/nb_energy_gate")

    md = ["# Boson-cutoff ENERGY GATE (L=2) — the audit's required experiment\n",
          "_Core-converged E_0 and binding energy BE(A)=A·E(1)−(A−1)·E(0)−E(A) vs cutoff N_f=2^n_b, "
          "with the solver residual. Seeds (5) are seed-robust at L=2 (spread ~0); the uncertainty is "
          "the core-convergence residual (± = median last-doubling ΔE, propagated through BE's "
          "extensive cancellation)._\n",
          "| observable | n_b=2 (N_f=4) | n_b=3 (N_f=8) | n_b=4 (N_f=16) | 2→3 | 3→4 |",
          "|---|--:|--:|--:|--:|--:|"]
    for A in (0, 1, 32):
        r = [g.get((n, A), {}).get("E") for n in (2, 3, 4)]
        if None in r:
            continue
        md.append(f"| E_0(A={A}) tot | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {r[1]-r[0]:+.2f} | {r[2]-r[1]:+.2f} |")
    be = [binding(g, 32, n) for n in (2, 3, 4)]
    if all(b[0] is not None for b in be):
        md.append(f"| **BE(A=32)** | **{be[0][0]:.0f}±{be[0][1]:.0f}** | **{be[1][0]:.0f}±{be[1][1]:.0f}** | "
                  f"**{be[2][0]:.0f}±{be[2][1]:.0f}** | **{be[1][0]-be[0][0]:+.0f}** | {be[2][0]-be[1][0]:+.1f} |")
    md.append("\n**Reading:** n_b=2 is NOT converged — E_0 shifts 4–7 MeV and binding energy ~91 MeV "
              "going to n_b=3, far above a 1 MeV budget — while n_b=3 IS converged (3→4 negligible). "
              "The occupation tail (<1% at n_b=2) is a PROBABILITY, not an energy: the rare "
              "high-occupation states carry multi-MeV energy (amplified ~30× in BE by the extensive "
              "cancellation). **Implication:** the physically-converged cutoff at L=2 is n_b=3, not "
              "n_b=2; the quantum anchor should move to n_b=3 or carry the (now-quantified) n_b=2→3 "
              "penalty. Caveat: L=2 only; the per-mode cutoff is ~L-independent so this likely "
              "generalizes, but larger-L confirmation hits the extensivity/H-build wall.\n")
    open(f"{args.out_dir}/nb_energy_gate_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_energy_gate_table.md")
    print("[done] BE: " + " ".join(f"n_b{n}={binding(g,32,n)[0]:.0f}" for n in (2, 3, 4)))


if __name__ == "__main__":
    main()
