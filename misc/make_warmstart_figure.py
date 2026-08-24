"""Warm-start overlap figure (APPENDIX / methods-validation, per warmstart_campaign_audit 2026-08-23).

Reframed after audit: this is NOT a physical-L resource headline. It shows two honest things on the
ED-tractable ONE-DIMENSIONAL TOY systems where the exact ground manifold is available:

  * panel a — three matched-D preparation arms per system: a single best determinant (COLD), the
    equally-loaded BARE-H selected-CI core, and the frame core mapped through U (WARM). This isolates
    the two interventions: multideterminant loading vs the squeeze frame. Result: multideterminant
    loading reaches near-unit ground-manifold overlap; the frame does NOT add (it slightly lowers it
    at n_b=2). The demonstrable claim is "a compact classical multideterminant trial state is a good
    QPE warm start vs a single determinant", not "the frame improves overlap".
  * panel b — repetition SENSITIVITY to the warm-start overlap p0 (exact Bernoulli, 99% confidence),
    a hypothetical curve with the measured toy points marked. NOT an extrapolation of the toy overlap
    to every physical L (the physical-L overlap is unmeasured/unknown).

Writes warmstart_overlap_appendix.{pdf,png} + warmstart_overlap_appendix_table.md.
    python -m misc.make_warmstart_figure
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
from src_PI.estimation.qpe_cost import overlap_repetition_factor   # noqa: E402

BLUE, LBLUE, ORANGE, CRIT = "#2a78d6", "#8fbce8", "#eb6834", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load_warmstart(d):
    out = []
    for f in sorted(glob.glob(f"{d}/wf_*.json")):
        j = json.load(open(f))
        deep = j["rows"][-1]                                  # deepest-D row
        out.append(dict(L=j["L"], dim=j["dim"], n_b=j["n_b"], A=j["A"], degen=j["ground_degeneracy"],
                        p0_cold=j["p0_cold_bestdet"], p0_warm=deep["p0_warm"],
                        p0_bare=deep["p0_bare_core"], D=deep["D_warm"], r_norm=j["r_norm"]))
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(ws, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.patch.set_facecolor(SURFACE)

    # (a) three matched-D arms per toy system
    labels = [f"$L$={w['L']} d{w['dim']}\n$n_b$={w['n_b']} ($D$={w['D']})" for w in ws]
    x = np.arange(len(ws)); bw = 0.26
    axA.bar(x - bw, [w["p0_cold"] for w in ws], bw, color=MUTED, label="cold: best 1 determinant", zorder=3)
    axA.bar(x, [w["p0_bare"] for w in ws], bw, color=LBLUE, label="bare-H $D$-core (no frame)", zorder=3)
    axA.bar(x + bw, [w["p0_warm"] for w in ws], bw, color=BLUE,
            label="frame $D$-core, $U|\\tilde\\psi\\rangle$", zorder=3)
    for i, w in enumerate(ws):
        axA.annotate(f"{w['p0_warm']:.2f}", (i + bw, w["p0_warm"]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7.5, color=INK2)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("QPE warm-start overlap $p_0=\\|P_{g}\\,\\psi_0\\|^2$", color=INK2, fontsize=9.5)
    axA.set_ylim(0, 1.08)
    axA.set_title("a  Overlap by preparation arm (1-D toy, ED-exact)", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=7.6, loc="lower center", labelcolor=INK2)
    _style(axA)

    # (b) repetition sensitivity to p0 (exact Bernoulli, 99% conf) — a hypothetical curve, toy points marked
    p = np.linspace(0.02, 0.999, 400)
    R = [overlap_repetition_factor(float(pi), confidence=0.99)["R"] for pi in p]
    axB.semilogy(p, R, "-", color=INK2, lw=1.8, zorder=2, label="exact Bernoulli, 99% conf.")
    ref = next((w for w in ws if w["n_b"] == 2), ws[-1])
    for tag, p0, col in (("cold (1 det)", ref["p0_cold"], MUTED),
                         ("bare $D$-core", ref["p0_bare"], LBLUE),
                         ("frame $D$-core", ref["p0_warm"], BLUE)):
        Rp = overlap_repetition_factor(float(min(p0, 0.999)), confidence=0.99)["R"]
        axB.semilogy([p0], [Rp], "o", color=col, ms=9, mec=SURFACE, mew=1.5, zorder=4)
        axB.annotate(f"{tag}: {Rp}", (p0, Rp), textcoords="offset points", xytext=(-6, 8),
                     ha="right", fontsize=8, color=col, weight="bold")
    axB.set_xlabel("assumed warm-start overlap $p_0$", color=INK2, fontsize=9.5)
    axB.set_ylabel("QPE repetitions $R$ (shots to $\\geq$1 hit)", color=INK2, fontsize=9.5)
    axB.set_title("b  Repetition sensitivity to $p_0$ (hypothetical)", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Compact classical multideterminant trial states as QPE warm starts (methods validation)",
                 fontsize=11.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", default="data/quantum/2026-08-23/warmstart_fidelity")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-23/warmstart_fidelity")
    args = ap.parse_args()
    ws = load_warmstart(args.warmstart)
    assert ws, "need warmstart data"
    make_figure(ws, f"{args.out_dir}/warmstart_overlap_appendix")

    md = ["# Warm-start overlap — methods validation (appendix)\n",
          "_ED-exact ground-manifold overlap on 1-D toy systems. Three matched-D preparation arms;"
          " isolates multideterminant loading vs the squeeze frame. NOT a physical-L resource claim"
          " (see warmstart_campaign_audit 2026-08-23)._\n",
          "| system | ground degen | p0 cold (1 det) | p0 bare $D$-core | p0 frame $D$-core | R cold | R frame | mean-shots ratio |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for w in ws:
        Rc = overlap_repetition_factor(min(w["p0_cold"], 0.999))["R"]
        Rf = overlap_repetition_factor(min(w["p0_warm"], 0.999))["R"]
        md.append(f"| L={w['L']} d{w['dim']} n_b={w['n_b']} A={w['A']} (D={w['D']}) | {w['degen']} "
                  f"| {w['p0_cold']:.3f} | {w['p0_bare']:.3f} | {w['p0_warm']:.3f} | {Rc} | {Rf} "
                  f"| {w['p0_warm']/w['p0_cold']:.2f}× |")
    md.append("\n**Honest reading:** multideterminant loading (bare *or* frame $D$-core) reaches "
              "near-unit overlap vs ~0.48 for the best single determinant; the frame does NOT add "
              "(bare $D$-core overlap ≈ 1.0 ≥ frame). Repetition savings are reported as two named "
              "metrics: fixed-confidence exact-Bernoulli $R$, and mean-shots $1/p_0$. Physical-L "
              "overlap is unmeasured; state prep is plausibly $T$-subdominant but its full "
              "configuration-word cost is not yet compiled.\n")
    open(f"{args.out_dir}/warmstart_overlap_appendix_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/warmstart_overlap_appendix_table.md")


if __name__ == "__main__":
    main()
