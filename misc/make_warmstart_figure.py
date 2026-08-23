"""Warm-start payoff figure + table: the classically-informed reduction in quantum GSEE resources.

Combines the classical γ measurement (`run_warmstart_fidelity` outputs) with the compiled PauliLCU
anchor (r3) via `gsee_total_cost`:
  * panel a — the MEASURED warm-start success probability p0: best single-determinant COLD start vs
    the classically-computed compact frame-core WARM start, at ED-tractable systems (genuine);
  * panel b — the resulting realistic TOTAL T-count vs L (cold vs warm), on the physical anchor,
    CONDITIONAL on the ED-measured overlap holding at scale (honest-claim: genuine at small L,
    expected-but-unverified at the physical L).

Writes warmstart_payoff.{pdf,png} + warmstart_payoff_table.md. dataviz reference palette.

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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src_PI.estimation.gsee_total_cost import total_gsee_cost   # noqa: E402

BLUE, ORANGE, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
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


def load_anchor(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/*fock_pauli*.json")):
        j = json.load(open(f))
        if not j.get("done") or not j["results"]:
            continue
        r = j["results"][0]; b = r.get("QPE_Budget") or {}
        if r["n_b"] == 2 and r["L"] >= 2 and "rep2" not in os.path.basename(f):
            rows.append(dict(L=r["L"], lam=r["Physical_Lambda"], walkT=r["Walk_T_Count"],
                             q=r["Logical_Qubits"], eps_qpe=b.get("eps_qpe")))
    rows.sort(key=lambda x: x["L"])
    return rows


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(ws, anchor, rep, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.patch.set_facecolor(SURFACE)

    # (a) measured p0: cold (single det) vs warm (frame core) per ED config
    labels = [f"$L$={w['L']} d{w['dim']}\n$n_b$={w['n_b']}" for w in ws]
    x = range(len(ws)); bw = 0.38
    axA.bar([i - bw / 2 for i in x], [w["p0_cold"] for w in ws], bw, color=MUTED,
            label="cold (best single det)", zorder=3)
    axA.bar([i + bw / 2 for i in x], [w["p0_warm"] for w in ws], bw, color=BLUE,
            label="warm (frame core, $U|\\tilde\\psi\\rangle$)", zorder=3)
    for i, w in enumerate(ws):
        axA.annotate(f"{w['p0_warm']:.2f}", (i + bw / 2, w["p0_warm"]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=8, color=INK2)
    axA.set_xticks(list(x)); axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("QPE warm-start success $p_0=|\\langle g|\\psi_0\\rangle|^2$", color=INK2, fontsize=9.5)
    axA.set_ylim(0, 1.4)
    axA.set_title("a  Measured overlap (ED-exact)", color=INK, fontsize=11, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=8, loc="upper center", labelcolor=INK2)
    _style(axA)

    # (b) realistic total T vs L on the physical anchor, cold vs warm (conditional on the p0)
    L = [a["L"] for a in anchor]
    axB.semilogy(L, [rep[a["L"]]["cold"]["total_T"] for a in anchor], "-o", color=CRIT, lw=2.0, ms=6,
                 mec=SURFACE, mew=1.5, zorder=3, label="cold start")
    axB.semilogy(L, [rep[a["L"]]["warm"]["total_T"] for a in anchor], "-o", color=BLUE, lw=2.0, ms=6,
                 mec=SURFACE, mew=1.5, zorder=3, label="warm start (frame)")
    axB.semilogy(L, [rep[a["L"]]["coherent_query_T"] for a in anchor], "--", color=MUTED, lw=1.3,
                 zorder=2, label="one coherent window")
    sv = rep[anchor[-1]["L"]]["warmstart_saving_x"]
    axB.annotate(f"~{sv:.1f}× fewer total $T$", (0.04, 0.05), xycoords="axes fraction",
                 fontsize=9, color=BLUE, weight="bold")
    axB.set_xlabel("lattice size $L$  (dim=3 anchor)", color=INK2, fontsize=9.5)
    axB.set_ylabel("realistic total GSEE $T$-count", color=INK2, fontsize=9.5)
    axB.set_title("b  Total cost, cold vs warm (conditional)", color=INK, fontsize=11, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Classically-informed warm start reduces QPE GSEE repetitions",
                 fontsize=12.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", default="data/quantum/2026-08-23/warmstart_fidelity")
    ap.add_argument("--anchor", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-23/warmstart_fidelity")
    args = ap.parse_args()
    ws = load_warmstart(args.warmstart)
    anchor = load_anchor(args.anchor)
    assert ws and anchor, "need warmstart + anchor data"

    # representative measured overlap for the anchor projection: the n_b=2 point (physical cutoff)
    # if present, else the deepest available. p0_cold from the same measurement; D likewise.
    ref = next((w for w in ws if w["n_b"] == 2), ws[-1])
    rep = {}
    for a in anchor:
        rep[a["L"]] = total_gsee_cost(
            physical_lambda=a["lam"], walk_T=a["walkT"], walk_register_qubits=a["q"],
            eps_qpe=a["eps_qpe"], p0_warm=ref["p0_warm"], p0_cold=ref["p0_cold"],
            D_warm=ref["D"], n_bos_modes=3 * a["L"] ** 3, N_f=4)
    make_figure(ws, anchor, rep, f"{args.out_dir}/warmstart_payoff")

    # --- table ---
    md = ["# Classically-informed warm start — measured overlap + total-cost impact\n",
          f"_Representative overlap for the anchor projection: L={ref['L']} d{ref['dim']} "
          f"n_b={ref['n_b']} (p0_cold={ref['p0_cold']:.3f}, p0_warm={ref['p0_warm']:.3f}, "
          f"D={ref['D']}). Conditional on this overlap holding at the physical L (genuine at the ED "
          f"sizes; expected-but-unverified at scale)._\n",
          "**Measured warm-start overlap (ED-exact):**\n",
          "| system | ground degen | r_norm | p0 cold (1 det) | p0 warm (frame core) | fewer reps |",
          "|---|--:|--:|--:|--:|--:|"]
    for w in ws:
        md.append(f"| L={w['L']} d{w['dim']} n_b={w['n_b']} A={w['A']} | {w['degen']} | {w['r_norm']:.3f} "
                  f"| {w['p0_cold']:.3f} | {w['p0_warm']:.3f} | {w['p0_warm']/w['p0_cold']:.2f}× |")
    md.append("\n**Realistic total GSEE cost on the physical anchor (conditional):**\n")
    md.append("| L | sites | total qubits | R cold | R warm | total-T cold | total-T warm | saving |")
    md.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for a in anchor:
        r = rep[a["L"]]
        md.append(f"| {a['L']} | {a['L']**3} | {r['warm']['total_qubits']:,} | {r['cold']['R']} "
                  f"| {r['warm']['R']} | {r['cold']['total_T']:.2e} | {r['warm']['total_T']:.2e} "
                  f"| {r['warmstart_saving_x']:.1f}× |")
    open(f"{args.out_dir}/warmstart_payoff_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/warmstart_payoff_table.md")
    print(f"[done] measured p0 warm/cold: " +
          ", ".join(f"n_b={w['n_b']}:{w['p0_cold']:.2f}->{w['p0_warm']:.2f}" for w in ws))


if __name__ == "__main__":
    main()
