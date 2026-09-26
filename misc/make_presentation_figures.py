"""Make the compact slide figures in docs/presentation from accepted NuQu data.

This is intentionally separate from the publication figures in results/.  The plots
trade audit-detail density for legibility, while the README beside them preserves the
qualifications that must travel with the claims.

    .venv/bin/python -m misc.make_presentation_figures
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nuqu-presentation-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from misc.aggregate_classical_energies import analyze as analyze_classical
from misc.aggregate_classical_energies import load as load_classical
from misc.make_nb3_headline import build as build_nb3
from misc.make_nb3_headline import load as load_quantum
from misc.make_nb_energy_gate import load as load_energy_gate
from misc.make_nb_energy_gate import shift as cutoff_shift
from src_PI.trotter_theory.trotter_exact import qpe_cost


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation"
BLUE = "#1769aa"
CYAN = "#36a9c9"
ORANGE = "#e76f51"
RED = "#c83e4d"
GREEN = "#2a9d66"
PURPLE = "#7451a6"
GRAY = "#72777d"
LIGHT = "#e9edf0"
INK = "#172026"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 18, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.edgecolor": "#9aa0a6", "axes.linewidth": 1.0,
        "axes.titleweight": "bold", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "legend.fontsize": 11,
        "savefig.dpi": 220,
    })


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(ygrid, axis="y", color=LIGHT, linewidth=1.0)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.tight_layout(pad=1.4, w_pad=2.6)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantum_rows():
    nb2 = load_quantum(str(ROOT / "data/quantum/2026-08-21/vertexfix_r3_290826"),
                       "*fock_pauli*nb2*.json", "wick")
    nb3 = load_quantum(str(ROOT / "data/quantum/nb3_anchor"),
                       "*fock_pauli_nb3*.json", "wick")
    rows, _ = build_nb3(nb2, nb3)
    return rows, nb2


def figure_quantum(rows):
    ls = np.arange(2, 11)
    compiled = ls <= 7
    t = np.array([rows[int(L)]["T"] for L in ls])
    q = np.array([rows[int(L)]["q"] for L in ls])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    ax1.plot(ls[compiled], q[compiled], "o-", color=BLUE, lw=3, ms=7)
    proj = ls >= 7
    ax1.plot(ls[proj], q[proj], "--", color=BLUE, lw=2.2)
    ax1.plot(ls[~compiled], q[~compiled], "o", color=BLUE, ms=7,
             markerfacecolor="white", markeredgewidth=2)
    ax1.annotate("13,032", (10, q[-1]), xytext=(-30, 12), textcoords="offset points",
                 ha="right", color=BLUE, fontweight="bold")
    ax1.set(title="Logical qubits", xlabel="Lattice length  $L$",
            ylabel="Total logical qubits")
    ax1.set_xticks(ls)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    style(ax1)

    ax2.semilogy(ls[compiled], t[compiled], "o-", color=RED, lw=3, ms=7,
                 label="Compiled")
    ax2.semilogy(ls[proj], t[proj], "--", color=RED, lw=2.2)
    ax2.semilogy(ls[~compiled], t[~compiled], "o", color=RED, ms=7,
                 markerfacecolor="white", markeredgewidth=2, label="Projected")
    ax2.annotate(r"$3.05\times10^{17}$", (10, t[-1]), xytext=(-38, -28),
                 textcoords="offset points", ha="right", color=RED, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    ax2.set(title="Fault-tolerant gate cost", xlabel="Lattice length  $L$",
            ylabel="Total T gates")
    ax2.set_xticks(ls)
    ax2.legend(loc="lower right")
    style(ax2)
    fig.suptitle(r"Qubitization resources grow rapidly with volume  ($n_b=3$, 1 MeV accuracy)", fontsize=20,
                 fontweight="bold")
    save(fig, "quantum_resources")


def figure_classical():
    groups, meta = load_classical([
        str(ROOT / "data/classical/2026-09-06/bare_baseline_nb3_292477")
    ], convention="wick")
    recs = {r["L"]: r for r in analyze_classical(groups, meta) if r["n_b"] == 3}
    ls = np.array(sorted(recs))
    bounds = np.array([recs[L]["E_var_bound_ps"] for L in ls])
    good = np.array([bool(recs[L]["ok"]) for L in ls])
    einf = np.array([recs[L].get("E_inf_ps", np.nan) or np.nan for L in ls])
    sig = np.array([recs[L].get("sigma_ps", np.nan) or np.nan for L in ls])

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    ax.plot(ls, bounds, "o--", color=GRAY, lw=2.2, ms=9, markerfacecolor="white",
            markeredgewidth=2, label="Variational upper bound")
    ax.errorbar(ls[good], einf[good], yerr=sig[good], fmt="s", ms=9, color=BLUE,
                capsize=6, elinewidth=2.2, label=r"Extrapolated $E_\infty\pm\sigma$")
    ax.plot(ls[~good], bounds[~good], "x", color=RED, ms=13, mew=3)
    ax.annotate("bound only", (ls[-1], bounds[-1]), xytext=(18, -12),
                textcoords="offset points", ha="left", color=RED)
    ax.set(title="Classical energy estimates lose precision with volume",
           xlabel="Lattice length  $L$", ylabel="Energy per site (MeV)")
    ax.set_xticks(ls)
    ax.legend(loc="upper left")
    style(ax)
    save(fig, "classical_baseline")


def cross_l(ls, ratio):
    for i in range(len(ls) - 1):
        if (ratio[i] - 1) * (ratio[i + 1] - 1) <= 0 and ratio[i] != ratio[i + 1]:
            f = -np.log(ratio[i]) / (np.log(ratio[i + 1]) - np.log(ratio[i]))
            return float(ls[i] + f * (ls[i + 1] - ls[i]))
    return np.nan


def figure_trotter(rows, nb2):
    ls = np.arange(2, 11)
    q = {2: np.array([nb2[int(L)]["T"] for L in ls]),
         3: np.array([rows[int(L)]["T"] for L in ls])}
    colors = {2: PURPLE, 3: BLUE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    tr3 = np.array([qpe_cost(int(L), A=1, dE=1.0, n_b_override=3)["total_T"] for L in ls])
    xc3 = cross_l(ls, tr3 / q[3])
    comp = ls <= 7
    ax1.semilogy(ls[comp], q[3][comp], "o-", color=BLUE, lw=2.8, ms=6,
                 label="Qubitization")
    ax1.semilogy(ls[ls >= 7], q[3][ls >= 7], "o--", color=BLUE, lw=2.0, ms=6,
                 markerfacecolor="white")
    ax1.semilogy(ls, tr3, "s--", color=ORANGE, lw=2.5, ms=5.5,
                 label="Trotter estimate")
    ax1.axvline(xc3, color=GRAY, lw=1.3, ls=":")
    ax1.annotate(fr"crossover  $L\approx{xc3:.1f}$", (xc3, 1.2e13),
                 xytext=(10, 18), textcoords="offset points", color=INK, fontsize=11)
    ax1.set(title=fr"Matched scenario  ($A=1$, $n_b=3$)", xlabel="Lattice length  $L$",
            ylabel="T gates")
    ax1.set_xticks(ls)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    style(ax1)

    As = np.arange(1, 41)
    for nb in (2, 3):
        xcs = []
        for A in As:
            tr = np.array([qpe_cost(int(L), A=int(A), dE=1.0, n_b_override=nb)["total_T"] for L in ls])
            xcs.append(cross_l(ls, tr / q[nb]))
        ok = np.isfinite(xcs)
        # The raw crossover has steps because the compiled PREPARE oracle pads to
        # powers of two.  For a talk, show the broad A trend rather than those
        # implementation-bin artifacts.
        coef = np.polyfit(np.log(As[ok]), np.log(np.asarray(xcs)[ok]), 1)
        trend = np.exp(np.polyval(coef, np.log(As)))
        ax2.plot(As, trend, color=colors[nb], lw=3, label=fr"$n_b={nb}$")
    ax2.set(title="Crossover shifts with nucleon number",
            xlabel="Nucleon number  $A$", ylabel="Crossover lattice length  $L$")
    ax2.legend(loc="upper left")
    style(ax2)
    ax2.text(.98, .04, "smooth trend; compiler-padding steps omitted", transform=ax2.transAxes,
             ha="right", color=GRAY, fontsize=9)
    fig.suptitle("The algorithm crossover depends on cutoff and particle number", fontsize=20,
                 fontweight="bold")
    save(fig, "trotter_crossover")


def figure_cutoff():
    tail = np.array([[32.7, .78], [24.2, .32], [9.4, .02], [22.1, .29],
                     [33.9, .87], [52.7, .56], [43.0, .33], [47.2, .26]])
    lad, _ = load_energy_gate(str(ROOT / "data/classical/nb_energy_gate"))
    shifts = {(pair, A): abs(cutoff_shift(lad, A, *pair)[-1][1])
              for pair in ((2, 3), (3, 4)) for A in (1, 32)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    rng = np.random.default_rng(7)
    for i, (nb, color) in enumerate(((1, RED), (2, ORANGE))):
        x = np.full(len(tail), i) + rng.uniform(-.08, .08, len(tail))
        ax1.scatter(x, tail[:, i], s=55, color=color, alpha=.9, zorder=3)
        ax1.plot([i-.18, i+.18], [np.median(tail[:, i])]*2, color=INK, lw=3)
    ax1.axhline(1, color=GREEN, ls="--", lw=1.6, label="1% reference")
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1], [r"$n_b=1$", r"$n_b=2$"])
    ax1.set_title("Occupation above the cutoff", fontsize=16)
    ax1.set_ylabel("Weight above cutoff (%)")
    ax1.legend(loc="upper right")
    style(ax1)

    xpos = np.array([0, 1]); width = .32
    for j, (A, color) in enumerate(((1, BLUE), (32, PURPLE))):
        vals = [shifts[((2, 3), A)], shifts[((3, 4), A)]]
        ax2.bar(xpos + (j-.5)*width, vals, width=width, color=color, label=fr"$A={A}$")
    ax2.axhline(1, color=GREEN, ls="--", lw=1.6, label="1 MeV target")
    ax2.set_yscale("log")
    ax2.set_xticks(xpos, [r"Compare $n_b=2$ vs 3", r"Compare $n_b=3$ vs 4"])
    ax2.set_title("Energy changes when the cutoff is raised", fontsize=16)
    ax2.set_ylabel("Absolute energy shift (MeV)")
    ax2.set_ylim(7e-4, 1.6e1)
    ax2.text(0, 8.2, r"$n_b=2$ is not enough", color=ORANGE, ha="center", fontweight="bold")
    ax2.text(1, .02, r"$n_b=3$ and 4 agree at $L=2$", color=GREEN, ha="center",
             fontweight="bold")
    ax2.legend(loc="upper right")
    style(ax2)
    fig.suptitle(r"Cutoff tests select $n_b=3$ as the smallest supported register", fontsize=19,
                 fontweight="bold")
    save(fig, "boson_cutoff_decision")


def load_frame(frame, L=3, filling=1.0, seed=0):
    pat = ROOT / "data/classical/2026-09-03/frame_isospectrality/conv2_shards" / (
        f"backeval_{frame}_L{L}d3nb3_f{filling:.1f}_s{seed}.json")
    j = json.loads(pat.read_text())
    return np.array([r["n_dets"] for r in j["results"]]), np.array([r["E_orig"] for r in j["results"]])


def figure_frame(caveat=False):
    nb, eb = load_frame("bare")
    ns, es = load_frame("gaussian")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.semilogx(nb, eb, "o-", color=GRAY, lw=3, ms=8, label="Bare frame")
    ax.semilogx(ns, es, "o-", color=GREEN, lw=3, ms=8, label="Gaussian squeeze")
    ax.axhline(eb[-1], color=GRAY, lw=1.3, ls="--")
    first = np.flatnonzero(es <= eb[-1])[0]
    ratio = int(nb[-1] / ns[first])
    ax.annotate(fr"Same or lower bound with $\geq {ratio}\times$ fewer determinants",
                xy=(ns[first], es[first]), xytext=(900, 8350),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, ls="--",
                                connectionstyle="arc3,rad=.12"), color=GREEN,
                fontsize=13, fontweight="bold")
    if caveat:
        ax.text(.02, .09, "Different truncated Hamiltonians: both target the same infinite-boson theory;\n"
                "squeezing changes the finite trial space and concentrates the state.",
                transform=ax.transAxes, fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=.4", facecolor="white", edgecolor=LIGHT, alpha=.95))
    ax.set(title="Gaussian squeezing makes the selected-CI state more compact",
           xlabel="Core size (number of states)", ylabel="Variational energy bound (MeV)")
    ax.legend(loc="upper right")
    style(ax)
    save(fig, "gaussian_frame_compaction_caveat" if caveat else "gaussian_frame_compaction")


def main():
    setup()
    rows, nb2 = quantum_rows()
    figure_quantum(rows)
    figure_classical()
    figure_trotter(rows, nb2)
    figure_cutoff()
    figure_frame(False)
    figure_frame(True)
    print(f"[presentation] wrote 6 PNG/PDF figure pairs to {OUT}")


if __name__ == "__main__":
    main()
