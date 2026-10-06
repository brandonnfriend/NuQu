"""Multi-A versions of the classical baseline figure, from the fixed-A n_b=3 sweep.

Cluster 293963 (+ 294184 / 294238 resubmits): bare TrimCI, n_b=3, A = 2..10 x L = 2..5 x
3 independent seeds, search v2 (stratified starts + 16k choice point). Every energy is the
Wick-ordered convention; each A is loaded on its own (`load(..., A=a)`).

Encoding (as in docs/presentation/classical_baseline.png):
  * open circle, dashed  = tightest variational upper bound over seeds (rigorous);
  * filled square + bar  = extrapolated E_inf +- sigma on the TrimCI/COO literature
                           convention (`extrapolation.DEFAULT_SIGMA_CONVENTION`);
  * hollow diamond       = the cross-check estimator (COO power law), where it resolves;
  * red x                = bound only (no estimator resolved a limit).
The last-doubling energy change is deliberately NOT shown: late drops can be the search
moving to a better nucleon arrangement, so it is not comparable across seeds.

    .venv/bin/python -m misc.make_Asweep_figures \
        [--data data/classical/2026-09-25/bare_Asweep_nb3_293963] [--out docs/presentation]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nuqu-presentation-mpl")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from misc.aggregate_classical_energies import analyze, load  # noqa: E402
from misc.make_presentation_figures import GRAY, RED, setup, style  # noqa: E402

AS = list(range(2, 11))
LS = [2, 3, 4, 5]


def collect(data_dir, sigma_convention=None):
    rec = {}
    for A in AS:
        g, m = load([str(data_dir)], convention="wick", A=A)
        for r in analyze(g, m, sigma_convention=sigma_convention):
            if r["n_b"] != 3:
                continue
            v = r["per_seed"].get(r.get("best_seed"), {}) if r.get("best_seed") is not None else {}
            cc = v.get("cross_check") or {}
            rob = r.get("seed_robustness") or {}
            bounds = [x["E_var_bound_ps"] for x in r["per_seed"].values()
                      if x.get("E_var_bound_ps") is not None]
            rec[(A, r["L"])] = dict(
                A=A, L=r["L"], sites=r["sites"], n_seeds=r["n_seeds"], ok=bool(r["ok"]),
                bound=r["E_var_bound_ps"], E_inf=r.get("E_inf_ps"), sigma=r.get("sigma_ps"),
                ci90=v.get("ci90_ps"), primary=v.get("primary"),
                xc_E=cc.get("E_inf_ps"), xc_gap=cc.get("gap_ps"),
                seed_spread=rob.get("spread_ps"),
                bound_spread=(max(bounds) - min(bounds)) if len(bounds) >= 2 else None,
                n_agree=rob.get("n_agreeing_with_best"),
                reason=None if r["ok"] else r.get("reason"),
                convention=r.get("sigma_convention"))
    return rec


def _series(rec, A):
    Ls = [L for L in LS if (A, L) in rec]
    get = lambda k: np.array([rec[(A, L)][k] if rec[(A, L)][k] is not None else np.nan
                              for L in Ls], dtype=float)
    ok = np.array([rec[(A, L)]["ok"] for L in Ls], dtype=bool)
    return np.array(Ls), get("bound"), get("E_inf"), get("sigma"), get("xc_E"), ok


def figure_overlay(rec, out):
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for A in AS:
        x, b, e, s, _, ok = _series(rec, A)
        if not len(x):
            continue
        c = cmap((A - 2) / 8 * 0.9)
        dx = x + (A - 6) * 0.018
        ax.plot(dx, b, "o--", color=c, lw=1.4, ms=6, mfc="white", mew=1.6)
        ax.errorbar(dx[ok], e[ok], yerr=s[ok], fmt="s", ms=6, color=c, capsize=3,
                    elinewidth=1.6, label=f"A={A}")
        ax.plot(dx[~ok], b[~ok], "x", color=RED, ms=9, mew=2)
    ax.plot([], [], "o--", color=GRAY, mfc="white", label="variational bound")
    ax.plot([], [], "x", color=RED, mew=2, label="bound only")
    ax.set(title="Classical n_b=3 baseline at fixed A",
           xlabel="Lattice length  $L$", ylabel="Energy per site (MeV)")
    ax.set_xticks(LS)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    style(ax)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"classical_baseline_multiA.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_per_A(rec, out):
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
    for ax, A in zip(axs.flat, AS):
        x, b, e, s, xc, ok = _series(rec, A)
        c = cmap((A - 2) / 8 * 0.9)
        ax.plot(x, b, "o--", color=GRAY, lw=1.8, ms=7, mfc="white", mew=1.8)
        ax.errorbar(x[ok], e[ok], yerr=s[ok], fmt="s", ms=7, color=c, capsize=4, elinewidth=1.8)
        m = ok & np.isfinite(xc)
        ax.plot(x[m] + 0.08, xc[m], "D", ms=6, mfc="none", mec=c, mew=1.4)
        ax.plot(x[~ok], b[~ok], "x", color=RED, ms=10, mew=2.4)
        ax.set_title(f"A = {A}", fontsize=13)
        ax.set_xticks(LS)
        style(ax)
    for ax in axs[-1]:
        ax.set_xlabel("$L$")
    for ax in axs[:, 0]:
        ax.set_ylabel("E / site (MeV)")
    fig.suptitle("Per-A classical baseline, n_b=3 bare  (○ bound · ■ E$_\\infty$±σ, TrimCI/COO "
                 "convention · ◇ power-law cross-check · × bound only)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"classical_baseline_perA.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table(rec, path):
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
    md = ["# Fixed-A n_b=3 classical sweep — per-(A, L) summary (MeV/site, Wick convention)", "",
          "_σ: TrimCI/COO literature convention (bootstrap std of the primary estimator). "
          "Cross-check gap and seed spread are reported beside σ, never added. The bound is the "
          "tightest over seeds._", "",
          "| A | L | seeds | bound | **E∞ ± σ** | 90% c.i. | primary | cross-check E∞ | gap | seed spread (E∞) | bound spread |",
          "|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|--:|"]
    for A in AS:
        for L in LS:
            r = rec.get((A, L))
            if r is None:
                md.append(f"| {A} | {L} | 0 | — | — | — | — | — | — | — | — |")
                continue
            ci = r["ci90"]
            md.append(f"| {A} | {L} | {r['n_seeds']} | {f(r['bound'])} | "
                      + (f"**{f(r['E_inf'])} ± {f(r['sigma'])}**" if r["ok"] else "bound only")
                      + f" | {'—' if not ci else f'[{ci[0]:.2f}, {ci[1]:.2f}]'} | {r['primary'] or '—'} | "
                      f"{f(r['xc_E'])} | {f(r['xc_gap'])} | {f(r['seed_spread'])} | {f(r['bound_spread'])} |")
    Path(path).write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/classical/2026-09-25/bare_Asweep_nb3_293963"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "presentation"))
    ap.add_argument("--sigma-convention", default=None, choices=["trimci-coo", "nuqu-2026-09"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    rec = collect(args.data, args.sigma_convention)
    figure_overlay(rec, out)
    figure_per_A(rec, out)
    table(rec, out / "classical_baseline_multiA_table.md")
    json.dump({f"A{A}_L{L}": v for (A, L), v in rec.items()},
              open(out / "classical_baseline_multiA.json", "w"), indent=1)
    n_ok = sum(v["ok"] for v in rec.values())
    print(f"[Asweep] {len(rec)} (A, L) cells, {n_ok} extrapolated, "
          f"{sum(v['n_seeds'] for v in rec.values())} seed-ladders -> {out}/classical_baseline_"
          f"{{multiA,perA}}.png + _table.md + .json")


if __name__ == "__main__":
    main()
