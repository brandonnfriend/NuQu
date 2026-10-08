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

FRAMES + BINDING (2026-10-08, infrastructure C5). `--frame gaussian` reads the squeeze
campaign's `gaussian_*.json` (outputs get a `_gaussian` suffix); `--compare-frame bare`
overlays the two frames per A; `--As 0 1 2 ...` includes the A=0 (pion vacuum) and A=1
(dressed nucleon) references, from which the binding-energy table
    BE(A) = A E(1) - (A-1) E(0) - E(A)   (sigma propagated, classical/trimci/binding.py)
is written per frame. Several --data directories may be given (the bare A=0/1 refs live in
the squeeze campaign directory; the deepest ladder per (n_b, L, seed) wins).
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

from classical.trimci.binding import binding_energy, binding_energy_sigma  # noqa: E402
from misc.aggregate_classical_energies import analyze, load  # noqa: E402
from misc.make_presentation_figures import GRAY, RED, setup, style  # noqa: E402

AS = list(range(2, 11))
LS = [2, 3, 4, 5]


def collect(data_dirs, sigma_convention=None, frame="bare", As=None):
    """{(A, L): summary} for one frame over the A list (default AS), from one or more
    shard directories."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    rec = {}
    for A in (AS if As is None else As):
        g, m = load([str(d) for d in data_dirs], convention="wick", A=A, frame=frame)
        if not g:
            continue
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
                convention=r.get("sigma_convention"), frame=frame)
    return rec


def _series(rec, A):
    Ls = [L for L in LS if (A, L) in rec]
    get = lambda k: np.array([rec[(A, L)][k] if rec[(A, L)][k] is not None else np.nan
                              for L in Ls], dtype=float)
    ok = np.array([rec[(A, L)]["ok"] for L in Ls], dtype=bool)
    return np.array(Ls), get("bound"), get("E_inf"), get("sigma"), get("xc_E"), ok


def _sfx(frame):
    return "" if frame == "bare" else f"_{frame}"


def _As(rec):
    return sorted({A for A, _ in rec})


def figure_overlay(rec, out, frame="bare"):
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for A in _As(rec):
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
    ax.set(title=f"Classical n_b=3 baseline at fixed A  ({frame} frame)",
           xlabel="Lattice length  $L$", ylabel="Energy per site (MeV)")
    ax.set_xticks(LS)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    style(ax)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"classical_baseline_multiA{_sfx(frame)}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def figure_per_A(rec, out, frame="bare"):
    cmap = plt.get_cmap("viridis")
    As = _As(rec)
    n = max(1, int(np.ceil(len(As) / 3)))
    fig, axs = plt.subplots(n, 3, figsize=(13.5, 3.5 * n), sharex=True, squeeze=False)
    for ax in axs.flat[len(As):]:
        ax.set_visible(False)
    for ax, A in zip(axs.flat, As):
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
    fig.suptitle(f"Per-A classical baseline, n_b=3 {frame}  (○ bound · ■ E$_\\infty$±σ, TrimCI/COO "
                 "convention · ◇ power-law cross-check · × bound only)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"classical_baseline_perA{_sfx(frame)}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def figure_frames(recs, out):
    """Overlay two (or more) frames per A: E_inf ± σ (or the bound, ×) vs L, one color per A,
    one marker/linestyle per frame. The frames are different truncated operators, so the
    offset between them is a reported shift, not a convergence difference."""
    cmap = plt.get_cmap("viridis")
    marks = [("s", "-"), ("o", "--"), ("^", ":"), ("D", "-.")]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    As = sorted(set().union(*[set(_As(r)) for r in recs.values()]))
    amin, amax = (min(As), max(As)) if As else (0, 1)
    for (frame, rec), (mk, ls) in zip(recs.items(), marks):
        for A in As:
            x, b, e, s, _, ok = _series(rec, A)
            if not len(x):
                continue
            c = cmap(0.9 * (A - amin) / max(1, amax - amin))
            dx = x + (A - (amin + amax) / 2) * 0.018
            ax.errorbar(dx[ok], e[ok], yerr=s[ok], fmt=mk, ls=ls, ms=6, color=c, capsize=3,
                        elinewidth=1.4, lw=1.2)
            ax.plot(dx[~ok], b[~ok], "x", color=c, ms=8, mew=2)
        ax.plot([], [], mk, ls=ls, color=GRAY, label=f"{frame} frame")
    for A in As:
        ax.plot([], [], "s", color=cmap(0.9 * (A - amin) / max(1, amax - amin)), label=f"A={A}")
    ax.plot([], [], "x", color=GRAY, mew=2, label="bound only")
    ax.set(title="Fixed-A baseline by frame: " + " vs ".join(recs),
           xlabel="Lattice length  $L$", ylabel="Energy per site (MeV)")
    ax.set_xticks(LS)
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    style(ax)
    name = "classical_baseline_multiA_" + "_vs_".join(recs)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def binding_rows(rec):
    """BE(A, L) = A E(1) - (A-1) E(0) - E(A) with propagated σ, per (A >= 2, L) where the A=0
    and A=1 references are present. Uses E_inf ± σ when all three sectors extrapolate;
    otherwise the variational bounds (σ = None, flagged `bounds`). Energies are TOTAL (per
    site x sites); BE is in MeV, BE/A in MeV per nucleon."""
    rows = []
    for (A, L), r in sorted(rec.items()):
        if A < 2 or (0, L) not in rec or (1, L) not in rec:
            continue
        secs = [rec[(0, L)], rec[(1, L)], r]
        n = r["sites"]
        if all(x["ok"] for x in secs):
            E0, E1, EA = [x["E_inf"] * n for x in secs]
            s0, s1, sA = [(x["sigma"] or 0.0) * n for x in secs]
            be, sig, src = binding_energy(E0, E1, EA, A), binding_energy_sigma(s0, s1, sA, A), "E_inf"
        else:
            E0, E1, EA = [x["bound"] * n for x in secs]
            be, sig, src = binding_energy(E0, E1, EA, A), None, "bounds"
        rows.append({"A": A, "L": L, "sites": n, "BE": be, "BE_sigma": sig, "BE_per_A": be / A,
                     "source": src, "E0_ps": E0 / n, "E1_ps": E1 / n, "EA_ps": EA / n,
                     "frame": r.get("frame", "bare")})
    return rows


def binding_table(rows, path, frame):
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
    md = [f"# Binding energies BE(A, L) — n_b=3, {frame} frame (MeV, Wick convention)", "",
          "_BE(A) = A·E(1) − (A−1)·E(0) − E(A); the extensive pion vacuum cancels. σ is "
          "propagated from the three sectors' TrimCI/COO bootstrap σ (uncorrelated, "
          "conservative; `classical/trimci/binding.py`). `bounds` rows use the variational "
          "bounds because a sector did not extrapolate, and carry no σ. BE > 0 means bound._",
          "", "| A | L | **BE ± σ** | BE/A | source | E(0)/site | E(1)/site | E(A)/site |",
          "|--:|--:|--:|--:|:--|--:|--:|--:|"]
    for r in rows:
        be = f"**{f(r['BE'])} ± {f(r['BE_sigma'])}**" if r["BE_sigma"] is not None else f(r["BE"])
        md.append(f"| {r['A']} | {r['L']} | {be} | {f(r['BE_per_A'])} | {r['source']} | "
                  f"{f(r['E0_ps'])} | {f(r['E1_ps'])} | {f(r['EA_ps'])} |")
    if not rows:
        md.append("| — | — | no (A=0, A=1, A) triple present | | | | | |")
    Path(path).write_text("\n".join(md) + "\n")


def figure_binding(rows, out, frame):
    if not rows:
        return
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    Ls = sorted({r["L"] for r in rows})
    for i, L in enumerate(Ls):
        rs = [r for r in rows if r["L"] == L]
        c = cmap(0.9 * i / max(1, len(Ls) - 1))
        x = np.array([r["A"] for r in rs], float)
        y = np.array([r["BE_per_A"] for r in rs], float)
        s = np.array([(r["BE_sigma"] / r["A"]) if r["BE_sigma"] is not None else np.nan for r in rs])
        ok = np.isfinite(s)
        ax.errorbar(x[ok], y[ok], yerr=s[ok], fmt="s-", color=c, ms=6, capsize=3, label=f"L={L}")
        ax.plot(x[~ok], y[~ok], "x", color=c, ms=8, mew=2)
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot([], [], "x", color=GRAY, mew=2, label="from bounds (no σ)")
    ax.set(title=f"Binding energy per nucleon, n_b=3 ({frame} frame)",
           xlabel="Nucleon number  $A$", ylabel="BE / A  (MeV)")
    ax.legend(fontsize=9)
    style(ax)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"classical_binding{_sfx(frame)}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def table(rec, path, frame="bare"):
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
    md = [f"# Fixed-A n_b=3 classical sweep — per-(A, L) summary (MeV/site, Wick convention, {frame} frame)", "",
          "_σ: TrimCI/COO literature convention (bootstrap std of the primary estimator). "
          "Cross-check gap and seed spread are reported beside σ, never added. The bound is the "
          "tightest over seeds._", "",
          "| A | L | seeds | bound | **E∞ ± σ** | 90% c.i. | primary | cross-check E∞ | gap | seed spread (E∞) | bound spread |",
          "|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|--:|"]
    for A in _As(rec):
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
    ap.add_argument("--data", nargs="+",
                    default=[str(ROOT / "data/classical/2026-09-25/bare_Asweep_nb3_293963")],
                    help="shard directories (several allowed; deepest ladder per seed wins)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "presentation"))
    ap.add_argument("--sigma-convention", default=None, choices=["trimci-coo", "nuqu-2026-09"])
    ap.add_argument("--frame", default="bare", help="shard frame to plot (bare | gaussian | ...)")
    ap.add_argument("--compare-frame", default=None,
                    help="a second frame to overlay (e.g. --frame gaussian --compare-frame bare)")
    ap.add_argument("--As", type=int, nargs="+", default=None,
                    help=f"nucleon numbers (default {AS[0]}..{AS[-1]}; include 0 1 for binding)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    sfx = _sfx(args.frame)
    rec = collect(args.data, args.sigma_convention, frame=args.frame, As=args.As)
    assert rec, f"no {args.frame}_*.json cells found in {args.data}"
    figure_overlay(rec, out, args.frame)
    figure_per_A(rec, out, args.frame)
    table(rec, out / f"classical_baseline_multiA{sfx}_table.md", args.frame)
    json.dump({f"A{A}_L{L}": v for (A, L), v in rec.items()},
              open(out / f"classical_baseline_multiA{sfx}.json", "w"), indent=1)
    n_ok = sum(v["ok"] for v in rec.values())
    print(f"[Asweep] {args.frame}: {len(rec)} (A, L) cells, {n_ok} extrapolated, "
          f"{sum(v['n_seeds'] for v in rec.values())} seed-ladders -> {out}/classical_baseline_"
          f"{{multiA,perA}}{sfx}.png + _table.md + .json")
    rows = binding_rows(rec)
    binding_table(rows, out / f"classical_binding{sfx}_table.md", args.frame)
    figure_binding(rows, out, args.frame)
    print(f"[binding] {args.frame}: {len(rows)} BE(A, L) rows "
          f"({sum(r['source'] == 'E_inf' for r in rows)} with σ) -> classical_binding{sfx}_table.md"
          + (" + .png" if rows else "  (no A=0/A=1 references: pass --As 0 1 ...)"))
    if args.compare_frame:
        rec2 = collect(args.data, args.sigma_convention, frame=args.compare_frame, As=args.As)
        assert rec2, f"no {args.compare_frame}_*.json cells found in {args.data}"
        name = figure_frames({args.frame: rec, args.compare_frame: rec2}, out)
        rows2 = binding_rows(rec2)
        binding_table(rows2, out / f"classical_binding{_sfx(args.compare_frame)}_table.md",
                      args.compare_frame)
        figure_binding(rows2, out, args.compare_frame)
        print(f"[frames] {args.frame} vs {args.compare_frame}: {len(rec2)} cells in the second "
              f"frame -> {name}.png; binding rows {len(rows2)}")


if __name__ == "__main__":
    main()
