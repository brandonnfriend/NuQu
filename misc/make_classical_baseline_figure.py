"""Classical bare-frame TrimCI baseline — the variational upper bound and the reported
extrapolated $E_\\infty$, plus the empirical cost growth of this solver.

WHAT CHANGED (audit 2026-09-05). Two edits, both about claim discipline:

  * The binary "converged / not converged" label is GONE (P0-2). It rested on the energy
    change over the last core-doubling, which is a convergence DIAGNOSTIC, not an error
    bar -- and at L=2 that change (1.6 MeV) exceeded the 1 MeV target it was supporting.
    Every L now carries the extrapolated E_infinity with its uncertainty
    (`extrapolation.einf_with_uncertainty`), and the last-doubling change is reported as
    what it is: a diagnostic column.
  * The "classically-hard / quantum-advantage regime" framing is GONE. That this
    particular selected-CI workflow stops descending at reachable cores is an empirical
    cost observation about THIS solver and implementation. It is not evidence of
    classical intractability: no controlled comparison against QMC, tensor networks,
    neural quantum states or other selected-CI codes has been run, and no lower bound
    exists. The figure says cost growth, and only that.

E_var remains a rigorous Ritz UPPER BOUND for the truncated (L, n_b, frame) Hamiltonian
at that core -- true regardless of convergence, and quotable as a bound in its own right.

    python -m misc.make_classical_baseline_figure \
        --data data/classical/<date>/bare_baseline_nb3_<cluster> \
        --out-dir results/02_classical_baseline
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from classical.trimci.extrapolation import combine_seeds  # noqa: E402

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
_PRE_FIX_DIRS = ("2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17")


def load(dirs, allow_pre_fix=False):
    """{(n_b, L): record} — pooled over seeds, deepest ladder wins a duplicate."""
    groups, meta, depth = defaultdict(dict), {}, {}
    for d in dirs:
        if not allow_pre_fix and any(t in d for t in _PRE_FIX_DIRS):
            print(f"[load] REFUSED pre-vertex-fix directory: {d}")
            continue
        for f in sorted(glob.glob(f"{d}/bare_*.json")):
            j = json.load(open(f))
            rungs = sorted([r for r in j.get("rungs", []) if r.get("E_var") is not None],
                           key=lambda r: r["core"])
            if not rungs:
                continue
            key, seed = (int(j["n_b"]), int(j["L"])), int(j["seed"])
            top = rungs[-1]["core"]
            if depth.get((key, seed), -1) >= top:
                continue
            depth[(key, seed)] = top
            groups[key][seed] = rungs
            meta[key] = dict(L=j["L"], n_b=int(j["n_b"]), sites=j["sites"], A=j["A"],
                             N_f=j.get("N_f"), done=j.get("done", False))
    recs = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        m, per_seed = meta[key], groups[key]
        pooled = combine_seeds(per_seed, sites=m["sites"])
        s0 = sorted(per_seed)[0]
        deep = max(per_seed.values(), key=lambda rr: rr[-1]["core"])
        v0 = pooled["per_seed"][s0]
        recs.append({**m, **pooled,
                     "cores": [r["core"] for r in per_seed[s0]],
                     "E": [r["E_var"] for r in per_seed[s0]],
                     "ladders": {s: ([r["core"] for r in rr], [r["E_var"] for r in rr])
                                 for s, rr in per_seed.items()},
                     "maxcore": deep[-1]["core"],
                     "wall_s": sum(r.get("wall_s", 0.0) for r in deep),
                     "dE_last": v0.get("dE_last_doubling"),
                     "dE_last_ps": v0.get("dE_last_doubling_ps")})
    return recs


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    fig.patch.set_facecolor(SURFACE)
    n_b_main = max({r["n_b"] for r in recs})            # plot the selected cutoff
    main = [r for r in recs if r["n_b"] == n_b_main]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(main), 1)))

    # (a) the ladders, with each L's extrapolated E_inf drawn as a band
    for r, col in zip(main, cmap):
        for k, (s, (cs, es)) in enumerate(sorted(r["ladders"].items())):
            axA.semilogx(cs, np.array(es) / r["sites"], "-o", color=col, lw=1.6, ms=4.0,
                         mec=SURFACE, mew=0.7, alpha=(1.0 if k == 0 else 0.45), zorder=3,
                         label=(f"$L$={r['L']} ({r['sites']} sites)" if k == 0 else None))
        if r["ok"]:
            lo = r["E_inf_ps"] - (r["sigma_ps"] or 0.0)
            hi = r["E_inf_ps"] + (r["sigma_ps"] or 0.0)
            axA.axhspan(lo, hi, color=col, alpha=0.16, zorder=1)
            axA.axhline(r["E_inf_ps"], ls="--", color=col, lw=1.1, zorder=2)
    axA.plot([], [], "--", color=INK2, lw=1.1, label="extrapolated $E_\\infty$ ± σ (band)")
    axA.set_xlabel("selected-CI core (# determinants)", color=INK2, fontsize=9.5)
    axA.set_ylabel("variational $E_\\mathrm{var}$ / site  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title(f"a  Core ladders and the extrapolated limit ($n_b$={n_b_main}, all seeds)",
                  color=INK, fontsize=10.5, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=7.4, loc="upper right", labelcolor=INK2)
    _style(axA)

    # (b) per-site: the rigorous bound and the reported extrapolation, per L
    Ls = sorted({r["L"] for r in main})
    axB.plot([r["L"] for r in main], [r["E_var_bound_ps"] for r in main], "--o", color=MUTED,
             lw=1.2, ms=8, mfc="none", mec=MUTED, mew=1.6, zorder=3,
             label="$E_\\mathrm{var}$ upper bound (deepest core)")
    ex = [r for r in main if r["ok"]]
    if ex:
        axB.errorbar([r["L"] for r in ex], [r["E_inf_ps"] for r in ex],
                     yerr=[(r["sigma_ps"] or 0.0) for r in ex], fmt="s", color=BLUE, ms=8,
                     mec=SURFACE, mew=1.0, capsize=4, elinewidth=1.6, zorder=5,
                     label="$E_\\infty$ extrapolated ± σ  (**reported**)")
    for r in main:
        if not r["ok"]:
            axB.plot([r["L"]], [r["E_var_bound_ps"]], "x", color=CRIT, ms=10, mew=2.0, zorder=6)
    if any(not r["ok"] for r in main):
        axB.plot([], [], "x", color=CRIT, ms=10, mew=2.0, label="bound only — no defensible extrapolation")
    axB.set_xlabel(f"lattice size $L$  (dim=3, filling 1.0, $n_b$={n_b_main})", color=INK2, fontsize=9.5)
    axB.set_ylabel("energy / site  (MeV)", color=INK2, fontsize=9.5)
    axB.set_title("b  Bound vs reported $E_\\infty$, per site", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axB.set_xticks(Ls)
    axB.legend(frameon=False, fontsize=7.8, loc="lower right", labelcolor=INK2)
    _style(axB)

    fig.suptitle("Classical bare-frame TrimCI: a rigorous variational bound at every $L$, and the "
                 "extrapolated $E_\\infty$ with its uncertainty",
                 fontsize=11.3, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def make_table(recs, out_path):
    md = ["# Classical bare-frame TrimCI baseline — variational bound and extrapolated $E_\\infty$\n",
          "_dim=3, filling 1.0. `E_var` is a rigorous VARIATIONAL upper bound (Ritz) on the ground "
          "state of the truncated $(L, n_b,$ frame$)$ Hamiltonian at that core — true whether or not "
          "the ladder converged. **The reported energy at every L is the extrapolated $E_\\infty$ with "
          "its error bar**, not a 'converged' value; see `classical_energy_aggregate.md` for the error "
          "budget and what σ does and does not cover. 'ΔE last doubling' is a convergence DIAGNOSTIC, "
          "not an uncertainty._\n",
          "**Cost growth is an observation about this solver, not a hardness claim.** That this "
          "selected-CI workflow stops descending at the cores we can reach says what it costs *here*; "
          "it is not evidence of classical intractability, and no controlled comparison against QMC, "
          "tensor networks, neural quantum states or other selected-CI implementations has been run.\n",
          "| $n_b$ | L | sites | seeds | max core | $E_\\mathrm{var}$ (MeV, bound) | /site | "
          "ΔE last doubling /site (diagnostic) | **$E_\\infty$/site ± σ (reported)** | wall (h, deepest seed) |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in recs:
        rep = (f"**{r['E_inf_ps']:,.1f} ± {(r['sigma_ps'] or 0):,.1f}**" if r["ok"]
               else "— (bound only)")
        dEs = "—" if r["dE_last_ps"] is None else f"{r['dE_last_ps']:.2f}"
        md.append(f"| {r['n_b']} | {r['L']} | {r['sites']} | {r['n_seeds']} | {r['maxcore']:,} | "
                  f"{r['E_var_bound']:,.0f} | {r['E_var_bound_ps']:.1f} | {dEs} | {rep} | "
                  f"{r['wall_s']/3600:.1f} |")
    md.append("\n**Reading:** TrimCI returns a rigorous variational bound at every L, cheaply. The "
              "per-site bound rises with L because the core needed to resolve a fixed per-site "
              "accuracy grows with the lattice (energy is extensive while the reachable core is "
              "not) — an empirical cost statement about this workflow. Where enough post-collapse "
              "rungs exist, the extrapolated $E_\\infty$ is reported with an uncertainty; where they "
              "do not, only the bound is quotable and the row says so.\n")
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-pre-vertex-fix", action="store_true",
                    help="NOT for release: lift the pre-2026-08-18 data refusal")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load(args.data, allow_pre_fix=args.allow_pre_vertex_fix)
    assert recs, "no bare_*.json found"
    make_figure(recs, f"{args.out_dir}/classical_baseline")
    make_table(recs, f"{args.out_dir}/classical_baseline_table.md")
    print("[done] " + " | ".join(
        f"nb{r['n_b']}/L{r['L']}: bound {r['E_var_bound']:,.0f} ({r['E_var_bound_ps']:.1f}/site)"
        + (f", E_inf {r['E_inf_ps']:.1f}±{(r['sigma_ps'] or 0):.1f}/site" if r["ok"] else ", bound only")
        for r in recs))


if __name__ == "__main__":
    main()
