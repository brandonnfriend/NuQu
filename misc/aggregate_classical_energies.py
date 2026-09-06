"""Aggregate the classical energy estimates: variational bound, PT2, and an
EXTRAPOLATED E_infinity with an honest error bar -- for EVERY L.

WHAT CHANGED (audit 2026-09-05, P0-2). This script used to report a binary label:
"L=2 converged (so quote it directly), L>=3 upper bound (so quote nothing)". The
convergence evidence was the energy change over the last core-doubling -- a
convergence DIAGNOSTIC, not an error bar -- and at L=2 that change was 1.6 MeV,
LARGER than the 1 MeV target it was being used to support. So the label is retired.

The defensible quantity, at every L including L=2, is the extrapolated
E_infinity carried with an uncertainty (`extrapolation.einf_with_uncertainty`):

  * E_var       -- rigorous Ritz UPPER BOUND for this (L, n_b, frame) truncated
                   Hamiltonian at this core. Always quotable, never an estimate of
                   the truth.
  * E_var+PT2   -- Epstein-Nesbet at the deepest post-collapse rung (non-variational).
  * E_inf ± σ   -- the N -> infinity (Full-CI-within-truncation) limit, ESTIMATED by
                   extrapolation over the POST-COLLAPSE rungs, with σ combining the fit,
                   the disagreement between two independent extrapolators, the
                   leave-one-out refit shift, and the seed-to-seed spread.
                   **This is the reported result.**

Large error bars are the truthful outcome for a variational heuristic that has not
converged. σ covers the N -> infinity limit of THIS ladder only -- NOT the boson cutoff
n_b, the lattice/finite-volume error, the EFT truncation, or the risk that the
selected-CI search never found the right basin at all.

THE BASIN CAVEAT (why we fit post-collapse only). The warm-grown ladder sits in a
delocalized exploration basin and escapes onto the compact ground-state basin at a
specific core. Fitting across that step mixes two different sequences: at L=2 a PT2
extrapolation from the pre-collapse basin overshoots the deep answer by ~19 MeV/site.
That cross-check is still reported -- as the failing diagnostic that justifies the split.

    python -m misc.aggregate_classical_energies \
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
from classical.trimci.extrapolation import (combine_seeds, fit_einf_pt2,  # noqa: E402
                                            split_at_collapse)

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
PURP, TEAL = "#7b5cd6", "#2f8f8a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NB_COLOR = {2: MUTED, 3: BLUE, 4: PURP}
SIGMA_COLOR = {"fit": BLUE, "method": ORANGE, "stability": TEAL, "seed": PURP}
# Pre-vertex-fix data is inadmissible (the nucleon spin-isospin vertex bug, fixed
# 2026-08-18 / 9404fac). Guard here too: a figure is the last place a retired shard
# should be able to reappear.
_VERTEX_FIX_DATE = "2026-08-18"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def load(dirs, require_post_fix=True):
    """Collect shards into {(n_b, L): {seed: rungs}} plus a per-group metadata record.

    Keeps the DEEPEST ladder when the same (n_b, L, seed) appears in more than one
    directory (a resubmit supersedes the run it recovered)."""
    groups, meta = defaultdict(dict), {}
    depth, skipped = {}, []
    for d in dirs:
        if require_post_fix and any(t in d for t in ("2026-08-13", "2026-08-14", "2026-08-15",
                                                     "2026-08-16", "2026-08-17")):
            skipped.append(d)
            continue
        for f in sorted(glob.glob(f"{d}/bare_*.json")):
            j = json.load(open(f))
            rungs = [r for r in j.get("rungs", []) if r.get("E_var") is not None]
            if not rungs:
                continue
            key, seed = (int(j["n_b"]), int(j["L"])), int(j["seed"])
            top = max(r["core"] for r in rungs)
            if depth.get((key, seed), -1) >= top:
                continue
            depth[(key, seed)] = top
            groups[key][seed] = rungs
            meta[key] = {"L": j["L"], "n_b": int(j["n_b"]), "N_f": j.get("N_f"),
                         "sites": j["sites"], "A": j["A"], "dim": j.get("dim", 3),
                         "filling": j.get("filling"), "n_terms": j.get("n_terms")}
    if skipped:
        print(f"[load] REFUSED {len(skipped)} pre-vertex-fix director{'y' if len(skipped)==1 else 'ies'}: {skipped}")
    return groups, meta


def analyze(groups, meta):
    """One record per (n_b, L): the pooled extrapolation + the pre-basin cross-check."""
    recs = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        n_b, L = key
        m = meta[key]
        pooled = combine_seeds(groups[key], sites=m["sites"])
        # the failing diagnostic: PT2 extrapolated from the PRE-collapse basin, which is
        # where 290832's PT2 cap forced every PT2 point to live.
        seed0 = sorted(groups[key])[0]
        rungs = sorted(groups[key][seed0], key=lambda r: r["core"])
        post, basin = split_at_collapse(rungs, sites=m["sites"])
        pre = [r for r in rungs[:basin["collapse_index"]] if r.get("dE_pt2") is not None]
        pre_fit = (fit_einf_pt2([r["E_var"] for r in pre], [r["dE_pt2"] for r in pre])
                   if len(pre) >= 3 else {"ok": False, "reason": f"{len(pre)} pre-basin PT2 rungs",
                                          "E_inf": None, "sigma": None})
        recs.append({**m, **pooled, "key": f"nb{n_b}_L{L}",
                     "collapse_core": basin["collapse_core"],
                     "collapse_drop_per_site": basin["collapse_drop_per_site"],
                     "pre_basin_pt2": {"ok": pre_fit.get("ok"), "n_pts": len(pre),
                                       "E_inf": pre_fit.get("E_inf"),
                                       "E_inf_ps": (pre_fit["E_inf"] / m["sites"]
                                                    if pre_fit.get("E_inf") is not None else None),
                                       "reason": pre_fit.get("reason")}})
    return recs


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.patch.set_facecolor(SURFACE)

    # (a) per-site: rigorous bound + the REPORTED extrapolated value with its error bar
    for n_b in sorted({r["n_b"] for r in recs}):
        rs = sorted([r for r in recs if r["n_b"] == n_b], key=lambda r: r["L"])
        c = NB_COLOR.get(n_b, INK2)
        axA.plot([r["L"] for r in rs], [r["E_var_bound_ps"] for r in rs], "--o", color=c,
                 lw=1.1, ms=6, mfc="none", mew=1.5, zorder=4,
                 label=f"$n_b$={n_b}  $E_\\mathrm{{var}}$ upper bound (deepest core)")
        ex = [r for r in rs if r["ok"]]
        if ex:
            axA.errorbar([r["L"] for r in ex], [r["E_inf_ps"] for r in ex],
                         yerr=[(r["sigma_ps"] or 0.0) for r in ex], fmt="s", color=c,
                         ms=7, mec=SURFACE, mew=1.0, capsize=4, elinewidth=1.5, zorder=6,
                         label=f"$n_b$={n_b}  $E_\\infty$ extrapolated ± σ  (**reported**)")
        miss = [r for r in rs if not r["ok"]]
        for r in miss:
            axA.plot([r["L"]], [r["E_var_bound_ps"]], "x", color=CRIT, ms=9, mew=2.0, zorder=7)
    if any(not r["ok"] for r in recs):
        axA.plot([], [], "x", color=CRIT, ms=9, mew=2.0,
                 label="no defensible extrapolation — bound only")
    axA.set_xlabel("lattice size $L$  (dim=3, filling 1.0)", color=INK2, fontsize=9.5)
    axA.set_ylabel("energy / site  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  Rigorous bound vs the reported extrapolated $E_\\infty$",
                  color=INK, fontsize=10.5, loc="left", weight="bold")
    axA.set_xticks(sorted({r["L"] for r in recs}))
    axA.legend(frameon=False, fontsize=7.2, loc="lower right", labelcolor=INK2)
    _style(axA)

    # (b) where the error bar comes from — the budget, per (n_b, L)
    ex = [r for r in recs if r["ok"] and r.get("sigma_ps") is not None]
    if ex:
        labels = [f"$n_b$={r['n_b']}\n$L$={r['L']}" for r in ex]
        x = np.arange(len(ex))
        bottom = np.zeros(len(ex))
        for term in ("fit", "method", "stability", "seed"):
            # the terms that actually build sigma: the BEST-LADDER seed's own fit/method/
            # stability, plus the between-seed spread (which is a property of the pool).
            vals = []
            for r in ex:
                if term == "seed":
                    vals.append(float(r.get("sigma_seed_ps") or 0.0))
                    continue
                t = (r["per_seed"].get(r.get("best_seed"), {}).get("sigma_terms_ps") or {})
                vals.append(float(t.get(term) or 0.0))
            vals = np.asarray(vals)
            axB.bar(x, vals, bottom=bottom, color=SIGMA_COLOR[term], width=0.62,
                    label=f"σ$_\\mathrm{{{term}}}$", edgecolor=SURFACE, lw=0.8)
            bottom += vals
        axB.plot(x, [r["sigma_ps"] or 0.0 for r in ex], "D", color=INK, ms=5, zorder=6,
                 label="σ (quadrature)")
        # explicit xlim: with a single (n_b, L) the auto-scaled axis stretches one bar
        # across the whole panel and it stops reading as a bar chart.
        axB.set_xlim(-0.6, len(ex) - 0.4)
        axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8)
        axB.set_ylim(0, max(1e-9, max(r["sigma_ps"] or 0.0 for r in ex)) * 1.45)
        axB.set_ylabel("uncertainty contribution  (MeV / site)", color=INK2, fontsize=9.5)
        axB.set_title("b  Error budget — what σ is made of (linear stack; σ adds in quadrature)",
                      color=INK, fontsize=10.5, loc="left", weight="bold")
        axB.legend(frameon=False, fontsize=7.4, loc="upper right", labelcolor=INK2, ncol=2)
        _style(axB)

    fig.suptitle("Classical energy aggregate — rigorous variational bound and the reported "
                 "extrapolated $E_\\infty$ with an honest error budget",
                 fontsize=11.2, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def _pm(x, s):
    if x is None:
        return "—"
    return f"{x:,.1f}" + (f" ± {s:,.1f}" if s is not None else " (no σ)")


def make_table(recs, out_path):
    md = [
        "# Classical energy aggregate — variational bound and extrapolated $E_\\infty$ ± σ\n",
        "_dim=3, filling 1.0. **The reported result at every L is the extrapolated $E_\\infty$ with "
        "its error bar** — not a 'converged' value. `E_var` is a rigorous Ritz UPPER BOUND for the "
        "truncated (L, n_b, frame) Hamiltonian at that core, and stays quotable as a bound in its "
        "own right. `E_var+PT2` is Epstein–Nesbet (non-variational). $E_\\infty$ is the "
        "$N\\to\\infty$ Full-CI-within-truncation limit, ESTIMATED by extrapolation over the "
        "POST-collapse rungs, never measured._\n",
        "**What σ covers:** the fit, the disagreement between two independent extrapolators "
        "(PT2/SHCI vs power-law) on the same rungs, the leave-one-out refit shift, and the "
        "seed-to-seed spread of independent solver trajectories. **What σ does NOT cover:** the "
        "boson cutoff $n_b$, lattice spacing / finite volume, the EFT truncation, or the "
        "possibility that the selected-CI search never found the right basin. Those are separate "
        "budget lines and are not folded in here.\n",
        "| $n_b$ | L | sites | seeds | $E_\\mathrm{var}$/site (bound) | ΔE last doubling | "
        "$E_\\mathrm{var}$+PT2/site | **$E_\\infty$/site ± σ (reported)** | extrapolator | post-basin rungs |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|",
    ]
    for r in recs:
        s0 = sorted(r["per_seed"])[0]
        v0 = r["per_seed"][s0]
        dE = v0.get("dE_last_doubling_ps")
        pt2 = v0.get("E_var_plus_pt2_ps")
        rep = (f"**{_pm(r['E_inf_ps'], r['sigma_ps'])}**" if r["ok"]
               else f"— ({r['reason']})")
        prim = ({"pt2": "SHCI/PT2 intercept", "power": "power law $E_\\infty+aN^{-b}$"}
                .get(v0.get("primary"), "—"))
        md.append(f"| {r['n_b']} | {r['L']} | {r['sites']} | {r['n_seeds']} | "
                  f"{r['E_var_bound_ps']:.1f} | {'—' if dE is None else f'{dE:.2f}'} | "
                  f"{'—' if pt2 is None else f'{pt2:.1f}'} | {rep} | {prim} | {v0.get('n_post', 0)} |")

    md += ["", "### Error budget (MeV/site)", "",
           "_fit / method / stability are the BEST-LADDER seed's own terms (the seed whose "
           "variational bound is tightest — that is the seed the reported $E_\\infty$ comes "
           "from); σ seed is the spread of $E_\\infty$ ACROSS seeds. σ total is their "
           "quadrature, so it is not the sum of the row._", "",
           "| $n_b$ | L | best seed | σ fit | σ method | σ stability | σ seed | **σ total** |",
           "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in recs:
        if not r["ok"]:
            continue
        best = r.get("best_seed")
        v = r["per_seed"].get(best, {})
        terms = v.get("sigma_terms_ps") or {}
        row = [(f"{terms[t]:.2f}" if terms.get(t) is not None else "—")
               for t in ("fit", "method", "stability")]
        row.append(f"{r['sigma_seed_ps']:.2f}" if r.get("sigma_seed_ps") is not None
                   else "— (1 seed)")
        tot = f"{r['sigma_ps']:.2f}" if r.get("sigma_ps") is not None else "—"
        md.append(f"| {r['n_b']} | {r['L']} | {best} | " + " | ".join(row) + f" | **{tot}** |")

    md += ["", "### Why we fit the POST-collapse basin (the failing diagnostic)", "",
           "The warm-grown ladder escapes a delocalized exploration basin onto the compact "
           "ground-state basin at a specific core. A PT2 extrapolation from the PRE-collapse side "
           "is unreliable — this table is the evidence, not a result.", "",
           "| $n_b$ | L | collapse core | drop at collapse (MeV/site) | PT2 extrap from PRE-basin /site | "
           "deepest $E_\\mathrm{var}$/site | error |",
           "|--:|--:|--:|--:|--:|--:|--:|"]
    for r in recs:
        pb = r["pre_basin_pt2"]
        if not pb["ok"]:
            continue
        err = pb["E_inf_ps"] - r["E_var_bound_ps"]
        md.append(f"| {r['n_b']} | {r['L']} | {r['collapse_core']:,} | "
                  f"{r['collapse_drop_per_site']:.1f} | {pb['E_inf_ps']:.1f} | "
                  f"{r['E_var_bound_ps']:.1f} | {err:+.1f} |")

    # n_b=2 -> 3 paired shift, when both cutoffs are present at the same L
    by = {(r["n_b"], r["L"]): r for r in recs}
    pairs = [(L, by[(2, L)], by[(3, L)]) for L in sorted({r["L"] for r in recs})
             if (2, L) in by and (3, L) in by]
    if pairs:
        md += ["", "### Cutoff shift $n_b$: 2 → 3 at matched settings", "",
               "| L | $E_\\mathrm{var}$/site $n_b$=2 | $n_b$=3 | Δ bound | "
               "$E_\\infty$/site $n_b$=2 | $n_b$=3 | Δ extrapolated |",
               "|--:|--:|--:|--:|--:|--:|--:|"]
        for L, a, b in pairs:
            ea = f"{a['E_inf_ps']:.1f}" if a["ok"] else "—"
            eb = f"{b['E_inf_ps']:.1f}" if b["ok"] else "—"
            de = (f"{b['E_inf_ps'] - a['E_inf_ps']:+.2f}"
                  if (a["ok"] and b["ok"]) else "—")
            md.append(f"| {L} | {a['E_var_bound_ps']:.1f} | {b['E_var_bound_ps']:.1f} | "
                      f"{b['E_var_bound_ps'] - a['E_var_bound_ps']:+.2f} | "
                      f"{ea} | {eb} | {de} |")
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="shard directories (bare_*.json); several may be given")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-pre-vertex-fix", action="store_true",
                    help="NOT for release: lift the pre-2026-08-18 data refusal")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    groups, meta = load(args.data, require_post_fix=not args.allow_pre_vertex_fix)
    assert groups, "no bare_*.json found"
    recs = analyze(groups, meta)
    make_figure(recs, f"{args.out_dir}/classical_energy_aggregate")
    make_table(recs, f"{args.out_dir}/classical_energy_aggregate.md")
    json.dump(recs, open(f"{args.out_dir}/classical_energy_aggregate.json", "w"), indent=2)
    print(f"[json] wrote {args.out_dir}/classical_energy_aggregate.json")
    print("[done] " + " | ".join(
        f"nb{r['n_b']}/L{r['L']}: bound {r['E_var_bound_ps']:.1f}"
        + (f", E_inf {r['E_inf_ps']:.1f}±{(r['sigma_ps'] or 0):.1f}/site" if r["ok"]
           else ", NO extrapolation")
        for r in recs))


if __name__ == "__main__":
    main()
