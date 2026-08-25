"""Aggregate the classical energy estimates: variational, PT2-corrected, and
extrapolated-with-error-bars — the honest three-tier energy picture.

Three quantities per L (see classical/trimci/extrapolation.py for the definitions):
  * E_var        — rigorous VARIATIONAL upper bound (Ritz) at the deepest core.
  * E_var + PT2  — Epstein-Nesbet second-order estimate (non-variational).
  * E_extrap±σ   — the N->infinity (Full-CI-within-truncation) limit, ESTIMATED by
                   extrapolation, NOT measured.

THE BASIN CAVEAT (why we can't just feed all rungs to the extrapolator). The bare
warm-grown ladder shows a "collapse" rung: the selected-CI search escapes a
delocalized exploration basin onto the compact ground-state basin (a sharp E drop +
participation-ratio collapse; see the results note). PT2 is only computed in the
PRE-collapse (exploration) basin, and a PT2 extrapolation from THERE is unreliable —
for L=2 it lands ~20 MeV/site ABOVE the converged variational answer. So we:
  * split each ladder at the collapse (largest per-site energy drop),
  * extrapolate E_var over the POST-collapse (consistent, deep) basin only,
  * report the PRE-basin PT2 extrapolation ONLY as a failing cross-check (it lies),
  * take the converged variational value as primary where the ladder converged (L=2).

    python -m misc.aggregate_classical_energies \
        --data data/classical/2026-08-23/bare_baseline_290832
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
from classical.trimci.extrapolation import fit_einf_power, fit_einf_pt2  # noqa: E402

BLUE, ORANGE, CRIT, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a"
PURP = "#7b5cd6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
_CONV_MEV = 2.0                                    # |ΔE| (total) per doubling below this = converged


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def analyze_one(j):
    """Return the honest per-L energy record (all energies in MeV, plus per-site)."""
    sites = j["sites"]
    rungs = [r for r in j.get("rungs", []) if r.get("E_var") is not None]
    rungs.sort(key=lambda r: r["core"])
    # collapse = the rung with the largest per-site drop from its predecessor
    drops = [((rungs[i - 1]["E_var"] - rungs[i]["E_var"]) / sites, i) for i in range(1, len(rungs))]
    d_collapse, i_col = max(drops)
    post = rungs[i_col:]                            # deep / consistent basin
    pt2r = [r for r in rungs if r.get("dE_pt2") is not None]   # exploration basin (PT2 lives here)

    E_var = rungs[-1]["E_var"]
    dE_last = abs(rungs[-1]["E_var"] - rungs[-2]["E_var"]) if len(rungs) >= 2 else float("nan")
    converged = dE_last < _CONV_MEV
    E_pt2 = (pt2r[-1]["E_var"] + pt2r[-1]["dE_pt2"]) if pt2r else None

    power = fit_einf_power([r["core"] for r in post], [r["E_var"] for r in post])   # post-basin
    pt2ex = (fit_einf_pt2([r["E_var"] for r in pt2r], [r["dE_pt2"] for r in pt2r])   # pre-basin (cross-check)
             if len(pt2r) >= 3 else {"ok": False, "reason": "<3 PT2 rungs", "E_inf": None, "sigma": None})

    # BEST ESTIMATE: converged direct value where the ladder converged (drift as σ);
    # else only a rigorous upper bound (post-basin too short to extrapolate reliably).
    if converged:
        best, best_sig, best_src = E_var, dE_last, "converged variational (drift σ)"
    elif power.get("ok"):
        best, best_sig, best_src = power["E_inf"], power.get("sigma"), "post-basin power-law extrap"
    else:
        best, best_sig, best_src = None, None, "bound only (insufficient post-basin rungs)"

    per = lambda x: (x / sites) if x is not None else None
    return dict(
        L=j["L"], sites=sites, A=j["A"], n_rungs=len(rungs),
        collapse_core=rungs[i_col]["core"], collapse_drop_per_site=d_collapse,
        maxcore=rungs[-1]["core"], cores=[r["core"] for r in rungs],
        E_series=[r["E_var"] for r in rungs],
        E_var=E_var, E_var_ps=per(E_var), dE_last=dE_last, converged=converged,
        E_pt2_pre=E_pt2, E_pt2_pre_ps=per(E_pt2),
        extrap_power=dict(ok=power.get("ok"), E_inf=power.get("E_inf"),
                          E_inf_ps=per(power.get("E_inf")), sigma=power.get("sigma"),
                          sigma_ps=per(power.get("sigma")), b=power.get("b"),
                          n_pts=len(post), reason=power.get("reason")),
        extrap_pt2_pre=dict(ok=pt2ex.get("ok"), E_inf=pt2ex.get("E_inf"),
                            E_inf_ps=per(pt2ex.get("E_inf")), sigma=pt2ex.get("sigma"),
                            sigma_ps=per(pt2ex.get("sigma")), n_pts=len(pt2r),
                            reason=pt2ex.get("reason")),
        best=best, best_ps=per(best), best_sigma=best_sig, best_sigma_ps=per(best_sig),
        best_source=best_src,
    )


def load(dirs):
    recs = []
    seen = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/bare_*.json")):
            j = json.load(open(f))
            r = analyze_one(j)
            if r["L"] not in seen or r["maxcore"] > seen[r["L"]]["maxcore"]:
                seen[r["L"]] = r
    return [seen[L] for L in sorted(seen)]


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    fig.patch.set_facecolor(SURFACE)

    # (a) per-site energies vs L: variational bound, PT2 (pre-basin, unreliable), extrapolation
    Ls = [r["L"] for r in recs]
    axA.plot(Ls, [r["E_var_ps"] for r in recs], "-", color=MUTED, lw=1.1, zorder=2)
    for r in recs:                                 # variational bound: filled=converged, hollow=bound
        f = r["converged"]
        axA.plot([r["L"]], [r["E_var_ps"]], "o", ms=9, color=(GREEN if f else "none"),
                 mec=(GREEN if f else CRIT), mew=1.8, zorder=5)
    axA.plot([r["L"] for r in recs], [r["E_pt2_pre_ps"] for r in recs], "x", color=ORANGE,
             ms=8, mew=1.8, zorder=4, label="$E_\\mathrm{var}$+PT2 (exploration basin — unreliable)")
    ex = [r for r in recs if r["extrap_power"]["ok"]]
    if ex:
        axA.errorbar([r["L"] for r in ex], [r["extrap_power"]["E_inf_ps"] for r in ex],
                     yerr=[(r["extrap_power"]["sigma_ps"] or 0) for r in ex], fmt="s", color=BLUE,
                     ms=7, mec=SURFACE, mew=1.0, capsize=4, elinewidth=1.4, zorder=6,
                     label="$E_\\infty$ extrapolated (post-basin, ±σ)")
    axA.plot([], [], "o", color=GREEN, label="$E_\\mathrm{var}$ converged (bound = $E_\\infty$)")
    axA.plot([], [], "o", color="none", mec=CRIT, mew=1.8, label="$E_\\mathrm{var}$ upper bound (not converged)")
    axA.set_xlabel("lattice size $L$  (dim=3, filling 1.0, $n_b$=2)", color=INK2, fontsize=9.5)
    axA.set_ylabel("energy / site  (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  Variational · PT2 · extrapolated, per site", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axA.set_xticks(Ls)
    axA.legend(frameon=False, fontsize=7.3, loc="lower right", labelcolor=INK2)
    _style(axA)

    # (b) L=2 detail: why the PRE-basin PT2 extrapolation lies
    r2 = next((r for r in recs if r["L"] == 2), None)
    if r2:
        c = np.array(r2["cores"]); e = np.array(r2["E_series"]) / r2["sites"]
        axB.semilogx(c, e, "-o", color=INK2, lw=1.6, ms=5, mec=SURFACE, mew=0.8, zorder=4,
                     label="$E_\\mathrm{var}$/site (ladder)")
        axB.axvline(r2["collapse_core"], ls=":", color=CRIT, lw=1.2, zorder=2)
        _yann = r2["E_var_ps"] + 0.75 * (e.max() - r2["E_var_ps"])   # clear of the PT2/green lines & legend
        axB.annotate("basin\ncollapse", (r2["collapse_core"], _yann), color=CRIT, fontsize=8,
                     ha="right", va="center", textcoords="offset points", xytext=(-10, 0))
        # pre-basin PT2 extrapolation target (the WRONG answer) vs converged plateau (right)
        if r2["extrap_pt2_pre"]["ok"]:
            axB.axhline(r2["extrap_pt2_pre"]["E_inf_ps"], ls="--", color=ORANGE, lw=1.6, zorder=3,
                        label=f"PT2 extrap from exploration basin → {r2['extrap_pt2_pre']['E_inf_ps']:.0f} (WRONG)")
        axB.axhline(r2["E_var_ps"], ls="-", color=GREEN, lw=1.6, zorder=3,
                    label=f"converged $E_\\mathrm{{var}}$ → {r2['E_var_ps']:.0f} (right)")
        axB.set_xlabel("selected-CI core (# determinants)", color=INK2, fontsize=9.5)
        axB.set_ylabel("$E_\\mathrm{var}$ / site  (MeV)", color=INK2, fontsize=9.5)
        axB.set_title("b  $L$=2: PT2 extrapolation from the wrong basin overshoots by ~20/site",
                      color=INK, fontsize=10.5, loc="left", weight="bold")
        axB.legend(frameon=False, fontsize=7.6, loc="upper right", labelcolor=INK2)
        _style(axB)

    fig.suptitle("Classical energy aggregate — variational bound, PT2, and extrapolated $E_\\infty$ "
                 "(basin-aware, honest error bars)", fontsize=11.3, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def _sig(x, ps=None):
    if x is None:
        return "—"
    return f"{x:,.1f}" + (f" ± {ps:,.1f}" if ps else "")


def make_table(recs, out_path):
    md = [
        "# Classical energy aggregate — variational, PT2, extrapolated ($E_\\infty$)\n",
        "_dim=3, filling 1.0, n_b=2. E_var = rigorous VARIATIONAL upper bound (Ritz). "
        "E_var+PT2 = Epstein–Nesbet (non-variational). E_∞ = N→∞ (Full-CI-within-truncation) limit, "
        "ESTIMATED by extrapolation, not measured — and still carrying the (L, n_b, EFT-truncation, "
        "lattice) error. All per-site (size-intensive)._\n",
        "**Basin caveat.** The warm-grown ladder collapses onto the compact ground-state basin at a "
        "specific core (below). PT2 is only computed in the PRE-collapse exploration basin, and a PT2 "
        "extrapolation from there is unreliable — see the L=2 cross-check (it overshoots the converged "
        "answer by ~20 MeV/site). We extrapolate E_var over the POST-collapse basin only, and take the "
        "converged variational value as primary where the ladder converged.\n",
        "| L | sites | E_var/site (bound) | conv? | E_var+PT2/site (pre-basin) | E_∞/site extrap ±σ | "
        "**best estimate/site** | source |",
        "|--:|--:|--:|:--:|--:|--:|--:|:--|",
    ]
    for r in recs:
        conv = "✅" if r["converged"] else "—"
        ex = r["extrap_power"]
        exs = (_sig(ex["E_inf_ps"], ex["sigma_ps"]) if ex["ok"]
               else f"— ({ex['n_pts']} pts)")
        best = _sig(r["best_ps"], r["best_sigma_ps"]) if r["best_ps"] is not None else f"< {r['E_var_ps']:.0f} (bound)"
        md.append(f"| {r['L']} | {r['sites']} | {r['E_var_ps']:.1f} | {conv} | "
                  f"{r['E_pt2_pre_ps']:.1f} | {exs} | **{best}** | {r['best_source']} |")
    md.append("")
    md.append("### PT2 extrapolation cross-check (the failing diagnostic)")
    md.append("| L | PT2 extrap from exploration basin /site | converged/deepest E_var /site | error |")
    md.append("|--:|--:|--:|--:|")
    for r in recs:
        px = r["extrap_pt2_pre"]
        if not px["ok"]:
            continue
        err = px["E_inf_ps"] - r["E_var_ps"]
        md.append(f"| {r['L']} | {px['E_inf_ps']:.1f} | {r['E_var_ps']:.1f}"
                  f"{' (converged)' if r['converged'] else ' (bound)'} | {err:+.1f} |")
    md.append("\n**Reading:** at L=2 (converged, so we know the truth) the PT2 extrapolation from the "
              "exploration basin lands ~20 MeV/site ABOVE the converged variational answer — a concrete "
              "demonstration that PT2 and PT2-extrapolation from an unconverged (wrong) basin are "
              "unreliable. This is why we report the rigorous variational bound (and the converged value "
              "where available) as primary, and treat PT2/extrapolation as diagnostics. It also flags "
              "that the L≥3 'unconverged' status may be partly a SEARCH-quality artifact (the warm-grow "
              "ladder biases toward its seed basin) rather than a pure extensivity wall — a heavy-restart "
              "/ wider-pool re-run at small core would decide this.\n")
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/classical/2026-08-23/bare_baseline_290832"])
    ap.add_argument("--out-dir", default="data/classical/2026-08-24/baseline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load(args.data)
    assert recs, "no bare_*.json found"
    make_figure(recs, f"{args.out_dir}/classical_energy_aggregate")
    make_table(recs, f"{args.out_dir}/classical_energy_aggregate.md")
    json.dump(recs, open(f"{args.out_dir}/classical_energy_aggregate.json", "w"), indent=2)
    print(f"[json] wrote {args.out_dir}/classical_energy_aggregate.json")
    print("[done] " + " | ".join(
        f"L{r['L']}:{'conv ' if r['converged'] else 'bnd '}{r['best_ps'] or r['E_var_ps']:.0f}/site" for r in recs))


if __name__ == "__main__":
    main()
