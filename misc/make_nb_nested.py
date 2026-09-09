"""Fixed-basis boson-cutoff shift `delta_shared` — the T3 nested A-sweep figure (audit P0-3).

WHAT THIS MEASURES. `delta_shared = E_4(core_3) - E_3(core_3)`: both cutoffs evaluated on the
IDENTICAL determinant set, so NO selection difference enters at all. It is the shared-basis
comparison the 2026-09-05 audit named, and it exists because the two cutoffs genuinely differ on a
core that touches the low cutoff's top level: the Hamiltonian carries `a a^dagger` terms and
`a^dagger|N_f-1> = 0` in a truncated space, so H_3 scores an occupation-7 determinant ~357.6 MeV
BELOW H_4. The low cutoff does not merely omit those states, it MIS-SCORES them -- in the
direction that flatters it.

WHAT THE FIGURE SHOWS, and the three things a reader must not misread:
  (a) delta_shared vs core. It is EXACTLY zero until the core reaches the boundary and then grows
      as a power of the core -- it has NOT plateaued. Projected to the core the L=2 baseline
      actually uses (1,024,000) it EXCEEDS the 0.001 MeV/site reference.
  (b) delta_shared vs A at matched core. There is NO monotonic filling trend.
  (c) L=4 is a NON-MEASUREMENT: its ladders never reached the boundary (max occupation 6 < 7), so
      its zero is null-by-non-arrival, not null-by-physics, and is drawn as such.

The 0.001 MeV/site line is `1 MeV / 1000 sites` -- a deliberately conservative uniform per-site
slice of the L=10 GSEE target, not a derived budget.

    python -m misc.make_nb_nested --data data/classical/<date>/nb_nested_<cluster> \
        --out-dir results/04_cutoff_nb
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, CRIT, GREEN, PURP, TEAL = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#7b5cd6", "#2f8f8a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
TARGET = 1e-3                      # MeV/site = 1 MeV / 1000 sites (L=10), conservative reference
L_COLOR = {2: BLUE, 3: GREEN, 4: PURP}
BASELINE_CORE = 1_024_000          # the depth the L=2 classical baseline actually reaches


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def load(dirs):
    """{(L, A): record}. Reports the BEST-BOUND seed, per the project rule that the trajectory
    taken furthest is the one quoted -- and because a worse search demonstrably UNDERSTATES the
    shift (measured: at L=2 A=8 the divergent seed has bound 1864.19 vs 1806.91 and delta 1.32e-4
    vs 6.94e-4)."""
    per = defaultdict(dict)
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/nested_*.json")):
            j = json.load(open(f))
            if not j.get("rungs"):
                continue
            per[(j["L"], j["A"])][j["seed"]] = j
    out = {}
    for key, seeds in per.items():
        best = min(seeds, key=lambda s: seeds[s]["rungs"][-1]["E_lo"])
        j = seeds[best]
        rungs = j["rungs"]
        last = rungs[-1]
        deltas = {s: seeds[s]["rungs"][-1]["delta_shared_per_site"] for s in seeds}
        out[key] = dict(
            L=j["L"], A=j["A"], sites=j["sites"], best_seed=best, n_seeds=len(seeds),
            cores=[r["core"] for r in rungs],
            ds=[r["delta_shared_per_site"] for r in rungs],
            dn=[r["delta_nested_per_site"] for r in rungs],
            dind=[r.get("delta_independent_per_site") for r in rungs],
            bw=[r["lo_boundary_weight"] for r in rungs],
            max_occ=last["lo_max_occ"], n_boundary=last["lo_n_boundary_dets"],
            top_core=last["core"], ds_top=last["delta_shared_per_site"],
            seed_spread=max(deltas.values()) - min(deltas.values()),
            reached=last["lo_max_occ"] >= (j["N_f_lo"] - 1),
            N_f_lo=j["N_f_lo"], N_f_hi=j["N_f_hi"])
    return out


def growth_projection(rec, to_core=BASELINE_CORE):
    """Power-law fit of delta_shared vs core over the NONZERO rungs, projected to `to_core`.
    Returns (exponent, projected value) or (None, None) with <3 nonzero points."""
    pts = [(c, v) for c, v in zip(rec["cores"], rec["ds"]) if v > 0]
    if len(pts) < 3:
        return None, None
    c = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    b = float(np.polyfit(np.log(c), np.log(v), 1)[0])
    return b, float(v[-1] * (to_core / c[-1]) ** b)


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.patch.set_facecolor(SURFACE)

    # (a) delta_shared vs core, with the projection to the baseline's own depth
    for (L, A), r in sorted(recs.items()):
        if not r["reached"]:
            continue
        pts = [(c, v) for c, v in zip(r["cores"], r["ds"]) if v > 0]
        if not pts:
            continue
        axA.loglog([p[0] for p in pts], [p[1] for p in pts], "-o", color=L_COLOR.get(L, INK2),
                   lw=1.5, ms=4, mec=SURFACE, mew=0.7, alpha=0.85,
                   label=f"$L$={L}, $A$={A}")
        b, proj = growth_projection(r)
        if proj is not None and L == 2 and A in (1, 32):
            axA.loglog([pts[-1][0], BASELINE_CORE], [pts[-1][1], proj], ":",
                       color=L_COLOR.get(L, INK2), lw=1.3)
            axA.plot([BASELINE_CORE], [proj], "*", color=CRIT, ms=13, zorder=6)
    axA.axhline(TARGET, color=CRIT, ls="--", lw=1.6, zorder=2)
    axA.axvline(BASELINE_CORE, color=MUTED, ls=":", lw=1.2, zorder=1)
    axA.set_xlim(1.2e4, 2.4e6)
    axA.set_ylim(6e-6, 3.5e-3)
    # annotations in AXES fractions so they cannot fall outside the view
    axA.annotate("0.001 MeV/site reference  (1 MeV / 1000 sites)", xy=(0.03, 0.845),
                 xycoords="axes fraction", color=CRIT, fontsize=8.0, weight="bold")
    axA.annotate("core the $L$=2\nbaseline reaches →", xy=(0.955, 0.97),
                 xycoords="axes fraction", color=INK2, fontsize=7.8, ha="right", va="top")
    axA.plot([], [], "*", color=CRIT, ms=13, label="projected there (power-law fit)")
    axA.set_xlabel("selected-CI core (# determinants)", color=INK2, fontsize=9.5)
    axA.set_ylabel("$\\Delta_\\mathrm{shared}$ / site   (MeV)", color=INK2, fontsize=9.5)
    axA.set_title("a  Fixed-basis shift vs core — not plateaued; the $L$=2 trend\n"
                  "    crosses the reference at the baseline's own depth",
                  color=INK, fontsize=10.2, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=6.8, loc="lower left", labelcolor=INK2, ncol=2,
               handlelength=1.4, columnspacing=0.9, borderaxespad=0.3)
    _style(axA)

    # (b) A-dependence at a matched core + the L=4 non-measurement
    match = 256000
    xs, ys, cs, labels = [], [], [], []
    for (L, A), r in sorted(recs.items()):
        if match in r["cores"]:
            i = r["cores"].index(match)
            xs.append(A); ys.append(r["ds"][i]); cs.append(L_COLOR.get(L, INK2))
            labels.append(L)
    for L in sorted(set(labels)):
        sel = [(x, y) for x, y, l in zip(xs, ys, labels) if l == L]
        if sel:
            axB.semilogx([p[0] for p in sel], [p[1] for p in sel], "o", ms=9,
                         color=L_COLOR.get(L, INK2), mec=SURFACE, mew=1.0,
                         label=f"$L$={L}  (core {match:,})")
    nm = [(r["A"], r["L"]) for r in recs.values() if not r["reached"]]
    NM_Y = 7e-6                     # a dedicated "no measurement" row, clear of the data
    for A, L in nm:
        axB.plot([A], [NM_Y], "x", color=CRIT, ms=12, mew=2.4, zorder=6)
    if nm:
        axB.plot([], [], "x", color=CRIT, ms=11, mew=2.2,
                 label="$L$=4: NO measurement — core never reached the\nboundary (max occ. 6 < 7); this is not a zero shift")
        axB.annotate("no measurement", xy=(0.30, 0.055), xycoords="axes fraction",
                     color=CRIT, fontsize=7.8, ha="center", style="italic")
    axB.axhline(TARGET, color=CRIT, ls="--", lw=1.6)
    axB.annotate("0.001 MeV/site reference", xy=(0.03, 0.93), xycoords="axes fraction",
                 color=CRIT, fontsize=8.0, weight="bold")
    axB.set_yscale("log")
    axB.set_ylim(4e-6, 2.2e-3)
    axB.set_xlim(0.7, 55)
    axB.set_xlabel("nucleon number $A$  (dilute → fully filled)", color=INK2, fontsize=9.5)
    axB.set_ylabel("$\\Delta_\\mathrm{shared}$ / site   (MeV)", color=INK2, fontsize=9.5)
    axB.set_title("b  Filling dependence at matched core — no monotonic trend",
                  color=INK, fontsize=10.2, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=7.2, loc="center left", labelcolor=INK2,
               handlelength=1.4, borderaxespad=0.3)
    _style(axB)

    fig.suptitle("Boson cutoff $n_b$=3 → 4 at FIXED BASIS: no selection noise, and still growing "
                 "with the calculation", fontsize=11.2, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def make_table(recs, out_path):
    md = ["# Fixed-basis boson-cutoff shift $\\Delta_\\mathrm{shared}$ ($n_b$: 3 → 4)\n",
          "_$\\Delta_\\mathrm{shared} = E_4(\\mathrm{core}_3) - E_3(\\mathrm{core}_3)$: both cutoffs "
          "on the **identical** determinant set, so **no selection difference enters**. Nonzero only "
          "because the two operators genuinely differ on a core that touches the low cutoff's top "
          "level — the Hamiltonian's `a a†` terms are truncated to zero at occupation $N_f-1$, so "
          "$H_3$ scores such a determinant ~357.6 MeV BELOW $H_4$. The low cutoff **mis-scores** "
          "boundary states, in the direction that flatters it._\n",
          "_Reference line 0.001 MeV/site = 1 MeV / 1000 sites: a conservative uniform per-site "
          "slice of the L=10 GSEE target, **not** a derived budget. Values are from the "
          "best-bound seed — a worse search understates the shift._\n",
          "| L | A | seeds | best seed | deepest core | boundary reached? | "
          "$\\Delta_\\mathrm{shared}$/site | % of reference | growth | projected @1,024,000 |",
          "|--:|--:|--:|--:|--:|:--|--:|--:|--:|--:|"]
    for (L, A), r in sorted(recs.items()):
        b, proj = growth_projection(r)
        reach = "yes" if r["reached"] else f"**NO** (max occ {r['max_occ']})"
        ds = f"{r['ds_top']:.2e}" if r["reached"] else "— (no measurement)"
        pct = f"{r['ds_top'] / TARGET * 100:.1f}%" if r["reached"] else "—"
        gr = f"core^{b:.2f}" if b is not None else "—"
        pr = (f"**{proj:.2e}** ({proj / TARGET * 100:.0f}%)" if proj is not None else "—")
        md.append(f"| {L} | {A} | {r['n_seeds']} | {r['best_seed']} | {r['top_core']:,} | {reach} | "
                  f"{ds} | {pct} | {gr} | {pr} |")
    md += ["", "### What this does and does not settle", "",
           "- **Settled:** the shift is real and cleanly resolved. It is *exactly* zero on every rung "
           "with no boundary population and turns on the moment the core reaches occupation "
           "$N_f-1$, with zero selection noise. On the same shards the legacy "
           "independently-selected arm swings to 2.0e-3 MeV/site with oscillating sign — **~2.8× "
           "larger than the signal it was meant to measure**, which is why that arm could not "
           "resolve this.",
           "- **Not settled:** the measured values **have not plateaued** over the observed core "
           "range and therefore cannot be treated as converged; the fitted L=2 trends project "
           "*above* the reference at the core the classical baseline actually reaches. (A "
           "difference of two Rayleigh quotients is not proved monotone as the shared subspace "
           "grows, so these are not mathematical lower bounds on their converged values — the "
           "observed sequence rises, which is a weaker and sufficient statement.)",
           "- **Not settled this way in practice:** the core needed to reach the boundary grows "
           "with volume (16k–64k at L=2, 64k–256k at L=3, **not reached by 128k at L=4**). At "
           "unlimited depth the method would in principle reach the boundary at any L; the point "
           "is that the observed growth of the required core makes certification at L=10 "
           "**computationally inaccessible with the present method and available resources**. "
           "Note also that this observable does not *bound* the true cutoff effect — both sides "
           "are variational energies, and their difference is not a bound.",
           "- **Filling:** no monotonic trend across A=1…32 at matched core (values scatter over 5×, "
           "with the fully-filled point among the largest).",
           ""]
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load(args.data)
    assert recs, "no nested_*.json found"
    make_figure(recs, f"{args.out_dir}/nb_nested_shared")
    make_table(recs, f"{args.out_dir}/nb_nested_shared_table.md")
    json.dump({f"L{k[0]}_A{k[1]}": v for k, v in sorted(recs.items())},
              open(f"{args.out_dir}/nb_nested_shared.json", "w"), indent=2)
    reached = sum(1 for r in recs.values() if r["reached"])
    print(f"[done] {len(recs)} (L,A) points, {reached} reached the cutoff boundary; "
          f"max delta_shared/site = {max(r['ds_top'] for r in recs.values()):.2e}")


if __name__ == "__main__":
    main()
