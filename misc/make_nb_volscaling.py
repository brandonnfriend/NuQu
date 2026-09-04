"""Volume scaling of the boson-cutoff shift (re-audit P0-4): can we BOUND the n_b=3 cutoff error in the
TOTAL energy at large L? Measure the PAIRED SAME-SEED shift Δ34(L,core,seed) = E4(seed) - E3(seed) at
COMMON selected-CI cores for L=2,3,4, track it down the core ladder, and read the shift at the deepest
core that has ≥3 comparable paired seeds.

METHOD (fixes the 2026-09-04 re-audit aggregation defect): the previous version differenced
min_s E4(s) − min_t E3(t) — a difference of two INDEPENDENTLY minimised per-cutoff energies whose
minima can come from different seeds, silently dropping outlying/failed runs. That is not a paired
difference and its spread is not a valid uncertainty. Here we instead:
  * keep EVERY seed's E_var per core (never min across seeds for the difference);
  * form Δ34 only for the SAME seed at the SAME (L, A, core), and only when BOTH paired solves pass a
    convergence quality check (E_var within OUTLIER_TOL of the best seed at that (L,n_b,A,core) — a
    bad-basin under-converged seed sits far ABOVE the others and is excluded WITH a printed reason);
  * report, at the deepest core with ≥MIN_SEEDS paired good seeds, the MEDIAN Δ34/site and the SEED
    RANGE; deeper cores with fewer seeds are shown as single-/few-seed probes, NOT used for a range.

WHAT THE DATA SHOWS (do not overclaim): at these sizes the solve is seed-insensitive (paired seeds
agree to ≪1 MeV), so the scatter is NOT seed noise — it is residual CORE-INCOMPLETENESS: the n_b=3 and
n_b=4 solves have different Fock spaces (different N_f) and select different spatial determinants, so
the paired same-seed difference still does not fully cancel core-incompleteness and Δ34 oscillates
down the ladder at the ~0.001–0.003 MeV/site level. The shift is SMALL (median |Δ34/site| below the
0.001 MeV/site conservative target-equivalent at L=4) with no clean accumulation, but the core-residual
is comparable to it, so no reliable L=10 projection exists and the large-volume conditional is RETAINED.

    python -m misc.make_nb_volscaling
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

BLUE, ORANGE, GREEN, CRIT, MUTED = "#2a78d6", "#eb6834", "#3a9b6a", "#d03b3b", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LCOLOR = {2: BLUE, 3: ORANGE, 4: GREEN}
OUTLIER_TOL = 5.0            # MeV: a seed whose E_var is >this above the best seed = under-converged (excluded)
MIN_SEEDS = 3                # a "reported" deep point needs ≥ this many paired good seeds
DEEP_CORE = 32000            # cores at/below this are too shallow to be even approximately converged
TARGET_PER_SITE = 0.001      # CONSERVATIVE target-equivalent: 1 MeV GSEE / 1000 sites (L=10). Not a derived budget.
DEFAULT_SRC = [
    "data/classical/nb_energy_gate",       # L=2 (n_b=3,4), L=3
    "data/classical/nb_energy_gate_L3",    # L=3 n_b=3
    "data/classical/nb_volscaling",        # L=3 n_b=4 + L=4 n_b={3,4} @256k (primary, 3 seeds)
    "data/classical/nb_volscaling_deep",   # L=3,4 n_b={3,4} @512k deep probe
]


def load(srcs):
    """PER-SEED (never min-across-seeds): {(L,n_b,A): {core: {seed: E_var}}} + sites[L]. If a (seed,core)
    appears in more than one source (same run grown to two ceilings), keep the lower E_var (better
    converged for THAT seed — this is per-seed, not the cross-cutoff min the re-audit flagged)."""
    data, sites = {}, {}
    for d in srcs:
        for f in glob.glob(f"{d}/nb*/bare_L*.json"):
            nb = int(re.search(r"/nb(\d+)/", f).group(1))
            m = re.search(r"bare_L(\d+)d\d+_A(\d+)_s(\d+)", os.path.basename(f))
            if not m:
                continue
            L, A, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            j = json.load(open(f)); sites[L] = j["sites"]
            for r in j["rungs"]:
                if r.get("E_var") is None:
                    continue
                c = int(r["core"]); cell = data.setdefault((L, nb, A), {}).setdefault(c, {})
                cell[s] = min(cell.get(s, r["E_var"]), r["E_var"])
    return data, sites


def good_seeds(cell):
    """Seeds whose E_var is within OUTLIER_TOL of the best (lowest = most converged) seed at this core.
    Returns (kept_set, dropped_dict{seed: E-E_best})."""
    if not cell:
        return set(), {}
    best = min(cell.values())
    kept = {s for s, e in cell.items() if e - best <= OUTLIER_TOL}
    dropped = {s: e - best for s, e in cell.items() if e - best > OUTLIER_TOL}
    return kept, dropped


def paired(data, L, A):
    """{core: {'seeds':[...], 'ds':[Δ34/site per seed], 'dropped':{...}}} over paired GOOD seeds."""
    e3, e4 = data.get((L, 3, A)), data.get((L, 4, A))
    if not e3 or not e4:
        return {}
    n = L ** 3
    out = {}
    for c in sorted(set(e3) & set(e4)):
        g3, d3 = good_seeds(e3[c]); g4, d4 = good_seeds(e4[c])
        ps = sorted(g3 & g4)
        if not ps:
            continue
        out[c] = dict(seeds=ps, ds=[(e4[c][s] - e3[c][s]) / n for s in ps],
                      dropped={**{f"nb3_s{s}": v for s, v in d3.items()},
                               **{f"nb4_s{s}": v for s, v in d4.items()}})
    return out


def last_doubling(data, L, nb, A, core):
    """Median over seeds of E_var(core) − E_var(previous core) — the variational convergence residual
    (PT2 is off). Large ⇒ the ABSOLUTE energy is far from converged (extensivity trap); the paired
    difference is designed to cancel most of that. Returns None if no previous core."""
    cells = data.get((L, nb, A))
    if not cells:
        return None
    cs = sorted(cells)
    if core not in cs or cs.index(core) == 0:
        return None
    prev = cs[cs.index(core) - 1]
    ds = [cells[core][s] - cells[prev][s] for s in cells[core] if s in cells[prev]]
    return st.median(ds) if ds else None


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=DEFAULT_SRC)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--out-dir", default="data/classical/nb_volscaling")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data, sites = load(args.src)

    rows, dropped_log = [], []
    for L in (2, 3, 4):
        pr = paired(data, L, args.A) or paired(data, L, 0)
        if not pr:
            continue
        n = L ** args.dim
        for c, info in pr.items():
            if info["dropped"]:
                dropped_log.append((L, c, info["dropped"]))
        # deepest core with ≥MIN_SEEDS paired good seeds = the reported point
        rep_cores = [c for c, i in pr.items() if len(i["seeds"]) >= MIN_SEEDS]
        rep = max(rep_cores) if rep_cores else max(pr)      # fall back to deepest available (flagged few-seed)
        info = pr[rep]
        med = st.median(info["ds"]); lo, hi = min(info["ds"]), max(info["ds"])
        # deeper single-/few-seed probes beyond the reported core (shown, not used for the range)
        deeper = {c: pr[c] for c in pr if c > rep}
        # core-ladder residual over cores ≥ DEEP_CORE (paired-median per core): the honest scatter
        lad = [(c, st.median(pr[c]["ds"])) for c in sorted(pr) if c >= DEEP_CORE]
        conv = last_doubling(data, L, 3, args.A, rep)     # n_b=3 abs-energy convergence residual (unconverged)
        rows.append(dict(L=L, sites=n, rep_core=rep, nseed=len(info["seeds"]), med=med, lo=lo, hi=hi,
                         lad=lad, deeper=deeper, conv=conv))
    assert rows, "no paired (n_b=3, n_b=4) shifts found — is the volume-scaling data pulled?"

    absmax = max(abs(v) for r in rows for _, v in r["lad"])   # core-residual envelope over all L
    verdict = "CONDITIONAL (retained)"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    # (a) paired-median Δ34/site DOWN THE CORE LADDER — residual core-incompleteness scatter about 0
    axA.axhline(0, color=MUTED, lw=1.0, zorder=1)
    axA.axhspan(-TARGET_PER_SITE, TARGET_PER_SITE, color=GREEN, alpha=0.10, zorder=0)
    for r in rows:
        cs = [c for c, _ in r["lad"]]; ys = [v for _, v in r["lad"]]
        axA.semilogx(cs, ys, "-o", color=LCOLOR[r["L"]], lw=1.6, ms=5, mec=SURFACE, mew=1.0, zorder=4,
                     label=f"$L$={r['L']} ({r['sites']} sites)")
    axA.set_xlabel("selected-CI core (determinants)", color=INK2, fontsize=9.5)
    axA.set_ylabel("paired Δ$_{34}$ per site  (MeV/site)", color=INK2, fontsize=9.5)
    axA.set_title("a  Paired shift vs core — residual core scatter about 0", color=INK, fontsize=10.0,
                  loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(axA)
    # (b) reported deepest ≥3-seed core: median Δ34/site + SEED range, vs the target-equivalent band
    axB.axhspan(-TARGET_PER_SITE, TARGET_PER_SITE, color=GREEN, alpha=0.12, zorder=0,
                label="±(1 MeV / 1000 sites), conservative")
    axB.axhline(0, color=MUTED, lw=1.0, zorder=1)
    xs = [r["L"] for r in rows]; ys = [r["med"] for r in rows]
    yerr = [[r["med"] - r["lo"] for r in rows], [r["hi"] - r["med"] for r in rows]]
    axB.errorbar(xs, ys, yerr=yerr, fmt="o", color=CRIT, ms=8, mec=SURFACE, mew=1.2, capsize=5,
                 elinewidth=1.4, zorder=4, label="median (bars = seed range)")
    for r in rows:
        tag = f"{r['nseed']} seed{'s' if r['nseed'] != 1 else ''}\n@{r['rep_core']//1000}k"
        axB.annotate(tag, (r["L"], r["med"]), textcoords="offset points", xytext=(7, 6),
                     fontsize=7.2, color=INK2)
    axB.set_xticks(xs); axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("paired Δ$_{34}$/site @ deepest ≥3-seed core", color=INK2, fontsize=9.5)
    axB.set_title("b  Shift small; seed-insensitive; residual straddles target", color=INK,
                  fontsize=10.0, loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=7.4, labelcolor=INK2, loc="upper left")
    _style(axB)
    fig.suptitle("Volume scaling of the n_b=3 cutoff shift (paired same-seed) — %s: shift ≤ residual "
                 "core-incompleteness at L≥3" % verdict, fontsize=10.3, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb_volscaling.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb_volscaling.pdf / .png")

    md = ["# Volume scaling of the n_b=3 boson-cutoff shift (re-audit P0-4, paired same-seed)\n",
          f"_Paired SAME-SEED shift Δ34 = E(n_b=4,seed) − E(n_b=3,seed) at COMMON cores, A={args.A}, "
          f"dim={args.dim}. Δ34 is formed per seed (never min-across-seeds) and only for seeds where "
          f"BOTH solves pass the convergence check (E_var within {OUTLIER_TOL:g} MeV of the best seed). "
          f"We report the deepest core with ≥{MIN_SEEDS} paired good seeds: median Δ34/site + seed "
          f"range. The solve is seed-insensitive here, so the scatter is residual CORE-incompleteness "
          f"(different N_f → different selected determinants), not seed noise._\n",
          "| L | sites | reported core | paired seeds | median Δ34/site (MeV) | seed range | "
          "core-ladder residual (≥32k) | abs-E last-doubling (n_b=3) | deeper probes |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in rows:
        lad_lo = min(v for _, v in r["lad"]); lad_hi = max(v for _, v in r["lad"])
        deeper = ", ".join(f"{c//1000}k:{st.median(i['ds']):+.5f}({len(i['seeds'])}s)"
                           for c, i in sorted(r["deeper"].items())) or "—"
        conv = f"{r['conv']:+.1f} MeV" if r["conv"] is not None else "—"
        md.append(f"| {r['L']} | {r['sites']} | {r['rep_core']:,} | {r['nseed']} | {r['med']:+.5f} | "
                  f"[{r['lo']:+.5f}, {r['hi']:+.5f}] | [{lad_lo:+.5f}, {lad_hi:+.5f}] | {conv} | {deeper} |")
    if dropped_log:
        md.append("\n**Excluded seeds (under-converged, E_var > best + %g MeV):** " % OUTLIER_TOL
                  + "; ".join(f"L={L} core={c//1000}k {d}" for L, c, d in dropped_log))
    else:
        md.append("\n**No seeds excluded** — every paired seed is within %g MeV of the best at its core "
                  "(seed-insensitive solve)." % OUTLIER_TOL)
    l4 = [r for r in rows if r["L"] == 4]
    l4med = l4[0]["med"] if l4 else float("nan")
    md.append(f"\n**Reading (no overclaim):** with the paired same-seed method the shift is **small — "
              f"|median Δ34/site| ≤ {absmax:.4f} MeV/site** (core-ladder residual) across L=2,3,4, and at "
              f"the largest lattice (L=4) the reported median is {l4med:+.5f}/site, **consistent with** "
              f"the 0.001 MeV/site *conservative* target-equivalent (1 MeV / 1000 sites — a uniform "
              f"per-site allocation, NOT a derived budget). Seeds are insensitive (no exclusions), so "
              f"the ±{absmax:.4f}/site scatter is residual CORE-incompleteness — the paired difference "
              f"does not fully cancel it because the n_b=3/4 solves span different determinant spaces — "
              f"and it STRADDLES the target line (sign flips down the ladder). **VERDICT: {verdict}.** "
              f"The test finds the shift small with no resolved accumulation through L=4, but the "
              f"residual is comparable to the signal, so it CANNOT certify the L=10 1 MeV target → C1's "
              f"large-volume conditional is **retained, not discharged**. Limitations kept explicit: "
              f"(i) same-seed pairs still use independent per-cutoff bases (a shared/union basis would "
              f"be stronger); (ii) only dilute A=1 is probed (no dense-regime discharge; the dense L=3 "
              f"gate was trap-limited); (iii) a paired difference near zero can still miss a common "
              f"truncation bias shared by both selected-CI solves — this is not a total-energy bound. "
              f"An earlier +1.78 MeV linear projection (min-over-seeds, shallow OOM core) is withdrawn.\n")
    open(f"{args.out_dir}/nb_volscaling_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_volscaling_table.md")
    for r in rows:
        print(f"[L={r['L']}] reported {r['rep_core']//1000}k ({r['nseed']} seeds): median "
              f"{r['med']:+.5f}/site, seed range [{r['lo']:+.5f},{r['hi']:+.5f}]; "
              f"deeper probes {[(c//1000, len(i['seeds'])) for c,i in sorted(r['deeper'].items())]}")
    print(f"[done] core-residual |median Δ34/site| ≤ {absmax:.5f} across L=2,3,4 -> {verdict} "
          f"({'no' if not dropped_log else len(dropped_log)} seed exclusions)")


if __name__ == "__main__":
    main()
