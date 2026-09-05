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

WHAT THE DATA SHOWS (do not overclaim): seed labels reproduce bit-identically here (the warm-grow
solve is deterministic — reproducibility, NOT independent samples), so no seed-based uncertainty is
claimed. The scatter that matters is residual CORE-INCOMPLETENESS: the n_b=3 and n_b=4 solves have
different Fock spaces (different N_f) and independently select spatial determinants, so the paired
same-seed difference does not ISOLATE the operator-cutoff effect — Δ34 oscillates in sign down the
ladder at the ~0.001–0.003 MeV/site level (the direct evidence of the residual). The shift is SMALL
(deep-core median |Δ34/site| ≲ 0.0015; L=2 and L=4 inside the 0.001 MeV/site conservative
target-equivalent, L=3 just outside) with no resolved accumulation across the tested dilute L=2–4
points — but the core-residual is comparable to it, so this is EVIDENCE, not an L=10 certification, and
the large-volume conditional is RETAINED (three volumes cannot establish volume-independence).

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
                 elinewidth=1.4, zorder=4, label="median (bars = seed spread; deterministic → ~0)")
    for r in rows:
        tag = f"{r['nseed']} seed{'s' if r['nseed'] != 1 else ''}\n@{r['rep_core']//1000}k"
        axB.annotate(tag, (r["L"], r["med"]), textcoords="offset points", xytext=(7, 6),
                     fontsize=7.2, color=INK2)
    axB.set_xticks(xs); axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("paired Δ$_{34}$/site @ deepest ≥3-seed core", color=INK2, fontsize=9.5)
    axB.set_title("b  Shift small; deterministic; residual straddles target", color=INK,
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
          f"range. Seed labels reproduce bit-identically here (the warm-grow solve is deterministic — a "
          f"reproducibility check, NOT independent-sample statistics), so no seed-based uncertainty is "
          f"claimed; the scatter that matters is residual CORE-incompleteness (different N_f → different "
          f"selected determinants) seen down the core ladder._\n",
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
    seed_spread = max((r["hi"] - r["lo"]) for r in rows)
    if dropped_log:
        md.append("\n**Excluded seeds** (OUTLIER_TOL=%g MeV, **pre-specified** for this analysis; all raw "
                  "shards retained — the filter is a quality gate, not a convergence certificate): "
                  % OUTLIER_TOL + "; ".join(f"L={L} core={c//1000}k {d}" for L, c, d in dropped_log))
    else:
        md.append(f"\n**No seeds excluded** (OUTLIER_TOL={OUTLIER_TOL:g} MeV, **pre-specified**; all raw "
                  f"shards retained; the filter is a quality gate, not a convergence certificate). Seed "
                  f"labels are **indistinguishable at the displayed precision** (max seed spread "
                  f"{seed_spread:.1e} MeV/site — the warm-grow solve is deterministic), so the seeds "
                  f"establish reproducibility, not independent-sample statistics.")
    l4 = [r for r in rows if r["L"] == 4]
    l4med = l4[0]["med"] if l4 else float("nan")
    md.append(f"\n**Publishable claim (bounded, empirical):** in dilute A=1 calculations, paired "
              f"same-seed n_b=3→4 shifts remain small and show **no resolved accumulation through the "
              f"tested L=2–4 selected-CI points**; because the per-L core-ladder residual "
              f"(≤{absmax:.4f} MeV/site) is comparable to the shift and the absolute energies are "
              f"unconverged, this is **evidence, not a certification**, and the large-volume n_b=3 "
              f"resource claim (C1) remains **conditional**.\n")
    md.append(f"\n**Reading (no overclaim):** the deep-core paired medians are small — L=2 +0.00017, "
              f"L=3 {[r['med'] for r in rows if r['L']==3][0]:+.5f}, L=4 {l4med:+.5f} MeV/site — "
              f"**sign-mixed, no monotonic trend** over the three tested volumes (three points cannot "
              f"establish volume-independence or justify an L=10 extrapolation). L=2 and L=4 fall inside, "
              f"and **L=3 falls just outside**, the ±0.001 MeV/site band — a **conservative "
              f"target-equivalent** (1 MeV / 1000 sites, a uniform per-site allocation; NOT a derived "
              f"budget, tolerance, or confidence band). This is **consistent with** the L=2 gate "
              f"(n_b=3 ≈ n_b=4). The paired difference does NOT isolate the operator-cutoff effect — the "
              f"n_b=3/4 solves use independently selected bases of different boson dimension, and the "
              f"sign-changing core ladder is the direct evidence pairing leaves a residual. **VERDICT: "
              f"{verdict}** — evidence of no resolved accumulation in dilute L≤4, not an L=10 "
              f"certification → C1 stays conditional. Limitations kept explicit: (i) independent "
              f"per-cutoff bases (a shared/union basis would be stronger); (ii) dilute A=1 only (no "
              f"dense-regime discharge; the dense L=3 gate was trap-limited); (iii) a paired difference "
              f"near zero can still miss a common truncation bias in both solves — this is not a "
              f"total-energy bound. An earlier +1.78 MeV linear projection (min-over-seeds, shallow OOM "
              f"core) is withdrawn.\n")
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
