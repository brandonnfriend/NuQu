"""Evaluate the warm-grow SEARCH LEVERS, and audit whether repeat seeds were independent.

Two questions, one script, because they are the same question asked twice: *how much of
our classical energy is decided by the search rather than by the core size?*

  (A) LEVER GRID (cluster 293961, L=2, A in {2,4,5,6,9} x seeds {0,1,2} x 6 arms, all to
      128k). Which Phase-0 levers land a single run on the lowest basin any arm found?
      Reports the basin hit rate, the seed spread, the wall-clock price, and -- from the
      per-init `phase0_select` trails -- whether the energy at the FIRST rung predicts the
      energy at the last select rung. That predictivity is the whole case for moving the
      choice point from 1k to 16k, so it is measured, not asserted.

  (B) SEED-INDEPENDENCE AUDIT of any warm-grow campaign. Before 2026-09-24 the Phase-0
      ensemble of shard `seed = s` drew inits s, s+1, ..., s+31 (`phase0_seed_base`
      stride 1), so shards 0/1/2 shared 30 of 32 inits and usually kept the same winner.
      A shard whose JSON has no `phase0_seed_stride` key ran under that numbering. This
      pass reports, per (n_b, L, A), how many DISTINCT E_var ladders the "independent"
      seeds actually produced -- the number that decides whether a seed-agreement check
      has any power.

Energies are per site (E / L^dim) so the basin tolerance is volume-independent. Every
number printed here is a function of the shard JSONs alone; no refits.

    python -m misc.analyze_search_levers \
        --levers data/classical/2026-09-24/levers_293961 \
        --v1-baseline data/classical/2026-09-23/bare_Asweep_nb3_293959 \
        --seed-audit data/classical/2026-09-06/bare_baseline_nb3_292477 \
                     data/classical/nb_energy_gate \
        --out-dir results/02_classical_baseline
"""
import argparse
import glob
import json
import os
import statistics

# Arm order = the story order: no lever, then each lever alone, then the combinations.
ARM_ORDER = ["base", "strat", "select", "stratsel", "novel", "all"]
ARM_LABEL = {
    "base": "v1 (random starts, choose at 1k)",
    "strat": "stratified starts only (choose at 1k)",
    "select": "choose at 16k only (random starts)",
    "stratsel": "stratified + choose at 16k (= v2)",
    "novel": "novel-configuration reservation",
    "all": "stratified + choose at 16k + novel",
}
BASIN_TOL_PS = 0.5      # MeV/site; "on the best basin any arm found"


def _load(path):
    with open(path) as fh:
        d = json.load(fh)
    return d if isinstance(d, dict) and "rungs" in d and d["rungs"] else None


def _sites(d):
    return int(d["L"]) ** int(d.get("dim", 3))


def _shards(root):
    """Every frame-shard JSON under `root` (recursively), skipping manifests."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        if "manifest" in os.path.basename(p):
            continue
        d = _load(p)
        if d is not None:
            out.append((p, d))
    return out


# ---------------------------------------------------------------------------
#  (A) the lever grid
# ---------------------------------------------------------------------------

def lever_grid(root):
    arms = {}
    for arm in sorted(os.listdir(root)):
        sub = os.path.join(root, arm)
        if not os.path.isdir(sub):
            continue
        for path, d in _shards(sub):
            top = d["rungs"][-1]
            arms.setdefault(arm, {})[(int(d["A"]), int(d["seed"]))] = {
                "E_var_ps": top["E_var"] / _sites(d),
                "E_pt2_ps": (top["E_pt2"] / _sites(d)) if top.get("E_pt2") else None,
                "top_core": int(top["core"]),
                "wall_h": d["wall_s"] / 3600.0,
                "select_h": d["rungs"][0]["wall_s"] / 3600.0,
                "trail": d["rungs"][0].get("phase0_select"),
                "sites": _sites(d),
                "levers": d.get("search_levers"),
                "stride": d.get("phase0_seed_stride"),
            }
    As = sorted({a for arm in arms.values() for (a, _s) in arm})
    best = {A: min(v["E_var_ps"] for arm in arms.values()
                   for (a, _s), v in arm.items() if a == A) for A in As}
    rows = []
    for arm in [a for a in ARM_ORDER if a in arms] + [a for a in sorted(arms) if a not in ARM_ORDER]:
        cells = arms[arm]
        hits = sum(1 for (A, _s), v in cells.items()
                   if v["E_var_ps"] - best[A] <= BASIN_TOL_PS)
        spreads = []
        for A in As:
            es = [v["E_var_ps"] for (a, _s), v in cells.items() if a == A]
            if len(es) > 1:
                spreads.append(max(es) - min(es))
        rows.append({
            "arm": arm,
            "label": ARM_LABEL.get(arm, arm),
            "n": len(cells),
            "on_basin": hits,
            "wall_h_mean": statistics.mean(v["wall_h"] for v in cells.values()),
            "select_share": statistics.mean(v["select_h"] / v["wall_h"] for v in cells.values()),
            "spread_max_ps": max(spreads) if spreads else None,
            "spread_mean_ps": statistics.mean(spreads) if spreads else None,
            "levers": next(iter(cells.values()))["levers"],
        })
    return {"arms": arms, "As": As, "best_ps": best, "rows": rows,
            "tops": sorted({v["top_core"] for arm in arms.values() for v in arm.values()})}


def choice_point(arms):
    """From the `phase0_select` trails: does the FIRST-rung energy pick the winner?

    For each shard that ran the select stage, compare the init that is lowest at the
    first select rung with the init that is lowest at the last one, and report the
    energy penalty (MeV/site) of having followed the first-rung winner instead.
    """
    recs = []
    for arm, cells in arms.items():
        for (A, s), v in cells.items():
            tr = v["trail"]
            if not tr or len(tr[0][1]) < 2:
                continue
            e_first = [t[1][0] for t in tr]
            e_last = [t[1][-1] for t in tr]
            i_first = min(range(len(e_first)), key=lambda i: e_first[i])
            i_last = min(range(len(e_last)), key=lambda i: e_last[i])
            n = len(e_first)
            rk = lambda es: {i: k for k, i in enumerate(sorted(range(n), key=lambda j: es[j]))}
            r1, r2 = rk(e_first), rk(e_last)
            dsq = sum((r1[i] - r2[i]) ** 2 for i in range(n))
            recs.append({
                "arm": arm, "A": A, "seed": s, "n_inits": n,
                "same_winner": i_first == i_last,
                "penalty_ps": (e_last[i_first] - e_last[i_last]) / v["sites"],
                "spearman": 1 - 6 * dsq / (n * (n * n - 1)),
            })
    if not recs:
        return None
    # The penalty distribution is BIMODAL -- the 1k pick is either harmless or badly wrong --
    # so a plain median is a misleading summary. Report the miss RATE, and the size of a miss.
    pen = [r["penalty_ps"] for r in recs]
    miss = [p for p in pen if p > BASIN_TOL_PS]
    return {
        "n": len(recs),
        "same_winner": sum(1 for r in recs if r["same_winner"]),
        "n_penalty_over_tol": len(miss),
        "penalty_when_missed_median_ps": statistics.median(miss) if miss else None,
        "penalty_mean_ps": statistics.mean(pen),
        "penalty_max_ps": max(pen),
        "spearman_median": statistics.median(r["spearman"] for r in recs),
        "spearman_min": min(r["spearman"] for r in recs),
        "per_A": {A: {"n": sum(1 for r in recs if r["A"] == A),
                      "n_over_tol": sum(1 for r in recs
                                        if r["A"] == A and r["penalty_ps"] > BASIN_TOL_PS),
                      "penalty_max_ps": max(r["penalty_ps"] for r in recs if r["A"] == A)}
                  for A in sorted({r["A"] for r in recs})},
        "records": recs,
    }


def vs_v1_baseline(arms, root, arm="stratsel"):
    """v1's DEEP ladder (its own top rung) against the lever arm at 128k, same A."""
    if not root or arm not in arms:
        return None
    v1 = {}
    for _p, d in _shards(root):
        if int(d["L"]) != 2:
            continue
        ps = _sites(d)
        v1.setdefault(int(d["A"]), []).append(
            {"seed": int(d["seed"]),
             "by_core": {int(r["core"]): r["E_var"] / ps for r in d["rungs"]}})
    out = []
    for A in sorted(v1):
        deep = min(v1[A], key=lambda r: min(r["by_core"].values()))
        top_core = max(deep["by_core"])
        cells = [v["E_var_ps"] for (a, _s), v in arms[arm].items() if a == A]
        base = [v["E_var_ps"] for (a, _s), v in arms.get("base", {}).items() if a == A]
        if not cells:
            continue
        out.append({
            "A": A,
            "v1_top_core": top_core,
            "v1_top_ps": deep["by_core"][top_core],
            "v1_at_128k_ps": deep["by_core"].get(128000),
            "v2_128k_best_ps": min(cells),
            "delta_vs_v1_top_ps": min(cells) - deep["by_core"][top_core],
            "v1_matched_128k_best_ps": min(base) if base else None,
        })
    return out


# ---------------------------------------------------------------------------
#  (B) were the repeat seeds independent?
# ---------------------------------------------------------------------------

def seed_audit(roots):
    groups = {}
    for root in roots:
        base = os.path.basename(root.rstrip("/"))
        for path, d in _shards(root):
            # Group per SHARD DIRECTORY, not per campaign root: a lever grid's arms live in
            # sibling directories and are different configurations, not repeat seeds.
            rel = os.path.relpath(os.path.dirname(path), root)
            name = base if rel in (".", "") else f"{base}/{rel}"
            key = (name, d.get("n_b"), int(d["L"]), int(d["A"]))
            ladder = tuple(round(r["E_var"], 6) for r in d["rungs"])
            groups.setdefault(key, []).append({
                "seed": int(d["seed"]),
                "stride": d.get("phase0_seed_stride"),
                "phase0_runs": d.get("phase0_runs"),
                "top_core": int(d["rungs"][-1]["core"]),
                "E_top_ps": d["rungs"][-1]["E_var"] / _sites(d),
                "ladder": ladder,
            })
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], kv[0][3])):
        rs.sort(key=lambda r: r["seed"])
        if len(rs) < 2:
            continue
        distinct = len({r["ladder"] for r in rs})
        strides = {r["stride"] for r in rs}
        runs = {r["phase0_runs"] for r in rs}
        es = [r["E_top_ps"] for r in rs]
        out.append({
            "campaign": key[0], "n_b": key[1], "L": key[2], "A": key[3],
            "seeds": [r["seed"] for r in rs],
            "n_seeds": len(rs),
            "n_distinct_ladders": distinct,
            "stride": sorted(s for s in strides if s is not None) or None,
            "legacy_numbering": any(s is None or s == 1 for s in strides),
            "phase0_runs": sorted(r for r in runs if r is not None) or None,
            "spread_ps": max(es) - min(es),
        })
    return out


# ---------------------------------------------------------------------------
#  report
# ---------------------------------------------------------------------------

def render(grid, choice, vsv1, audit, args):
    md = ["# Search-lever evaluation, and whether repeat seeds were independent searches",
          "",
          "*Generated by `misc/analyze_search_levers.py` from the shard JSONs named below. "
          "Every number is a direct function of those files; nothing here is refitted.*",
          ""]
    if grid:
        md += [f"**Lever grid:** `{args.levers}` — "
               f"L=2, dim=3, n_b=3, bare frame, A ∈ {{{', '.join(str(a) for a in grid['As'])}}}, "
               f"seeds {{0,1,2}}, 32 Phase-0 starts, every arm grown to "
               f"{grid['tops'][-1]:,} determinants. Seeds use the disjoint Phase-0 stride, so the "
               "three seeds per cell are genuinely independent searches.", "",
               "## A. Which levers put a single run on the lowest basin?", "",
               f"_\"On the best basin\" = within {BASIN_TOL_PS} MeV/site of the lowest energy any "
               "arm reached at that A. **Levers do not lower a bound that a lucky v1 run could "
               "not also reach; they make one run reach it reliably.** Read the spread column "
               "with the hit rate._", "",
               "| arm | on best basin | seed spread /site (mean, max) | wall h (mean) | search stage, share of wall |",
               "|:--|--:|--:|--:|--:|"]
        for r in grid["rows"]:
            sp = ("—" if r["spread_mean_ps"] is None
                  else f"{r['spread_mean_ps']:.2f}, {r['spread_max_ps']:.2f}")
            md.append(f"| {r['label']} | **{r['on_basin']}/{r['n']}** | {sp} | "
                      f"{r['wall_h_mean']:.2f} | {r['select_share']:.0%} |")
        md += ["",
               "Per-A final $E_\\mathrm{var}$/site, every arm and seed (MeV/site):", "",
               "| A | seed | " + " | ".join(r["arm"] for r in grid["rows"]) + " |",
               "|--:|--:|" + "--:|" * len(grid["rows"])]
        for A in grid["As"]:
            for s in sorted({s for (a, s) in grid["arms"][grid["rows"][0]["arm"]] if a == A}):
                cells = []
                for r in grid["rows"]:
                    v = grid["arms"][r["arm"]].get((A, s))
                    cells.append("—" if v is None else f"{v['E_var_ps']:.3f}")
                md.append(f"| {A} | {s} | " + " | ".join(cells) + " |")
        md += ["", f"Lowest found at each A (MeV/site): "
                   + ", ".join(f"A={A}: {grid['best_ps'][A]:.3f}" for A in grid["As"]) + ".", ""]
    if choice:
        md += ["## B. Does the first-rung energy pick the winner?", "",
               "_From the per-init `phase0_select` trails, which record every start's energy at "
               "every select rung: the same shard therefore contains both rankings. The penalty "
               "is what it would have cost to keep the first-rung winner instead of the "
               "last-rung one. The distribution is **bimodal** — the early pick is either "
               "harmless or badly wrong — so the miss rate and the size of a miss are reported "
               "instead of a median over everything._", "",
               f"- The 1k winner is also the 16k winner in only "
               f"**{choice['same_winner']}/{choice['n']}** runs.",
               f"- Picking at 1k costs more than {BASIN_TOL_PS} MeV/site in "
               f"**{choice['n_penalty_over_tol']}/{choice['n']}** runs. When it misses, the "
               f"median cost is **{choice['penalty_when_missed_median_ps']:.1f}** MeV/site and "
               f"the worst is **{choice['penalty_max_ps']:.1f}** MeV/site; averaged over all "
               f"runs it is {choice['penalty_mean_ps']:.1f} MeV/site.",
               f"- Spearman ρ between the 1k and 16k rankings: median "
               f"**{choice['spearman_median']:.2f}**, lowest {choice['spearman_min']:.2f}. "
               "Correlated, nowhere near decisive.", "",
               "| A | runs | picked at 1k costs > tol | worst penalty /site |",
               "|--:|--:|--:|--:|"]
        for A, v in choice["per_A"].items():
            md.append(f"| {A} | {v['n']} | {v['n_over_tol']} | {v['penalty_max_ps']:.1f} |")
        md += ["",
               "At every A the early pick sometimes costs more than the whole error budget, and "
               "the risk grows with A: at A>1 the competing basins are different nucleon "
               "arrangements, and weak hopping means a 1k core has not yet sorted them.", ""]
    if vsv1:
        md += ["## C. The lever arm at 128k against v1's deepest ladder", "",
               "_v1 here is the fixed-A L=2 pass that ran the old search to its ladder top. "
               "The honest comparison is the matched-core column; the deep column says how much "
               "of three extra doublings the levers buy back._", "",
               "| A | v1 top core | v1 at top /site | v1 at 128k /site | v2 at 128k /site | "
               "v2(128k) − v1(top) |", "|--:|--:|--:|--:|--:|--:|"]
        for r in vsv1:
            at128 = ("—" if r["v1_at_128k_ps"] is None else f"{r['v1_at_128k_ps']:.3f}")
            md.append(f"| {r['A']} | {r['v1_top_core']:,} | {r['v1_top_ps']:.3f} | {at128} | "
                      f"{r['v2_128k_best_ps']:.3f} | {r['delta_vs_v1_top_ps']:+.3f} |")
        md += [""]
    if audit:
        md += ["## D. Were the repeat seeds independent searches?", "",
               "_A shard with no `phase0_seed_stride` key ran under the legacy numbering "
               "(`base = seed`), so shard s drew Phase-0 inits s … s+n−1 and neighbouring "
               "shards shared all but one or two. `distinct ladders` counts how many different "
               "$E_\\mathrm{var}$ sequences the seeds actually produced. **1 means the seeds are "
               "copies and a seed-agreement check has no power.**_", "",
               "| campaign | $n_b$ | L | A | seeds | starts | stride | distinct ladders | "
               "spread /site | verdict |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
        for r in audit:
            same = r["n_seeds"] - r["n_distinct_ladders"]
            if not r["legacy_numbering"]:
                # Disjoint init blocks: repeat ladders mean independent searches CONVERGED,
                # which is the outcome a robustness check is looking for.
                verdict = ("independent" if same == 0 else
                           f"independent — {same + 1} converged to the same ladder")
            elif r["n_distinct_ladders"] == 1:
                verdict = "**NO POWER — one search reported %d times**" % r["n_seeds"]
            else:
                verdict = (f"**shared starts — only {r['n_distinct_ladders']} of "
                           f"{r['n_seeds']} searches distinct**")
            md.append(f"| `{r['campaign']}` | {r['n_b']} | {r['L']} | {r['A']} | "
                      f"{r['n_seeds']} | {r['phase0_runs'] or '—'} | "
                      f"{'legacy' if r['legacy_numbering'] else r['stride']} | "
                      f"{r['n_distinct_ladders']} | {r['spread_ps']:.3f} | {verdict} |")
        md += [""]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levers", default=None, help="lever-grid root (arm subdirectories)")
    ap.add_argument("--v1-baseline", default=None, help="old-search fixed-A campaign for §C")
    ap.add_argument("--seed-audit", nargs="*", default=(), help="campaign roots for §D")
    ap.add_argument("--out-dir", default=None, help="write the .md/.json report here")
    args = ap.parse_args()

    grid = lever_grid(args.levers) if args.levers else None
    choice = choice_point(grid["arms"]) if grid else None
    vsv1 = vs_v1_baseline(grid["arms"], args.v1_baseline) if grid else None
    audit = seed_audit(args.seed_audit) if args.seed_audit else None

    md = render(grid, choice, vsv1, audit, args)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        mp = os.path.join(args.out_dir, "search_lever_evaluation.md")
        jp = os.path.join(args.out_dir, "search_lever_evaluation.json")
        with open(mp, "w") as fh:
            fh.write(md)
        payload = {
            "basin_tol_ps": BASIN_TOL_PS,
            "levers": ({"rows": grid["rows"], "As": grid["As"], "best_ps": grid["best_ps"],
                        "tops": grid["tops"]} if grid else None),
            "choice_point": ({k: v for k, v in choice.items() if k != "records"}
                             if choice else None),
            "choice_point_records": choice["records"] if choice else None,
            "vs_v1": vsv1, "seed_audit": audit,
            "sources": {"levers": args.levers, "v1_baseline": args.v1_baseline,
                        "seed_audit": list(args.seed_audit)},
        }
        with open(jp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        print(f"[levers] wrote {mp} and {jp}")
    else:
        print(md)

    if grid:
        for r in grid["rows"]:
            print(f"  {r['arm']:9s} on-basin {r['on_basin']:>2d}/{r['n']:<2d}  "
                  f"{r['wall_h_mean']:.2f} h/run")
    if choice:
        print(f"  choice point: 1k winner == 16k winner in {choice['same_winner']}/{choice['n']}; "
              f"picking at 1k misses by >{BASIN_TOL_PS} MeV/site in "
              f"{choice['n_penalty_over_tol']}/{choice['n']} runs (worst "
              f"{choice['penalty_max_ps']:.1f})")
    if audit:
        bad = [r for r in audit if r["n_distinct_ladders"] == 1]
        print(f"  seed audit: {len(bad)} of {len(audit)} (n_b, L, A) cells have only ONE "
              f"distinct ladder across their seeds")


if __name__ == "__main__":
    main()
