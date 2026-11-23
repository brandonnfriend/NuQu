"""A/B of the basin-collapse rule (2026-11-22): largest single drop vs an OUTLIER drop.

`extrapolation.split_at_collapse` used to cut every ladder at its largest single E_var drop.
Most ladders have no collapse (the largest drop is 1.04-1.15x the next at L=3-5), so that cut
fell at an arbitrary rung -- on the smooth squeezed L=4 ladders (cluster 294246) at the peak of
the drop curve, leaving 2-3 PT2 rungs and 8 of 11 cells bound-only. The new rule splits only
when the largest drop is >= COLLAPSE_OUTLIER_RATIO (3x) the next; otherwise the ladder is kept
whole and the late fit window drops the early curvature. Only the literature-based conventions
use it (`trimci-coo`, and `nuqu-2026-10` whose per-seed fits are the literature ones); the
legacy `nuqu-2026-09` path is untouched and is checked to stay bit-identical here.

For every (directory, frame, A) under --data-root this pools the seeds under each convention
twice -- old rule (ratio None) and new rule (3.0) -- in one process, in parallel over groups,
and writes:
    <out>/ab_collapse_rule.json   every cell, old and new
    <out>/ab_collapse_rule.md     what moved: counts per (frame, L), the changed cells

    python -m misc.ab_collapse_rule --data-root data/classical --out-dir <dir> [--workers N]
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CONVS = ("nuqu-2026-10", "trimci-coo")
LEGACY = "nuqu-2026-09"


def _groups(root):
    combos = set()
    for f in glob.glob(os.path.join(root, "**", "*_L*d*_*.json"), recursive=True):
        b = os.path.basename(f)
        if not b.startswith(("bare_", "gaussian_")) or "pre_extend" in f:
            continue
        try:
            combos.add((os.path.dirname(f), b.split("_")[0], int(json.load(open(f))["A"])))
        except Exception:
            pass
    return sorted(combos)


def _cell(r):
    v = r["per_seed"].get(r.get("best_seed"), {}) if r.get("best_seed") is not None else {}
    return {"ok": bool(r["ok"]), "E": r.get("E_inf_ps"), "s": r.get("sigma_ps"),
            "bound": r.get("E_var_bound_ps"), "best": r.get("best_seed"),
            "primary": v.get("primary"), "n_post": v.get("n_post"),
            "collapse_core": v.get("collapse_core"),
            "reason": (r.get("reason") or "")[:100]}


def _one(job):
    d, fr, A, root = job
    from classical.trimci import extrapolation as ex
    from misc.aggregate_classical_energies import analyze, load
    out = {}
    try:
        g, m = load([d], convention="wick", A=A, frame=fr)
    except Exception as e:
        return {f"{os.path.relpath(d, root)}|{fr}|A{A}": {"error": str(e)[:200]}}
    if not g:
        return out
    rel = os.path.relpath(d, root)
    for tag, ratio in (("old", None), ("new", 3.0)):
        ex.COLLAPSE_OUTLIER_RATIO = ratio
        for conv in CONVS + (LEGACY,):
            try:
                recs = analyze(g, m, sigma_convention=conv)
            except Exception as e:
                out[f"{rel}|{fr}|A{A}|{conv}|{tag}"] = {"error": str(e)[:200]}
                continue
            for r in recs:
                out[f"{rel}|{fr}|A{A}|L{r['L']}|nb{r['n_b']}|{conv}|{tag}"] = _cell(r)
    ex.COLLAPSE_OUTLIER_RATIO = 3.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(d, fr, A, a.data_root) for d, fr, A in _groups(a.data_root)]
    print(f"[ab] {len(jobs)} (dir, frame, A) groups, {a.workers} workers", flush=True)
    res = {}
    with Pool(a.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_one, jobs), 1):
            res.update(part)
            if i % 20 == 0:
                print(f"[ab] {i}/{len(jobs)} groups", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, "ab_collapse_rule.json"), "w"), indent=1,
              default=str)

    # ---- summary ---------------------------------------------------------------------
    f = lambda x: "—" if x is None else f"{x:.2f}"
    # a cell CHANGES if the reported result does (not just diagnostics like the collapse core)
    sig = lambda c: (c.get("ok"), c.get("E"), c.get("s"), c.get("best"), c.get("primary"),
                     c.get("error"))
    keys = sorted({k.rsplit("|", 1)[0] for k in res if k.endswith(("|old", "|new"))})
    legacy_bad = [k for k in keys if k.split("|")[-1] == LEGACY
                  and sig(res.get(k + "|old", {})) != sig(res.get(k + "|new", {}))]
    md = ["# Basin-collapse rule A/B — largest drop (old) vs outlier drop ≥ 3× next (new)", "",
          f"Groups: {len(jobs)}. Legacy `{LEGACY}` cells changed: **{len(legacy_bad)}** "
          "(must be 0: the legacy path does not use the rule).", ""]
    for conv in CONVS:
        ck = [k for k in keys if k.split("|")[-1] == conv]
        moved = [k for k in ck if sig(res.get(k + "|old", {})) != sig(res.get(k + "|new", {}))]
        stat = Counter()
        for k in ck:
            o, n = res.get(k + "|old", {}), res.get(k + "|new", {})
            fr, L = k.split("|")[1], k.split("|")[3]
            stat[(fr, L, "cells")] += 1
            stat[(fr, L, "old_ok")] += bool(o.get("ok"))
            stat[(fr, L, "new_ok")] += bool(n.get("ok"))
            stat[(fr, L, "moved")] += (sig(o) != sig(n))
        md += [f"## `{conv}`: {len(moved)} of {len(ck)} cells change", "",
               "| frame | L | cells | extrapolated old → new | changed |", "|:--|--:|--:|--:|--:|"]
        for fr, L in sorted({(fr, L) for fr, L, _ in stat}):
            md.append(f"| {fr} | {L[1:]} | {stat[(fr, L, 'cells')]} | {stat[(fr, L, 'old_ok')]} → "
                      f"{stat[(fr, L, 'new_ok')]} | {stat[(fr, L, 'moved')]} |")
        md += ["", "| cell | old E∞ ± σ | new E∞ ± σ | old n_post / collapse | new n_post / collapse | new reason |",
               "|:--|--:|--:|--:|--:|:--|"]
        for k in sorted(moved):
            o, n = res.get(k + "|old", {}), res.get(k + "|new", {})
            cell = "|".join(k.split("|")[:5])
            ov = f"{f(o.get('E'))} ± {f(o.get('s'))}" if o.get("ok") else "bound only"
            nv = f"{f(n.get('E'))} ± {f(n.get('s'))}" if n.get("ok") else "bound only"
            md.append(f"| `{cell}` | {ov} | {nv} | {o.get('n_post')} / {o.get('collapse_core')} | "
                      f"{n.get('n_post')} / {n.get('collapse_core')} | {n.get('reason', '')[:60]} |")
        md.append("")
    open(os.path.join(a.out_dir, "ab_collapse_rule.md"), "w").write("\n".join(md) + "\n")
    print(f"[ab] done: {len(res)} records; legacy changed {len(legacy_bad)}; "
          f"-> {a.out_dir}/ab_collapse_rule.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()
