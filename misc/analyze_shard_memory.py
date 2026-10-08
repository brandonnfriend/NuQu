"""Per-rung MEMORY of classical shards -> a per-row memory request (infrastructure C3, 2026-10-08).

Shards written since 2026-10-08 record, per rung, `mem` = {peak_rss_mb (this process),
children_peak_rss_mb (the forked select-stage workers), solver_peak_rss_mb, max_pool,
max_surv, rounds, trace} where `trace` rows are [round, N_core, P_pool, S_survivors,
RSS after expand, RSS after trim+diag] in MB (`graph_arrays.ground_state_arrays`). Peak
RSS is monotone, so the trace row where it first jumps names the stage that set the
high-water mark; Condor's 5-minute MemoryUsage samples miss it (293963: holds 100-250 GB
above the last sample).

This prints, per shard, the peak over rungs and the rung/stage it happened in, then per
(frame, L) the max and median peak and a suggested request:
    MEM_GB = ceil(HEADROOM x max peak + FLOOR_GB)       (defaults 1.3 and 16)
Use it on the oomtest / test shards first, then on a finished grid to set MEM per row
instead of per L.

    .venv/bin/python -m misc.analyze_shard_memory --data <shard dir> [--frame '*'] [--json out]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def shard_peak(j):
    """(peak_mb, rung_core, stage, detail) for one shard JSON; None if it has no `mem`."""
    best = None
    for r in j.get("rungs", []):
        m = r.get("mem") or {}
        vals = [v for v in (m.get("peak_rss_mb"), m.get("children_peak_rss_mb"),
                            m.get("solver_peak_rss_mb")) if v]
        if not vals:
            continue
        peak = max(vals)
        if best is None or peak > best[0]:
            stage, detail = "rss", ""
            tr = m.get("trace") or []
            # the first trace row that reaches the solver's final high-water mark
            top = m.get("solver_peak_rss_mb") or 0
            for row in tr:
                if len(row) >= 6 and row[5] >= top - 1e-6:
                    stage = "expand" if row[4] >= top - 1e-6 else "trim+diag"
                    detail = f"round {row[0]}: core {row[1]:,} pool {row[2]:,} surv {row[3]:,}"
                    break
            if peak == m.get("children_peak_rss_mb") and peak > (m.get("peak_rss_mb") or 0):
                stage = "select-worker"
            best = (float(peak), int(r["core"]), stage, detail, r.get("phase"))
    return best


def summarize(dirs, frame="*"):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, f"{frame}_*.json"))):
            j = json.load(open(f))
            pk = shard_peak(j)
            rows.append({"file": os.path.basename(f), "frame": j.get("frame", "bare"),
                         "L": int(j["L"]), "A": int(j["A"]), "seed": int(j["seed"]),
                         "top_core": max((r["core"] for r in j.get("rungs", [])), default=0),
                         "done": bool(j.get("done")), "restarts": len(j.get("resumed", [])),
                         "peak_mb": None if pk is None else pk[0],
                         "peak_core": None if pk is None else pk[1],
                         "peak_stage": None if pk is None else pk[2],
                         "peak_detail": None if pk is None else pk[3]})
    return rows


def suggest(rows, headroom=1.3, floor_gb=16.0):
    by = defaultdict(list)
    for r in rows:
        if r["peak_mb"] is not None:
            by[(r["frame"], r["L"])].append(r["peak_mb"])
    out = []
    for (fr, L), v in sorted(by.items()):
        v = sorted(v)
        mx, med = v[-1], v[len(v) // 2]
        out.append({"frame": fr, "L": L, "n": len(v), "max_gb": mx / 1024, "median_gb": med / 1024,
                    "suggest_gb": int(math.ceil(headroom * mx / 1024 + floor_gb))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--frame", default="*")
    ap.add_argument("--headroom", type=float, default=1.3)
    ap.add_argument("--floor-gb", type=float, default=16.0)
    ap.add_argument("--json", default=None, help="write the per-shard rows + suggestions here")
    a = ap.parse_args()
    rows = summarize(a.data, a.frame)
    print("| shard | top core | done | restarts | peak GB | at core | stage | detail |")
    print("|:--|--:|:--|--:|--:|--:|:--|:--|")
    for r in rows:
        pk = "—" if r["peak_mb"] is None else f"{r['peak_mb'] / 1024:.1f}"
        print(f"| {r['file']} | {r['top_core']:,} | {r['done']} | {r['restarts']} | {pk} | "
              f"{r['peak_core'] or '—'} | {r['peak_stage'] or '—'} | {r['peak_detail'] or ''} |")
    sug = suggest(rows, a.headroom, a.floor_gb)
    print()
    print(f"| frame | L | shards | max peak GB | median GB | suggest MEM GB ({a.headroom}x + {a.floor_gb:g}) |")
    print("|:--|--:|--:|--:|--:|--:|")
    for s in sug:
        print(f"| {s['frame']} | {s['L']} | {s['n']} | {s['max_gb']:.1f} | {s['median_gb']:.1f} | "
              f"**{s['suggest_gb']}** |")
    n_mem = sum(r["peak_mb"] is not None for r in rows)
    print(f"\n[mem] {len(rows)} shards, {n_mem} with per-rung memory records")
    if a.json:
        json.dump({"shards": rows, "suggest": sug}, open(a.json, "w"), indent=1)
        print(f"[mem] wrote {a.json}")


if __name__ == "__main__":
    main()
