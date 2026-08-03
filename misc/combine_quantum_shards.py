"""Combine per-(L, series) quantum shard JSONs into one sweep file (task 34, I4).

Globs a campaign's `shards/` dir, concatenates every shard's `results` (tagging
each row with its `series` so the amplitude/energy_bound vs ns vs fock/sparse
columns stay distinguishable), and writes a combined JSON. Reports which shards
are incomplete (`done: false`) so a truncated shard is never silently treated as
complete — mirrors the classical `combine_detsvsL.py` honesty rule.

    python -m misc.combine_quantum_shards --shard-dir campaign_X/shards --out campaign_X/combined.json
"""
import argparse
import glob
import json
import os


def combine(shard_dir):
    paths = sorted(glob.glob(os.path.join(shard_dir, '*.json')))
    results = []
    shards = []
    incomplete = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  WARN: could not read {p}: {e}")
            continue
        meta = d.get('metadata', {})
        series = meta.get('series')
        L = meta.get('L')
        done = d.get('done', False)
        n = len(d.get('results', []))
        shards.append({'file': os.path.basename(p), 'L': L, 'series': series,
                       'done': done, 'n_points': n,
                       'manifest': meta.get('manifest')})
        if not done:
            incomplete.append(os.path.basename(p))
        for r in d.get('results', []):
            row = dict(r)
            row.setdefault('series', series)
            results.append(row)
    return results, shards, incomplete


def main():
    ap = argparse.ArgumentParser(description="Combine quantum shard JSONs.")
    ap.add_argument('--shard-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results, shards, incomplete = combine(args.shard_dir)
    out = {
        'metadata': {'kind': 'quantum_combined', 'shard_dir': args.shard_dir,
                     'n_shards': len(shards), 'n_points': len(results),
                     'incomplete_shards': incomplete, 'shards': shards},
        'results': results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"combined {len(shards)} shards -> {len(results)} points -> {args.out}")
    by_series = {}
    for r in results:
        by_series.setdefault(r.get('series'), []).append(r.get('L'))
    for s, Ls in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        print(f"  series={s}: {len(Ls)} points, L in {sorted(set(Ls))}")
    if incomplete:
        print(f"  WARNING: {len(incomplete)} incomplete shard(s) (done=false): {incomplete}")


if __name__ == '__main__':
    main()
