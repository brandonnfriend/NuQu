"""
Standalone total-QPE-cost computation (Phase E of the block-encoder refactor).

The total T-cost of a qubitized QPE run is the per-walk-step T-count times
the number of walk-operator queries:

    QPE_Total_T_Count = Total_T_Count · N_walk
    N_walk            = √2 · π · Λ / ΔE          (Eq. 9; Λ = Physical_Lambda)

This used to be computed inside `plot_sweep_data.py` at plot time. Per the
pseudocode (`human_knowledge/pseudocode/Block Encoders.md` step 7) the
computation now lives here as a standalone, file-in/file-out function so:
  * `run_nucleon_sweep.py` calls it once after each sweep and the totals are
    saved into the JSON;
  * `plot_sweep_data.py` reads the precomputed totals instead of recomputing;
  * it can be re-applied to *legacy* sweep files that predate the field.

`ΔE` is the QPE energy-precision target in MeV. The Watson reference report
uses 1 MeV, which is the default here.

CLI:
    python -m src_PI.estimation.qpe_cost <sweep.json> [--delta-e 1.0]
"""

import argparse
import json
import math


DEFAULT_DELTA_E_MEV = 1.0


def walk_queries(physical_lambda, delta_E=DEFAULT_DELTA_E_MEV):
    """N_walk = √2 · π · Λ / ΔE — the qubitized-walk query count for QPE."""
    return (math.sqrt(2.0) * math.pi * physical_lambda) / delta_E


def total_qpe_t_count(total_t_count, physical_lambda, delta_E=DEFAULT_DELTA_E_MEV):
    """Total QPE T-cost = per-step T-count · N_walk."""
    return total_t_count * walk_queries(physical_lambda, delta_E)


def compute_total_qpe_cost(filepath, delta_E=DEFAULT_DELTA_E_MEV, write=True):
    """Read a sweep JSON, compute the total QPE cost per result entry, write back.

    For each entry in `data['results']` adds:
      * 'QPE_Walk_Queries'   = √2·π·Λ / ΔE
      * 'QPE_Total_T_Count'  = Total_T_Count · QPE_Walk_Queries

    Records the `ΔE` used under `data['metadata']['delta_E_MeV']`. Idempotent:
    re-running recomputes from `Total_T_Count` and `Physical_Lambda`, which the
    function never mutates, so repeated calls converge to the same values.

    Args:
        filepath: path to a sweep JSON (the `save_sweep_data` schema).
        delta_E: QPE energy precision in MeV (default 1.0).
        write: if True, write the updated JSON back to `filepath` (indent=4,
            matching `DataIO.save_sweep_data`). If False, just return the dict.

    Returns:
        The updated data dict.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    for r in results:
        t_step = r.get('Total_T_Count')
        lam = r.get('Physical_Lambda')
        if t_step is None or lam is None:
            # Entry predates the Total_T_Count / Physical_Lambda fields; skip
            # rather than guess.
            continue
        nq = walk_queries(lam, delta_E)
        r['QPE_Walk_Queries'] = nq
        r['QPE_Total_T_Count'] = t_step * nq

    data.setdefault('metadata', {})['delta_E_MeV'] = delta_E

    if write:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Compute total QPE T-cost (N_walk · T_per_step) for a sweep JSON, '
                    'writing QPE_Total_T_Count back into the file.'
    )
    parser.add_argument('filepath', help='Path to a sweep JSON file.')
    parser.add_argument('--delta-e', type=float, default=DEFAULT_DELTA_E_MEV,
                        help=f'QPE energy precision ΔE in MeV (default {DEFAULT_DELTA_E_MEV}).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and print but do not write the file back.')
    args = parser.parse_args()

    data = compute_total_qpe_cost(
        args.filepath, delta_E=args.delta_e, write=not args.dry_run
    )
    results = data.get('results', [])
    written = 'NOT written (dry run)' if args.dry_run else 'written back'
    print(f"QPE total cost for {len(results)} entries ({written}, ΔE={args.delta_e} MeV):")
    for r in results[:8]:
        if 'QPE_Total_T_Count' in r:
            print(f"  A={r.get('A')!s:>4}  n_b={r.get('n_b')!s:>3}  "
                  f"T_step={r['Total_T_Count']:.3e}  Λ={r['Physical_Lambda']:.3e}  "
                  f"QPE_total={r['QPE_Total_T_Count']:.3e}")
    if len(results) > 8:
        print(f"  ... ({len(results) - 8} more)")


if __name__ == '__main__':
    main()
