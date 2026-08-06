"""Architecture-A driver — the warm-start total-T comparison (task 34, Stage 1).

Reads the classical frame-core HPC shards (`p0`, `mean_occ` per rung), joins each
`(L, A)` to a bare-H `src_PI` resource point (Λ, T_step, n_b), and folds them through
the doc-faithful total-T (`docs/frame_on_quantum_side.md` §4) via
`classical.trimci.frame_arch_study`. No `src_PI` / walk changes — Architecture A
qubitizes the BARE H; the frame is only a warm start.

Quantum resource source (pick one):
  * `--from-sweep GLOB` (default: today's watson shards): join by NEAREST A. Λ is
    A-dependent, so a nearest-A join is approximate for the ABSOLUTE total-T, but the
    warm-start ratio (p0_bare/p0_frame) is Λ- and T_step-independent, so the verdict
    is robust regardless. The actual A used is recorded per row.
  * `--reestimate`: call `evaluate_resources` at the EXACT classical A (needs the
    venv + pyLIQTR). Exact absolute total-T; slower.

Run:
  python -m misc.run_frame_arch_study                       # ratio table, venv-free
  python -m misc.run_frame_arch_study --reestimate          # exact absolute total-T
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci import frame_arch_study as fas

# Curated classical dirs that carry BOTH bare and framed cores (2026-08-04 campaign).
DEFAULT_CLASSICAL_DIRS = [
    'data/classical/hpc/2026-08-04/isospectral_L2',   # L=2 d=3, A=4,8
    'data/classical/hpc/2026-08-04/agrid_L3_64k',     # L=3 d=3, A=1..7 (dilute)
    'data/classical/hpc/2026-08-04/deepL3_128k',      # L=3 d=3, A=14,27,54
    'data/classical/hpc/2026-08-04/tail',             # L=3 d=3, A=3,27,108 (high fill)
    'data/classical/hpc/2026-08-04/L4grid',           # L=4 d=3, A=6..64 (extension)
]
DEFAULT_SWEEP_GLOB = 'data/quantum/2026-08-05/curve3_290387/shards/L*_watson_*.json'


def _fock_heuristic_n_b(A):
    """The Fock-occupation-register bare cutoff (heuristic), used as the
    apples-to-apples baseline for the boson-qubit saving (NOT the field-amplitude
    n_b, which measures a different register)."""
    return max(4, math.ceil(4 + math.log2(1 + A)))


def _nearest_sweep_lookup(sweep_glob):
    """Build a `(L, A) -> resource point` lookup by NEAREST A from sweep JSONs."""
    by_L: dict = {}
    for path in glob.glob(sweep_glob):
        with open(path) as f:
            data = json.load(f)
        for r in data.get('results', []):
            L = r.get('L')
            if L is None or r.get('Physical_Lambda') is None:
                continue
            by_L.setdefault(int(L), []).append({
                'A': int(r['A']), 'physical_lambda': r['Physical_Lambda'],
                'total_t_count': r['Total_T_Count'], 'n_b': r['n_b']})

    def lookup(L, A):
        pts = by_L.get(int(L))
        if not pts:
            return None
        best = min(pts, key=lambda p: abs(p['A'] - A))
        return {**best, 'A_quantum_used': best['A'],
                'A_match': 'exact' if best['A'] == A else 'nearest'}
    return lookup


def _reestimate_lookup(dim=3):
    """`(L, A) -> resource point` by calling `evaluate_resources` at the EXACT A
    (watson series: amplitude / energy_bound / pauli_lcu). Caches per (L, A)."""
    from run_nucleon_sweep import _compute_cutoffs, get_sweep_config
    from src_PI.estimation.EstimateResources import evaluate_resources
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.utils.Config import Config
    params = get_physical_parameters()
    config = Config(pion_basis='amplitude', walk_mode='series',
                    cutoff_method='energy_bound', boson_cutoff_method='heuristic',
                    block_encoder='pauli_lcu')
    cache: dict = {}

    def lookup(L, A):
        key = (int(L), int(A))
        if key in cache:
            return cache[key]
        run_cfg = get_sweep_config(L=L, dim=dim, pion_basis='amplitude',
                                   cutoff_method='energy_bound',
                                   boson_cutoff_method='heuristic',
                                   block_encoder='pauli_lcu')
        n_b, pi_max, _ = _compute_cutoffs(L, dim, A, params, run_cfg, config)
        norm = evaluate_resources(L, dim, n_b, pi_max, params, config)
        pt = {'A': int(A), 'physical_lambda': norm['Physical_Lambda'],
              'total_t_count': norm['Total_T_Count'], 'n_b': n_b,
              'A_quantum_used': int(A), 'A_match': 'exact'}
        cache[key] = pt
        print(f"  [reestimate] L={L} A={A}: n_b={n_b} Λ={pt['physical_lambda']:.3e} "
              f"T_step={pt['total_t_count']:.3e}")
        return pt
    return lookup


def main():
    ap = argparse.ArgumentParser(description="Architecture-A warm-start total-T study.")
    ap.add_argument('--dirs', nargs='*', default=DEFAULT_CLASSICAL_DIRS,
                    help="classical frame-core dirs (default: the 2026-08-04 campaign)")
    ap.add_argument('--from-sweep', default=DEFAULT_SWEEP_GLOB,
                    help="quantum sweep JSON glob for the nearest-A resource join")
    ap.add_argument('--reestimate', action='store_true',
                    help="re-estimate the bare-H point at the EXACT A (needs venv/pyLIQTR)")
    ap.add_argument('--frames', nargs='*', default=None,
                    help="frames to compare (default: all present except bare)")
    ap.add_argument('--delta-E', type=float, default=1.0)
    ap.add_argument('--out', default=None, help="output JSON (default: data/quantum/<date>/)")
    args = ap.parse_args()

    records = fas.collect_frame_records(args.dirs)
    n_frames = len({k[3] for k in records})
    print(f"[arch-A] collected {len(records)} (L,dim,A,frame) records "
          f"({n_frames} frame types) from {len(args.dirs)} dirs")

    if args.reestimate:
        print("[arch-A] re-estimating bare-H resource points at exact A (pyLIQTR) ...")
        qlookup = _reestimate_lookup()
    else:
        qlookup = _nearest_sweep_lookup(args.from_sweep)

    out = fas.build_architecture_A(
        records, qlookup, frames=args.frames, delta_E=args.delta_E,
        n_b_fock_lookup=lambda L, A: _fock_heuristic_n_b(A))
    rows = out['rows']

    # stamp the A-provenance onto each row from the resource point
    for r in rows:
        qp = qlookup(r['L'], r['A'])
        if qp:
            r['A_quantum_used'] = qp.get('A_quantum_used', r['A'])
            r['A_match'] = qp.get('A_match', 'exact')

    print("\n" + fas.format_table(rows))

    wins = [r for r in rows if r['total_T_ratio'] < 1.0]
    saves = [r for r in rows if (r['boson_qubit_saving_per_mode'] or 0) > 0]
    print(f"\n[verdict] {len(wins)}/{len(rows)} points are a warm-start WIN "
          f"(ratio<1); {len(saves)}/{len(rows)} show a boson-qubit saving.")
    if wins:
        best = min(wins, key=lambda r: r['total_T_ratio'])
        print(f"  best warm-start: L={best['L']} A={best['A']} {best['frame']} "
              f"-> {best['total_T_ratio']:.2f}× total-T (p0 {best['p0_bare']:.3f}"
              f"->{best['p0_frame']:.3f})")
    if any(r['A_match'] == 'nearest' for r in rows) and not args.reestimate:
        print("  NOTE: absolute total-T uses nearest-A Λ (join gaps) — the ratio "
              "column is A-exact (Λ,T cancel). Use --reestimate for exact absolute-T.")
    if out['skipped']:
        print(f"  skipped {len(out['skipped'])} (no bare partner or no resource point)")

    date = os.path.basename(os.path.dirname(
        glob.glob(args.from_sweep)[0])) if glob.glob(args.from_sweep) else '2026-08-05'
    # write next to today's quantum data
    default_dir = 'data/quantum/2026-08-05'
    os.makedirs(default_dir, exist_ok=True)
    outpath = args.out or os.path.join(
        default_dir, f"frame_arch_A_{'reest' if args.reestimate else 'nearestA'}.json")
    with open(outpath, 'w') as f:
        json.dump({'kind': 'architecture_A_study', 'delta_E_MeV': args.delta_E,
                   'source_dirs': args.dirs, 'sweep_glob': args.from_sweep,
                   'reestimated': args.reestimate,
                   'rows': rows, 'skipped': out['skipped']}, f, indent=2)
    print(f"\n[arch-A] wrote {len(rows)} rows -> {outpath}")


if __name__ == '__main__':
    main()
