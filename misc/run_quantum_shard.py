"""One quantum resource-estimation SHARD — (L, series) with an A-sweep (task 34, I4).

The HPC-facing counterpart of `run_nucleon_sweep.run_sweep`: instead of auto-saving
into `data/quantum/<date>/`, it writes ONE self-describing JSON to an explicit
`--out` path and RE-SAVES after every A (atomic tmp+rename), so a shard that OOMs
or times out on a deep (L, A) still keeps everything it finished — the same
robustness pattern as the classical `run_frame_shard.py`.

A "series" is one design-axis column of the campaign (basis + encoder + cutoff):

    fock_pauli = fock  / (n_b) / pauli_lcu   (THE compiled PauliLCU anchor, task N4)
    watson  = amplitude / energy_bound / pauli_lcu   (EXPERIMENTAL — not paper-grade)
    ns      = amplitude / ns          / pauli_lcu     (EXPERIMENTAL — not paper-grade)
    sparse  = fock      / sparse      / tong          (FROZEN feasibility path; not a headline)
    sparse_heuristic = fock / sparse  / heuristic     (comparison against tong)

⚠️ The amplitude series (watson/ns) are EXPERIMENTAL: their split-oracle walk is not a
validated block encoding (H_WT is mis-represented; codex amplitude_combined_walk_audit_
2026-08-20). Do NOT report their QPE totals or amplitude-vs-Fock comparisons — the paper's
quantum anchor is `fock_pauli`. Amplitude is retained only as a component-cost diagnostic.

The paper's quantum anchor (REMEDIATION_PLAN N4) is `fock_pauli`: the Fock-basis
Hamiltonian materialized as a Pauli sum and block-encoded by pyLIQTR's PauliLCU
(genuinely compiler-derived). It is **A-independent at a fixed n_b** (the block
encoding encodes the operator, not the state — A enters only through the cutoff),
so the deep-L anchor runs one estimate per (L, n_b) with `--n-b` set. Sweep `--n-b`
for the resource-vs-cutoff curve; a measured framed ⟨n⟩ picks the physical n_b via
`--frame-occupation` (recommended_n_b_from_occupation).

Smoke test (run FIRST on a qis node before any real campaign):

    python -m misc.run_quantum_shard --test

imports pyLIQTR through a real L=2, A=1 sparse estimate — this is what surfaces the
`juliacall`/`juliapkg` Julia-runtime auto-provision and `gmpy2`/GMP issues that the
heavier quantum dependency tree can hit inside a fresh sandbox.

    python -m misc.run_quantum_shard --L 6 --series sparse --A-values 1,2,4 \
        --out campaign_X/shards/L6_sparse.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_nucleon_sweep import _compute_cutoffs, get_sweep_config
from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.qpe_cost import (
    DEFAULT_DELTA_E_MEV, total_qpe_t_count, walk_queries,
)
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.Config import Config
from src_PI.utils.manifest import build_manifest


def _reaction_depth(breakdown, n_b, n_walk):
    """Fold the walk-step Toffoli-depth band + adaptive (N_walk·D_walk) band into a
    sparse-family entry (task 30/34). Depth is an ALGORITHM resource; the τ_react
    multiply (-> wall-clock) is a hardware choice applied at figure-assembly time.
    Best-effort: returns None if the breakdown is missing (non-sparse series) or the
    atom build fails, so a shard never dies on the depth add-on."""
    if not breakdown:
        return None
    try:
        # local import: keeps the (cirq atom-build) dependency off the non-sparse path.
        from src_PI.estimation.hardware import walk_depth_from_breakdown
        band, atom = walk_depth_from_breakdown(breakdown, n_b)
        return {
            'atom_toffoli_depth': atom.toffoli_depth,
            'atom_toffoli_count': atom.toffoli_count,
            'atom_clamped': atom.clamped,          # True at n_b=1 (conservative n_b=2 proxy)
            'p_max': band.p_max,
            'D_walk': {'serial': band.serial, 'qroam': band.qroam, 'log': band.log},
            'adaptive_toffoli_depth': {
                'serial': n_walk * band.serial,
                'qroam': n_walk * band.qroam,
                'log': n_walk * band.log,
            },
        }
    except Exception as e:                        # noqa: BLE001 — never fail the shard
        return {'error': f"{type(e).__name__}: {e}"}


# Campaign design-axis columns (basis + encoder + cutoff prescription).
SERIES = {
    # THE compiled PauliLCU anchor (N4): Fock basis, PauliLCU encoder. n_b is set
    # by --n-b (anchor / convergence sweep) or --frame-occupation; the tong cutoff
    # is only the fallback when neither is given. A-independent at fixed n_b.
    'fock_pauli': dict(pion_basis='fock', cutoff_method='energy_bound',
                       block_encoder='pauli_lcu', boson_cutoff_method='tong'),
    'watson': dict(pion_basis='amplitude', cutoff_method='energy_bound',
                   block_encoder='pauli_lcu', boson_cutoff_method='heuristic'),
    'ns': dict(pion_basis='amplitude', cutoff_method='ns',
               block_encoder='pauli_lcu', boson_cutoff_method='tong'),
    'sparse': dict(pion_basis='fock', cutoff_method='energy_bound',
                   block_encoder='sparse', boson_cutoff_method='tong'),
    'sparse_heuristic': dict(pion_basis='fock', cutoff_method='energy_bound',
                             block_encoder='sparse', boson_cutoff_method='heuristic'),
}


def _config_from_series(series, walk_composition='combined_lcu'):
    s = SERIES[series]
    return Config(pion_basis=s['pion_basis'], walk_mode='series',
                  cutoff_method=s['cutoff_method'],
                  boson_cutoff_method=s['boson_cutoff_method'],
                  block_encoder=s['block_encoder'],
                  walk_composition=walk_composition)


def run_shard(L, series, A_values, dim=3, frame_occupation=None,
              delta_E=DEFAULT_DELTA_E_MEV, out=None, extra_manifest=None,
              epsilon_cut=None, n_b_override=None, walk_composition='combined_lcu',
              optimize_qpe_budget=False):
    s = SERIES[series]
    config = _config_from_series(series, walk_composition=walk_composition)
    cfg_kw = dict(L=L, dim=dim, frame_occupation=frame_occupation, **s)
    # epsilon_cut override (Option A: Watson budget-derived ε_cut for the amplitude
    # cutoff, so the amplitude n_b matches the Trotter baseline). Only affects the
    # energy_bound/ns amplitude paths; ignored by fock/tong.
    if epsilon_cut is not None:
        cfg_kw['epsilon_cut'] = epsilon_cut
    # n_b override: fixes the boson register size directly (wins over the series
    # cutoff AND over frame_occupation, per _compute_cutoffs). This is how the
    # compiled PauliLCU anchor is run A-independently at a chosen n_b, and how the
    # resource-vs-cutoff convergence curve is swept.
    if n_b_override is not None:
        cfg_kw['n_b_override'] = int(n_b_override)
    run_cfg = get_sweep_config(**cfg_kw)
    params = get_physical_parameters()

    out_data = {
        'metadata': {
            'kind': 'quantum_shard', 'L': L, 'dim': dim, 'series': series,
            'series_config': s, 'delta_E_MeV': delta_E,
            'optimize_qpe_budget': optimize_qpe_budget,
            'frame_occupation': frame_occupation, 'epsilon_cut': epsilon_cut,
            'n_b_override': int(n_b_override) if n_b_override is not None else None,
            'params': params,
            'config': config.to_dict(),
            'manifest': build_manifest(extra=extra_manifest),
        },
        'results': [], 'done': False,
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    def save():
        if not out:
            return
        tmp = out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out_data, f, indent=2)
        os.replace(tmp, out)          # atomic — a reader never sees a half file

    save()                            # header immediately
    t0 = time.time()
    for A in A_values:
        n_b, pi_max, Pi_max = _compute_cutoffs(L, dim, A, params, run_cfg, config)
        t_pt = time.time()
        norm = evaluate_resources(L, dim, n_b, pi_max, params, config,
                                  delta_E=delta_E,
                                  optimize_qpe_budget=optimize_qpe_budget)
        dt = time.time() - t_pt
        lam = norm['Physical_Lambda']
        t_step = norm['Total_T_Count']
        # When the budget optimizer ran, it already computed N_walk / QPE-T at the
        # optimal eps_qpe (not the full delta_E). Otherwise fall back to the naive
        # full-delta_E-to-QPE convention (labeled default-precision diagnostic).
        budget = norm.get('QPE_Budget')
        qpe_nwalk = norm['QPE_Walk_Queries'] if budget else walk_queries(lam, delta_E)
        qpe_total = norm['QPE_Total_T_Count'] if budget else total_qpe_t_count(t_step, lam, delta_E)
        entry = {
            'A': int(A), 'L': L, 'dim': dim, 'n_b': n_b,
            'pi_max': float(pi_max) if pi_max == pi_max else None,
            'Pi_max': float(Pi_max) if Pi_max == Pi_max else None,
            'Estimator_Wall_Seconds': round(dt, 3),   # classical estimator runtime (NOT quantum)
            'Physical_Lambda': lam,
            'Logical_Qubits': norm['Logical_Qubits'],  # one walk/block-encoding register
            'Walk_Clifford_Count': norm['Walk_Clifford_Count'],
            'Walk_T_Count': norm['Walk_T_Count'],
            'QFT_T_Count': norm['QFT_T_Count'],
            'Total_T_Count': t_step,
            'Per_Sub_Walk': norm.get('Per_Sub_Walk', []),
            # precision/error budget + pruning provenance (audit issues 1+2). None when
            # optimize_qpe_budget is off (default-precision diagnostic run).
            'QPE_Budget': budget,
            'Pruned_One_Norm_MeV': norm.get('pruned_one_norm_MeV'),
            'Pauli_Term_Count': norm.get('Pauli_Term_Count'),   # audit item 5
            'Rotation_Count': norm.get('Rotation_Count'),        # audit item 5
            # sparse LCU breakdown (L_eff, select_T, single_mode_walk_T) — feeds the
            # walk-depth / reaction-limited runtime model (task 30/34) downstream.
            'Sparse_Breakdown': norm.get('Sparse_Breakdown'),
            # QPE totals: coherent walk-query cost (multiply by repetitions ~1/p0 separately)
            'QPE_Walk_Queries': qpe_nwalk,
            'QPE_Total_T_Count': qpe_total,
        }
        # reaction-limited depth band (sparse-family only; None otherwise).
        entry['Reaction_Depth'] = _reaction_depth(
            norm.get('Sparse_Breakdown'), n_b, entry['QPE_Walk_Queries'])
        out_data['results'].append(entry)
        out_data['wall_s'] = time.time() - t0
        save()                        # INCREMENTAL: survive an OOM on the next A
        bs = ""
        if budget:
            bs = (f" f*={budget['qpe_fraction']:.2f} cp*={budget['circuit_precision']:.2e} "
                  f"pruned={budget['pruned_one_norm_MeV']:.1e}MeV"
                  f"{'' if budget['prune_within_budget'] else ' PRUNE-OVER-BUDGET'}")
        print(f"[qshard] L={L} series={series} A={A} n_b={n_b} "
              f"Lambda={lam:.3e} QPE_T={entry['QPE_Total_T_Count']:.3e} ({dt:.1f}s){bs}")

    out_data['done'] = True
    out_data['wall_s'] = time.time() - t0
    save()
    return out_data


def smoke_test():
    """Import pyLIQTR through a real tiny estimate — surfaces Julia/gmpy2 issues."""
    print("[qshard:test] importing pyLIQTR via a real L=2 A=1 sparse estimate ...")
    import pyLIQTR
    print(f"[qshard:test] pyLIQTR {getattr(pyLIQTR, '__version__', '?')}")
    data = run_shard(L=2, series='sparse', A_values=[1], out=None)
    r = data['results'][0]
    print(f"[qshard:test] OK  Lambda={r['Physical_Lambda']:.3e}  "
          f"logical_qubits={r['Logical_Qubits']}  QPE_T={r['QPE_Total_T_Count']:.3e}")
    print(f"[qshard:test] manifest: {data['metadata']['manifest']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="One quantum resource-estimation shard.")
    ap.add_argument('--test', action='store_true',
                    help="pyLIQTR/Julia/gmpy2 smoke test (L=2 A=1 sparse); no --out needed.")
    ap.add_argument('--L', type=int)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--series', choices=sorted(SERIES), default='sparse')
    ap.add_argument('--A-values', default='1,2,4',
                    help="comma-separated nucleon counts (default 1,2,4)")
    ap.add_argument('--frame-occupation', type=float, default=None,
                    help="per-mode <n> -> frame-reduced Fock register (task 34 seam a)")
    ap.add_argument('--n-b', type=int, default=None, dest='n_b_override',
                    help="fix the boson register size n_b directly (wins over the "
                         "series cutoff and over --frame-occupation). Used to run the "
                         "compiled PauliLCU anchor A-independently and to sweep the "
                         "resource-vs-cutoff convergence curve.")
    ap.add_argument('--walk-composition', default='combined_lcu',
                    choices=['combined_lcu', 'split_sum'],
                    help="amplitude split-oracle composition: 'combined_lcu' (default, "
                         "QPE-valid controlled-sum LCU walk) or 'split_sum' (legacy "
                         "invalid two-walk sum, for the A/B methods delta only). No "
                         "effect on the single-walk Fock/sparse series.")
    ap.add_argument('--optimize-budget', action='store_true', dest='optimize_qpe_budget',
                    help="single-walk Fock/PauliLCU: split delta_E between QPE resolution "
                         "and block-encoding synthesis and pick the total-T-minimizing "
                         "allocation (audit issue 1); record pruned one-norm vs budget "
                         "(issue 2). The publication-grade accuracy contract.")
    ap.add_argument('--epsilon-cut', type=float, default=None,
                    help="override the amplitude-basis field-cutoff error (Option A: the "
                         "Watson budget-derived value, e.g. 6.275e-6, so amplitude n_b "
                         "matches the Trotter baseline). Ignored by fock/tong paths.")
    ap.add_argument('--delta-E', type=float, default=DEFAULT_DELTA_E_MEV)
    ap.add_argument('--out', default=None, help="output JSON path (per-shard)")
    args = ap.parse_args()

    if args.test:
        sys.exit(smoke_test())
    if args.L is None or args.out is None:
        ap.error("--L and --out are required unless --test")
    A_values = [int(x) for x in args.A_values.split(',') if x.strip()]
    data = run_shard(args.L, args.series, A_values, dim=args.dim,
                     frame_occupation=args.frame_occupation, delta_E=args.delta_E,
                     epsilon_cut=args.epsilon_cut, n_b_override=args.n_b_override,
                     walk_composition=args.walk_composition,
                     optimize_qpe_budget=args.optimize_qpe_budget,
                     out=args.out, extra_manifest={'run_args': vars(args)})
    n = len(data['results'])
    print(f"[qshard] done: {n} points, wall={data.get('wall_s', 0):.1f}s -> {args.out}")


if __name__ == '__main__':
    main()
