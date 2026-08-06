"""Tests for the Architecture-A study glue (`classical/trimci/frame_arch_study.py`).

Pure Python — no HPC, no pyLIQTR, no venv-only deps. Builds synthetic classical
frame shards on disk, aggregates them, and checks the doc-faithful total-T formula
against hand-computed numbers.

Run:  python -m classical.trimci.tests.test_frame_arch_study
"""

import json
import math
import os
import tempfile

from classical.trimci import frame_arch_study as fas
from classical.trimci.frame_qpe import stateprep_tcount
from src_PI.estimation.qpe_cost import walk_queries


def _write_shard(d, frame, L, dim, A, seed, rungs, sites=None, n_b=2, N_f=4,
                 filling=None):
    sites = sites if sites is not None else L ** dim
    shard = {'kind': 'frame_shard', 'L': L, 'dim': dim, 'A': A, 'filling': filling,
             'frame': frame, 'seed': seed, 'sites': sites, 'n_b': n_b, 'N_f': N_f,
             'rungs': rungs, 'done': True}
    path = os.path.join(d, f"{frame}_L{L}_f{filling}_s{seed}.json")
    with open(path, 'w') as f:
        json.dump(shard, f)
    return path


def test_deepest_rung_and_seed_aggregation():
    """collect_frame_records: central metrics come from the deepest rung of the
    lowest-E_pt2 seed; the p0 band spans all seeds; a non-physical <n> outlier is
    dropped."""
    with tempfile.TemporaryDirectory() as d:
        # bare, L=3 d=3 A=27, two seeds. seed 0 more converged (lower E_pt2).
        _write_shard(d, 'bare', 3, 3, 27, 0, sites=27, filling=1.0, rungs=[
            {'core': 1000, 'p0': 0.30, 'mean_occ': 0.025, 'E_pt2': 9000.0},
            {'core': 8000, 'p0': 0.21, 'mean_occ': 0.020, 'E_pt2': 8700.0}])  # deepest
        _write_shard(d, 'bare', 3, 3, 27, 1, sites=27, filling=1.0, rungs=[
            {'core': 8000, 'p0': 0.18, 'mean_occ': 0.022, 'E_pt2': 8750.0}])  # higher E
        recs = fas.collect_frame_records([d])
        rec = recs[(3, 3, 27, 'bare')]
        assert rec.core_best == 8000, "central must be the deepest rung"
        assert abs(rec.p0_best - 0.21) < 1e-12, "central seed = lowest E_pt2 (seed 0)"
        assert rec.p0_lo == 0.18 and rec.p0_hi == 0.21, "band spans both seeds"
        assert rec.n_seeds == 2
        assert abs(rec.mean_occ_best - 0.020) < 1e-12

        # outlier <n>: a broken run at 0.77 must not become the central occupation
        _write_shard(d, 'gaussian', 3, 3, 27, 0, sites=27, filling=1.0, rungs=[
            {'core': 8000, 'p0': 0.09, 'mean_occ': 0.77, 'E_pt2': 8600.0}])
        recs = fas.collect_frame_records([d], mean_occ_max=0.5)
        g = recs[(3, 3, 27, 'gaussian')]
        assert g.mean_occ_best is None, "0.77 outlier dropped, no fallback -> None"
    print("[1] seed aggregation: deepest rung, lowest-E_pt2 central, band, outlier OK")


def test_architecture_A_docformula():
    """architecture_A_point reproduces the doc's total-T = (T_prep + N_walk·T_step)/p0
    exactly, and flags a WIN vs LOSS by the sign of p0_frame - p0_bare."""
    lam, t_step, n_b = 6.5e5, 2.6e6, 5
    q = {'physical_lambda': lam, 'total_t_count': t_step, 'n_b': n_b}
    n_walk = walk_queries(lam, 1.0)

    def rec(frame, p0, occ):
        return fas.FrameRecord(L=3, dim=3, A=27, frame=frame, sites=27, fill=1.0,
                               n_b_classical=2, N_f=4, p0_best=p0, p0_lo=p0, p0_hi=p0,
                               mean_occ_best=occ, E_pt2_best=8700.0, core_best=8000,
                               n_seeds=1, source_dir='synthetic')

    # (a) a frame that RAISES p0 (dilute-like) -> warm-start win
    bare = rec('bare', 0.26, 0.025)
    win = rec('gaussian', 0.50, 0.010)
    row = fas.architecture_A_point(bare, win, q)
    prep = stateprep_tcount(3 * 27, 4 * 27, n_b, squeeze=True, displace=False,
                            orbital=False)['T_total']
    exp_bare = n_walk * t_step / 0.26
    exp_A = (prep + n_walk * t_step) / 0.50
    assert abs(row['total_T_bare'] - exp_bare) / exp_bare < 1e-12
    assert abs(row['total_T_A'] - exp_A) / exp_A < 1e-12
    assert abs(row['total_T_ratio'] - exp_A / exp_bare) < 1e-12
    assert row['total_T_ratio'] < 1.0 and row['verdict'] == 'warm-start WIN'
    assert row['prep_vs_walk'] < 1e-6, "T_prep must be negligible vs the walk"

    # (b) a frame that LOWERS p0 (mid-filling reality) -> honest warm-start LOSS
    lose = rec('gaussian+lf', 0.05, 0.014)
    row2 = fas.architecture_A_point(bare, lose, q)
    assert row2['total_T_ratio'] > 1.0
    assert row2['verdict'].startswith('warm-start LOSS')
    assert row2['repetition_factor'] > 1.0
    # gaussian+lf charges both squeeze and displace state-prep
    assert row2['layers'] == ['squeeze', 'displace']

    # (c) boson-qubit saving uses the Fock/tong baseline when provided
    row3 = fas.architecture_A_point(bare, win, q, n_b_fock_bare=5)
    assert row3['n_b_fock_bare'] == 5
    # <n>_frame=0.010 -> recommended n_b ~ 3, so saving = 5 - 3 = 2
    assert row3['boson_qubit_saving_per_mode'] == 5 - fas.recommended_n_b(0.010)
    print(f"[2] doc total-T formula exact; WIN ratio={row['total_T_ratio']:.3f}, "
          f"LOSS ratio={row2['total_T_ratio']:.2f}, Δn_b={row3['boson_qubit_saving_per_mode']} OK")


def test_build_and_skip():
    """build_architecture_A pairs bare+frame and skips (L,A) with no resource point."""
    recs = {
        (2, 3, 4, 'bare'): fas.FrameRecord(2, 3, 4, 'bare', 8, 0.5, 2, 4,
                                           0.05, 0.04, 0.07, 0.065, 900.0, 8000, 2, 'd'),
        (2, 3, 4, 'gaussian'): fas.FrameRecord(2, 3, 4, 'gaussian', 8, 0.5, 2, 4,
                                               0.024, 0.022, 0.025, 0.065, 890.0, 8000, 2, 'd'),
        (2, 3, 8, 'gaussian'): fas.FrameRecord(2, 3, 8, 'gaussian', 8, 1.0, 2, 4,
                                               0.007, 0.006, 0.008, 0.098, 950.0, 8000, 2, 'd'),
    }
    q = {(2, 3): None}  # note: lookup is by (L, A)
    lookup = {(2, 4): {'physical_lambda': 1e13, 'total_t_count': 2.6e6, 'n_b': 33}}
    out = fas.build_architecture_A(recs, lookup)
    assert len(out['rows']) == 1 and out['rows'][0]['A'] == 4
    # A=8 gaussian has no bare partner AND no resource point -> skipped
    reasons = {(s['A'], s['reason'].split()[0]) for s in out['skipped']}
    assert (8, 'no') in reasons
    print(f"[3] build pairs bare+frame, skips {len(out['skipped'])} unmatched OK")


if __name__ == '__main__':
    test_deepest_rung_and_seed_aggregation()
    test_architecture_A_docformula()
    test_build_and_skip()
    print("\nall Architecture-A study tests passed")
