"""Tests for the walk-step Toffoli-depth model + reaction-limited floor (task 30/34).

Load-bearing: the MEASURED atom depth must agree with the production analytic T-count
(same primitive `resources.py` composes), and the depth band must bracket honestly
(serial >= qroam >= log) with the serial end equal to the walk Toffoli count.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.estimation.hardware import (
    atomic_depths, walk_depth_band, walk_toffoli_count,
    walk_depth_from_breakdown, reaction_runtime_s, reaction_band_s,
)
from src_PI.estimation.sparse_oracle.block_encoding import (
    SingleLadderProblemInstance, SparseSingleLadderBlockEncoding,
)


def test_atom_tcount_matches_analytic():
    """Measured exact-T count == the primitive's own analytic _t_complexity_ t --
    proves the lowered circuit is the same object resources.py costs (n_b=2:57, n_b=3:129)."""
    for n_b in (2, 3):
        ad = atomic_depths(n_b)
        analytic_t = SparseSingleLadderBlockEncoding(
            SingleLadderProblemInstance(n_b))._t_complexity_().t
        assert ad.t_count == analytic_t, (n_b, ad.t_count, analytic_t)
        assert 0 < ad.toffoli_depth <= ad.toffoli_count
        assert 0 < ad.t_depth <= ad.t_count


def test_atom_depth_grows_with_nb():
    assert atomic_depths(3).toffoli_depth > atomic_depths(2).toffoli_depth


def test_atom_nb1_clamps_conservatively():
    """n_b=1 trips a qualtran AddK(bitsize=1) decompose bug -> we measure the n_b=2 atom
    as a conservative proxy (n_b=1 atom is strictly cheaper). Must not crash; flagged."""
    a1 = atomic_depths(1)
    assert a1.clamped is True
    assert a1.measured_at_n_b == 2
    assert a1.toffoli_depth == atomic_depths(2).toffoli_depth   # same proxy build
    assert atomic_depths(2).clamped is False


def test_walk_toffoli_count_formula():
    # 2*L_eff PREPARE Toffolis + (sum_P) * atom_toffoli_count SELECT Toffolis.
    assert walk_toffoli_count(l_eff=100, total_atom_applications=250,
                              atom_toffoli_count=26) == 2 * 100 + 250 * 26


def test_band_orders_and_bounds():
    ad = atomic_depths(3)
    wtc = walk_toffoli_count(500, 1200, ad.toffoli_count)
    b = walk_depth_band(500, ad.toffoli_depth, wtc, p_max=2)
    assert b.serial >= b.qroam >= b.log
    assert b.log >= 2 * ad.toffoli_depth          # selected-term atom path (p_max=2) is a floor
    assert b.serial == wtc


def test_qroam_scales_sub_serial_as_Leff_grows():
    """The whole point: at large L_eff the QROAM band sits far below serial (parallelism)."""
    ad = atomic_depths(3)
    small = walk_depth_from_breakdown(
        {'L_eff': 50, 'select_T': 50 * 2 * 129, 'single_mode_walk_T': 129}, n_b=3)[0]
    big = walk_depth_from_breakdown(
        {'L_eff': 5000, 'select_T': 5000 * 2 * 129, 'single_mode_walk_T': 129}, n_b=3)[0]
    # qroam grows slowly (~sqrt) while serial grows ~linearly -> ratio widens with L_eff.
    assert big.qroam > small.qroam
    assert (big.serial / big.qroam) > (small.serial / small.qroam)


def test_reaction_runtime_formula_and_band():
    assert reaction_runtime_s(1e6, 200, 1e-6) == 1e6 * 200 * 1e-6
    ad = atomic_depths(3)
    wtc = walk_toffoli_count(500, 1200, ad.toffoli_count)
    b = walk_depth_band(500, ad.toffoli_depth, wtc, p_max=2)
    rb = reaction_band_s(1e6, b, 1e-6)
    assert rb['serial'] >= rb['qroam'] >= rb['log']


def test_from_breakdown_recovers_atom_applications():
    """select_T / single_mode_walk_T must recover sum_P; here 300 terms x P=2 -> 600."""
    bd = {'L_eff': 300, 'select_T': 600 * 129, 'single_mode_walk_T': 129}
    band, atom = walk_depth_from_breakdown(bd, n_b=3, p_max=2)
    assert band.serial == 2 * 300 + 600 * atom.toffoli_count


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
    print(f"test_walk_depth: {len(fns)} passed")
