"""Unit guards for the DMRG area-law analysis (`misc.make_dmrg_arealaw`).

The campaign these support is expensive, so the reduction from raw shards to the two
plotted observables must be right before the run, not after. Pure functions only —
no block2, no cluster data.

The load-bearing behaviour is the LOWER-BOUND flag: a shard that hit its per-chi wall
cap before the discarded weight reached the target has NOT measured chi, and reporting
its deepest rung as if it had would understate the area-law wall — i.e. it would bias
the result toward the claim we are trying to test. It must come back flagged.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from misc.make_dmrg_arealaw import DW_TARGET, chi_at_dw, cut_area, s_max


def _rec(rungs, **kw):
    base = dict(L=2, dim=3, A=2, N_f=2, sites=8, rungs=rungs, done=True)
    base.update(kw)
    return base


@pytest.mark.parametrize("L,dim,expect", [
    (2, 1, 1), (3, 1, 1), (4, 1, 1),        # 1D cut area is 1, flat in L
    (2, 2, 2), (3, 2, 3), (4, 2, 4),        # 2D cut area is L
    (2, 3, 4), (3, 3, 9), (10, 3, 100),     # 3D cut area is L^2 -- the claim
])
def test_cut_area_is_L_to_the_D_minus_one(L, dim, expect):
    assert cut_area(L, dim) == expect


def test_area_vs_volume_control_pair_shares_an_area():
    """L=4 2D (16 sites) and L=2 3D (8 sites): same cut area, different volume.
    That pair is what separates an area law from a volume law."""
    assert cut_area(4, 2) == cut_area(2, 3) == 4
    assert 4 ** 2 != 2 ** 3                      # ... at different volumes


def test_chi_at_dw_takes_the_first_rung_that_clears_the_target():
    rungs = [dict(chi=50, discarded_weight=1e-3), dict(chi=100, discarded_weight=1e-5),
             dict(chi=200, discarded_weight=1e-7), dict(chi=400, discarded_weight=1e-9)]
    chi, lower_bound = chi_at_dw(_rec(rungs), target=DW_TARGET)
    assert (chi, lower_bound) == (200, False), "must bracket at the FIRST clearing rung"


def test_capped_shard_is_reported_as_a_lower_bound():
    """No rung reached the target -> the true chi is LARGER than the deepest rung.
    Silently reporting 200 here would understate the wall."""
    rungs = [dict(chi=50, discarded_weight=1e-2), dict(chi=100, discarded_weight=1e-3),
             dict(chi=200, discarded_weight=1e-4)]
    chi, lower_bound = chi_at_dw(_rec(rungs, done=False), target=DW_TARGET)
    assert (chi, lower_bound) == (200, True)


def test_missing_discarded_weight_does_not_count_as_clearing():
    """A rung with no recorded dw (older shard / block2 API change) must be skipped,
    not treated as a zero."""
    rungs = [dict(chi=50, discarded_weight=None), dict(chi=100, discarded_weight=1e-8)]
    assert chi_at_dw(_rec(rungs), target=DW_TARGET) == (100, False)
    only_missing = [dict(chi=50, discarded_weight=None)]
    assert chi_at_dw(_rec(only_missing), target=DW_TARGET) == (50, True)


def test_s_max_takes_the_max_and_tolerates_missing():
    assert s_max(_rec([dict(chi=1, S_max_bond=0.5), dict(chi=2, S_max_bond=1.7)])) == 1.7
    assert s_max(_rec([dict(chi=1, S_max_bond=None)])) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
