"""Tests for the frame-adjusted ⟨n⟩ -> n_b seam in the sweep driver
(task 34, I1 seam a) and the shared occupation->register-size helper.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_nucleon_sweep import _compute_cutoffs, get_sweep_config
from src_PI.estimation.qpe_cost import recommended_n_b_from_occupation
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.Config import Config


def test_recommended_n_b_monotone_and_vacuum():
    # Near-vacuum (the verified ~0.045) lands at a small, honest cutoff.
    n_b_vac = recommended_n_b_from_occupation(0.045)
    assert n_b_vac == 3                                   # ceil(log2(0.045+5√1.045+1))
    # Monotone non-decreasing in occupation.
    seq = [recommended_n_b_from_occupation(x) for x in (0.0, 0.05, 0.5, 2.0, 8.0, 32.0)]
    assert all(b <= a for b, a in zip(seq, seq[1:]))
    assert seq[0] >= 1


def _cutoffs(frame_occupation=None, boson_cutoff_method='heuristic'):
    run_cfg = get_sweep_config(pion_basis='fock', block_encoder='sparse',
                               boson_cutoff_method=boson_cutoff_method,
                               frame_occupation=frame_occupation)
    config = Config(pion_basis='fock', block_encoder='sparse',
                    boson_cutoff_method=boson_cutoff_method)
    return _compute_cutoffs(2, 3, 10, get_physical_parameters(), run_cfg, config)


def test_frame_occupation_reduces_n_b():
    n_b_base, _, _ = _cutoffs(frame_occupation=None)
    n_b_frame, _, _ = _cutoffs(frame_occupation=0.045)
    assert n_b_frame == recommended_n_b_from_occupation(0.045)
    assert n_b_frame < n_b_base                           # the frame shrinks the register


def test_n_b_override_wins_over_frame_occupation():
    run_cfg = get_sweep_config(pion_basis='fock', block_encoder='sparse',
                               boson_cutoff_method='heuristic',
                               frame_occupation=0.045, n_b_override=6)
    config = Config(pion_basis='fock', block_encoder='sparse',
                    boson_cutoff_method='heuristic')
    n_b, _, _ = _compute_cutoffs(2, 3, 10, get_physical_parameters(), run_cfg, config)
    assert n_b == 6                                        # hard override wins last


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
    print(f"test_frame_occupation_seam: {len(fns)} passed")
