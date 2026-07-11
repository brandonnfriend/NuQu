"""
Tests for the D-DMRG baseline demos (classical/baselines/dmrg_*.py):

  1. BLOCK2 ADAPTER IS CORRECT — the mixed fermion-boson complex MPO built for
     block2 reproduces the exact Lanczos ground energy at L=2 (1D). This is the
     correctness gate: only if it passes are the 3D DMRG numbers (out of exact
     reach) trustworthy.
  2. AREA-LAW ONSET (exact) — the exact bond-dimension lower bound chi*(eps) grows
     with the cut area: a 2D bisection (area 2) needs more than a 1D bisection
     (area 1); and it stays flat in 1D as L grows (L=2 vs L=3, both area 1). This
     is the mechanism of the DMRG wall, computed exactly from the Lanczos state.

Run from the project root:
    python -m tests.test_dmrg
"""

import numpy as np

from classical.baselines.dmrg_block2 import validate_vs_lanczos
from classical.baselines.dmrg_entanglement import (
    ground_state, schmidt_values, entanglement_report,
)


def test_block2_matches_lanczos():
    ok, E_lanc, E_dmrg = validate_vs_lanczos(L=2, dim=1, A=2, N_f=2, tol=1e-4)
    assert ok, f"block2 DMRG {E_dmrg} != Lanczos {E_lanc}"
    print("  PASS: block2 mixed fermion-boson MPO reproduces exact Lanczos\n")


def _chi_star(L, dim, A=2, N_f=2, eps=1e-6):
    E0, vec, H = ground_state(L, dim, A, N_f=N_f)
    sv, area = schmidt_values(vec, H, L, dim)
    rep = entanglement_report(sv)
    return rep['chi_star'][eps], area, rep['S_vonNeumann']


def test_area_law_onset():
    chi_1d, a1, _ = _chi_star(2, 1)         # cut area 1
    chi_2d, a2, _ = _chi_star(2, 2)         # cut area 2
    chi_1d_L3, a1b, _ = _chi_star(3, 1)     # cut area 1 (flat in 1D)
    print(f"  chi*(1e-6): L=2 1D (area {a1})={chi_1d}, "
          f"L=2 2D (area {a2})={chi_2d}, L=3 1D (area {a1b})={chi_1d_L3}")
    # bond dimension grows with the cut area (the area-law wall)
    assert chi_2d > chi_1d, "chi* must grow going from a 1D cut to a 2D cut"
    # in 1D it does NOT grow with L (cut area stays 1)
    assert chi_1d_L3 <= 2 * chi_1d, "1D chi* should stay flat as L grows"
    print("  PASS: chi* grows with cut area (2D>1D), flat in 1D — the area-law wall\n")


if __name__ == '__main__':
    print("=== D-DMRG tests ===\n")
    test_block2_matches_lanczos()
    test_area_law_onset()
    print("ALL TESTS PASSED")
