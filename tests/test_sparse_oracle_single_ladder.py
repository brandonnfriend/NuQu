"""
Classical-simulation tests for the single-mode `(â + â†)` BCK block
encoding (sub-phase C2 of the sparse-oracle work).

Implements pseudocode step 3 for the sparse encoder at the single-mode
level: build the prototype Cirq circuit, run cirq's unitary simulator,
extract `α · ⟨0|_anc U |0⟩_anc`, and compare to the explicit numpy
matrix of `(â + â†)`. Validates at every `n_b ∈ {2, 3, 4, 5}` (the
pseudocode's recommended range).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math

import numpy as np

from src_PI.estimation.sparse_oracle.single_ladder import (
    alpha_for,
    build_single_ladder_circuit,
    extracted_block_matrix,
    single_ladder_matrix,
)


_TOLERANCE = 1e-9  # the actual achieved precision is ~1e-15 (machine eps).


def _classical_sim_check(n_b):
    """Build, simulate, extract, compare — returns the Frobenius error."""
    circuit, ancilla, sysq, alpha = build_single_ladder_circuit(n_b)
    assert alpha == alpha_for(n_b)
    extracted = extracted_block_matrix(circuit, ancilla, sysq, n_b)
    reference = single_ladder_matrix(n_b)
    return np.linalg.norm(extracted - reference)


def test_single_ladder_n_b_2():
    err = _classical_sim_check(n_b=2)
    assert err < _TOLERANCE, f"n_b=2: ||extracted - reference||_F = {err:.3e} > {_TOLERANCE}"


def test_single_ladder_n_b_3():
    err = _classical_sim_check(n_b=3)
    assert err < _TOLERANCE, f"n_b=3: ||extracted - reference||_F = {err:.3e} > {_TOLERANCE}"


def test_single_ladder_n_b_4():
    err = _classical_sim_check(n_b=4)
    assert err < _TOLERANCE, f"n_b=4: ||extracted - reference||_F = {err:.3e} > {_TOLERANCE}"


def test_single_ladder_n_b_5():
    err = _classical_sim_check(n_b=5)
    assert err < _TOLERANCE, f"n_b=5: ||extracted - reference||_F = {err:.3e} > {_TOLERANCE}"


def test_alpha_matches_bck_formula():
    """α = 2 · √(N_f − 1) for d=2 sparse Hermitian (â + â†)."""
    for n_b in (2, 3, 4, 5):
        N_f = 1 << n_b
        assert abs(alpha_for(n_b) - 2.0 * math.sqrt(N_f - 1)) < 1e-12


def test_extracted_block_has_correct_tridiagonal_structure():
    """At n_b=3 the extracted block should be tridiagonal with √j entries."""
    circuit, ancilla, sysq, _ = build_single_ladder_circuit(n_b=3)
    extracted = extracted_block_matrix(circuit, ancilla, sysq, n_b=3)
    N_f = 8
    # Tridiagonal: zero everywhere except on the immediate sub/super-diagonals.
    for r in range(N_f):
        for c in range(N_f):
            if abs(r - c) > 1:
                assert abs(extracted[r, c]) < 1e-9, (
                    f"off-tridiagonal entry [{r},{c}] = {extracted[r,c]} "
                    "(expected 0)"
                )
            elif r == c:
                assert abs(extracted[r, c]) < 1e-9, (
                    f"diagonal entry [{r},{c}] = {extracted[r,c]} (expected 0)"
                )
    # Off-diagonal nonzeros match √j on the super-diagonal (and symmetric).
    for j in range(1, N_f):
        assert abs(extracted[j - 1, j] - math.sqrt(j)) < 1e-9
        assert abs(extracted[j, j - 1] - math.sqrt(j)) < 1e-9


def test_qualtran_shift_matches_hand_rolled_shift_matrix():
    """C3b: composing two `AddK` bloqs (inc + dec) should produce the same
    unitary as the hand-rolled `_shift_matrix` cirq MatrixGate, at every
    n_b in the validation range. Frobenius diff must be exactly zero —
    Qualtran's add-K is a permutation, so no floating-point drift.
    """
    import numpy as np
    from qualtran.bloqs.arithmetic.addition import AddK
    from src_PI.estimation.sparse_oracle.single_ladder import _shift_matrix

    for n_b in (2, 3, 4, 5):
        N_f = 1 << n_b
        dec = AddK(bitsize=n_b, k=N_f - 1, cvs=(0,), signed=False)
        inc = AddK(bitsize=n_b, k=1, cvs=(1,), signed=False)
        U_q = inc.tensor_contract() @ dec.tensor_contract()
        U_h = _shift_matrix(n_b)
        diff = np.linalg.norm(U_q - U_h)
        assert diff == 0.0, f"n_b={n_b}: Qualtran shift vs hand-rolled diff = {diff}"


def test_boundary_rotations_do_not_contribute_spurious_entries():
    """Regression guard for the boundary bug we hit during C2.

    Skipping the rotation at (s=0, j=0) / (s=1, j=N_f−1) leaves the amp
    qubit in |0⟩, so the conditional shift's wrap-around column (j=−1 →
    N_f−1; j=N_f → 0) would leak into ⟨0|_amp and corrupt the block.
    The fix is to apply Ry(π) at boundaries (rotating amp to |1⟩);
    verify no spurious row-0 column-(N_f−1) or row-(N_f−1) column-0
    entries appear at any n_b.
    """
    for n_b in (2, 3, 4):
        circuit, ancilla, sysq, _ = build_single_ladder_circuit(n_b)
        extracted = extracted_block_matrix(circuit, ancilla, sysq, n_b)
        N_f = 1 << n_b
        assert abs(extracted[0, N_f - 1]) < 1e-9
        assert abs(extracted[N_f - 1, 0]) < 1e-9


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll sparse-oracle single-ladder classical-sim tests passed.")
