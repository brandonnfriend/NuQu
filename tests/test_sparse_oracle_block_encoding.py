"""
End-to-end pyLIQTR-wrap tests for the single-mode sparse-oracle encoder
(sub-phase C3a of Phase C).

Validates that the BCK prototype from C2 wraps as a pyLIQTR
`BlockEncoding`, builds a `QubitizedWalkOperator`, and produces a
finite T-count via `estimate_resources`. The classical-sim sanity is
covered by `tests/test_sparse_oracle_single_ladder.py` (C2); this file
focuses on the pyLIQTR plumbing.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.estimation.sparse_oracle import (
    SingleLadderProblemInstance,
    SparseSingleLadderBlockEncoding,
    alpha_for,
)

# pyLIQTR walk + resource estimator.
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources


def test_problem_instance_carries_alpha_and_n_qubits():
    pi = SingleLadderProblemInstance(n_b=3)
    assert pi.n_qubits() == 3
    assert pi.n_b == 3
    assert abs(pi.alpha - alpha_for(3)) < 1e-12


def test_block_encoding_signature_shape():
    pi = SingleLadderProblemInstance(n_b=3)
    be = SparseSingleLadderBlockEncoding(pi)
    # selection = 2 qubits (s + amp), target = n_b qubits, no controls.
    assert be.control_registers == ()
    sel = be.selection_registers
    tgt = be.target_registers
    assert len(sel) == 1 and sel[0].total_bits() == 2
    assert len(tgt) == 1 and tgt[0].total_bits() == 3
    sig = be.signature
    assert sum(r.total_bits() for r in sig) == 5


def test_block_encoding_reports_t_complexity():
    """The TComplexity is finite, with the right shape for the C3c prototype.

    C3c uses ProgrammableRotationGateArray (QROM-loaded angle + B controlled
    rotations) on the amplitude oracle side, plus the C3b two-AddK shift on
    the column-oracle side. Total cost is a sum.
    """
    pi = SingleLadderProblemInstance(n_b=3)
    be = SparseSingleLadderBlockEncoding(pi)
    tc = be._t_complexity_()
    # Rotations: now bounded by `kappa = 8` (not 2·N_f). The actual count
    # comes from ProgrammableRotationGateArray's internals; expect O(kappa).
    assert 0 < tc.rotations <= 64, (
        f"rotations should be O(kappa)=O(8), got {tc.rotations}"
    )
    # T: at least the shift's 2 · (4·n_b − 4) = 16 T at n_b=3, plus the
    # amplitude oracle's contribution.
    assert tc.t >= 2 * (4 * 3 - 4), (
        f"t should include at least the shift cost (16 T at n_b=3), got {tc.t}"
    )
    # Clifford: at least 2 from the diffusion Hadamards plus AddK + QROM
    # contributions.
    assert tc.clifford >= 2


def test_full_pyliqtr_pipeline_returns_finite_resources():
    """Walk operator + estimate_resources runs without exception."""
    pi = SingleLadderProblemInstance(n_b=3)
    be = SparseSingleLadderBlockEncoding(pi)
    walk = QubitizedWalkOperator(be)
    results = estimate_resources(walk)
    # pyLIQTR's estimator must return a dict with T, Clifford, LogicalQubits.
    for key in ('T', 'Clifford', 'LogicalQubits'):
        assert key in results, f"missing key in estimate_resources output: {key}"
        assert results[key] > 0, f"non-positive {key} = {results[key]}"


def test_full_pyliqtr_pipeline_scales_with_n_b():
    """Larger n_b -> more rotations -> larger T-count."""
    t_counts = {}
    for n_b in (2, 3, 4):
        pi = SingleLadderProblemInstance(n_b=n_b)
        be = SparseSingleLadderBlockEncoding(pi)
        walk = QubitizedWalkOperator(be)
        results = estimate_resources(walk)
        t_counts[n_b] = results['T']
    # T-count monotonically increases with n_b (more rotations from the
    # per-(s,j) controlled Ry loop).
    assert t_counts[2] < t_counts[3] < t_counts[4], (
        f"T-counts not monotone: {t_counts}"
    )


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll sparse-oracle C3a pyLIQTR-wrap tests passed.")
