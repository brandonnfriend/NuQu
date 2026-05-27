"""
pyLIQTR `BlockEncoding` wrapper for the single-mode `(â + â†)` BCK encoder
(Phase C3a of the sparse-oracle refactor).

Subclasses `pyLIQTR.BlockEncodings.BlockEncoding.BlockEncoding` directly
(per `01_pyliqtr_audit.md` §7c — "trivial to moderate" plumbing). Lets
pyLIQTR's `QubitizedWalkOperator` consume the prototype Cirq circuit and
`estimate_resources` produce a T-count.

**Status (C3a):** wraps the C2 prototype unchanged. Internal primitives
(`cirq.MatrixGate` for the shift; explicit per-(s, j) controlled `Ry` for
the amplitude oracle) will be replaced by Qualtran `Add`/`Subtract` and
QROM + phase-kickback rotations in a follow-on sub-phase for a tighter
T-count. The point of this commit is to establish the pyLIQTR end-to-end
path; the T-count it reports is **a conservative analytical upper bound**
on the realistic Qualtran-bloq version.
"""

import math
from typing import Tuple

import qualtran as qt
# pyLIQTR 1.3.4's `QubitizedReflection._t_complexity_` references
# `qt.bloqs.mcmt.MultiControlPauli` via dotted-attribute access, but
# `qualtran.bloqs` is *not* imported by `import qualtran`. Pre-load it
# here so any caller of the sparse encoder gets a working pipeline.
import qualtran.bloqs.mcmt  # noqa: F401

from qualtran._infra.data_types import QAny
from qualtran._infra.registers import Register, Signature
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance

from src_PI.estimation.sparse_oracle.single_ladder import (
    alpha_for,
    count_amplitude_rotations,
    yield_ops_on_qubits,
)


class SingleLadderProblemInstance(ProblemInstance):
    """Minimal `ProblemInstance` for the single-mode `(â + â†)` BCK encoder.

    Carries `n_b` (the bits per pion mode) and the BCK rescale factor
    `α = 2·√(N_f − 1)`. No Pauli expansion is held — this instance is
    only consumed by the sparse-oracle encoder, which uses the operator
    structure directly (BCK construction, not LCU-of-Paulis).
    """

    def __init__(self, n_b):
        self._n_b = int(n_b)
        self._N_f = 1 << self._n_b
        self._alpha = alpha_for(self._n_b)

    def n_qubits(self) -> int:
        return self._n_b

    @property
    def n_b(self):
        return self._n_b

    def __str__(self):
        return f"SingleLadderProblemInstance(n_b={self._n_b})"

    def get_alpha(self, **kwargs):
        return self._alpha


class SparseSingleLadderBlockEncoding(BlockEncoding):
    """pyLIQTR `BlockEncoding` for the single-mode `(â + â†)` BCK encoder.

    Register layout:
      * selection: 2 qubits — `s` (sparsity ancilla) then `amp` (amplitude
        ancilla). The "block encoding zero" is `|0⟩_s ⊗ |0⟩_amp`.
      * system:    `n_b` qubits — the bosonic Fock register.

    `α · ⟨0|_selection U |0⟩_selection = (â + â†)` on the system register
    (classical-sim validated to ~1e-15 in `tests/test_sparse_oracle_single_ladder.py`).
    """

    def __init__(self, problem_instance, control_val=None, **kwargs):
        if not isinstance(problem_instance, SingleLadderProblemInstance):
            raise TypeError(
                "SparseSingleLadderBlockEncoding requires a "
                "SingleLadderProblemInstance; got "
                f"{type(problem_instance).__name__}"
            )
        super().__init__(problem_instance, control_val=control_val, **kwargs)
        self._encoding_type = None  # not in VALID_ENCODINGS (custom encoder)

    @property
    def n_b(self):
        return self.PI.n_b

    # --- Qualtran register signature -----------------------------------

    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        # 2 qubits = s (sparsity) + amp (amplitude ancilla)
        return (Register('selection', QAny(2)),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('system', QAny(self.n_b)),)

    @property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    # --- Decomposition --------------------------------------------------

    def decompose_from_registers(self, *, context, **quregs):
        """Yield the BCK ops on the qubits Qualtran assigns to each register.

        The decomposition uses Qualtran's `AddK` bloqs for the conditional
        shift (rather than the `cirq.MatrixGate` used in the classical-sim
        path) so pyLIQTR's call graph picks up the realistic shift T-cost
        directly from each AddK's `t_complexity()`. Amplitude oracle still
        emits explicit per-(s, j) controlled `Ry` rotations; replacement
        with QROM + phase-kickback rotation lands in C3c.
        """
        sel = list(quregs['selection'])
        sys = list(quregs['system'])
        assert len(sel) == 2, f"selection register must have 2 qubits, got {len(sel)}"
        assert len(sys) == self.n_b, (
            f"system register must have {self.n_b} qubits, got {len(sys)}"
        )
        s, amp = sel
        yield from yield_ops_on_qubits(s, amp, sys, self.n_b, use_qualtran_shift=True)

    # --- Cost --------------------------------------------------------------

    def _t_complexity_(self) -> TComplexity:
        """Conservative analytical T-count for the C3b prototype.

        Breakdown:
          * 2 Hadamards on `s`: 0 T, 2 Clifford (charged manually below
            since pyLIQTR sees `_t_complexity_` rather than walking the
            full circuit).
          * `2·N_f` controlled `Ry(θ_{s,j})` rotations (one per (s, j)
            pair, including the 2 boundary `Ry(π)`). Each ends up as an
            arbitrary-angle rotation after the controlled-gate
            decomposition; charged under `rotations` so pyLIQTR's
            Selinger synthesis converts them to T at default precision.
          * Conditional shift: two Qualtran `AddK` bloqs (one inc with
            `cvs=(1,)`, one dec with `cvs=(0,)` and `k = N_f − 1`).
            Each costs `4·n_b − 4` T, so the shift contributes
            `8·n_b − 8` T total. This is the realistic Qualtran cost
            (verified Frobenius-zero against the hand-rolled
            `cirq.MatrixGate` permutation).
        """
        n_rotations = count_amplitude_rotations(self.n_b)
        # Per-AddK T-cost from Qualtran's analytical formula.
        per_addk = AddK(bitsize=self.n_b, k=1, cvs=(1,), signed=False).t_complexity()
        shift_t = 2 * per_addk.t           # 2 controlled AddKs (inc + dec)
        shift_clifford = 2 * per_addk.clifford
        return TComplexity(
            t=shift_t,
            clifford=2 + shift_clifford,
            rotations=n_rotations,
        )
