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
    _amplitude_oracle_ops,
    alpha_for,
    count_amplitude_rotations,
    make_qrom_amplitude_bloq,
    yield_qualtran_shift_ops,
)


# Bit precision and kappa-batch size for the QROM-based amplitude oracle.
# B=8 gives angle resolution ~π/256 ≈ 0.012 rad, well below the ε_QPE typical
# of MeV-scale problems. kappa = B uses a single batch, matching the simplest
# von Burg layout.
_AMPLITUDE_BIT_PRECISION = 8
_AMPLITUDE_KAPPA = 8


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

        C3a: pyLIQTR `BlockEncoding` wrap of the C2 prototype.
        C3b: conditional shift replaced with two Qualtran `AddK` bloqs.
        C3c: amplitude oracle replaced with `ProgrammableRotationGateArray`
             — QROM-loads `B`-bit angles indexed by `(s · N_f + j)` and
             applies them via Y^t phase-kickback rotation on the `amp`
             qubit. `kappa_load_target` ancillae are allocated locally
             via the decomposition context.
        """
        sel = list(quregs['selection'])
        sys = list(quregs['system'])
        assert len(sel) == 2, f"selection register must have 2 qubits, got {len(sel)}"
        assert len(sys) == self.n_b, (
            f"system register must have {self.n_b} qubits, got {len(sys)}"
        )
        s, amp = sel

        # 1. Diffusion on the sparsity ancilla.
        yield cirq.H(s)

        # 2. QROM-loaded angle + phase-kickback rotation on amp. Allocate the
        #    kappa-load ancilla via the decomposition context (Qualtran will
        #    return these qubits to the pool after the bloq finishes).
        amp_bloq = make_qrom_amplitude_bloq(
            self.n_b,
            bit_precision=_AMPLITUDE_BIT_PRECISION,
            kappa=_AMPLITUDE_KAPPA,
        )
        kappa_qubits = context.qubit_manager.qalloc(_AMPLITUDE_KAPPA)
        # The selection register of the Programmable bloq is a single
        # BoundedQUInt of size (n_b + 1); pass `[s, *j_qubits]` (s = MSB,
        # j = remaining n_b qubits) to encode `s · N_f + j`.
        yield amp_bloq.on_registers(
            selection=[s, *sys],
            kappa_load_target=kappa_qubits,
            rotations_target=[amp],
        )
        context.qubit_manager.qfree(kappa_qubits)

        # 3. Conditional shift via two Qualtran AddK bloqs.
        yield from yield_qualtran_shift_ops(s, sys, self.n_b)

        # 4. Diffusion closes.
        yield cirq.H(s)

    # --- Cost --------------------------------------------------------------

    def _t_complexity_(self) -> TComplexity:
        """Analytical T-count for the C3c prototype.

        All components delegated to Qualtran's analytical bloq cost so the
        report tracks Qualtran's analytics automatically:

          * 2 Hadamards on the sparsity ancilla: 2 Cliffords.
          * Conditional shift: 2 × `AddK(n_b, k=±1).t_complexity()`
            (inc + dec). At n_b=3: 16 T, 76 Cliffords.
          * Amplitude oracle: `ProgrammableRotationGateArray.t_complexity()`
            with 2·N_f angles at `B = _AMPLITUDE_BIT_PRECISION` precision
            and batch-size `kappa = _AMPLITUDE_KAPPA`. This is the QROM
            +`B` controlled-Y-rotations construction (von Burg 2021).
        """
        # Shift contribution (two controlled AddK bloqs).
        per_addk = AddK(bitsize=self.n_b, k=1, cvs=(1,), signed=False).t_complexity()
        shift_t = 2 * per_addk.t
        shift_clifford = 2 * per_addk.clifford
        shift_rotations = 2 * per_addk.rotations

        # Amplitude-oracle contribution (Programmable rotation array).
        amp_bloq = make_qrom_amplitude_bloq(
            self.n_b,
            bit_precision=_AMPLITUDE_BIT_PRECISION,
            kappa=_AMPLITUDE_KAPPA,
        )
        amp_tc = amp_bloq.t_complexity()

        return TComplexity(
            t=shift_t + amp_tc.t,
            clifford=2 + shift_clifford + amp_tc.clifford,
            rotations=shift_rotations + amp_tc.rotations,
        )
