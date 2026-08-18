"""
Sparse block encoding of a *single boson monomial* (C1 step 1).

A boson monomial is an OpenFermion `BosonOperator` term key such as
`((3, 1), (3, 0))` (`â†_3 â_3`, the number operator on mode 3) or
`((0, 1), (2, 0))` (`â†_0 â_2`, a two-mode product). Every such monomial
factorizes across modes (distinct modes commute), and **on each mode the
product of ladder operators is a single net-shift operator** — one nonzero
per column of the truncated `N_f × N_f` matrix (d = 1 sparse). Concretely,
grouping the monomial's factors by mode gives, per touched mode `m`, an
operator `M_m` with

    M_m |n⟩ = v_m(n) |n + Δn_m⟩            (0 if n + Δn_m ∉ [0, N_f))

where `Δn_m = (#creations − #annihilations)` on that mode and `v_m(n) ≥ 0`
is the product of the `√·` ladder factors (real, non-negative). The full
monomial is the tensor product `⊗_m M_m` on disjoint registers.

**Encoder (per mode, no sparsity ancilla).** Because each `M_m` is d = 1
there is no superposition over shift directions — we drop the `s` ancilla of
`SparseSingleLadderBlockEncoding` (which encodes the d = 2 `â + â†`). One amp
qubit per touched mode suffices:

    U_m = Shift_m(Δn_m) · AmpOracle_m ,
    AmpOracle_m : |0⟩_amp |n⟩ ↦ (cos(θ_n/2)|0⟩ + sin(θ_n/2)|1⟩)_amp |n⟩,
    θ_n = 2·arccos(v_m(n) / α_m),     α_m = max_n |v_m(n)|.

Then `α_m · ⟨0|_amp U_m |0⟩_amp = M_m` (boundary columns whose shifted row
leaves the register have `v_m(n)=0 → θ=π`, so the `AddK` wrap-around column
is killed by the `⟨0|_amp` projection — exactly the single-ladder boundary
fix). Across modes the block factor is `∏_m α_m` and the encoded operator is
`⊗_m M_m`, so

    α · ⟨0…0|_amps U |0…0|_amps = monomial,   α = ∏_m α_m.

**Normalization invariant.** `α = ∏_m α_m = _monomial_max_amplitude(monomial,
n_b)` in `lambda_compute.py` *by construction* — the per-mode `α_m` is the
exact truncated-matrix max, i.e. the d = 1 sparse-oracle rescale. This is
what makes the full-bundle `α_tot = Σ_l |coeff_l|·α_l` equal
`compute_native_lambda(...)['physical_lambda']` (C1 §2 invariant).

The overall term sign / complex phase (ε-tensor, chiral, gradient) is *not*
folded here — a boson monomial's entries are non-negative real. Signs fold
into the outer SELECT at bundle-assembly time (design §2), which is where the
alias-sampling non-negativity constraint lives.

Status (C1 step 1): the classical-sim path (`build_boson_monomial_circuit`,
`extracted_monomial_block`) is the validation gate; the pyLIQTR
`BlockEncoding` subclass declares the realistic `_t_complexity_` roll-up
(ProgrammableRotationGateArray amp oracle + AddK shift, per mode) for the
Step-3 bundle assembly.
"""

import math
from typing import Tuple

import cirq
import numpy as np
import qualtran as qt
# pyLIQTR 1.3.4's QubitizedReflection._t_complexity_ dotted-accesses
# qt.bloqs.mcmt.MultiControlPauli, which `import qualtran` does not load.
import qualtran.bloqs.mcmt  # noqa: F401

from qualtran._infra.data_types import QAny
from qualtran._infra.registers import Register, Signature
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
    ProgrammableRotationGateArray,
)
from qualtran.cirq_interop import BloqAsCirqGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance


_AMPLITUDE_BIT_PRECISION = 8
_AMPLITUDE_KAPPA = 8
_TOL = 1e-12


# --------------------------------------------------------------------------- #
# numpy references                                                            #
# --------------------------------------------------------------------------- #


def _a_matrices(n_b):
    """Return `(a, a_dag)` truncated ladder matrices on `N_f = 2^n_b` levels."""
    N_f = 1 << n_b
    a = np.zeros((N_f, N_f))
    for n in range(1, N_f):
        a[n - 1, n] = math.sqrt(n)          # â|n⟩ = √n |n-1⟩
    return a, a.T


def single_mode_monomial_matrix(actions, n_b):
    """Exact truncated `N_f × N_f` matrix for a product of ladder ops on ONE mode.

    `actions` is the left-to-right tuple of 1 (creation â†) / 0 (annihilation â)
    as they appear in the OpenFermion term tuple. The product is formed in tuple
    order: `((m,1),(m,0)) → â† â → a_dag @ a` (the number operator). Matches
    `lambda_compute._single_mode_factor_max`'s ordering exactly.
    """
    a, a_dag = _a_matrices(n_b)
    N_f = 1 << n_b
    M = np.eye(N_f)
    for act in actions:
        M = M @ (a_dag if act == 1 else a)
    return M


def monomial_mode_groups(monomial):
    """Group a boson monomial by mode, preserving per-mode factor order.

    Returns a list of `(mode_idx, actions_tuple)` sorted by `mode_idx`
    (deterministic register layout). `actions_tuple` is the ordered list of
    1/0 ladder actions on that mode as they appear left-to-right in `monomial`.
    """
    per_mode = {}
    for mode_idx, action in monomial:
        per_mode.setdefault(mode_idx, []).append(action)
    return [(m, tuple(per_mode[m])) for m in sorted(per_mode)]


def _column_shift_and_values(M):
    """For a d=1 (single-net-shift) matrix `M`, return `(delta, values)`.

    `values[n]` is the (non-negative) magnitude of the single nonzero in column
    `n` (0 if the column is empty — truncated away). `delta` is the net row
    shift `row = n + delta` common to all nonzero columns. Raises if `M` is not
    a pure shift (should never happen for a ladder-product monomial).
    """
    N_f = M.shape[0]
    values = np.zeros(N_f)
    delta = None
    for n in range(N_f):
        col = M[:, n]
        nz = np.nonzero(np.abs(col) > _TOL)[0]
        if len(nz) == 0:
            continue
        if len(nz) > 1:
            raise ValueError(
                f"column {n} of a boson monomial is not d=1 sparse "
                f"(nonzeros at rows {nz.tolist()}); ladder products must be "
                "pure shifts"
            )
        row = int(nz[0])
        this_delta = row - n
        if delta is None:
            delta = this_delta
        elif this_delta != delta:
            raise ValueError(
                f"inconsistent net shift: column {n} shifts by {this_delta}, "
                f"earlier columns by {delta}"
            )
        values[n] = abs(col[row])
    if delta is None:
        delta = 0                                # identically-zero mode operator
    return delta, values


def monomial_alpha(monomial, n_b):
    """`α = ∏_m max_n |v_m(n)|` — the d=1 sparse-oracle rescale for the monomial.

    Equal to `lambda_compute._monomial_max_amplitude(monomial, n_b)` by
    construction (product of exact per-mode truncated-matrix maxima).
    """
    alpha = 1.0
    for _mode, actions in monomial_mode_groups(monomial):
        M = single_mode_monomial_matrix(actions, n_b)
        alpha *= float(np.abs(M).max())
    return alpha


def monomial_reference_matrix(monomial, n_b):
    """Exact operator `⊗_m M_m` on the touched-mode registers (sorted mode order).

    Used as the classical-sim reference. For a `K`-mode monomial this is an
    `N_f^K × N_f^K` matrix in the same big-endian layout as
    `extracted_monomial_block` (mode 0 most significant).
    """
    ref = np.array([[1.0]])
    for _mode, actions in monomial_mode_groups(monomial):
        ref = np.kron(ref, single_mode_monomial_matrix(actions, n_b))
    return ref


# --------------------------------------------------------------------------- #
# Classical-sim path (the C1 step-1 validation gate)                          #
# --------------------------------------------------------------------------- #


def _amp_oracle_ops(amp, mreg, values, alpha_m, n_b):
    """Yield controlled-Ry(θ_n) on `amp`, controlled on the mode register = n.

    θ_n = 2·arccos(values[n] / alpha_m). Mirrors `single_ladder._amplitude_oracle_ops`
    but without the sparsity control (d=1). `mreg` is big-endian: `mreg[0]` is
    the MSB, so control bit k = (n >> (n_b-1-k)) & 1.
    """
    for n in range(1 << n_b):
        amp_value = 0.0 if alpha_m == 0.0 else values[n] / alpha_m
        theta = 2.0 * math.acos(max(0.0, min(1.0, amp_value)))
        ctrl_values = [(n >> (n_b - 1 - k)) & 1 for k in range(n_b)]
        yield cirq.ControlledGate(
            cirq.ry(theta), control_values=ctrl_values
        ).on(*mreg, amp)


def _shift_gate_matrix(delta, n_b):
    """`N_f × N_f` permutation for `|n⟩ → |(n+delta) mod N_f⟩` (big-endian mreg)."""
    N_f = 1 << n_b
    M = np.zeros((N_f, N_f))
    for n in range(N_f):
        M[(n + delta) % N_f, n] = 1.0
    return M


def build_boson_monomial_circuit(monomial, n_b):
    """Build the classical-sim BCK circuit for a single boson monomial.

    Returns `(circuit, amp_qubits, mode_registers, alpha)`:
      * `amp_qubits`      — one amp ancilla per touched mode (block-encoding
                            zero = all |0⟩), in sorted-mode order.
      * `mode_registers`  — list of `n_b`-qubit lists, one per touched mode,
                            sorted-mode order (mode 0 first / most significant).
      * `alpha`           — `∏_m α_m`.

    `α · ⟨0…0|_amps U |0…0|_amps = ⊗_m M_m` (validated by
    `extracted_monomial_block`).
    """
    groups = monomial_mode_groups(monomial)
    amp_qubits = [cirq.NamedQubit(f'amp_{i}') for i in range(len(groups))]
    mode_registers = [
        [cirq.NamedQubit(f'm{i}_{k}') for k in range(n_b)]
        for i in range(len(groups))
    ]
    alpha = 1.0
    ops = []
    for i, (_mode, actions) in enumerate(groups):
        M = single_mode_monomial_matrix(actions, n_b)
        alpha_m = float(np.abs(M).max())
        alpha *= alpha_m
        delta, values = _column_shift_and_values(M)
        amp, mreg = amp_qubits[i], mode_registers[i]
        ops.extend(_amp_oracle_ops(amp, mreg, values, alpha_m, n_b))
        if delta % (1 << n_b) != 0:
            shift = cirq.MatrixGate(
                _shift_gate_matrix(delta, n_b), name=f'SHIFT{delta}'
            )
            ops.append(shift.on(*mreg))
    circuit = cirq.Circuit(ops)
    return circuit, amp_qubits, mode_registers, alpha


def extracted_monomial_block(circuit, amp_qubits, mode_registers, n_b, alpha):
    """Return `α · ⟨0…0|_amps U |0…0|_amps` — the `N_f^K × N_f^K` encoded block.

    Big-endian layout: amps are the most-significant qubits (so the all-|0⟩ amp
    subspace is the top-left block); within the system, mode 0 is most
    significant. Comparable directly to `monomial_reference_matrix`. `alpha` is
    the `∏_m α_m` returned by `build_boson_monomial_circuit`.
    """
    K = len(amp_qubits)
    N_f = 1 << n_b
    sys_qubits = [q for reg in mode_registers for q in reg]
    qubit_order = list(amp_qubits) + sys_qubits
    U = circuit.unitary(qubit_order=qubit_order)
    dim = N_f ** K
    block = U[:dim, :dim] * alpha
    return np.real_if_close(block, tol=1e-9)


# --------------------------------------------------------------------------- #
# Qualtran-bloq path (realistic _t_complexity_ for the bundle assembly)       #
# --------------------------------------------------------------------------- #


def _mode_angle_int_data(values, alpha_m, n_b, bit_precision):
    """`N_f`-entry integer angle table θ_n = 2·arccos(v_n/α_m), indexed by n."""
    N_f = 1 << n_b
    cap = (1 << bit_precision) - 1
    data = []
    for n in range(N_f):
        amp_value = 0.0 if alpha_m == 0.0 else values[n] / alpha_m
        theta = 2.0 * math.acos(max(0.0, min(1.0, amp_value)))
        angle_int = int(round(theta * (1 << bit_precision) / math.pi))
        data.append(max(0, min(cap, angle_int)))
    return data


def make_mode_amp_bloq(values, alpha_m, n_b, bit_precision=_AMPLITUDE_BIT_PRECISION,
                       kappa=_AMPLITUDE_KAPPA):
    """QROM-loaded amplitude oracle (ProgrammableRotationGateArray) for one mode.

    Selection register is the mode's `n_b` qubits; loads θ_n and applies it on
    the amp qubit via Y^t phase kickback (von Burg 2021), matching the C3c
    single-ladder amplitude oracle.
    """
    data = _mode_angle_int_data(values, alpha_m, n_b, bit_precision)
    return ProgrammableRotationGateArray(tuple(data), kappa=kappa, rotation_gate=cirq.Y)


class BosonMonomialProblemInstance(ProblemInstance):
    """Minimal `ProblemInstance` carrying one boson monomial's structure."""

    def __init__(self, monomial, n_b):
        self._monomial = tuple(monomial)
        self._n_b = int(n_b)
        self._groups = monomial_mode_groups(monomial)
        self._alpha = monomial_alpha(monomial, n_b)

    def n_qubits(self):
        return self._n_b * len(self._groups)

    @property
    def n_b(self):
        return self._n_b

    @property
    def groups(self):
        return self._groups

    @property
    def monomial(self):
        return self._monomial

    def __str__(self):
        return f"BosonMonomialProblemInstance({self._monomial}, n_b={self._n_b})"

    def get_alpha(self, **kwargs):
        return self._alpha


class SparseBosonMonomialBlockEncoding(BlockEncoding):
    """pyLIQTR `BlockEncoding` for a single boson monomial (d=1, per-mode).

    Register layout:
      * selection: `K` amp qubits (K = # touched modes). Block-encoding zero =
        all |0⟩.
      * system:    `K · n_b` qubits — the touched modes' Fock registers, in
        sorted-mode order (mode 0 first).

    `α · ⟨0…0|_selection U |0…0|_selection = ⊗_m M_m`, α = ∏_m α_m
    (classical-sim validated in `tests/test_sparse_full_bundle.py`).
    """

    def __init__(self, problem_instance, control_val=None, **kwargs):
        if not isinstance(problem_instance, BosonMonomialProblemInstance):
            raise TypeError(
                "SparseBosonMonomialBlockEncoding requires a "
                "BosonMonomialProblemInstance; got "
                f"{type(problem_instance).__name__}"
            )
        super().__init__(problem_instance, control_val=control_val, **kwargs)
        self._encoding_type = None

    @property
    def n_b(self):
        return self.PI.n_b

    @property
    def _K(self):
        return len(self.PI.groups)

    # --- Qualtran register signature -----------------------------------

    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', QAny(self._K)),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('system', QAny(self._K * self.n_b)),)

    @property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    # --- per-mode bloq/cost data ---------------------------------------

    def _mode_data(self):
        """Yield `(amp_bloq, delta, alpha_m)` per touched mode."""
        for _mode, actions in self.PI.groups:
            M = single_mode_monomial_matrix(actions, self.n_b)
            alpha_m = float(np.abs(M).max())
            delta, values = _column_shift_and_values(M)
            amp_bloq = make_mode_amp_bloq(values, alpha_m, self.n_b)
            yield amp_bloq, delta, alpha_m

    # --- Decomposition --------------------------------------------------

    def decompose_from_registers(self, *, context, **quregs):
        sel = list(quregs['selection'])
        sys = list(quregs['system'])
        assert len(sel) == self._K, f"selection must be {self._K} qubits"
        assert len(sys) == self._K * self.n_b
        n_b = self.n_b
        for i, (amp_bloq, delta, _alpha_m) in enumerate(self._mode_data()):
            amp = sel[i]
            mreg = sys[i * n_b:(i + 1) * n_b]
            kappa_qubits = context.qubit_manager.qalloc(_AMPLITUDE_KAPPA)
            yield amp_bloq.on_registers(
                selection=mreg,
                kappa_load_target=kappa_qubits,
                rotations_target=[amp],
            )
            context.qubit_manager.qfree(kappa_qubits)
            if delta % (1 << n_b) != 0:
                yield BloqAsCirqGate(
                    AddK(bitsize=n_b, k=delta % (1 << n_b), signed=False)
                ).on(*mreg)

    # --- Cost -----------------------------------------------------------

    def _t_complexity_(self) -> TComplexity:
        """Roll-up: Σ_modes (amp_oracle.t_complexity() + shift AddK.t_complexity())."""
        t = clifford = rotations = 0
        n_b = self.n_b
        for amp_bloq, delta, _alpha_m in self._mode_data():
            amp_tc = amp_bloq.t_complexity()
            t += amp_tc.t
            clifford += amp_tc.clifford
            rotations += amp_tc.rotations
            if delta % (1 << n_b) != 0:
                shift_tc = AddK(
                    bitsize=n_b, k=delta % (1 << n_b), signed=False
                ).t_complexity()
                t += shift_tc.t
                clifford += shift_tc.clifford
                rotations += shift_tc.rotations
        return TComplexity(t=t, clifford=clifford, rotations=rotations)
