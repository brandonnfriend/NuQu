"""
Single-mode `(â + â†)` BCK block encoding — C2 prototype.

This module builds a Cirq circuit that block-encodes the tridiagonal
`(â + â†)` matrix on a single bosonic mode at given `n_b`. The
construction is the symmetric BCK form for a d=2-sparse Hermitian
matrix with **real, positive** entries (Berry-Childs-Kothari 2015,
specialized for our case):

    U = (H_s ⊗ I) · CtrlAmp_{s,j} · Shift_s · (H_s ⊗ I)

where
  * H_s is the Hadamard "diffusion" on the 1-qubit sparsity ancilla,
  * CtrlAmp loads the matrix element value into the amplitude qubit:
        for s=0, amplitude on |0⟩_amp is √j   / α_max;
        for s=1, amplitude on |0⟩_amp is √(j+1) / α_max.
  * Shift conditionally increments / decrements the system register:
        s=0 → j ← j - 1   (mod N_f)
        s=1 → j ← j + 1   (mod N_f)

After tracing:
    α · ⟨0|_anc U |0⟩_anc |j⟩ = (â + â†) |j⟩
with α = 2 · α_max = 2 · √(N_f − 1).

**Status:** prototype only. Uses `cirq.MatrixGate` for the modular shift
and explicit per-(s, j) controlled `Ry` rotations for the amplitude
oracle; both will be replaced by Qualtran `Add` and QROM + phase-
kickback rotations in a follow-on sub-phase for a realistic T-count.
The math is the same. The point of this version is to make the
classical-sim sanity (pseudocode step 3 of the refactor) pass.

The reported `t_count` (see `circuit_t_count`) is the **analytical**
T-cost of the Qualtran-bloq replacement, not the cost of the prototype
circuit itself — that lets resource-estimate harnesses preview the
"realistic" cost without us yet having the realistic circuit.
"""

import math

import cirq
import numpy as np
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
    ProgrammableRotationGateArray,
)
from qualtran.cirq_interop import BloqAsCirqGate


# --- numpy reference --------------------------------------------------


def single_ladder_matrix(n_b):
    """`(â + â†)` on n_b qubits: N_f × N_f tridiagonal, with √j on each off-diagonal."""
    N_f = 1 << n_b
    M = np.zeros((N_f, N_f), dtype=float)
    for j in range(1, N_f):
        M[j - 1, j] = math.sqrt(j)
        M[j, j - 1] = math.sqrt(j)
    return M


def alpha_for(n_b):
    """BCK rescale factor α = 2·√(N_f − 1) for `(â + â†)` at d=2 sparsity."""
    N_f = 1 << n_b
    return 2.0 * math.sqrt(N_f - 1)


# --- Cirq primitives --------------------------------------------------


def _shift_matrix(n_b):
    """The conditional-shift unitary on (s, j) registers.

    Builds the 2·N_f × 2·N_f permutation matrix that maps
        |s, j⟩  →  |s, j − 1⟩   if s = 0 (mod N_f)
                 →  |s, j + 1⟩   if s = 1 (mod N_f).
    """
    N_f = 1 << n_b
    dim = 2 * N_f
    M = np.zeros((dim, dim), dtype=complex)
    # Cirq ordering: when we apply MatrixGate on [s, *j_qubits],
    # qubit s is the most significant. So basis index = s·N_f + j with
    # j's bit ordering matching cirq's big-endian convention.
    for s in (0, 1):
        for j in range(N_f):
            if s == 0:
                j_new = (j - 1) % N_f
            else:
                j_new = (j + 1) % N_f
            row = s * N_f + j_new   # output index
            col = s * N_f + j       # input index
            M[row, col] = 1.0
    return M


def _shift_gate(n_b):
    """Build the conditional shift as a `cirq.MatrixGate` on (1 + n_b) qubits.

    The matrix here is exactly the numerics produced by composing two
    Qualtran `AddK` bloqs (`yield_qualtran_shift_ops`); use this when the
    consumer is a Cirq simulator that needs an explicit unitary.
    """
    M = _shift_matrix(n_b)
    return cirq.MatrixGate(M, name=f'SHIFT_n{n_b}')


def _angle_int_data(n_b, bit_precision):
    """Build the 2·N_f-entry integer angle table indexed by (s, j) = s·N_f + j.

    Encoding: integer k corresponds to angle (k/2^B)·π applied via
    `cirq.Y ** angle_fraction = Ry(π·angle_fraction)`.
    """
    N_f = 1 << n_b
    alpha_max = math.sqrt(N_f - 1)
    B = bit_precision
    cap = (1 << B) - 1
    data = []
    for idx in range(2 * N_f):
        s_val = idx >> n_b
        j_val = idx & (N_f - 1)
        if s_val == 0:
            amp_value = math.sqrt(j_val) / alpha_max
        else:
            # boundary: j+1 entry is undefined → rotate to |1⟩_amp (θ=π).
            amp_value = 0.0 if j_val == N_f - 1 else math.sqrt(j_val + 1) / alpha_max
        theta = 2.0 * math.acos(max(0.0, min(1.0, amp_value)))   # θ ∈ [0, π]
        angle_int = int(round(theta * (1 << B) / math.pi))
        data.append(max(0, min(cap, angle_int)))
    return data


def make_qrom_amplitude_bloq(n_b, bit_precision=8, kappa=None):
    """Build the QROM-loaded-angle amplitude oracle as a Qualtran bloq.

    Returns a `ProgrammableRotationGateArray` configured to apply the
    angle θ_{s,j} on the amplitude qubit when the selection register
    encodes the index `s·N_f + j`. The bloq's signature is:

      * `selection`        — log₂(2·N_f) qubits = n_b + 1
      * `kappa_load_target` — kappa qubits (temporary; we allocate them in
                              the BlockEncoding's `decompose_from_registers`)
      * `rotations_target` — 1 qubit (the amplitude qubit)
    """
    if kappa is None:
        kappa = bit_precision
    data = _angle_int_data(n_b, bit_precision)
    return ProgrammableRotationGateArray(
        tuple(data),
        kappa=kappa,
        rotation_gate=cirq.Y,
    )


def yield_qualtran_shift_ops(s, j_qubits, n_b):
    """Yield the conditional shift as two Qualtran `AddK` bloqs (decompose-friendly).

    Output sequence:
      * `AddK(k=N_f − 1, cvs=(0,))` — controlled on s=0; adds (2^n_b − 1)
        which is ≡ −1 (mod 2^n_b), i.e. decrements j.
      * `AddK(k=+1, cvs=(1,))`       — controlled on s=1; increments j.

    The composite is mathematically identical to `_shift_gate(n_b)`'s
    `cirq.MatrixGate` (verified: Frobenius diff exactly zero). This path
    is used inside the pyLIQTR `BlockEncoding.decompose_from_registers`
    so pyLIQTR's call graph can introspect realistic T-counts from the
    AddK bloqs (4·n_b − 4 T per bloq, 8 T at n_b=3).
    """
    N_f = 1 << n_b
    dec = AddK(bitsize=n_b, k=N_f - 1, cvs=(0,), signed=False)
    inc = AddK(bitsize=n_b, k=1, cvs=(1,), signed=False)
    yield BloqAsCirqGate(dec).on(s, *j_qubits)
    yield BloqAsCirqGate(inc).on(s, *j_qubits)


def _ry_angle_for_amplitude(amp_value):
    """Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩.

    We want `cos(θ/2) = amp_value`, so `θ = 2 · arccos(amp_value)`.
    Clamp amp_value into [0, 1] to handle the boundary cases robustly.
    """
    clamped = max(0.0, min(1.0, amp_value))
    return 2.0 * math.acos(clamped)


def _amplitude_oracle_ops(s, amp, j_qubits, n_b):
    """Yield ops that apply Ry(θ_{s,j}) on `amp` controlled on (s, j) =
    (s_val, j_val), for every (s_val, j_val) pair that has a nonzero entry.

    s_val=0 → entry √j → applies for j ≥ 1;
    s_val=1 → entry √(j+1) → applies for j ≤ N_f − 2.

    Boundary entries (j=0 with s=0; j=N_f-1 with s=1) have zero amplitude —
    we skip the rotation (it's the identity for amp_value=0).
    """
    N_f = 1 << n_b
    alpha_max = math.sqrt(N_f - 1)
    for s_val in (0, 1):
        for j_val in range(N_f):
            if s_val == 0:
                # entry √j ∈ [0, √(N_f−1)]
                amp_value = math.sqrt(j_val) / alpha_max
            else:
                # entry √(j+1) ∈ [√1, √N_f] — at j=N_f−1 this is √N_f,
                # which exceeds α_max=√(N_f−1). At the boundary we want
                # amp_value = 0 (rotate amp to |1⟩) so the wraparound
                # column doesn't contribute to ⟨0|_amp.
                if j_val == N_f - 1:
                    amp_value = 0.0
                else:
                    amp_value = math.sqrt(j_val + 1) / alpha_max

            theta = _ry_angle_for_amplitude(amp_value)
            ry_op = cirq.ry(theta).on(amp)

            # Construct the controlled rotation: control on s == s_val AND
            # j == j_val. Cirq is big-endian in `circuit.unitary(qubit_order=...)`
            # — the first qubit in qubit_order is the most-significant bit of
            # the basis index. So `j_qubits[k]` (position k in the j-register
            # sub-list) carries bit value `n_b − 1 − k` of the integer j.
            ctrl_qubits = [s] + list(j_qubits)
            ctrl_values = [s_val] + [
                (j_val >> (n_b - 1 - k)) & 1 for k in range(n_b)
            ]
            yield cirq.ControlledGate(
                ry_op.gate,
                control_values=ctrl_values,
            ).on(*ctrl_qubits, amp)


# --- The full BCK circuit ---------------------------------------------


def yield_ops_on_qubits(s, amp, j_qubits, n_b, *, use_qualtran_shift=False):
    """Yield the BCK block-encoding ops on the supplied qubits.

    Args:
        use_qualtran_shift: when False (default), use the `cirq.MatrixGate`
            shift — convenient for Cirq's native simulator. When True,
            decompose the conditional shift into two Qualtran `AddK`
            bloqs — required for pyLIQTR's call graph to introspect the
            realistic shift T-cost. The two paths produce mathematically
            identical unitaries (Frobenius diff exactly zero).
    """
    yield cirq.H(s)
    yield from _amplitude_oracle_ops(s, amp, j_qubits, n_b)
    if use_qualtran_shift:
        yield from yield_qualtran_shift_ops(s, j_qubits, n_b)
    else:
        yield _shift_gate(n_b).on(s, *j_qubits)
    yield cirq.H(s)


def build_single_ladder_circuit(n_b):
    """Build the prototype BCK block encoding circuit for `(â + â†)` at n_b qubits.

    Returns:
        circuit:        `cirq.Circuit` operating on (1 + 1 + n_b) qubits in
                        the order [s, amp, j_0, j_1, ..., j_{n_b-1}].
        ancilla:        tuple of the ancilla qubits (s, amp). The "block
                        encoding zero" is `|0⟩_s ⊗ |0⟩_amp`.
        system_qubits:  list of the n_b system qubits (j register).
        alpha:          the rescale factor `2·√(N_f − 1)`.
    """
    s = cirq.NamedQubit('s')
    amp = cirq.NamedQubit('amp')
    j_qubits = [cirq.NamedQubit(f'j_{k}') for k in range(n_b)]
    circuit = cirq.Circuit(yield_ops_on_qubits(s, amp, j_qubits, n_b))
    return circuit, (s, amp), j_qubits, alpha_for(n_b)


def count_amplitude_rotations(n_b):
    """Number of non-trivial controlled-Ry rotations in the prototype amplitude oracle.

    The oracle has 2·N_f controls (one per (s, j) pair). At every (s, j) we
    apply Ry(θ_{s,j}) — including the two boundary cases where θ = π
    (rotating amp to |1⟩ so the wraparound column doesn't contribute).
    There are no skipped rotations in the current implementation.
    """
    return 2 * (1 << n_b)


# --- Classical-sim block-encoding extraction --------------------------


def extracted_block_matrix(circuit, ancilla, system_qubits, n_b):
    """Run cirq's unitary simulator and return α · ⟨0|_anc U |0⟩_anc.

    The result is the N_f × N_f matrix the encoder is *supposed* to
    block-encode; comparing it to `single_ladder_matrix(n_b)` is the
    classical-sim sanity gate.
    """
    qubit_order = list(ancilla) + list(system_qubits)
    U = circuit.unitary(qubit_order=qubit_order)

    # Cirq's `cirq.unitary` returns U with qubit_order interpreted big-endian:
    # the first qubit in qubit_order is the most significant. So the basis
    # ordering is: bit_0 (s) · 2 + bit_1 (amp) · 1 + j-bits — meaning the
    # |0⟩_anc subspace is the top-left N_f × N_f block.
    N_f = 1 << n_b
    block = U[:N_f, :N_f] * alpha_for(n_b)
    # Numerical cleanup: drop the imaginary noise floor.
    return np.real_if_close(block, tol=1e-9)
