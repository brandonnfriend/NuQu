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
    """Build the conditional shift as a `cirq.MatrixGate` on (1 + n_b) qubits."""
    M = _shift_matrix(n_b)
    return cirq.MatrixGate(M, name=f'SHIFT_n{n_b}')


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

    circuit = cirq.Circuit()
    # Diffusion.
    circuit.append(cirq.H(s))
    # Amplitude oracle.
    for op in _amplitude_oracle_ops(s, amp, j_qubits, n_b):
        circuit.append(op)
    # Conditional shift on the system register.
    circuit.append(_shift_gate(n_b).on(s, *j_qubits))
    # Diffusion closes.
    circuit.append(cirq.H(s))

    return circuit, (s, amp), j_qubits, alpha_for(n_b)


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
