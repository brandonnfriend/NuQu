"""
Decomposable matching-dilation circuit (C1 sparse P0-1).

Codex's audit (2026-08-18) established that the Hermitian bundle's resource
number was a hand-assembled `_t_complexity_`, not a compiled circuit: the block
encoding had **no** `decompose_from_registers`. This module builds the first
piece of the real, decomposable composite — the matching-dilation for one
1-sparse Hermitian matching component `M_k` — as an **executable gate-level
circuit** whose `cirq.unitary`-extracted matrix matches the dense reference
(including unused states), is Hermitian, and is self-inverse.

Construction (proven, `tests/test_matching_dilation.py`)
--------------------------------------------------------
A matching `M_k` couples `n ↔ n+Δ` within edges. Its contraction dilation
`B_k = [[A, S], [S, −A]]` (`A=M_k/α`, `S=√(I−A²)` diagonal) acts within each
`edge ⊗ b_dil` 4-dim subspace as a fixed 2-qubit gate

    G(a) = [[0, a, s, 0],
            [a, 0, 0, s],
            [s, 0, 0, −a],
            [0, s, −a, 0]]     (basis index 2·b_dil + q,  s = √(1−a²)),

with `a = v(edge)/α` the edge amplitude and `q` the edge's low bit. Summing the
per-edge `G(a_edge)` over the (disjoint) edges reproduces `B_k` exactly. `G(a)`
has the closed rotation form

    G(a) = R_K(−φ) · X_b · R_K(+φ) ,   K = X_q·Y_b ,  φ = arcsin(a),

so it is a Clifford-framed single-parameter rotation — the angle `φ_edge` is the
only Hamiltonian-dependent datum (QROM-loadable in the cost-optimised form).

Status: this is the **base case** — single-mode, Δ a power of two, real
amplitudes, one matching. It decomposes to elementary gates (no
`DecomposeNotImplementedError`) and its extracted matrix == the dense dilation.
Generalisation (multi-shift, two-mode/non-power-of-two Δ, complex phase, the
inner-LCU over components, and the QROM-multiplexed cost form) + the bundle
composite + a precision budget are the remaining P0 work.
"""

import math

import cirq
import numpy as np

from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
    _dilation,
    _split_into_components,
)

_TOL = 1e-12


def _edges_of_matching(M_k):
    """Return `[(lo, hi, amplitude), ...]` for a 1-sparse Hermitian matching `M_k`.

    `lo < hi` are the coupled basis states; `amplitude = |M_k[hi, lo]|` (real,
    non-negative — a boson matching has non-negative amplitudes)."""
    N = M_k.shape[0]
    edges = []
    for lo in range(N):
        for hi in range(lo + 1, N):
            if abs(M_k[hi, lo]) > _TOL:
                edges.append((lo, hi, abs(M_k[hi, lo])))
    return edges


def _r_k_ops(q, b, phi):
    """`R_K(phi) = exp(-i·phi·(X_q·Y_b)/2)` as elementary gates (CNOT-framed Rz).

    Basis-change `X_q → Z_q` (H on q) and `Y_b → Z_b` (S†,H on b), fold
    `Z_q Z_b → Z_b` with a CNOT, apply `Rz(phi)`, undo."""
    yield cirq.H(q)
    yield cirq.S(b) ** -1
    yield cirq.H(b)
    yield cirq.CNOT(q, b)
    yield cirq.rz(phi).on(b)
    yield cirq.CNOT(q, b)
    yield cirq.H(b)
    yield cirq.S(b)
    yield cirq.H(q)


def _g_ops(q, b, a):
    """`G(a) = R_K(-phi)·X_b·R_K(phi)` on (edge-qubit `q`, dilation ancilla `b`)."""
    phi = math.asin(max(-1.0, min(1.0, a)))
    yield from _r_k_ops(q, b, phi)
    yield cirq.X(b)
    yield from _r_k_ops(q, b, -phi)


def matching_dilation_ops(b_dil, sys_qubits, M_k, alpha, shift):
    """Yield the decomposable matching-dilation ops for one matching component.

    `b_dil` is the 1-qubit dilation ancilla; `sys_qubits` the `n_b` system qubits
    (big-endian, `sys_qubits[-1]` = LSB). `M_k` is the 1-sparse Hermitian matching
    on `N=2^n_b` states with a single fixed `shift = Δ` (a power of two here); the
    edges pair `n ↔ n+Δ`. Emits, per edge, `G(a_edge)` on (edge-LSB, b_dil)
    controlled by the edge index (the system bits above the LSB block).

    `α·⟨0|_{b_dil} U |0⟩_{b_dil} = M_k` and `U = U† = U⁻¹` (validated by the
    extraction test).
    """
    n_b = len(sys_qubits)
    lsb_bit = int(round(math.log2(shift)))          # Δ = 2^lsb_bit → the edge low bit
    q = sys_qubits[n_b - 1 - lsb_bit]                # the qubit that flips within an edge
    other = [sys_qubits[n_b - 1 - k] for k in range(n_b) if k != lsb_bit]  # edge-index bits
    for lo, hi, v in _edges_of_matching(M_k):
        a = v / alpha
        # edge index = the value of `lo` on the non-`q` bits (lo has q-bit = 0).
        ctrl_vals = [(lo >> k) & 1 for k in range(n_b) if k != lsb_bit]
        g = list(_g_ops(q, b_dil, a))
        for op in g:
            yield op.controlled_by(*other, control_values=ctrl_vals) if other else op


def extract_matching_dilation(M_k, alpha, shift, n_b):
    """Build the ops on named qubits, return `α·⟨0|_{b_dil} U |0⟩_{b_dil}`."""
    b_dil = cirq.NamedQubit('b_dil')
    sysq = [cirq.NamedQubit(f's{i}') for i in range(n_b)]
    circ = cirq.Circuit(matching_dilation_ops(b_dil, sysq, M_k, alpha, shift))
    U = circ.unitary(qubit_order=[b_dil, *sysq])
    N = 1 << n_b
    return U, U[:N, :N] * alpha


def dense_matching_dilation(M_k, alpha):
    """Reference dense dilation `[[A, S],[S, −A]]` (b_dil outer)."""
    return _dilation(M_k, alpha)
