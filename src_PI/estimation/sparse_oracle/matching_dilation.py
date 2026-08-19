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
    """Return `[(lo, hi, magnitude, phase), ...]` for a 1-sparse Hermitian matching.

    `lo < hi` are the coupled basis states; `M_k[hi, lo] = magnitude·e^{i·phase}`
    (H_WT's conjugate-momentum pieces give imaginary matchings, so the phase is
    load-bearing)."""
    N = M_k.shape[0]
    edges = []
    for lo in range(N):
        for hi in range(lo + 1, N):
            z = M_k[hi, lo]
            if abs(z) > _TOL:
                edges.append((lo, hi, abs(z), float(np.angle(z))))
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


def _aligned_ops(b_dil, sys_qubits, M_k, alpha, j):
    """Yield the aligned matching-dilation ops (edges pair `n ↔ n+2^j` within a
    `2^{j+1}` block, i.e. every `lo` has bit `j` = 0).

    Loops over **all** aligned pairs, applying `G(a)` on (bit-`j` qubit, b_dil)
    controlled by the pair's other bits. Non-edge pairs (both states unmatched)
    get `a=0 → G(0)=X_b`, flipping `b_dil` to |1⟩ so the projected block is 0 —
    the boundary/unmatched handling."""
    n_b = len(sys_qubits)
    shift = 1 << j
    q = sys_qubits[n_b - 1 - j]                      # the bit-j qubit (flips in a pair)
    other = [sys_qubits[n_b - 1 - k] for k in range(n_b) if k != j]
    for lo in range(1 << n_b):
        if (lo >> j) & 1:                           # only lower endpoints (bit j = 0)
            continue
        hi = lo | shift
        z = M_k[hi, lo]
        a = abs(z) / alpha                          # 0 for unmatched pairs → X_b
        psi = float(np.angle(z)) if abs(z) > _TOL else 0.0
        ctrl_vals = [(lo >> k) & 1 for k in range(n_b) if k != j]
        # complex amplitude v·e^{iψ}: the edge block must be (v/α)(cosψ·X+sinψ·Y),
        # i.e. the unitary Rz(ψ)_q·G(a)·Rz(−ψ)_q. cirq applies ops first→last, so
        # emit Rz(−ψ), then G(a), then Rz(+ψ).
        edge_ops = []
        if abs(psi) > _TOL:
            edge_ops.append(cirq.rz(-psi).on(q))
        edge_ops.extend(_g_ops(q, b_dil, a))
        if abs(psi) > _TOL:
            edge_ops.append(cirq.rz(psi).on(q))
        for op in edge_ops:
            yield op.controlled_by(*other, control_values=ctrl_vals) if other else op


def _is_aligned(M_k, shift):
    """True iff every edge's lower endpoint has bit `log2(shift)` = 0."""
    j = int(round(math.log2(shift)))
    return all(not ((lo >> j) & 1) for lo, *_rest in _edges_of_matching(M_k))


def _cyclic_shift_gate(k, n_b):
    """`|n⟩ → |(n+k) mod 2^n_b⟩` as a MatrixGate on the `n_b` system qubits."""
    N = 1 << n_b
    P = np.zeros((N, N))
    for n in range(N):
        P[(n + k) % N, n] = 1.0
    return cirq.MatrixGate(P, name=f'Shift{k:+d}')


def matching_dilation_ops(b_dil, sys_qubits, M_k, alpha, shift):
    """Yield the decomposable matching-dilation ops for one matching component.

    `b_dil` is the 1-qubit dilation ancilla; `sys_qubits` the `n_b` system qubits
    (big-endian). `M_k` is a 1-sparse Hermitian matching with a single fixed
    `shift = Δ = 2^j`; edges pair `n ↔ n+Δ`. Aligned matchings (lower endpoints
    on the `Δ` boundary) compile directly; **misaligned** matchings (the second
    edge-colour, edges crossing the block boundary) are reduced to aligned by
    shift-conjugation `AddK(+Δ)·aligned(M')·AddK(−Δ)`, `M'[m,n]=M_k[(m+Δ)%N,(n+Δ)%N]`.

    `α·⟨0|_{b_dil} U |0⟩_{b_dil} = M_k` and `U = U† = U⁻¹`.
    """
    n_b = len(sys_qubits)
    j = int(round(math.log2(shift)))
    if _is_aligned(M_k, shift):
        yield from _aligned_ops(b_dil, sys_qubits, M_k, alpha, j)
        return
    # misaligned → cyclic shift so lower endpoints land on the Δ boundary.
    N = 1 << n_b
    M_shift = np.array([[M_k[(m + shift) % N, (n + shift) % N] for n in range(N)]
                        for m in range(N)])
    yield _cyclic_shift_gate(-shift, n_b).on(*sys_qubits)
    yield from _aligned_ops(b_dil, sys_qubits, M_shift, alpha, j)
    yield _cyclic_shift_gate(+shift, n_b).on(*sys_qubits)


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
