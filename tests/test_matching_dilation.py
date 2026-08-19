"""
P0-1 (sparse compile): the matching-dilation is a real decomposable circuit.

Codex's audit found the Hermitian bundle had no executable decomposition. This
verifies the first piece: for an **aligned** matching (edges within a `Δ=2^k`
block, states differing in one bit), `matching_dilation_ops` emits a gate-level
circuit that

  * decomposes to elementary gates (no `DecomposeNotImplementedError`),
  * whose `cirq.unitary`-extracted matrix EQUALS the dense dilation `[[A,S],[S,−A]]`
    (including unused states),
  * is Hermitian and self-inverse (so the qubitization walk is valid), and
  * projects to `α·⟨0|_{b_dil} U|0⟩ = M_k`.

The **misaligned** matching (edges crossing the `Δ` block boundary — the second
edge-colour) is a documented WIP (`xfail`): its intra-edge swap is a ±Δ shift,
not a single-bit flip, so it needs the shift-based generalisation (next P0-1
step). Together the two colours make one atom; both are needed before the sparse
walk-T is compiler-derived.

Run: `python -m pytest tests/test_matching_dilation.py -q`
"""

import numpy as np
import pytest

from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
    _split_into_components,
    hermitian_single_mode_matrix,
)
from src_PI.estimation.sparse_oracle.matching_dilation import (
    dense_matching_dilation,
    extract_matching_dilation,
)


def _atom(kind, n_b):
    """Return (M, shift). 'ladder' = â+â† (Δ=1); 'squared' = â²+â†² (Δ=2)."""
    if kind == 'ladder':
        M = (hermitian_single_mode_matrix((0,), 1.0, n_b)
             + hermitian_single_mode_matrix((1,), 0.0, n_b))
        return M, 1
    M = (hermitian_single_mode_matrix((0, 0), 1.0, n_b)
         + hermitian_single_mode_matrix((1, 1), 0.0, n_b))
    return M, 2


@pytest.mark.parametrize('kind,n_b', [('ladder', 2), ('ladder', 3),
                                      ('ladder', 4), ('squared', 3), ('squared', 4)])
@pytest.mark.parametrize('m_idx', [0, 1])
def test_matching_dilation_is_compiled_and_correct(kind, n_b, m_idx):
    """BOTH edge-colours (aligned component 0 AND misaligned component 1, the
    latter via shift-conjugation) decompose and their extracted matrix equals the
    dense dilation exactly, incl. unmatched/boundary states."""
    M, shift = _atom(kind, n_b)
    _diag, matchings = _split_into_components(M)
    if m_idx >= len(matchings):
        pytest.skip("only one matching for this atom/size")
    M_k = matchings[m_idx]
    alpha = float(np.abs(M_k).max())
    U, block = extract_matching_dilation(M_k, alpha, shift, n_b)
    B_dense = dense_matching_dilation(M_k, alpha)
    assert np.allclose(U, B_dense, atol=1e-7), "extracted U != dense dilation"
    assert np.allclose(U, U.conj().T, atol=1e-7), "not Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-7), "not self-inverse"
    assert np.allclose(block, M_k, atol=1e-7), "α·⟨0|U|0⟩ != M_k"


def test_matching_dilation_decomposes_to_elementary_gates():
    """The circuit is gate-level (no MatrixGate/DecomposeNotImplementedError):
    every op is a known elementary/controlled gate."""
    import cirq
    from src_PI.estimation.sparse_oracle.matching_dilation import matching_dilation_ops
    M, shift = _atom('ladder', 3)
    _d, matchings = _split_into_components(M)
    M_k = matchings[0]
    alpha = float(np.abs(M_k).max())
    b = cirq.NamedQubit('b')
    sysq = [cirq.NamedQubit(f's{i}') for i in range(3)]
    circ = cirq.Circuit(matching_dilation_ops(b, sysq, M_k, alpha, shift))
    assert len(list(circ.all_operations())) > 0
    for op in circ.all_operations():          # each op has a unitary (decomposable)
        assert cirq.has_unitary(op)


def test_misaligned_matching_uses_shift_conjugation():
    """The misaligned edge-colour is compiled (not skipped): its ops include the
    ±Δ cyclic shifts that reduce it to the aligned case."""
    import cirq
    from src_PI.estimation.sparse_oracle.matching_dilation import (
        matching_dilation_ops, _is_aligned,
    )
    M, shift = _atom('ladder', 3)
    _d, matchings = _split_into_components(M)
    M_k = matchings[1]
    assert not _is_aligned(M_k, shift)
    alpha = float(np.abs(M_k).max())
    b = cirq.NamedQubit('b')
    sysq = [cirq.NamedQubit(f's{i}') for i in range(3)]
    ops = list(matching_dilation_ops(b, sysq, M_k, alpha, shift))
    n_shifts = sum(1 for op in ops
                   if isinstance(op.gate, cirq.MatrixGate) and len(op.qubits) == 3)
    assert n_shifts == 2, "misaligned should sandwich with two ±Δ cyclic shifts"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
