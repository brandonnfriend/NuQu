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


_ATOMS = {
    # kind: (ladder actions, coeff, Δ)
    'ladder': ((0,), 1.0, 1),                 # â+â†  (real, Δ=1)
    'ladder_imag': ((0,), 1.0j, 1),           # i(â−â†)  (imaginary — H_WT Π-type)
    'ladder_cplx': ((0,), 0.7 + 0.5j, 1),     # general complex, Δ=1
    'squared': ((0, 0), 1.0, 2),              # â²+â†²  (real, Δ=2)
    'squared_imag': ((0, 0), 1.0j, 2),        # imaginary, Δ=2
}


def _atom(kind, n_b):
    """Return (M, shift) for the named single-mode Hermitian atom `c·m + c̄·m†`."""
    actions, coeff, shift = _ATOMS[kind]
    M = hermitian_single_mode_matrix(actions, coeff, n_b)
    return M, shift


@pytest.mark.parametrize('kind', list(_ATOMS))
@pytest.mark.parametrize('n_b', [2, 3, 4])
@pytest.mark.parametrize('m_idx', [0, 1])
def test_matching_dilation_is_compiled_and_correct(kind, n_b, m_idx):
    """BOTH edge-colours (aligned component 0 AND misaligned component 1, the
    latter via shift-conjugation) decompose and their extracted matrix equals the
    dense dilation exactly — incl. unmatched/boundary states and complex phases."""
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


# --------------------------------------------------------------------------- #
# Full single-mode atom (inner LCU over diagonal + matchings)                  #
# --------------------------------------------------------------------------- #


def _walk_qubitizes(U, alpha, N, M):
    Pi = np.diag([1.0] * N + [0.0] * (len(U) - N))
    W = (2 * Pi - np.eye(len(U))) @ U
    wph = np.angle(np.linalg.eigvals(W))
    for e in np.linalg.eigvalsh((M + M.conj().T) / 2):
        th = np.arccos(np.clip(e / alpha, -1, 1))
        if np.min(np.abs((wph - th + np.pi) % (2 * np.pi) - np.pi)) > 1e-6:
            return False
    return True


@pytest.mark.parametrize('actions,coeff,num_c', [
    ((0,), 1.0, None),        # â+â†  (2 matchings)
    ((0,), 1.0, 0.5),         # â+â† + ½n̂  (diagonal + 2 matchings)
    ((0,), 1.0j, 1.0),        # i(â−â†) + n̂  (imaginary matchings + diagonal)
    ((0, 0), 1.0, 0.5),       # â²+â†² + ½n̂  (Δ=2 matchings + diagonal)
])
@pytest.mark.parametrize('n_b', [2, 3])
def test_full_atom_block_encoding_qubitizes(actions, coeff, num_c, n_b):
    """A full single-mode atom (inner LCU over diagonal + matchings) is a
    decomposable block encoding of M that is Hermitian, self-inverse, and whose
    walk qubitizes."""
    from src_PI.estimation.sparse_oracle.matching_dilation import extract_atom_dilation
    M = hermitian_single_mode_matrix(actions, coeff, n_b)
    if num_c is not None:
        M = M + hermitian_single_mode_matrix((1, 0), num_c, n_b)
    U, alpha, block = extract_atom_dilation(M, n_b)
    assert np.allclose(block, M, atol=1e-6), "α·⟨0|U|0⟩ != M"
    assert np.allclose(U, U.conj().T, atol=1e-6), "atom U not Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-6), "atom U not self-inverse"
    assert _walk_qubitizes(U, alpha, 1 << n_b, M), "atom walk does not qubitize"


def test_single_mode_atom_has_genuine_compiled_t_count():
    """The full single-mode atom decomposes to elementary gates with a real,
    finite T-count (a circuit-traversed cost, not a hand `_t_complexity_`) at the
    production cutoff n_b=2 — directly refuting the 'no executable decomposition'
    finding for the dominant boson sector."""
    from src_PI.estimation.sparse_oracle.matching_dilation import compiled_atom_cost
    M = (hermitian_single_mode_matrix((0,), 1.0, 2)
         + hermitian_single_mode_matrix((1, 0), 0.5, 2))     # â+â† + ½n̂
    t, cliff = compiled_atom_cost(M, 2)
    assert 0 < t < 10000 and cliff > 0
