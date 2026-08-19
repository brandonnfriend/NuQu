"""
Feasibility spike (2026-08-18): a VALID (Hermitian, qubitizing) sparse block
encoding of the bosonic ladder operator `(â + â†)`.

Context — the walk-validity defect
-----------------------------------
The sparse-oracle block encoders shipped so far (`single_ladder.py`, and hence
the C1 `SparseFullBundleBlockEncoding`) are NOT Hermitian (`U ≠ U†`). pyLIQTR's
`QubitizedWalkOperator` is a single-reflection walk `W = (2Π−I)·U`, which
qubitizes only a Hermitian `U`. So those encoders' block encodings are correct
(`α·⟨0|U|0⟩ = H`) but the *walk* built from them does NOT run QPE — its spectrum
lacks the qubitization phases `e^{±i·arccos(E_k/α)}` (see
`tests/test_sparse_full_bundle.py::test_bundle_walk_qubitizes_hermitian_H`,
xfail).

The valid construction
----------------------
`(â + â†)` is a real-symmetric tridiagonal (2-sparse) Hermitian matrix.
Edge-color it into two 1-sparse Hermitian **matchings** `M_a + M_b` (even/odd
nearest-neighbour edges). For a 1-sparse matching, `M²` is *diagonal*, so the
Hermitian contraction dilation

    D = [[M/α, √(I−M²/α²)], [√(I−M²/α²), −M/α]]

is **sparse** (1-sparse `M/α` + diagonal `√`) AND Hermitian AND self-inverse
(`D²=I`). LCU-combine `D_a, D_b` with a Hermitian SELECT (`|0⟩⟨0|⊗D_a +
|1⟩⟨1|⊗D_b`) and a 1-qubit PREP → a Hermitian, self-inverse, sparse block
encoding `U` of `(â+â†)`, whose walk qubitizes.

This test asserts, for `n_b ∈ {2,3,4}`: `U=U†`, `U²=I`, `α_tot·⟨0|U|0⟩ =
(â+â†)`, and `W=(2Π−I)U` has the exact qubitization spectrum. It also records
that `α_tot = α_a+α_b` is *tighter* than `single_ladder`'s `2√(N−1)` (edge
colouring gives a better 1-norm).

**Feasibility verdict:** a valid sparse-Hermitian boson encoder exists and is
sparse-compilable; the "cost of Hermiticity" is ≈4× the per-atom boson SELECT
(two matchings × two amplitude oracles each vs single_ladder's one), with
fermion atoms unchanged (PauliLCU already Hermitian) and diagonal `n̂` atoms
cheap (diagonal dilation). This is the basis for the C1 Hermitization rebuild.
"""

import numpy as np
import pytest


def _ladder_matrix(n_b):
    N = 1 << n_b
    H = np.zeros((N, N))
    for n in range(1, N):
        H[n - 1, n] = H[n, n - 1] = np.sqrt(n)
    return H


def _matching_dilation(M, alpha):
    """Sparse Hermitian self-inverse dilation of a 1-sparse matching M."""
    N = M.shape[0]
    A = M / alpha
    S = np.diag(np.sqrt(np.clip(np.diag(np.eye(N) - A @ A), 0.0, None)))
    return np.block([[A, S], [S, -A]])


def _build_hermitian_ladder_be(n_b):
    """Return (U, alpha_tot, N) — the matching-dilation block encoding of (â+â†)."""
    N = 1 << n_b
    Ma, Mb = np.zeros((N, N)), np.zeros((N, N))
    for n in range(N - 1):
        target = Ma if n % 2 == 0 else Mb
        target[n, n + 1] = target[n + 1, n] = np.sqrt(n + 1)
    aa = max(np.abs(Ma).max(), 1e-12)
    ab = max(np.abs(Mb).max(), 1e-12)
    Da, Db = _matching_dilation(Ma, aa), _matching_dilation(Mb, ab)
    atot = aa + ab
    pa, pb = np.sqrt(aa / atot), np.sqrt(ab / atot)
    prep = np.array([[pa, -pb], [pb, pa]])            # first column = (pa, pb)
    d = Da.shape[0]
    sel = np.zeros((2 * d, 2 * d), dtype=complex)
    sel[:d, :d], sel[d:, d:] = Da, Db
    P = np.kron(prep, np.eye(d))
    U = P.conj().T @ sel @ P                           # (b_LCU, b_dil, system)
    return U, atot, N


@pytest.mark.parametrize('n_b', [2, 3, 4])
def test_matching_dilation_is_hermitian_and_block_correct(n_b):
    U, atot, N = _build_hermitian_ladder_be(n_b)
    assert np.allclose(U, U.conj().T, atol=1e-9), "block encoding must be Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-9), "must be self-inverse (U²=I)"
    block = U[:N, :N] * atot
    assert np.allclose(block, _ladder_matrix(n_b), atol=1e-9), "α·⟨0|U|0⟩ = (â+â†)"


@pytest.mark.parametrize('n_b', [2, 3, 4])
def test_matching_dilation_walk_qubitizes(n_b):
    """W = (2Π−I)·U has the qubitization spectrum e^{±i·arccos(E_k/α)}."""
    U, atot, N = _build_hermitian_ladder_be(n_b)
    Pi = np.diag([1.0] * N + [0.0] * (len(U) - N))
    W = (2 * Pi - np.eye(len(U))) @ U
    wph = np.angle(np.linalg.eigvals(W))
    for e in np.linalg.eigvalsh(_ladder_matrix(n_b)):
        th = np.arccos(np.clip(e / atot, -1, 1))
        dist = np.min(np.abs((wph - th + np.pi) % (2 * np.pi) - np.pi))
        assert dist < 1e-6, f"n_b={n_b}: qubitization phase {th:.4f} absent from walk"


def test_matching_dilation_alpha_is_tighter_than_single_ladder():
    """Edge colouring gives α_tot = α_a+α_b < single_ladder's 2√(N−1)."""
    for n_b in (2, 3, 4):
        _U, atot, N = _build_hermitian_ladder_be(n_b)
        assert atot < 2.0 * np.sqrt(N - 1)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
