"""
Sparse **Hermitian** boson block encoding (C1 walk-validity rebuild, sub-step 1).

Replaces the non-Hermitian d=1 monomial encoder (`boson_monomial_encoding.py`)
for the qubitization path. pyLIQTR's `QubitizedWalkOperator` is a single-
reflection walk `W = (2Π−I)·U` that qubitizes only a **Hermitian** block
encoding `U`; the BCK/ladder encoders (`single_ladder`, the d=1 atoms) are NOT
Hermitian, so their walks do not run QPE (see `tests/test_hermitian_sparse_spike.py`
for the diagnosis + the valid construction this module implements).

Construction (matching-dilation)
--------------------------------
The Hermitian boson atoms are `M = c·m + c̄·m†` (m a single-mode ladder
monomial; e.g. `c·â + c̄·â†`, `c·â² + c̄·â†²`) and real diagonals (`n̂`, `ââ†`).
Each such `M` is a Hermitian matrix whose off-diagonal is a single fixed shift
`±Δ`. Decompose:

    M = D (diagonal) + Σ_matchings M_k (1-sparse Hermitian matchings),

edge-colouring the ±Δ off-diagonal graph (a union of paths) into ≤2 matchings by
`(n//Δ) mod 2`. For a **matching**, `M_k²` is *diagonal*, so the contraction
dilation

    B_k = [[M_k/α_k, √(I−M_k²/α_k²)], [√(I−M_k²/α_k²), −M_k/α_k]]   (1 ancilla)

is **sparse** (1-sparse `M_k/α_k` + diagonal `√`), Hermitian, and self-inverse.
The diagonal `D` gets the same dilation (purely diagonal → cheap). LCU-combine
the components with a Hermitian SELECT (`Σ_k |k⟩⟨k|⊗B_k`) and a small PREP over
`√(α_k/α_tot)`:

    U = PREP† · SELECT · PREP ,   α_tot·⟨0|_{flag} U |0⟩_{flag} = M ,

with `flag = (LCU-select, dilation ancilla)`. Every component is Hermitian and
the LCU is symmetric, so `U = U†`, `U² = I` — a valid qubitization iterate.

`α_tot = Σ_k α_k` is the sparse-oracle subnormalization; for `(â+â†)` it is
`α_a+α_b`, *tighter* than single_ladder's `2√(N−1)` (edge colouring gives a
better 1-norm).

Status: this module provides the exact numpy/cirq simulation path + the
Hermiticity / block / walk-qubitization validation. The compiled `_t_complexity_`
+ pyLIQTR wrapper are added on top (matching-dilation = amplitude oracles +
conditional shifts, all Qualtran bloqs).
"""

import numpy as np

from src_PI.estimation.sparse_oracle.boson_monomial_encoding import (
    monomial_mode_groups,
    single_mode_monomial_matrix,
)

_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Hermitian atom matrices                                                     #
# --------------------------------------------------------------------------- #


def hermitian_single_mode_matrix(actions, coeff, n_b):
    """`M = c·m + c̄·m†` for a single-mode ladder monomial m (Hermitian, N_f×N_f).

    `actions` is the left-to-right 1/0 (â†/â) tuple of m; `coeff` = c (complex).
    For a diagonal m (net shift 0, e.g. â†â) this is `2·Re(c)·m` (still Hermitian).
    """
    m = single_mode_monomial_matrix(actions, n_b).astype(complex)
    c = complex(coeff)
    return c * m + np.conj(c) * m.conj().T


def monomial_flat_matrix(monomial, n_b):
    """Flattened matrix of a (multi-mode) boson monomial on the touched modes.

    Modes are laid out in sorted order with the first (lowest-index) mode as the
    MSB — the index is `Σ_k n_{mode_k}·N_f^(K−1−k)`, matching the reference
    layout in `boson_monomial_encoding.monomial_reference_matrix`. Returns
    `(prod, sorted_modes)`."""
    prod = np.array([[1.0 + 0j]])
    modes = []
    for mode, actions in monomial_mode_groups(monomial):
        prod = np.kron(prod, single_mode_monomial_matrix(actions, n_b))
        modes.append(mode)
    return prod.astype(complex), modes


def hermitian_monomial_matrix(monomial, coeff, n_b):
    """`M = c·prod + c̄·prod†` for a general boson monomial (Hermitian).

    Handles single-mode (`â`, `n̂`) and multi-mode (H_WT `â_b â_c†`) alike — on
    the flattened touched-mode register the monomial is a single fixed shift, so
    `build_hermitian_boson_be(M, abs_shift(M))` gives a valid qubitization iterate.
    Returns `(M, sorted_modes)`."""
    prod, modes = monomial_flat_matrix(monomial, n_b)
    c = complex(coeff)
    return c * prod + np.conj(c) * prod.conj().T, modes


def build_hermitian_monomial_be(monomial, coeff, n_b):
    """Sparse-Hermitian block encoding of `c·monomial + c̄·monomial†`.

    Returns `(U, alpha_tot, N, sorted_modes)`; `N = N_f^(#touched modes)`."""
    M, modes = hermitian_monomial_matrix(monomial, coeff, n_b)
    U, alpha, N = build_hermitian_boson_be(M, abs_shift(M))
    return U, alpha, N, modes


def abs_shift(matrix):
    """|net shift| of a fixed-shift matrix (|row−col| of any nonzero); 0 if diagonal."""
    nz = np.argwhere(np.abs(matrix) > _TOL)
    if len(nz) == 0:
        return 0
    shifts = {abs(int(r - c)) for r, c in nz}
    nonzero = {s for s in shifts if s != 0}
    if len(nonzero) > 1:
        raise ValueError(f"matrix is not a single fixed shift: shifts={sorted(shifts)}")
    return nonzero.pop() if nonzero else 0


# --------------------------------------------------------------------------- #
# Matching decomposition + dilation                                           #
# --------------------------------------------------------------------------- #


def _split_into_components(M, shift):
    """Decompose Hermitian `M` into (diagonal_vector, [matching matrices]).

    The ±`shift` off-diagonal graph (a union of paths within each residue class
    mod shift) is edge-2-coloured by `(n // shift) mod 2`, giving ≤2 disjoint
    Hermitian matchings. `shift == 0` ⇒ no matchings (pure diagonal).
    """
    N = M.shape[0]
    diag = np.real(np.diag(M)).copy()
    if shift == 0:
        return diag, []
    Ma = np.zeros((N, N), dtype=complex)
    Mb = np.zeros((N, N), dtype=complex)
    for n in range(N - shift):
        val = M[n + shift, n]                       # sub-diagonal (row>col)
        if abs(val) <= _TOL:
            continue
        target = Ma if (n // shift) % 2 == 0 else Mb
        target[n + shift, n] = val
        target[n, n + shift] = np.conj(val)         # keep each matching Hermitian
    matchings = [X for X in (Ma, Mb) if np.abs(X).max() > _TOL]
    return diag, matchings


def _dilation(component, alpha):
    """Hermitian self-inverse contraction dilation `[[A,S],[S,−A]]`, A=component/α.

    Valid (unitary) because for a matching `component²` is diagonal (so `A` and
    `S=√(I−A²)` commute); for a diagonal component this is trivially true."""
    A = np.asarray(component, dtype=complex) / alpha
    N = A.shape[0]
    S = np.diag(np.sqrt(np.clip(np.diag(np.eye(N) - A @ A.conj().T).real, 0.0, None)))
    return np.block([[A, S], [S, -A.conj().T]])


def _diag_matrix(diag_vec):
    return np.diag(diag_vec.astype(complex))


def build_hermitian_boson_be(M, shift):
    """Build the sparse-Hermitian block encoding of fixed-shift Hermitian `M`.

    Returns `(U, alpha_tot, N)` where `U` (on `flag = LCU-select ⊕ dilation
    ancilla`, then the `N`-dim system) satisfies `α_tot·⟨0|_flag U |0⟩_flag = M`,
    `U = U†`, `U² = I`. `N = M.shape[0]`.
    """
    N = M.shape[0]
    diag, matchings = _split_into_components(M, shift)

    components = []                                 # list of (component_matrix, alpha)
    if np.abs(diag).max() > _TOL:
        a = float(np.abs(diag).max())
        components.append((_diag_matrix(diag), a))
    for Mk in matchings:
        a = float(np.abs(Mk).max())
        components.append((Mk, a))
    if not components:                              # M ≡ 0
        components = [(np.zeros((N, N), dtype=complex), 1.0)]

    dilations = [_dilation(comp, a) for comp, a in components]
    alphas = np.array([a for _comp, a in components], dtype=float)
    alpha_tot = float(alphas.sum())

    d = dilations[0].shape[0]                       # 2N (dilation ancilla + system)
    n_lcu = len(dilations)
    # SELECT over the LCU-select register (ceil-log2 padded), Hermitian by parts.
    import math
    b_lcu = max(1, int(math.ceil(math.log2(max(1, n_lcu)))))
    L = 1 << b_lcu
    sel = np.zeros((L * d, L * d), dtype=complex)
    for k, Dk in enumerate(dilations):
        sel[k * d:(k + 1) * d, k * d:(k + 1) * d] = Dk
    for k in range(n_lcu, L):                       # pad unused branches with I
        sel[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.eye(d)

    # PREP on the LCU-select register: |0> -> Σ √(α_k/α_tot)|k>.
    amps = np.zeros(L)
    amps[:n_lcu] = np.sqrt(alphas / alpha_tot)
    prep = _householder(amps)
    P = np.kron(prep, np.eye(d))
    U = P.conj().T @ sel @ P
    return U, alpha_tot, N


def _householder(amps):
    """Real symmetric-orthogonal matrix with first column = unit(amps)."""
    dim = len(amps)
    v = np.array(amps, dtype=float)
    nrm = np.linalg.norm(v)
    if nrm < _TOL:
        return np.eye(dim)
    v = v / nrm
    e0 = np.zeros(dim)
    e0[0] = 1.0
    u = e0 - v
    nu = np.linalg.norm(u)
    if nu < _TOL:
        return np.eye(dim)
    u = u / nu
    return np.eye(dim) - 2.0 * np.outer(u, u)


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #


def extracted_block(U, alpha_tot, N):
    """`α_tot·⟨0|_flag U |0⟩_flag` — the encoded operator (flag = MSBs)."""
    return U[:N, :N] * alpha_tot


def walk_qubitizes(U, alpha_tot, N, M, tol=1e-6):
    """True iff `W=(2Π−I)U` has the qubitization spectrum e^{±i·arccos(E_k/α)}."""
    Pi = np.diag([1.0] * N + [0.0] * (len(U) - N))
    W = (2 * Pi - np.eye(len(U))) @ U
    wph = np.angle(np.linalg.eigvals(W))
    for e in np.linalg.eigvalsh((M + M.conj().T) / 2):
        th = np.arccos(np.clip(e / alpha_tot, -1.0, 1.0))
        if np.min(np.abs((wph - th + np.pi) % (2 * np.pi) - np.pi)) > tol:
            return False
    return True
