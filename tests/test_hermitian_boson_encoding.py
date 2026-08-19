"""
Validation for the sparse **Hermitian** boson encoder (C1 walk-validity rebuild).

`hermitian_boson_encoding.build_hermitian_boson_be` must, for every single-mode
Hermitian atom `c·m + c̄·m†` (and real diagonals), produce a block encoding `U`
that is Hermitian (`U=U†`), self-inverse (`U²=I`), block-correct
(`α_tot·⟨0|U|0⟩ = M`), and — the property the non-Hermitian d=1 encoder lacked —
whose single-reflection walk `W=(2Π−I)U` has the qubitization spectrum
`e^{±i·arccos(E_k/α)}`. Covers real / imaginary / general-complex coeffs and
shift-1 (â), shift-2 (â²), diagonal (n̂) atoms.

Run: `python -m pytest tests/test_hermitian_boson_encoding.py -q`
"""

import numpy as np
import pytest

from src_PI.estimation.sparse_oracle.boson_monomial_encoding import (
    single_mode_monomial_matrix,
)
from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
    abs_shift,
    build_hermitian_boson_be,
    extracted_block,
    hermitian_single_mode_matrix,
    walk_qubitizes,
)

# (label, ladder actions, coeff) covering every single-mode atom shape + phases.
_ATOMS = [
    ('a_real', (0,), 1.0),
    ('a_imag', (0,), 1.0j),
    ('a_cplx', (0,), 0.7 + 0.5j),
    ('adag_real', (1,), -1.3),
    ('aa_real', (0, 0), 0.9),
    ('aa_cplx', (0, 0), 0.3 - 0.4j),
    ('adagadag', (1, 1), 0.6),
    ('number', (1, 0), 1.3),
    ('n_plus_1', (0, 1), 0.8),
]


@pytest.mark.parametrize('label,actions,coeff', _ATOMS)
@pytest.mark.parametrize('n_b', [2, 3])
def test_hermitian_atom_is_valid_qubitization(label, actions, coeff, n_b):
    M = hermitian_single_mode_matrix(actions, coeff, n_b)
    shift = abs_shift(single_mode_monomial_matrix(actions, n_b))
    U, alpha, N = build_hermitian_boson_be(M, shift)
    assert np.allclose(U, U.conj().T, atol=1e-9), f"{label}: not Hermitian"
    assert np.allclose(U @ U, np.eye(len(U)), atol=1e-9), f"{label}: not self-inverse"
    assert np.allclose(extracted_block(U, alpha, N), M, atol=1e-9), f"{label}: block wrong"
    assert walk_qubitizes(U, alpha, N, M), f"{label}: walk does not qubitize"


def test_alpha_tighter_than_single_ladder_for_ladder_sum():
    """(â+â†) via matching-dilation has α = α_a+α_b < single_ladder's 2√(N−1)."""
    for n_b in (2, 3, 4):
        M = hermitian_single_mode_matrix((0,), 1.0, n_b) \
            + hermitian_single_mode_matrix((1,), 0.0, n_b)  # = â + â†
        _U, alpha, _N = build_hermitian_boson_be(M, 1)
        assert alpha < 2.0 * np.sqrt((1 << n_b) - 1)


def test_diagonal_atom_has_no_matchings():
    """A diagonal (number) atom uses a single diagonal component (shift 0)."""
    from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
        _split_into_components,
    )
    M = hermitian_single_mode_matrix((1, 0), 1.0, 3)
    diag, matchings = _split_into_components(M, 0)
    assert len(matchings) == 0
    assert np.abs(diag).max() > 0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
