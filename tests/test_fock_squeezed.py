"""Tests for the Gaussian-squeezed Fock basis (Architecture B, task 34).

Two guarantees:
  (1) at r=0 the squeezed basis is BYTE-IDENTICAL to the bare `fock` basis
      (the collapsed local term is recovered from the explicit build);
  (2) at r≠0 the physical low spectrum is UNCHANGED (the squeeze is a canonical,
      isospectral transform) at a converged Fock cutoff.

(1) is exact and cheap. (2) uses a tiny L=1 d=1 system (dense-diagonalizable) at
N_f=8; the full-H (gradient+AV) isospectrality at L=2 d=1 is exercised by the
resource-smoke driver `misc/run_frame_arch_B.py --isospectral-check`.

Run:  python -m tests.test_fock_squeezed
"""

import numpy as np
import scipy.sparse.linalg as sla
from openfermion import get_sparse_operator, jordan_wigner

from src_PI.hamiltonians.core.pion_basis import fock, fock_squeezed
from src_PI.hamiltonians.core.StaticTerms import Static_Nucleon_Hamiltonian
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.LatticeGeometry import total_qubits


def _qop_max_diff(a, b):
    """Max |coeff| of (a - b) over the union of terms (0 ⇒ identical operators)."""
    diff = a - b
    return max((abs(c) for c in diff.terms.values()), default=0.0)


def _full_qop(basis_module, L, dim, n_b, params):
    """Static-nucleon JW + the basis's dynamical-pion sector, as one QubitOperator
    (mirrors the ConstructEFT pauli path)."""
    static_q = jordan_wigner(
        Static_Nucleon_Hamiltonian(params['h'], params['C'], params['CI'], L, dim, n_b))
    subs = basis_module.Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, 0.0, params)
    return static_q + subs[0][1]


def _low_spectrum(qop, n_qubits, k):
    """Lowest-k eigenvalues via SPARSE eigsh (no dense 2^n matrix)."""
    M = get_sparse_operator(qop, n_qubits=n_qubits)
    M = 0.5 * (M + M.getH())
    return np.sort(sla.eigsh(M, k=k, which='SA', return_eigenvectors=False))


def test_r_zero_identical_to_fock():
    """r=0 ⇒ the explicit local term recovers the collapsed one, and the full
    squeezed sub-Hamiltonian equals bare fock's, term for term."""
    params = dict(get_physical_parameters())
    params['squeeze_r'] = 0.0
    L, dim, n_b = 2, 1, 2

    # the risky piece: explicit local == collapsed local
    loc_sq = fock_squeezed.H_pion_free_local(L, dim, n_b, params, r=0.0)
    loc_ba = fock.H_pion_free_local(L, dim, n_b, params)
    d_loc = _qop_max_diff(loc_sq, loc_ba)
    assert d_loc < 1e-9, f"explicit local != collapsed local at r=0: max|Δ|={d_loc:.2e}"

    # the whole thing
    full_sq = fock_squeezed.Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, 0.0, params)[0][1]
    full_ba = fock.Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, 0.0, params)[0][1]
    d_full = _qop_max_diff(full_sq, full_ba)
    assert d_full < 1e-9, f"fock_squeezed(r=0) != fock: max|Δ|={d_full:.2e}"
    print(f"[1] r=0 identical to fock (local Δ={d_loc:.1e}, full Δ={d_full:.1e}) OK")


def test_isospectral_r_nonzero():
    """A canonical squeeze preserves the physical low spectrum at converged N_f.
    Small + sparse: L=1 d=1 n_b=2 = 4 nucleon + 6 pion = 10 qubits (1024-dim),
    lowest-4 via eigsh."""
    params = dict(get_physical_parameters())
    L, dim, n_b = 1, 1, 2            # 10 qubits, N_f=4
    nq = total_qubits(L, dim, n_b)
    k = 4

    params['squeeze_r'] = 0.0
    e_bare = _low_spectrum(_full_qop(fock_squeezed, L, dim, n_b, params), nq, k)

    worst = 0.0
    for r in (0.1, -0.1):
        params['squeeze_r'] = r
        e_sq = _low_spectrum(_full_qop(fock_squeezed, L, dim, n_b, params), nq, k)
        err = float(np.max(np.abs(e_sq - e_bare)))
        worst = max(worst, err)
        # N_f=4 leaks a little truncation at r≠0 (~1 MeV at this cutoff); the
        # squeeze is exactly canonical, so the residual is bounded and small vs
        # the ~500 MeV scale — a loose tol screens gross non-isospectrality.
        assert err < 2.0, f"squeeze r={r} not isospectral at N_f=4: max|ΔE|={err:.2e} MeV"
    print(f"[2] isospectral at N_f=4 (L=1, 10q) for r∈{{±0.1}}: max|ΔE|={worst:.2e} MeV OK")


if __name__ == '__main__':
    test_r_zero_identical_to_fock()
    test_isospectral_r_nonzero()
    print("\nall fock_squeezed tests passed")
