"""Validate the TRUE warm-start fidelity primitive `frame_qpe.warmstart_fidelity`.

The sparse–sparse bilinear `Σ g*_a U_ab ψ̃_b` (per-mode squeeze matrix elements, no fan-out)
must equal the dense `⟨g|expm(G)|ψ̃⟩` on a tiny controlled system — this pins the convention
against `frame.squeeze_generator_terms` and confirms tractability without the dense state.

Run: `python -m pytest -q tests/test_warmstart_fidelity.py`  (or `python -m tests.test_warmstart_fidelity`).
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.hamiltonian import MixedH
from classical.trimci.hij import build_dense
from classical.trimci.state import enumerate_basis
from classical.trimci.frame import squeeze_generator_terms
from classical.trimci.frame_qpe import warmstart_fidelity, warmstart_overlap

# Tiny controlled shape: 2 fermion modes, 2 boson modes, N_f=4 (nontrivial squeeze), 1 electron.
N_FERM, N_BOS, N_F, N_ELEC = 2, 2, 4, 1
R = np.array([0.4, 0.3])           # per-mode squeeze amplitudes (nonzero -> nontrivial U)
PHI = np.array([0.0, 0.7])


def _setup():
    H = MixedH([], N_FERM, N_BOS, N_F)                       # shell: only shape is used
    basis = enumerate_basis(N_FERM, N_BOS, N_F, N_ELEC)
    G = build_dense(squeeze_generator_terms(H, R, PHI), basis)
    U = expm(G)                                             # the exact truncated squeeze unitary
    return basis, U


def _sparse(basis, vec, tol=1e-14):
    return {basis[i]: complex(vec[i]) for i in range(len(basis)) if abs(vec[i]) > tol}


def test_sparse_bilinear_matches_dense():
    """Random g, ψ̃: the sparse fidelity's overlap and p0 equal the dense expm result."""
    basis, U = _setup()
    rng = np.random.default_rng(0)
    g = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
    psi = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
    overlap_dense = np.vdot(g, U @ psi)                     # ⟨g|U|ψ⟩
    p0_dense = abs(overlap_dense) ** 2 / (np.vdot(g, g).real * np.vdot(psi, psi).real)

    res = warmstart_fidelity(_sparse(basis, g), _sparse(basis, psi), R, PHI, N_F, N_BOS)
    assert abs(res["overlap"] - overlap_dense) < 1e-10, \
        f"sparse overlap {res['overlap']} != dense {overlap_dense}"
    assert abs(res["p0"] - p0_dense) < 1e-12, f"p0 {res['p0']} != dense {p0_dense}"


def test_exact_framed_core_gives_unit_fidelity():
    """Preparing the EXACT framed core ψ̃ = U†|g⟩ recovers |g⟩ -> p0 = 1 (U unitary here)."""
    basis, U = _setup()
    rng = np.random.default_rng(1)
    g = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
    psi = U.conj().T @ g                                    # U|ψ̃⟩ = U U†|g⟩ = |g⟩
    res = warmstart_fidelity(_sparse(basis, g), _sparse(basis, psi), R, PHI, N_F, N_BOS)
    assert abs(res["p0"] - 1.0) < 1e-9, f"expected p0=1 for exact framed core, got {res['p0']}"


def test_full_core_beats_single_determinant():
    """The tightening: loading the FULL core beats the crude single-dominant-det warm start."""
    basis, U = _setup()
    rng = np.random.default_rng(2)
    g = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
    psi_full = U.conj().T @ g                               # exact framed core (many dets)
    # crude proxy: keep ONLY the dominant determinant of the framed core
    k = int(np.argmax(np.abs(psi_full)))
    psi_single = np.zeros_like(psi_full); psi_single[k] = psi_full[k]

    p0_full = warmstart_fidelity(_sparse(basis, g), _sparse(basis, psi_full), R, PHI, N_F, N_BOS)["p0"]
    p0_single = warmstart_fidelity(_sparse(basis, g), _sparse(basis, psi_single), R, PHI, N_F, N_BOS)["p0"]
    # and the *legacy* crude proxy computed the OTHER way (max weight of the framed core):
    p0_legacy = warmstart_overlap(list(psi_full))
    assert p0_full > p0_single + 1e-6, f"full core ({p0_full}) should beat single det ({p0_single})"
    assert p0_full >= p0_legacy - 1e-9, "full-core fidelity should be >= the |c_dominant|^2 proxy"


if __name__ == "__main__":
    test_sparse_bilinear_matches_dense()
    test_exact_framed_core_gives_unit_fidelity()
    test_full_core_beats_single_determinant()
    print("all warmstart_fidelity checks passed")
