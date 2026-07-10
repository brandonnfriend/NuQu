"""
Sparse full-space Lanczos ground state.

The exact-diagonalization reference, scaled past dense ED. `build_dense` forms
an N x N complex matrix (O(N^2) memory) and dies around N ~ 6000. Lanczos
(`scipy.sparse.linalg.eigsh`) needs only the lowest eigenpair of a *sparse*
Hermitian matrix, so it reaches much larger N — the only O(N^2)-free reference
we have for "meaningful but still exactly checkable" sizes.

This is NOT the TrimCI selected-CI path (`graph.ground_state`): it diagonalizes
the FULL sector exactly, giving the true E0 to validate TrimCI against where
dense ED can't reach.

SAFETY (this module exists partly to *avoid* OOM crashes, so every allocation
is guarded up front):
  * the full basis size is checked before enumeration (`enumerate_basis` cap),
  * the number of nonzeros is *estimated from a sample* and the projected
    sparse-matrix memory is checked before the (expensive) build,
  * both limits are explicit kwargs with conservative defaults.

A system too large for the nnz/memory budget raises cleanly with guidance —
it never silently tries to allocate terabytes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence

from .hij import build_dense, connections_nocache
from .state import basis_size, enumerate_basis

# Defaults chosen so a run stays comfortably within a couple GB. For our H the
# connectivity is a few×10 nonzeros/column, so MAX_NNZ ~ 4×10^7 admits several
# ×10^5 states — well past dense ED's ~6000. "real N_f" (heuristic N_f=32) at a
# single site is ~1.3×10^5 states / ~4×10^6 nnz, comfortably inside this.
DEFAULT_MAX_STATES = 400_000
DEFAULT_MAX_NNZ = 40_000_000
DEFAULT_MAX_MEM_GB = 3.0

# Peak bytes per stored nonzero during the build. The matrix is held as CSC
# (int32 indices 4 + complex128 data 16 = 20 B), and the symmetrization
# transient (M + M^H) roughly doubles it — budget ~48 B/nnz to stay safe.
_BYTES_PER_NNZ = 48
# Below this many states, just use dense eigh (eigsh needs k << N and is
# pointless on tiny matrices).
_DENSE_FALLBACK_N = 64


def _grow(arr, new_cap):
    """Return a larger copy of `arr` (size new_cap), preserving its contents."""
    out = np.empty(new_cap, dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    return out


def _estimate_nnz(H, basis, sample=256):
    """Project total nonzeros from a sample of columns.

    Connectivity varies across states (interior states couple to more
    neighbors than cutoff-boundary states, which lose terms to truncation), so
    a strided sample can under/over-shoot. This estimate only sizes the initial
    buffer — the build grows it as needed up to max_nnz — so being off is cheap,
    not fatal. We still sample a healthy spread (strided + the dense low-occ
    region) to start close.
    """
    N = len(basis)
    step = max(1, N // sample)
    sampled = basis[::step][:sample]
    counts = [len(connections_nocache(H, s)) for s in sampled]
    avg = sum(counts) / len(counts)
    return int(np.ceil(avg * N)), avg


def lanczos_ground_state(H, n_elec, k=1, which="SA",
                         max_states=DEFAULT_MAX_STATES,
                         max_nnz=DEFAULT_MAX_NNZ,
                         max_mem_gb=DEFAULT_MAX_MEM_GB,
                         return_vec=False, verbose=False):
    """Lowest `k` eigenvalue(s) of H over the full A=n_elec sector via Lanczos.

    Args:
        H: MixedH.
        n_elec: nucleon-number sector A (conserved).
        k: number of lowest eigenvalues to return (1 = ground state).
        which: eigsh selection ('SA' = smallest algebraic; correct for E0).
        max_states: refuse to enumerate a basis larger than this.
        max_nnz: refuse to build a sparse matrix with more nonzeros than this.
        max_mem_gb: refuse if the projected CSR matrix would exceed this.
        return_vec: also return the ground-state vector as {state: amplitude}.

    Returns:
        (E0, info) where info has n_states, nnz, and (if k>1) all_evals; or
        (E0, coeffs_dict, info) when return_vec=True.

    Raises:
        MemoryError if the sector / matrix would exceed the configured budget
        (cleanly, before any large allocation).
    """
    # --- guard 1: basis size (no allocation yet) --------------------------
    size = basis_size(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec)
    if size > max_states:
        raise MemoryError(
            f"lanczos: full sector has {size:,} states (cap max_states="
            f"{max_states:,}). Reduce N_f/n_bos/n_elec, raise max_states if you "
            f"have the RAM, or use the TrimCI selected-CI path for this size."
        )
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec,
                            max_states=max_states)
    N = len(basis)
    if verbose:
        print(f"  lanczos: sector A={n_elec} has N={N:,} states")

    # --- tiny systems: dense is fine and eigsh is finicky -----------------
    if N <= max(_DENSE_FALLBACK_N, 2 * k + 1):
        M = build_dense(H, basis)
        M = 0.5 * (M + M.conj().T)
        evals, evecs = np.linalg.eigh(M)
        E0 = float(evals[0].real)
        info = {"n_states": N, "nnz": None, "method": "dense",
                "all_evals": evals[:k].real.tolist()}
        if return_vec:
            return E0, {basis[i]: evecs[i, 0] for i in range(N)}, info
        return E0, info

    # --- guard 2: project nnz + memory before building --------------------
    est_nnz, avg = _estimate_nnz(H, basis)
    proj_gb = est_nnz * _BYTES_PER_NNZ / 1e9
    if verbose:
        print(f"  lanczos: ~{avg:.0f} nonzeros/column -> est nnz {est_nnz:,} "
              f"(~{proj_gb:.2f} GB CSR)")
    if est_nnz > max_nnz:
        raise MemoryError(
            f"lanczos: estimated {est_nnz:,} nonzeros (cap max_nnz={max_nnz:,}). "
            f"Reduce the system size or raise max_nnz if you have the RAM."
        )
    if proj_gb > max_mem_gb:
        raise MemoryError(
            f"lanczos: projected sparse matrix ~{proj_gb:.2f} GB exceeds "
            f"max_mem_gb={max_mem_gb}. Reduce the system size or raise the budget."
        )

    # --- build CSC directly into a growable buffer (memory-lean + safe) ---
    # Iterating columns in order and writing (row-index, value) with a
    # per-column indptr yields CSC with no COO intermediate (~half the
    # transient memory of coo->csr). The buffer GROWS on demand if the nnz
    # estimate was low (the sample can miss high-connectivity interior states),
    # but never past the hard max_nnz ceiling — so a bad estimate costs a
    # reallocation, not a crash, and memory stays bounded by max_nnz.
    index = {s: i for i, s in enumerate(basis)}
    hard_cap = int(max_nnz)
    cap = min(max(int(est_nnz * 1.5) + N, 8 * N + 1024), hard_cap)
    indices = np.empty(cap, dtype=np.int32)   # row index of each entry
    data = np.empty(cap, dtype=np.complex128)
    indptr = np.zeros(N + 1, dtype=np.int64)
    p = 0
    for j, sj in enumerate(basis):
        conns = connections_nocache(H, sj)
        if p + len(conns) > cap:
            new_cap = min(max(int(cap * 1.7), p + len(conns)), hard_cap)
            if p + len(conns) > new_cap:
                raise MemoryError(
                    f"lanczos: nonzeros exceeded the max_nnz={max_nnz:,} "
                    f"ceiling mid-build. Reduce the system size or raise "
                    f"max_nnz deliberately if you have the RAM."
                )
            indices = _grow(indices, new_cap)
            data = _grow(data, new_cap)
            cap = new_cap
        for si, val in conns.items():
            i = index.get(si)
            if i is not None:
                indices[p] = i
                data[p] = val
                p += 1
        indptr[j + 1] = p
    M = sp.csc_matrix((data[:p].copy(), indices[:p].copy(), indptr),
                      shape=(N, N))
    del indices, data   # free the (possibly over-sized) buffers promptly
    nnz = M.nnz
    M = 0.5 * (M + M.getH())   # enforce exact Hermiticity for eigsh

    # --- Lanczos ----------------------------------------------------------
    # Our spectrum is wide (boson-number diagonal up to ~N_f*m_pi per mode),
    # so give Arnoldi a roomy subspace and retry harder on non-convergence.
    ncv = min(N - 1, max(2 * k + 1, 40))
    try:
        evals, evecs = eigsh(M, k=k, which=which, ncv=ncv)
    except ArpackNoConvergence:
        if verbose:
            print("  lanczos: eigsh retry with larger ncv / maxiter")
        evals, evecs = eigsh(M, k=k, which=which,
                             ncv=min(N - 1, ncv * 3), maxiter=10000, tol=1e-9)
    order = np.argsort(evals.real)
    evals = evals.real[order]
    evecs = evecs[:, order]
    E0 = float(evals[0])
    info = {"n_states": N, "nnz": int(nnz), "method": "lanczos",
            "all_evals": evals[:k].tolist()}
    if verbose:
        print(f"  lanczos: nnz={nnz:,}  E0={E0:.8f}")
    if return_vec:
        return E0, {basis[i]: evecs[i, 0] for i in range(N)}, info
    return E0, info
