"""
Hybrid integration with the official (compiled) TrimCI C++ backend.

What plugs in, and what doesn't (the honest integration surface — see TODO.md):

  * The official backend's GRAPH SELECTION — `pool_build`, `run_trim`,
    `run_expansion`, `generate_excitations`, and the fast matrix-free Davidson
    `davidson_solve_matfree` — is hardwired to *fermionic Slater-Condon over
    (Determinant bitstrings, h1, eri)*. There is NO custom-H_ij / provider hook,
    so we cannot offload our mixed-state selection to it without a C++ fork.

  * BUT the high-performance C++ Davidson eigensolver IS injectable through one
    entry point: `trimci_core.fast_expansion.davidson_solve_dense(H, diag, ...)`
    accepts an ARBITRARY matrix (not h1/eri). That is our hook — it lets us run
    the official optimized Davidson on the projected Hamiltonian of any selected
    determinant set, i.e. offload the O(N_dets^3) dense-`eigh` diagonalization
    that is our current bottleneck.

  * Caveat: `davidson_solve_dense` is REAL-symmetric only; our H is complex
    Hermitian (the Weinberg-Tomozawa term carries an i). We use the standard
    2N real-symmetric embedding
        M = [[Re H, -Im H], [Im H, Re H]]
    whose lowest eigenvalue equals H's (its spectrum is H's, doubled) and from
    whose real eigenvector [x; y] the complex eigenvector is v = x + i y.

So this module is a HYBRID: our Python expansion/trim for selection + the
official C++ Davidson for diagonalization. It is the achievable "modular"
integration today; full offload of the selection needs the Otten group's
mixed-mode generalization (or a provider-interface fork). See `00_literature_review.md` §6.4.
"""

from __future__ import annotations

import numpy as np

from .hij import build_dense

# Below this complex dimension we just use numpy.eigh — Davidson on a tiny
# (2N) matrix is pointless and its subspace params can exceed the dimension.
_SMALL_N = 16

_FE = None  # cached trimci_core.fast_expansion module


def load_backend():
    """Import the compiled backend; raise a clear error if it isn't built."""
    global _FE
    if _FE is None:
        try:
            from trimci import trimci_core
        except ImportError as e:
            raise ImportError(
                "Official TrimCI not installed. Build it into the venv:\n"
                "  .venv/bin/python -m ensurepip --upgrade   # if pip is missing\n"
                "  .venv/bin/python -m pip install /path/to/TrimCI"
            ) from e
        _FE = trimci_core.fast_expansion
    return _FE


def backend_available():
    try:
        load_backend()
        return True
    except ImportError:
        return False


def _real_embed(H):
    """Complex Hermitian H (N×N) → real symmetric M (2N×2N) with the same spectrum
    (each eigenvalue appears once in H, twice in M)."""
    N = H.shape[0]
    A = np.ascontiguousarray(H.real)
    B = np.ascontiguousarray(H.imag)
    M = np.empty((2 * N, 2 * N), dtype=np.float64)
    M[:N, :N] = A
    M[:N, N:] = -B
    M[N:, :N] = B
    M[N:, N:] = A
    return M


def davidson_lowest(H, tol=1e-10):
    """Lowest eigenpair of complex-Hermitian H via the official C++ Davidson.

    Returns (E0: float, v: complex ndarray length N). Falls back to numpy.eigh
    for small matrices, where Davidson has no advantage.
    """
    H = np.asarray(H)
    N = H.shape[0]
    if N < _SMALL_N:
        evals, evecs = np.linalg.eigh(H)
        return float(evals[0].real), evecs[:, 0]

    fe = load_backend()
    is_complex = np.iscomplexobj(H) and np.abs(H.imag).max() > 1e-14
    if is_complex:
        M = _real_embed(H)                       # 2N real symmetric
    else:
        M = np.ascontiguousarray(H.real)
    diag = np.ascontiguousarray(np.diag(M))
    p = fe.DavidsonParams()
    p.n_states = 1
    p.energy_tol = tol
    p.residual_tol = tol   # else stops at the loose 1e-7 default (review #1)
    p.verbose = 0          # silence per-solve stdout in the hot loop (review #2)
    res = fe.davidson_solve_dense(np.ascontiguousarray(M), diag, p,
                                  np.zeros((0, 0), dtype=np.float64))
    E0 = float(res.eigenvalues[0])
    w = np.asarray(res.eigenvectors)[0]          # (2N,) or (N,)
    if is_complex:
        v = w[:N] + 1j * w[N:]                    # [x; y] -> x + i y
    else:
        v = w.astype(complex)
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    return E0, v


def backend_diagonalize(H_op, states):
    """`graph.diagonalize`-compatible drop-in that uses the official C++ Davidson.

    Builds the projected (complex Hermitian) matrix of `states` via our mixed
    H_ij (`build_dense`), then diagonalizes it with the backend. Returns
    (E0, {state: amplitude}). Forms the dense matrix — use
    `backend_diagonalize_sparse` past a few thousand states.
    """
    states = list(states)
    if not states:
        return float("inf"), {}
    M = build_dense(H_op, states)
    M = 0.5 * (M + M.conj().T)
    E0, v = davidson_lowest(M)
    coeffs = {s: v[k] for k, s in enumerate(states)}
    return E0, coeffs


# ---------------------------------------------------------------------------
#  Sparse path — requires the Tier-1 fork (davidson_solve_sparse). O(nnz)
#  memory: builds the projected H directly as a scipy CSR via `connections`
#  (never forming the dense matrix) and runs the official C++ sparse Davidson.
#  See classical/trimci/backend_fork/.
# ---------------------------------------------------------------------------

def has_sparse_davidson():
    return backend_available() and hasattr(load_backend(), "davidson_solve_sparse")


def _build_sparse(H_op, states, index):
    """Projected complex-Hermitian H over `states` as a scipy CSR (no dense matrix)."""
    import scipy.sparse as sp
    from .hij import connections
    rows, cols, data = [], [], []
    for j, sj in enumerate(states):
        for si, val in connections(H_op, sj).items():
            i = index.get(si)
            if i is not None:
                rows.append(i)
                cols.append(j)
                data.append(val)
    N = len(states)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=complex)


def _real_embed_sparse(Hs):
    """Complex Hermitian CSR → real symmetric 2N CSR with the same spectrum."""
    import scipy.sparse as sp
    A, B = Hs.real, Hs.imag
    return sp.bmat([[A, -B], [B, A]], format="csr")


def davidson_lowest_sparse(Hs, tol=1e-10):
    """Lowest eigenpair of complex-Hermitian H (scipy sparse) via the official
    C++ SPARSE Davidson. Returns (E0, complex eigenvector length N)."""
    import scipy.sparse as sp
    fe = load_backend()
    if not hasattr(fe, "davidson_solve_sparse"):
        raise RuntimeError(
            "davidson_solve_sparse not in backend — apply + rebuild the Tier-1 fork "
            "(classical/trimci/backend_fork/)."
        )
    Hs = sp.csr_matrix(Hs)
    N = Hs.shape[0]
    if N < _SMALL_N:
        return davidson_lowest(Hs.toarray())
    is_complex = np.iscomplexobj(Hs.data) and np.abs(Hs.data.imag).max() > 1e-14
    M = _real_embed_sparse(Hs) if is_complex else Hs.real.tocsr()
    M.sort_indices()
    diag = np.ascontiguousarray(M.diagonal(), dtype=np.float64)
    p = fe.DavidsonParams()
    p.n_states = 1
    p.energy_tol = tol
    p.residual_tol = tol   # else stops at the loose 1e-7 default (review #1)
    p.verbose = 0          # silence per-solve stdout in the hot loop (review #2)
    res = fe.davidson_solve_sparse(
        np.ascontiguousarray(M.data, dtype=np.float64),
        np.ascontiguousarray(M.indices, dtype=np.int32),
        np.ascontiguousarray(M.indptr, dtype=np.int32),
        diag, p, np.zeros((0, 0), dtype=np.float64),
    )
    E0 = float(res.eigenvalues[0])
    w = np.asarray(res.eigenvectors)[0]
    v = (w[:N] + 1j * w[N:]) if is_complex else w.astype(complex)
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    return E0, v


def backend_diagonalize_sparse(H_op, states):
    """`graph.diagonalize`-compatible drop-in: sparse build + official C++ sparse
    Davidson. O(nnz) memory — the scaling path. Returns (E0, {state: amplitude})."""
    states = list(states)
    if not states:
        return float("inf"), {}
    index = {s: k for k, s in enumerate(states)}
    Hs = _build_sparse(H_op, states, index)
    Hs = (Hs + Hs.getH()) * 0.5
    E0, v = davidson_lowest_sparse(Hs)
    return E0, {s: v[k] for k, s in enumerate(states)}


# ---------------------------------------------------------------------------
#  Full C++ hot-path — the Tier-2 connections port (mixed_ci) doing the matrix
#  build (`build_coo`) and the expansion (`expand`) in C++, plus the official
#  C++ sparse Davidson. Python only orchestrates ranking / set ops. This is the
#  "fork working fully" path for larger-L solves. Needs the standalone module
#  built (classical/trimci/backend_fork/build_mixed_ci.sh).
# ---------------------------------------------------------------------------

_CPP = None


def _load_cpp():
    global _CPP
    if _CPP is None:
        import os
        import sys
        fork = os.path.join(os.path.dirname(__file__), "backend_fork")
        if fork not in sys.path:
            sys.path.insert(0, fork)
        try:
            import mixed_ci as _m
        except ImportError as e:
            raise ImportError(
                "mixed_ci C++ module not built — run "
                "`bash classical/trimci/backend_fork/build_mixed_ci.sh` "
                "(needs `uv pip install pybind11`)."
            ) from e
        _CPP = _m
    return _CPP


def cpp_available():
    try:
        _load_cpp()
        return has_sparse_davidson()
    except ImportError:
        return False


def _cpp_provider(H):
    """Build + cache the C++ MixedProvider on the Hamiltonian object."""
    if H.N_f > 65536:
        # Boson occupations are passed as uint16 to C++. N_f > 65536 (n_b > 16)
        # would wrap silently — use the pure-Python path. (In practice the
        # physical cutoff is n_b ~ 5-11, far below this; see _states_to_arrays.)
        raise ValueError(
            f"C++ path supports N_f <= 65536 (n_b <= 16); got N_f={H.N_f}. "
            "Use the pure-Python solver for larger cutoffs."
        )
    prov = getattr(H, "_cpp_provider", None)
    if prov is None:
        mod = _load_cpp()
        terms = [(complex(t.coeff),
                  [(int(m), int(a)) for (m, a) in t.ferm_ops],
                  [(int(m), int(a)) for (m, a) in t.bos_ops])
                 for t in H.terms]
        cache_cap = int(getattr(H, "_cpp_cache_cap", 100000))
        # RAM safeguard: bound the C++ connection cache by a byte budget. Default
        # 128 MiB — MEASURED (2026-07-02, L=2 3d, 50k core): dropping the budget
        # from 1 GiB to 64 MiB cut peak RSS 3.47 GB -> 1.44 GB (2.4x) with NO time
        # change and an identical energy. The cache THRASHES at large core (each
        # cached state holds a Hamiltonian-sized ConnMap, so 1 GiB fills and clears
        # wholesale without earning reuse) — connections aren't the bottleneck, so
        # a big cache is pure RAM waste. Raise H._cpp_cache_bytes only if a profile
        # shows connection recompute dominating; set 0 to disable the byte bound.
        cache_bytes = int(getattr(H, "_cpp_cache_bytes", 128 << 20))
        prov = mod.MixedProvider(terms, H.N_f, cache_cap, cache_bytes)
        H._cpp_provider = prov
    return prov


_MASK64 = (1 << 64) - 1


def _ferm_words(H):
    """Number of 64-bit words needed to hold H's fermion bitmask (>=1)."""
    return max(1, (H.n_ferm_modes + 63) // 64)


def _states_to_arrays(states, n_words):
    # Fermion masks can be arbitrarily wide (e.g. L=4 in 3d => 256 modes), so we
    # ship each as `n_words` little-endian uint64 words — numpy has no native
    # >64-bit integer. The C++ side reassembles the mask (mixed_ci_pybind
    # read_ferm). n_words = ceil(n_ferm_modes/64).
    n = len(states)
    ferm = np.empty((n, n_words), dtype=np.uint64)
    for w in range(n_words):
        shift = 64 * w
        ferm[:, w] = np.fromiter(((s.ferm >> shift) & _MASK64 for s in states),
                                 dtype=np.uint64, count=n)
    bos = np.ascontiguousarray([s.bos for s in states], dtype=np.uint16)
    return ferm, bos


def _diagonalize_arrays_davidson(H, ferm, bos):
    """Lowest eigenpair of H projected onto the states (ferm, bos) arrays, via
    the C++ 2N real-embedded build + official C++ sparse Davidson. Array in,
    (E0, complex eigenvector) out — the array-native core shared by
    `cpp_diagonalize` and the small-N branch of `cpp_diagonalize_matfree_arrays`.
    Fast + robust at small-to-moderate N; the 2N embedding doubles memory, so
    very large N should use the matrix-free eigsh path instead."""
    import scipy.sparse as sp
    ferm = np.ascontiguousarray(ferm, dtype=np.uint64)
    bos = np.ascontiguousarray(bos, dtype=np.uint16)
    N = ferm.shape[0]
    prov = _cpp_provider(H)

    if 2 * N < _SMALL_N:               # tiny: dense numpy is faster than Davidson
        rows, cols, re, im = prov.build_coo(ferm, bos)
        Hc = sp.csr_matrix((np.asarray(re) + 1j * np.asarray(im),
                            (np.asarray(rows), np.asarray(cols))),
                           shape=(N, N), dtype=complex).toarray()
        evals, evecs = np.linalg.eigh(0.5 * (Hc + Hc.conj().T))
        return float(evals[0].real), evecs[:, 0]

    # 2N real embedding built in C++; scipy assembles the CSR (its COO->CSR sort
    # gives Davidson cache-friendly sorted indices — faster than an unsorted
    # direct CSC). No complex CSR / Hermitization / bmat in Python.
    rows, cols, data, n2 = prov.build_real_embedded_coo(ferm, bos)
    M = sp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))),
                      shape=(n2, n2))
    fe = load_backend()
    p = fe.DavidsonParams()
    p.n_states = 1
    p.energy_tol = 1e-10
    p.residual_tol = 1e-10
    p.verbose = 0
    res = fe.davidson_solve_sparse(
        np.ascontiguousarray(M.data, dtype=np.float64),
        np.ascontiguousarray(M.indices, dtype=np.int32),
        np.ascontiguousarray(M.indptr, dtype=np.int32),
        np.ascontiguousarray(M.diagonal(), dtype=np.float64),
        p, np.zeros((0, 0), dtype=np.float64))
    E0 = float(res.eigenvalues[0])
    w = np.asarray(res.eigenvectors)[0]
    v = w[:N] + 1j * w[N:]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    return E0, v


def cpp_diagonalize(H, states):
    """`graph.diagonalize` via the C++ real-embedded matrix build + official C++
    sparse Davidson. The 2N real embedding of the complex-Hermitian H is built
    in C++ (no complex CSR / Hermitization / bmat in Python), then a single
    real sparse Davidson solve; the complex eigenvector is reconstructed."""
    states = list(states)
    if not states:
        return float("inf"), {}
    ferm, bos = _states_to_arrays(states, _ferm_words(H))
    E0, v = _diagonalize_arrays_davidson(H, ferm, bos)
    return E0, {s: v[k] for k, s in enumerate(states)}


def cpp_expand(H, core_coeffs, pool_factor=10):
    """`graph.expand` via the C++ neighbor-generation + scoring. Returns a pool set."""
    from .state import MixedState
    core = list(core_coeffs)
    if not core:
        return set()
    prov = _cpp_provider(H)
    ferm, bos = _states_to_arrays(core, _ferm_words(H))
    coeffs = np.array([core_coeffs[s] for s in core], dtype=complex)
    cf, cb, sc = prov.expand(ferm, bos,
                             np.ascontiguousarray(coeffs.real),
                             np.ascontiguousarray(coeffs.imag))
    sc = np.asarray(sc)
    if sc.size == 0:
        return set(core)
    keep = max(pool_factor * len(core), 1)
    idx = (np.argpartition(sc, -keep)[-keep:] if keep < sc.size
           else np.arange(sc.size))
    # .tolist() converts to native Python ints in C — avoids the per-element
    # int(x) comprehension in the hot expansion loop (review #6). cf is (M, W)
    # uint64 words; reassemble the arbitrary-width fermion mask as a Python int.
    cf_sel = np.asarray(cf)[idx].tolist()
    cb_sel = np.asarray(cb)[idx].tolist()
    pool = set(core)
    for words, brow in zip(cf_sel, cb_sel):
        f = 0
        for w, word in enumerate(words):
            f |= word << (64 * w)
        pool.add(MixedState(f, tuple(brow)))
    return pool


# ---------------------------------------------------------------------------
#  Matrix-free eigensolver — SubspaceContext + scipy eigsh.
#
#  Instead of building a 2N real-embedded sparse matrix (build_real_embedded_coo
#  -> scipy COO->CSR -> davidson_solve_sparse), we:
#   1. Call prov.build_context(ferm, bos): builds the complex CSC of the projected
#      H entirely in C++, populating the provider cache for all N states. O(N*k_conn)
#      work, O(N*k_conn) memory, ~same as the matrix build but:
#        * NO 2x real embedding  ->  2x fewer entries, 2x less memory
#        * NO scipy COO->CSR assembly  ->  removes ~30% of previous hot-path time
#        * CSC reused across ALL eigsh iterations (no per-call rebuild)
#   2. Wrap ctx.matvec in a scipy LinearOperator and call eigsh(k=1, which='SA').
#      Each eigsh iteration calls the C++ CSC matvec directly (pure array loops,
#      no hash lookups at matvec time).
#
#  Break-even vs sparse Davidson (build_real_embedded_coo path): roughly N~2000
#  (below that the matrix build is fast; above, the 2x memory savings and no-scipy
#  cost dominate). The threshold _MATFREE_N controls the crossover.
#
#  For eigsh convergence problems (rare; extremely ill-conditioned spectra), the
#  function falls back automatically to cpp_diagonalize.
# ---------------------------------------------------------------------------

_MATFREE_N = 2000   # switch to eigsh+CSC above this subspace size


def cpp_diagonalize_matfree(H, states, tol=1e-10):
    """Lowest eigenpair of the projected H over `states` via C++ CSC + scipy eigsh.

    Builds the complex CSC of H|_states once in C++ (SubspaceContext), then runs
    scipy.sparse.linalg.eigsh with a LinearOperator wrapping the C++ CSC matvec.
    No 2N real embedding, no scipy COO->CSR conversion. Falls back to
    cpp_diagonalize for small N or if eigsh fails to converge.

    This is the matrix-free path: O(N*k_conn) memory for the CSC (vs 2x that for
    the real-embedded matrix), and the CSC is built once and reused for all
    eigsh Lanczos iterations.
    """
    from scipy.sparse.linalg import eigsh, LinearOperator

    states = list(states)
    N = len(states)
    if N == 0:
        return float("inf"), {}
    if N < _MATFREE_N:
        return cpp_diagonalize(H, states)

    prov = _cpp_provider(H)
    ferm, bos = _states_to_arrays(states, _ferm_words(H))

    # Build complex CSC once — populates provider cache for all N states.
    # Subsequent matvec calls iterate CSC arrays (no hash lookups).
    ctx = prov.build_context(ferm, bos)

    def _mv(v):
        v = np.asarray(v, dtype=complex)
        vr = np.ascontiguousarray(v.real, dtype=np.float64)
        vi = np.ascontiguousarray(v.imag, dtype=np.float64)
        or_, oi_ = ctx.matvec(vr, vi)
        return np.asarray(or_) + 1j * np.asarray(oi_)

    op = LinearOperator((N, N), matvec=_mv, dtype=complex)

    # Seed eigsh with the lowest-diagonal-entry unit vector (cheap, usually
    # close to GS for Hamiltonian systems and reduces Lanczos iterations).
    diag = np.asarray(ctx.diagonal(), dtype=np.float64)
    v0 = np.zeros(N, dtype=complex)
    v0[int(np.argmin(diag))] = 1.0

    try:
        vals, vecs = eigsh(op, k=1, which="SA", tol=tol, v0=v0,
                           maxiter=max(300, N))
    except Exception:
        # Fall back to sparse Davidson (e.g. ARPACK convergence failure).
        return cpp_diagonalize(H, states)

    E0 = float(vals[0].real)
    v = vecs[:, 0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    return E0, {s: v[k] for k, s in enumerate(states)}


def cpp_diagonalize_smart(H, states):
    """Auto-select between sparse Davidson and matfree eigsh based on N.

    For N < _MATFREE_N: cpp_diagonalize (sparse Davidson, faster for small N).
    For N >= _MATFREE_N: cpp_diagonalize_matfree (C++ CSC + eigsh, 2x less memory,
    no scipy assembly overhead, scales to 100k+ states).
    """
    N = len(states) if not isinstance(states, int) else states
    if N < _MATFREE_N:
        return cpp_diagonalize(H, states)
    return cpp_diagonalize_matfree(H, states)


# ---------------------------------------------------------------------------
#  Tier 2 — ARRAY-NATIVE hot path. States live as compact numpy arrays
#  (ferm (N,W) uint64, bos (N,n_bos) uint16, coeffs (N,) complex) end-to-end,
#  never as Python MixedState objects. This removes the RAM wall: a MixedState's
#  boson tuple alone is ~n_bos*8 + 56 bytes (3 KB/state at L=5 in 3d), and the
#  per-round pool holds ~pool_factor x more of them; the array form is
#  n_bos*2 + W*8 bytes/state contiguous (no per-object / GC / set overhead), and
#  the pool never leaves C++/numpy. See graph_arrays.py for the loop.
#
#  These take/return arrays directly (no MixedState round-trip) so the whole
#  expand -> trim -> diagonalize cycle stays in the compact representation.
# ---------------------------------------------------------------------------

def cpp_expand_topk(H, ferm, bos, coeffs, keep):
    """Top-`keep` scored expansion candidates (array in, array out).

    Returns (cand_ferm (K,W) uint64, cand_bos (K,n_bos) uint16, scores (K,)).
    The candidates are unique and disjoint from the core (guaranteed in C++), so
    the caller can concatenate them onto the core arrays with no deduplication.
    Selection (top-`keep` by |H_ij c_j|) happens in C++ — only `keep` rows cross
    the boundary, not the (potentially millions of) full candidate set.
    """
    prov = _cpp_provider(H)
    coeffs = np.ascontiguousarray(coeffs, dtype=complex)
    cf, cb, sc = prov.expand_topk(
        np.ascontiguousarray(ferm, dtype=np.uint64),
        np.ascontiguousarray(bos, dtype=np.uint16),
        np.ascontiguousarray(coeffs.real),
        np.ascontiguousarray(coeffs.imag),
        int(keep))
    return np.asarray(cf), np.asarray(cb), np.asarray(sc)


def cpp_pt2_external(H, states, coeffs):
    """Epstein-Nesbet PT2 pass-1 in C++ (the array-marshaling half of the port).

    `states` is the variational space V (list[MixedState]); `coeffs` a complex
    ndarray of the NORMALIZED amplitudes aligned to `states` (as prepared by
    epstein_nesbet_pt2 after re-diagonalization). Runs the C++
    `MixedProvider.pt2_accumulate`, which does the one-pass connections sweep and
    the diagonal H_aa evaluation, and returns per-external-determinant arrays for
    the vectorized reduction on the Python side.

    Returns (amp_re, amp_im, Haa, e_ray_re, e_ray_im):
      amp_re, amp_im : Re/Im of A_a = <a|H|psi> per distinct external state (M,)
      Haa            : Epstein-Nesbet diagonal <a|H|a> (real) per external (M,)
      e_ray_re/_im   : the internal Rayleigh quotient <psi|H|psi> (imag ~ 0)
    """
    prov = _cpp_provider(H)                       # raises if N_f > 65536
    ferm, bos = _states_to_arrays(states, _ferm_words(H))
    coeffs = np.ascontiguousarray(coeffs, dtype=complex)
    amp_re, amp_im, Haa, e_ray_re, e_ray_im = prov.pt2_accumulate(
        ferm, bos,
        np.ascontiguousarray(coeffs.real),
        np.ascontiguousarray(coeffs.imag),
    )
    return (np.asarray(amp_re), np.asarray(amp_im), np.asarray(Haa),
            float(e_ray_re), float(e_ray_im))


def cpp_diagonalize_matfree_arrays(H, ferm, bos, tol=1e-8):
    """Lowest eigenpair of H projected onto the states (ferm, bos) — array in,
    coeffs-array out (aligned to the input row order).

    Returns (E0: float, coeffs: complex ndarray length N). Small N goes through a
    dense complex build (build_coo -> eigh); large N through the C++ CSC context
    + scipy eigsh (matrix-free). This is the array-native twin of
    cpp_diagonalize_matfree — no MixedState objects, no dict keyed by state.

    eigsh is seeded from the lowest-diagonal unit vector (NOT from a prior round's
    eigenvector: a warm start was tried and dropped — it did not speed ARPACK's
    smallest-algebraic mode on this wide spectrum, and biasing the solve subtly
    perturbed the trimmed set, cutting the exploration the random-init ensemble
    relies on). `tol` is the eigsh relative eigenvalue tolerance (1e-8 ≈ 2e-5 MeV
    here — far below the selected-CI convergence uncertainty).
    """
    ferm = np.ascontiguousarray(ferm, dtype=np.uint64)
    bos = np.ascontiguousarray(bos, dtype=np.uint16)
    N = ferm.shape[0]
    if N == 0:
        return float("inf"), np.zeros(0, dtype=complex)
    prov = _cpp_provider(H)

    if N < _MATFREE_N:
        # Small N: the fast official sparse Davidson over the 2N real embedding
        # (same primitive the object path uses) — the 2N memory cost is trivial
        # here, and it's far faster than dense eigh. Only above _MATFREE_N does
        # the 2N embedding's memory matter enough to switch to matrix-free eigsh.
        return _diagonalize_arrays_davidson(H, ferm, bos)

    from scipy.sparse.linalg import eigsh, LinearOperator
    ctx = prov.build_context(ferm, bos)

    def _mv(v):
        v = np.asarray(v, dtype=complex)
        vr = np.ascontiguousarray(v.real, dtype=np.float64)
        vi = np.ascontiguousarray(v.imag, dtype=np.float64)
        or_, oi_ = ctx.matvec(vr, vi)
        return np.asarray(or_) + 1j * np.asarray(oi_)

    op = LinearOperator((N, N), matvec=_mv, dtype=complex)
    diag = np.asarray(ctx.diagonal(), dtype=np.float64)
    v0 = np.zeros(N, dtype=complex)
    v0[int(np.argmin(diag))] = 1.0
    try:
        vals, vecs = eigsh(op, k=1, which="SA", tol=tol, v0=v0,
                           maxiter=max(300, N))
    except Exception:
        # Fall back to the sparse Davidson build on ARPACK convergence failure.
        return _diagonalize_arrays_davidson(H, ferm, bos)

    E0 = float(vals[0].real)
    v = vecs[:, 0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v = v / nrm
    return E0, v


def cpp_ground_state(H, n_elec, **kw):
    """TrimCI ground state with the entire hot path (build + expand + Davidson)
    in C++. Convenience wrapper over `graph.ground_state`."""
    from .graph import ground_state
    return ground_state(H, n_elec, diag_fn=cpp_diagonalize_smart,
                        expand_fn=cpp_expand, **kw)


def cpp_ground_state_ensemble(H, n_elec, **kw):
    from .graph import ground_state_ensemble
    return ground_state_ensemble(H, n_elec, diag_fn=cpp_diagonalize_smart,
                                 expand_fn=cpp_expand, **kw)


def cpp_ground_state_arrays(H, n_elec, **kw):
    """Array-native TrimCI ground state (Tier 2): states stay as compact numpy
    arrays throughout — the scaling path toward 1e6 states. See graph_arrays."""
    from .graph_arrays import ground_state_arrays
    return ground_state_arrays(H, n_elec, **kw)


def cpp_ground_state_ensemble_arrays(H, n_elec, **kw):
    """Array-native ensemble TrimCI (Tier 2)."""
    from .graph_arrays import ground_state_ensemble_arrays
    return ground_state_ensemble_arrays(H, n_elec, **kw)
