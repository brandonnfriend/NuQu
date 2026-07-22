"""
Mixed-state H_ij evaluator.

Computes matrix elements <Phi_i | H | Phi_j> of the dynamical-pion EFT
between mixed fermion-boson Fock states (`state.MixedState`), where H is a
list of elementary ladder-operator terms (`hamiltonian.MixedH`).

Strategy: apply-operator-to-ket. Each term is a product of fermion and
boson ladder ops acting on disjoint registers, so a term maps a basis ket
to (amplitude) x (single basis ket) or to zero. Summing over terms gives,
for a fixed column |Phi_j>, the full set of connected rows {Phi_i: H_ij} —
which is exactly the "generate excitations + matrix elements" primitive
the TrimCI expansion step consumes (`graph.py`).

Fermion signs follow the OpenFermion convention (modes ordered ascending;
applying a (c)reation/(a)nnihilation at mode p carries parity = number of
occupied modes with index < p). This makes the fermion sector match
`openfermion.get_sparse_operator` up to basis relabeling (validated in
tests/test_hij.py against the spectrum).

Boson ladder ops use a|n> = sqrt(n)|n-1>, a_dag|n> = sqrt(n+1)|n+1>, with
matrix elements that would leave [0, N_f-1] dropped (the Fock cutoff).
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from .state import MixedState

# Hard cap on the dense matrix `build_dense` will allocate. A complex128 NxN
# matrix is 16*N**2 bytes and np.linalg.eigh roughly triples peak memory, so
# 6000 -> ~0.6 GB matrix + workspace. Past this, use the sparse Lanczos path
# (`graph.lanczos_ground_state`). This guard exists to prevent the OOM crashes
# that an unguarded dense build can trigger on realistic sizes.
MAX_DENSE = 6_000

# Per-Hamiltonian connection cache cap (entries). connections() is a pure
# function of `state` for a fixed H, so caching pays off across the many
# repeated diagonalize() calls. Bounded + cleared-on-overflow so it can never
# grow without limit (another OOM guard).
_CONN_CACHE_CAP = 200_000


def _apply_fermion_ops(occ, ops):
    """Apply fermion ladder ops (right-to-left) to bitmask `occ`.

    ops: tuple of (mode, action), action 1=create, 0=annihilate.
    Returns (sign, new_occ) or None if annihilated.
    """
    sign = 1
    for (p, action) in reversed(ops):
        bit = (occ >> p) & 1
        if action == 0:           # annihilation a_p
            if not bit:
                return None
            # parity = occupied modes with index < p
            below = occ & ((1 << p) - 1)
            if below.bit_count() & 1:
                sign = -sign
            occ &= ~(1 << p)
        else:                     # creation a_p^dagger
            if bit:
                return None
            below = occ & ((1 << p) - 1)
            if below.bit_count() & 1:
                sign = -sign
            occ |= (1 << p)
    return sign, occ


def _apply_boson_ops(bos, ops, N_f):
    """Apply boson ladder ops (right-to-left) to occupation tuple `bos`.

    Returns (amplitude, new_bos) or None if annihilated / pushed past cutoff.
    """
    bos = list(bos)
    amp = 1.0
    for (m, action) in reversed(ops):
        n = bos[m]
        if action == 0:           # annihilation a_m
            if n == 0:
                return None
            amp *= sqrt(n)
            bos[m] = n - 1
        else:                     # creation a_m^dagger
            if n + 1 >= N_f:       # would leave the cutoff box -> truncate
                return None
            amp *= sqrt(n + 1)
            bos[m] = n + 1
    return amp, tuple(bos)


def apply_term(term, state, N_f):
    """Apply one OperatorTerm to a MixedState ket.

    Returns (amplitude, MixedState) or None.
    """
    fres = _apply_fermion_ops(state.ferm, term.ferm_ops)
    if fres is None:
        return None
    sign, new_ferm = fres
    bres = _apply_boson_ops(state.bos, term.bos_ops, N_f)
    if bres is None:
        return None
    bamp, new_bos = bres
    amp = term.coeff * sign * bamp
    return amp, MixedState(new_ferm, new_bos)


def connections_nocache(H, state):
    """Connections of |state> WITHOUT touching the cache.

    Used by bulk one-pass builders (e.g. the Lanczos sparse-matrix assembly)
    that visit every state exactly once, where caching would only waste memory.
    """
    N_f = H.N_f
    out = {}
    for term in H.terms:
        res = apply_term(term, state, N_f)
        if res is None:
            continue
        amp, s2 = res
        out[s2] = out.get(s2, 0.0 + 0.0j) + amp
    return {s: v for s, v in out.items() if abs(v) > 1e-14}


def connections(H, state):
    """All rows connected to column |state>: dict{MixedState: H_row,state}.

    Includes the diagonal; near-zero (cancelling) entries are pruned. This is
    the matrix-free column oracle used by the TrimCI expansion + diagonalize.

    Memoized per-H: connections(H, state) is a pure function of `state` for a
    fixed (immutable) H, and the same states recur constantly across expand /
    local_trim / global_trim / ensemble runs. The cache is bounded
    (`_CONN_CACHE_CAP`) and cleared on overflow so it cannot leak.
    """
    cache = getattr(H, "_conn_cache", None)
    if cache is None:
        cache = {}
        H._conn_cache = cache
    hit = cache.get(state)
    if hit is not None:
        return hit
    result = connections_nocache(H, state)
    if len(cache) >= _CONN_CACHE_CAP:
        cache.clear()
    cache[state] = result
    return result


def h_ij(H, state_i, state_j):
    """Single matrix element <state_i | H | state_j>."""
    return connections(H, state_j).get(state_i, 0.0 + 0.0j)


def build_dense(H, basis, max_dense=MAX_DENSE):
    """Dense Hermitian matrix of H over an explicit `basis` (list[MixedState]).

    Toy / exact-diagonalization use only. Returns a complex ndarray.

    Guarded against OOM: refuses to allocate an N x N complex matrix past
    `max_dense` states. For larger systems use `graph.lanczos_ground_state`,
    which never forms the dense matrix.
    """
    n = len(basis)
    if n > max_dense:
        raise MemoryError(
            f"build_dense: {n} states -> {16 * n * n / 1e9:.2f} GB dense "
            f"complex matrix (cap {max_dense}). Use the sparse Lanczos path "
            f"(graph.lanczos_ground_state) for this size, or raise max_dense "
            f"explicitly if you have the RAM."
        )
    index = {s: k for k, s in enumerate(basis)}
    M = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(basis):
        for si, val in connections(H, sj).items():
            i = index.get(si)
            if i is not None:
                M[i, j] += val
    return M
