"""
Tier-2 ARRAY-NATIVE TrimCI hot loop.

Same algorithm as `graph.ground_state` (Zhang & Otten 2025: random core ->
[expand -> local trim -> global trim] x rounds), but the core / pool / survivor
state sets live as **compact numpy arrays** the whole way through, never as
Python `MixedState` objects:

    ferm : (N, W) uint64   little-endian fermion-bitmask words  (W = ceil(modes/64))
    bos  : (N, n_bos) uint16   boson occupations per pion mode  (N_f <= 65536)
    coeffs : (N,) complex128   ground-state amplitude of each state

WHY: a `MixedState`'s boson tuple alone costs ~n_bos*8 + 56 bytes (≈3 KB/state at
L=5 in 3d), plus object/GC/set overhead; the per-round pool holds ~pool_factor×
more of them. That is the RAM wall that capped the object path near ~10^4 states.
The array form is n_bos*2 + W*8 bytes/state, contiguous, with no per-object cost,
and the (largest) pool/candidate sets never leave C++/numpy — so we scale toward
10^6 states on a laptop (10^9 on HPC).

HOW the set algebra stays cheap: the C++ `expand_topk` returns candidates that are
unique AND disjoint from the core, so the pool is just `concat(core, candidates)`
with the core occupying rows [0, N). Every subsequent "keep" / "union" is then an
operation on integer ROW INDICES into that pool array (`np.argpartition`,
`np.union1d`) — never a hash/dedup over the (heavy) state keys. Diagonalization is
the array-native matrix-free path (`backend.cpp_diagonalize_matfree_arrays`:
complex CSC built once in C++ + scipy eigsh), so no state ever becomes a dict key.

The object path in `graph.py` is kept intact as the reference / comparison switch;
this module is validated to agree with it (see tests) and is the default only in
the array drivers (`backend.cpp_ground_state_*_arrays`, `run_cpp --arrays`).
"""

from __future__ import annotations

import math

import numpy as np

from .backend import cpp_diagonalize_matfree_arrays, cpp_expand_topk, _ferm_words
from .graph import (GroundStateResult, halving_drop, random_core,
                    _sector_size)
from .state import fermion_determinants  # noqa: F401  (parity with graph API)


# ---------------------------------------------------------------------------
#  Array-native initial core
# ---------------------------------------------------------------------------

def random_core_arrays(H, n_elec, n_init, rng, boson_init_mean=0.5):
    """Sample `n_init` near-vacuum random states as (ferm (n,W), bos (n,n_bos)).

    Reuses `graph.random_core` (same truncated-geometric boson seeding, same
    vacuum anchor) and converts the small initial set to arrays once — n_init is
    tiny (~20–2000), so the one-time MixedState materialization is negligible.
    """
    from .backend import _states_to_arrays
    core = list(random_core(H, n_elec, n_init, rng, boson_init_mean=boson_init_mean))
    ferm, bos = _states_to_arrays(core, _ferm_words(H))
    return ferm, bos


# ---------------------------------------------------------------------------
#  Array-native expansion / trimming — all "keep" ops are on ROW INDICES
# ---------------------------------------------------------------------------

def expand_arrays(H, core_ferm, core_bos, coeffs, pool_factor=3):
    """Pool = core ⊕ top-(pool_factor·N) scored candidates, as arrays.

    Returns (pool_ferm, pool_bos) with the core occupying the first N rows
    (candidates after). Candidates are unique and disjoint from the core
    (guaranteed by `expand_topk` in C++), so the concatenation needs no dedup.
    """
    N = core_ferm.shape[0]
    keep = max(pool_factor * N, 1)
    cf, cb, _sc = cpp_expand_topk(H, core_ferm, core_bos, coeffs, keep)
    if cf.shape[0] == 0:
        return core_ferm, core_bos
    pool_ferm = np.concatenate([core_ferm, cf], axis=0)
    pool_bos = np.concatenate([core_bos, cb], axis=0)
    return pool_ferm, pool_bos


def local_trim_arrays(H, pool_ferm, pool_bos, num_groups, keep_per_group, rng):
    """Local trim: shuffle the pool, split into `num_groups`, diagonalize each,
    keep the top-`keep_per_group` rows (by |amplitude|) from each group.

    Returns the survivor ROW INDICES into the pool arrays. Groups are disjoint
    index slices, so the survivor indices are automatically unique (plain
    concatenation — no dedup).
    """
    P = pool_ferm.shape[0]
    if P <= num_groups:
        return np.arange(P)
    idx = np.arange(P)
    rng.shuffle(idx)
    survivors = []
    for g in range(num_groups):
        grp = idx[g::num_groups]
        if grp.size == 0:
            continue
        _E, coeffs = cpp_diagonalize_matfree_arrays(H, pool_ferm[grp], pool_bos[grp])
        amp = np.abs(coeffs)
        k = min(keep_per_group, grp.size)
        top = np.argpartition(amp, -k)[-k:] if k < grp.size else np.arange(grp.size)
        survivors.append(grp[top])
    return np.concatenate(survivors) if survivors else np.arange(P)


def global_trim_arrays(H, ferm, bos, keep):
    """Global trim: one diagonalization, keep the top-`keep` rows by |amplitude|.

    Returns (core_ferm, core_bos, core_coeffs, energy).
    """
    E0, coeffs = cpp_diagonalize_matfree_arrays(H, ferm, bos)
    amp = np.abs(coeffs)
    P = ferm.shape[0]
    k = min(keep, P)
    top = np.argpartition(amp, -k)[-k:] if k < P else np.arange(P)
    return ferm[top], bos[top], coeffs[top], E0


# ---------------------------------------------------------------------------
#  Driver — mirrors graph.ground_state (FIXED + ADAPTIVE), array-native
# ---------------------------------------------------------------------------

def ground_state_arrays(H, n_elec, n_dets=200, n_init=20, pool_factor=3,
                        num_groups=5, local_keep_ratio=4, max_rounds=12,
                        tol=1e-9, seed=None, boson_init_mean=0.5, verbose=False,
                        max_n_dets=None, conv_tol_rel=None, conv_patience=2,
                        target_gs_rel=None,
                        # accepted + ignored for drop-in parity with the object
                        # path (which threads diag_fn/expand_fn hooks):
                        diag_fn=None, expand_fn=None):
    """Array-native twin of `graph.ground_state`. Identical selection algorithm
    and stopping logic; states carried as compact arrays. `diag_fn`/`expand_fn`
    are accepted (for a uniform call signature) but ignored — the array path is
    hardwired to the C++ matrix-free diagonalizer + `expand_topk`.

    Returns a GroundStateResult whose `.ferm_arr`/`.bos_arr` carry the final core
    compactly (so io can save the wavefunction without materializing MixedStateS);
    `.states` is left empty. `.coeffs` is the amplitude array.
    """
    adaptive = conv_tol_rel is not None or target_gs_rel is not None
    ceiling = (max_n_dets if max_n_dets is not None else n_dets)
    if adaptive:
        need = int(math.ceil(math.log(max(ceiling, 2) / max(n_init, 1))
                             / math.log(1.5))) + conv_patience + 3
        max_rounds = max(max_rounds, need)

    rng = np.random.default_rng(seed)
    core_ferm, core_bos = random_core_arrays(H, n_elec, n_init, rng,
                                             boson_init_mean=boson_init_mean)
    energy, coeffs = cpp_diagonalize_matfree_arrays(H, core_ferm, core_bos)
    history = [(core_ferm.shape[0], energy)]
    target = max(n_init, 1)
    sector = _sector_size(H, n_elec)
    below = 0
    cap_round = None
    converged = False
    stop_reason = "max_rounds"

    for rnd in range(max_rounds):
        target = min(ceiling, max(target + n_init, int(np.ceil(target * 1.5))))
        N = core_ferm.shape[0]

        pool_ferm, pool_bos = expand_arrays(H, core_ferm, core_bos, coeffs,
                                            pool_factor)
        P = pool_ferm.shape[0]
        keep_per_group = max(1, (target * local_keep_ratio) // num_groups)
        # local_trim only helps when the pool is much larger than the target
        # (it cheaply prunes before the big global diagonalization). When each
        # group would keep ALL its members (keep_per_group >= max group size) it
        # is a pure no-op that still costs ~num_groups diagonalizations of the
        # whole pool — which is exactly the regime at the default pool_factor=3
        # (pool ~= (1+pf)*target, groups ~= pool/num_groups; keep_per_group ~=
        # target*local_keep_ratio/num_groups >= group size iff 1+pf <=
        # local_keep_ratio). Skip it there: global_trim then sees the identical
        # survivor set (the whole pool), so the result is unchanged — we only
        # drop the wasted prefilter diagonalization (~halving per-round cost).
        group_max = -(-P // num_groups)   # ceil(P / num_groups)
        if keep_per_group >= group_max:
            surv_idx = np.arange(P)        # local_trim would keep everything
        else:
            surv_idx = local_trim_arrays(H, pool_ferm, pool_bos, num_groups,
                                         keep_per_group, rng)
            # always keep the current core (its rows are [0, N) of the pool).
            surv_idx = np.union1d(surv_idx, np.arange(N))
        surv_ferm = pool_ferm[surv_idx]
        surv_bos = pool_bos[surv_idx]
        core_ferm, core_bos, coeffs, energy = global_trim_arrays(
            H, surv_ferm, surv_bos, target)

        history.append((core_ferm.shape[0], energy))
        dE = abs(history[-1][1] - history[-2][1])
        n_now = core_ferm.shape[0]
        at_ceiling = n_now >= min(ceiling, sector)
        if verbose:
            print(f"  round {rnd:2d}: n_dets={n_now:6d}  E={energy:.8f}  dE={dE:.2e}")

        if adaptive:
            if target_gs_rel is not None:
                drop = halving_drop(history)
                signal = (drop is not None and n_now >= 600 and drop < target_gs_rel)
            else:
                rel = dE / max(abs(history[-1][1]), 1e-12)
                signal = rel < conv_tol_rel
            below = below + 1 if signal else 0
            if below >= conv_patience:
                converged, stop_reason = True, "converged"
                break
            if at_ceiling:
                if cap_round is None:
                    cap_round = rnd
                elif rnd - cap_round >= conv_patience:
                    converged, stop_reason = bool(signal), "capped"
                    break
        else:
            if len(history) >= 2 and dE < tol and n_now >= min(n_dets, sector):
                converged, stop_reason = True, "converged"
                break

    coeffs = np.ascontiguousarray(coeffs, dtype=complex)
    return GroundStateResult(
        energy=energy,
        states=[],                       # array path: not materialized (see io)
        coeffs=coeffs,
        n_dets=int(core_ferm.shape[0]),
        history=history,
        converged=converged,
        stop_reason=stop_reason,
        ferm_arr=np.ascontiguousarray(core_ferm, dtype=np.uint64),
        bos_arr=np.ascontiguousarray(core_bos, dtype=np.uint16),
    )


def ground_state_ensemble_arrays(H, n_elec, n_runs=8, seed=None, **kwargs):
    """Array-native ensemble TrimCI: `n_runs` independent random inits, keep the
    best (lowest energy). Mirrors `graph.ground_state_ensemble`."""
    base = 0 if seed is None else int(seed)
    best = None
    per_run = []
    for k in range(n_runs):
        s = None if seed is None else base + k
        res = ground_state_arrays(H, n_elec, seed=s, **kwargs)
        per_run.append((s, res.energy, res.n_dets))
        if best is None or res.energy < best.energy:
            best = res
    best.history = list(best.history) + [("ensemble", per_run)]
    return best
