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
import multiprocessing as mp
import os

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


def fixed_ferm_core_arrays(H, n_init, ferm_occ, rng, boson_init_mean=0.5):
    """Initial core that holds ONE assigned nucleon arrangement `ferm_occ` (an int
    fermion bitmask) under `n_init` random near-vacuum boson configurations, with the
    all-vacuum state as the anchor. The stratified-start lever: `random_core_arrays`
    draws a fresh random nucleon placement for every initial state, so which
    arrangement a run explores is left to the first few selection rounds."""
    from .backend import _states_to_arrays
    from .graph import boson_occupation_weights
    from .state import MixedState
    weights = boson_occupation_weights(H.N_f, boson_init_mean)
    states = {MixedState(int(ferm_occ), tuple([0] * H.n_bos_modes))}
    guard = 0
    while len(states) < n_init and guard < 50 * n_init:
        guard += 1
        bos = tuple(int(x) for x in rng.choice(H.N_f, size=H.n_bos_modes, p=weights))
        states.add(MixedState(int(ferm_occ), bos))
    return _states_to_arrays(list(states), _ferm_words(H))


def mode_sites(H):
    """Lattice site of each compact fermion mode. `build_from_eft` compacts the
    quantum register's nucleon qubits (site*(4+3*n_b) + slot) in sorted order, so
    compact mode m sits on site m // 4; read it from the index map when present."""
    imap = H.meta.get("fermion_index_map") if getattr(H, "meta", None) else None
    n_b = H.meta.get("n_b") if getattr(H, "meta", None) else None
    if imap and n_b is not None:
        stride = 4 + 3 * int(n_b)
        sites = np.empty(H.n_ferm_modes, dtype=int)
        for orig, k in imap.items():
            sites[k] = int(orig) // stride
        return sites
    return np.arange(H.n_ferm_modes) // 4


def site_partitions(n_elec, max_per_site, max_sites):
    """Every way to split `n_elec` nucleons into per-site counts (each <= max_per_site,
    at most max_sites occupied sites), as descending tuples: the arrangement TYPES,
    from fully spread (1,1,...,1) to maximally clustered (4,4,...)."""
    out = []

    def rec(left, cap, parts):
        if left == 0:
            out.append(tuple(parts))
            return
        if len(parts) >= max_sites:
            return
        for c in range(min(cap, left), 0, -1):
            rec(left - c, c, parts + [c])
    rec(int(n_elec), int(max_per_site), [])
    return out


def nucleon_arrangement_starts(H, n_elec, n_runs, rng):
    """`n_runs` fermion bitmasks that cover the nucleon-arrangement TYPES evenly.

    Types are the site-occupancy partitions (`site_partitions`); the runs cycle
    through them in a random order, so every type gets floor/ceil(n_runs/n_types)
    starts. Within a type the occupied sites and each site's spin-isospin slots are
    drawn at random, so repeated starts of one type land in different places."""
    sites = mode_sites(H)
    n_sites = int(sites.max()) + 1
    by_site = [np.flatnonzero(sites == j) for j in range(n_sites)]
    cap = min(len(m) for m in by_site)
    types = site_partitions(n_elec, cap, n_sites)
    order = rng.permutation(len(types))
    starts = []
    for k in range(int(n_runs)):
        parts = types[order[k % len(types)]]
        occ = 0
        for site, c in zip(rng.choice(n_sites, size=len(parts), replace=False), parts):
            for m in rng.choice(by_site[int(site)], size=c, replace=False):
                occ |= 1 << int(m)
        starts.append(occ)
    return starts


def _rows_isin(a, b):
    """Boolean mask: which rows of 2-D array `a` also occur as rows of `b`."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return np.isin(a.view(dt).ravel(), b.view(dt).ravel())


# ---------------------------------------------------------------------------
#  Array-native expansion / trimming — all "keep" ops are on ROW INDICES
# ---------------------------------------------------------------------------

def expand_arrays(H, core_ferm, core_bos, coeffs, pool_factor=3,
                  novel_ferm_frac=0.0, novel_oversample=4):
    """Pool = core ⊕ top-(pool_factor·N) scored candidates, as arrays.

    Returns (pool_ferm, pool_bos) with the core occupying the first N rows
    (candidates after). Candidates are unique and disjoint from the core
    (guaranteed by `expand_topk` in C++), so the concatenation needs no dedup.

    `novel_ferm_frac` > 0 (exploration lever, default OFF = the path above, unchanged):
    ALSO admit up to novel_ferm_frac·N candidates whose NUCLEON configuration is not
    in the core at all (a hop, or a spin/isospin flip), best score first, drawn from
    the next novel_oversample·keep candidates below the cut. Nucleon hopping is weak
    (h = 1/2Ma²), so those candidates score low and are otherwise almost never let
    into the diagonalization; the diagonalization still decides whether they stay.
    """
    N = core_ferm.shape[0]
    keep = max(pool_factor * N, 1)
    cf, cb, _sc = cpp_expand_topk(H, core_ferm, core_bos, coeffs, keep)
    if novel_ferm_frac and novel_ferm_frac > 0 and cf.shape[0] > 0:
        # a second, wider call; the default pool above is kept EXACTLY (C++ breaks
        # score ties its own way), so lever ON = default pool + the added rows.
        wf, wb, ws = cpp_expand_topk(H, core_ferm, core_bos, coeffs,
                                     keep * (1 + int(novel_oversample)))
        cand = np.flatnonzero(~_rows_isin(wf, core_ferm))
        cand = cand[np.argsort(-ws[cand], kind="stable")]
        rows = lambda f, b: np.hstack([f, b.astype(np.uint64)])
        cand = cand[~_rows_isin(rows(wf[cand], wb[cand]), rows(cf, cb))]
        add = cand[:int(math.ceil(novel_ferm_frac * N))]
        cf = np.concatenate([cf, wf[add]], axis=0)
        cb = np.concatenate([cb, wb[add]], axis=0)
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


def global_trim_arrays(H, ferm, bos, keep, novel_keep_frac=0.0, prev_core_ferm=None):
    """Global trim: one diagonalization, keep the top-`keep` rows by |amplitude|.

    `novel_keep_frac` > 0 (exploration lever, default OFF): up to novel_keep_frac·keep
    of the kept rows are RESERVED for the highest-amplitude rows whose nucleon
    configuration is not in `prev_core_ferm`, so a newly reached arrangement survives
    long enough for later rounds to grow its pion cloud. Any kept subset is still a
    valid variational space, so E stays an upper bound.

    Returns (core_ferm, core_bos, core_coeffs, energy).
    """
    E0, coeffs = cpp_diagonalize_matfree_arrays(H, ferm, bos)
    amp = np.abs(coeffs)
    P = ferm.shape[0]
    k = min(keep, P)
    if novel_keep_frac and novel_keep_frac > 0 and prev_core_ferm is not None and k < P:
        novel = np.flatnonzero(~_rows_isin(ferm, prev_core_ferm))
        n_res = min(int(math.ceil(novel_keep_frac * k)), novel.size)
        res = novel[np.argsort(-amp[novel], kind="stable")[:n_res]]
        mask = np.ones(P, dtype=bool)
        mask[res] = False
        others = np.flatnonzero(mask)
        k2 = k - n_res
        if k2 <= 0:
            top_o = others[:0]
        elif k2 >= others.size:
            top_o = others
        else:
            top_o = others[np.argpartition(amp[others], -k2)[-k2:]]
        top = np.concatenate([res, top_o])
        return ferm[top], bos[top], coeffs[top], E0
    top = np.argpartition(amp, -k)[-k:] if k < P else np.arange(P)
    return ferm[top], bos[top], coeffs[top], E0


# ---------------------------------------------------------------------------
#  Driver — mirrors graph.ground_state (FIXED + ADAPTIVE), array-native
# ---------------------------------------------------------------------------

def ground_state_arrays(H, n_elec, n_dets=200, n_init=20, pool_factor=3,
                        num_groups=5, local_keep_ratio=4, max_rounds=12,
                        tol=1e-9, seed=None, boson_init_mean=0.5, verbose=False,
                        max_n_dets=None, conv_tol_rel=None, conv_patience=2,
                        target_gs_rel=None, initial_core=None,
                        init_ferm=None, novel_ferm_frac=0.0, novel_oversample=4,
                        novel_keep_frac=0.0,
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

    Exploration levers (all default OFF -> the unchanged algorithm):
      init_ferm       -- start from this one nucleon arrangement (int bitmask) under
                         random boson clouds (`fixed_ferm_core_arrays`), not a random
                         placement per initial state;
      novel_ferm_frac -- reserve pool slots for new nucleon configurations
                         (`expand_arrays`);
      novel_keep_frac -- reserve core slots for them through the global trim.
    """
    adaptive = conv_tol_rel is not None or target_gs_rel is not None
    ceiling = (max_n_dets if max_n_dets is not None else n_dets)
    if adaptive:
        need = int(math.ceil(math.log(max(ceiling, 2) / max(n_init, 1))
                             / math.log(1.5))) + conv_patience + 3
        max_rounds = max(max_rounds, need)

    rng = np.random.default_rng(seed)
    if initial_core is not None:
        # WARM START: grow from a given (ferm, bos) core (the previous rung's core)
        # instead of a fresh random seed — the "grow, don't redo" ladder. `target`
        # then continues the x1.5 ramp from the warm core size, not from n_init.
        core_ferm = np.ascontiguousarray(initial_core[0], dtype=np.uint64)
        core_bos = np.ascontiguousarray(initial_core[1], dtype=np.uint16)
    elif init_ferm is not None:
        core_ferm, core_bos = fixed_ferm_core_arrays(H, n_init, init_ferm, rng,
                                                     boson_init_mean=boson_init_mean)
    else:
        core_ferm, core_bos = random_core_arrays(H, n_elec, n_init, rng,
                                                 boson_init_mean=boson_init_mean)
    energy, coeffs = cpp_diagonalize_matfree_arrays(H, core_ferm, core_bos)
    history = [(core_ferm.shape[0], energy)]
    target = max(n_init, core_ferm.shape[0])
    sector = _sector_size(H, n_elec)
    below = 0
    cap_round = None
    converged = False
    stop_reason = "max_rounds"

    for rnd in range(max_rounds):
        target = min(ceiling, max(target + n_init, int(np.ceil(target * 1.5))))
        N = core_ferm.shape[0]

        pool_ferm, pool_bos = expand_arrays(H, core_ferm, core_bos, coeffs,
                                            pool_factor, novel_ferm_frac=novel_ferm_frac,
                                            novel_oversample=novel_oversample)
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
            H, surv_ferm, surv_bos, target, novel_keep_frac=novel_keep_frac,
            prev_core_ferm=(pool_ferm[:N] if novel_keep_frac else None))

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


_FORK_ENSEMBLE_STATE = {}


def _ensemble_worker(s):
    """Runs in a forked child. Reads H/n_elec/kwargs from INHERITED module state so
    the un-picklable C++-backed H is never sent through the pipe -- only the seed `s`
    (an int) is pickled. Returns the pure-numpy GroundStateResult (picklable back)."""
    st = _FORK_ENSEMBLE_STATE
    return ground_state_arrays(st["H"], st["n_elec"], seed=s, **st["kwargs"])


def _select_worker(arg):
    """Forked child for the Phase-0 SELECT lever: solve one init at the first rung,
    then warm-grow it through `rungs`, returning every rung's result (the parent keeps
    the lowest at the last rung). Only (seed, init_ferm) crosses the pipe."""
    s, init_ferm = arg
    st = _FORK_ENSEMBLE_STATE
    H, n_elec, rungs, kw = st["H"], st["n_elec"], st["rungs"], st["kwargs"]
    res = ground_state_arrays(H, n_elec, n_dets=rungs[0], seed=s, init_ferm=init_ferm, **kw)
    out = [res]
    for i, r in enumerate(rungs[1:], start=1):
        res = ground_state_arrays(H, n_elec, n_dets=r, initial_core=(res.ferm_arr, res.bos_arr),
                                  seed=s + i, **kw)
        out.append(res)
    return out


def select_phase0_arrays(H, n_elec, rungs, n_runs, seed=0, init_ferms=None,
                         n_workers=None, **kwargs):
    """Phase-0 SELECT lever: grow EVERY one of `n_runs` inits (seeds seed+k) through
    `rungs` (e.g. 1k..16k) and keep the run that is lowest at the LAST rung, instead
    of the one lowest at the first. At L=2 A=5 the 1k energy barely predicts the
    16k one (the best-at-1k pick ends in a basin 16 MeV/site above the best), because
    nucleon-arrangement changes happen during growth. Returns (per-rung results of the
    winner, [(seed, E at each rung) for every run])."""
    seeds = [int(seed) + k for k in range(int(n_runs))]
    ferms = list(init_ferms) if init_ferms is not None else [None] * len(seeds)
    args = list(zip(seeds, ferms))
    nw = _ensemble_workers(n_runs) if n_workers is None else max(1, min(n_workers, n_runs))
    _FORK_ENSEMBLE_STATE.update(H=H, n_elec=n_elec, rungs=list(rungs), kwargs=kwargs)
    if nw == 1:
        trails = [_select_worker(a) for a in args]
    else:
        with mp.get_context("fork").Pool(nw) as pool:
            trails = pool.map(_select_worker, args)
    summary = [(s, [float(r.energy) for r in t]) for s, t in zip(seeds, trails)]
    best = min(range(len(trails)), key=lambda i: trails[i][-1].energy)
    return trails[best], summary


def _ensemble_workers(n_runs):
    """How many processes to fan the ensemble across -- picked ADAPTIVELY from the
    resources the job actually has. Priority: NUQU_NUM_WORKERS (explicit override) ->
    _CONDOR_REQUEST_CPUS (the Condor cpu allocation) -> the cores this process may use
    (sched_getaffinity / cpu_count). Capped at n_runs (no point in more workers than
    independent inits). Set NUQU_NUM_WORKERS=1 to force serial. NOTE: qis is 48
    physical cores + 2-way SMT (96 logical); compute-bound solves scale on physical
    cores, so request up to ~48 -- past that the SMT siblings add little."""
    for var in ("NUQU_NUM_WORKERS", "_CONDOR_REQUEST_CPUS"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1, min(int(val), n_runs))
            except ValueError:
                pass
    try:
        avail = len(os.sched_getaffinity(0))        # cores in this job's cpuset
    except AttributeError:
        avail = os.cpu_count() or 1                  # macOS: no sched_getaffinity
    return max(1, min(avail, n_runs))


def ground_state_ensemble_arrays(H, n_elec, n_runs=8, seed=None, n_workers=None,
                                 **kwargs):
    """Array-native ensemble TrimCI: `n_runs` independent random inits, keep the
    best (lowest energy). Mirrors `graph.ground_state_ensemble`.

    The `n_runs` inits are INDEPENDENT (each fully determined by its own seed
    `base+k`), so they fan across processes when NUQU_NUM_WORKERS>1 -- a ~min(
    n_runs, cores)x speedup that is BIT-IDENTICAL to the serial path (same seeds ->
    same random cores -> same energies). Every ensemble solve flows through here
    (Phase-0 discovery, and each LF-scan / COO-cycle solve), so this one change
    multithreads essentially all of TrimCI's cost on the cluster."""
    base = 0 if seed is None else int(seed)
    seeds = [None if seed is None else base + k for k in range(n_runs)]
    nw = _ensemble_workers(n_runs) if n_workers is None else max(1, min(n_workers, n_runs))
    if nw == 1 or n_runs == 1:
        results = [ground_state_arrays(H, n_elec, seed=s, **kwargs) for s in seeds]
    else:
        # fork so H (read-only, un-picklable C++ provider inside) is INHERITED, not
        # pickled -- workers read it from module state set just before the fork; only
        # the int seed crosses the pipe. BLAS is pinned single-thread -> fork-safe.
        _FORK_ENSEMBLE_STATE.update(H=H, n_elec=n_elec, kwargs=kwargs)
        with mp.get_context("fork").Pool(nw) as pool:
            results = pool.map(_ensemble_worker, seeds)
    best = None
    per_run = []
    for s, res in zip(seeds, results):
        per_run.append((s, res.energy, res.n_dets))
        if best is None or res.energy < best.energy:
            best = res
    best.history = list(best.history) + [("ensemble", per_run)]
    return best
