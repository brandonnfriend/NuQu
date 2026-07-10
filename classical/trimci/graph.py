"""
TrimCI core graph functions, adapted to the mixed fermion-boson EFT.

This mirrors the structure of the released TrimCI Python interface
(`py/trimci/interface.py`: generate_excitations / screening / trim) but
operates on our generalized mixed-state H_ij (`hij.connections`) instead of
the fermionic-only Slater-Condon kernel that the released C++ backend
hardwires to FCIDUMP integrals.

Why a reimplementation rather than a call into the package: released TrimCI
computes H_ij itself from one-/two-body integrals (h1[n_orb^2], eri[n_orb^4])
and has no bosonic modes and no custom-H_ij hook (see TODO.md "Integration
routes"). So to run the *dynamical-pion* system we either (a) drive our own
TrimCI over our H_ij — implemented here, validated against ED — or (b)
fermionize the bosons into a standard FCIDUMP and use the package as-is
(deferred; see TODO.md). This module is route (a).

Algorithm (Zhang & Otten 2025, arXiv:2511.14734):
  random core --> [ expansion --> local trim --> global trim ] x rounds.

Diagonalization is dense (numpy eigh) here — toy / ED scale. A sparse
Davidson matvec over `hij.connections` is the obvious next step for scale.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .hij import build_dense, connections
from .state import MixedState, fermion_determinants


# ---------------------------------------------------------------------------
#  Diagonalization helpers
# ---------------------------------------------------------------------------

def diagonalize(H, states):
    """Diagonalize H projected onto `states`. Returns (E0, coeffs_dict).

    coeffs_dict maps each state to its ground-state amplitude.
    """
    states = list(states)
    if not states:
        return float("inf"), {}
    M = build_dense(H, states)
    # Hermitize defensively against tiny asymmetries from float accumulation.
    M = 0.5 * (M + M.conj().T)
    evals, evecs = np.linalg.eigh(M)
    g = evecs[:, 0]
    coeffs = {s: g[k] for k, s in enumerate(states)}
    return float(evals[0].real), coeffs


# ---------------------------------------------------------------------------
#  Expansion (a.k.a. screening): grow the core along strong couplings
# ---------------------------------------------------------------------------

def expand(H, core_coeffs, pool_factor=10):
    """Augment the core with neighbors of largest |H_ij c_j|.

    Mirrors TrimCI's screening rule P = C u {i : |H_ij c_j| > theta}; we set
    theta implicitly by keeping the top (pool_factor * |C|) scored neighbors,
    which avoids hand-tuning theta while realizing "a reasonably large pool".

    Returns the pool as a set of MixedState (core included).
    """
    core = set(core_coeffs)
    scores = {}
    for j, cj in core_coeffs.items():
        for i, hij in connections(H, j).items():
            if i in core:
                continue
            s = abs(hij * cj)
            if s > scores.get(i, 0.0):
                scores[i] = s
    if not scores:
        return set(core)
    keep = max(pool_factor * len(core), 1)
    ranked = heapq.nlargest(keep, scores, key=scores.get)
    return core | set(ranked)


# ---------------------------------------------------------------------------
#  Trimming
# ---------------------------------------------------------------------------

def local_trim(H, pool, num_groups, keep_per_group, rng, diag_fn=None):
    """Local trim: random-group diagonalizations, keep top-k_a per group.

    Discards obviously negligible determinants cheaply while preserving
    representatives from diverse regions of the graph. `diag_fn` overrides the
    subspace diagonalizer (e.g. the official C++ Davidson; see `backend.py`).
    """
    diag_fn = diag_fn or diagonalize
    pool = list(pool)
    if len(pool) <= num_groups:
        return set(pool)
    rng.shuffle(pool)
    groups = [pool[g::num_groups] for g in range(num_groups)]
    survivors = set()
    for grp in groups:
        if not grp:
            continue
        _E, coeffs = diag_fn(H, grp)
        survivors.update(heapq.nlargest(keep_per_group, grp,
                                        key=lambda s: abs(coeffs[s])))
    return survivors


def global_trim(H, pool, keep, diag_fn=None):
    """Global trim: one diagonalization, keep top-k_b by |amplitude|.

    Returns (core_coeffs, energy). `diag_fn` overrides the diagonalizer.
    """
    diag_fn = diag_fn or diagonalize
    energy, coeffs = diag_fn(H, pool)
    ranked = heapq.nlargest(keep, coeffs, key=lambda s: abs(coeffs[s]))
    core = {s: coeffs[s] for s in ranked}
    return core, energy


# ---------------------------------------------------------------------------
#  Random initial core
# ---------------------------------------------------------------------------

def boson_occupation_weights(N_f, mean_occ):
    """Occupation weights over {0,..,N_f-1} for the random init.

    `mean_occ` (float): truncated-geometric P(n) ∝ p^n with p = mean_occ/(1+mean_occ)
      so the untruncated mean is `mean_occ`. Strictly decreasing in n — the
      "near-vacuum" prior. `mean_occ <= 0` or `N_f == 1` collapses to vacuum.
    `mean_occ is None`: UNIFORM P(n) = 1/N_f — the UNBIASED control (no
      low-occupation prior; every Fock level equally likely at init, so the
      high-N_f runs are seeded across the full occupation range). Used to verify
      the near-vacuum result isn't an artifact of the seed (Study D).
    """
    w = np.zeros(N_f)
    if N_f <= 1:
        w[0] = 1.0
        return w
    if mean_occ is None:
        return np.full(N_f, 1.0 / N_f)          # uniform: no near-vacuum prior
    if mean_occ <= 0:
        w[0] = 1.0
        return w
    p = mean_occ / (1.0 + mean_occ)
    w = p ** np.arange(N_f)
    return w / w.sum()


def random_core(H, n_elec, n_init, rng, boson_init_mean=0.5):
    """Sample `n_init` random mixed determinants from the A=n_elec sector.

    Boson occupations are drawn NEAR VACUUM: each pion mode is sampled from a
    truncated-geometric distribution with mean ~ `boson_init_mean`, so higher
    occupations are exponentially less likely. This seeds the search where the
    weakly-dressed physical pion ground state actually lives, rather than the
    uniform-[0,N_f) high-occupation cloud — which matters increasingly as N_f
    grows (at N_f=32 a uniform draw averages ~16 quanta/mode, nowhere near the
    GS). The all-boson-vacuum reference is always included as an anchor.

    `boson_init_mean` is tunable: ~0 hugs vacuum, larger spreads occupation up
    for systems whose ground state is more strongly dressed. `boson_init_mean=None`
    selects UNIFORM sampling with NO vacuum anchor — the unbiased control (Study D):
    no low-occupation prior at all, so the search must FIND the near-vacuum GS
    rather than being seeded there.
    """
    uniform = boson_init_mean is None
    ferm_pool = list(fermion_determinants(H.n_ferm_modes, n_elec))
    weights = boson_occupation_weights(H.N_f, boson_init_mean)
    states = set()
    # Anchor: a fermion determinant on the boson vacuum. Skipped for the uniform
    # (unbiased) init so vacuum gets no special seeding.
    if ferm_pool and not uniform:
        states.add(MixedState(int(rng.choice(ferm_pool)),
                              tuple([0] * H.n_bos_modes)))
    guard = 0
    while len(states) < n_init and guard < 50 * n_init:
        guard += 1
        occ = int(rng.choice(ferm_pool)) if ferm_pool else 0
        bos = tuple(int(x) for x in
                    rng.choice(H.N_f, size=H.n_bos_modes, p=weights))
        states.add(MixedState(occ, bos))
    return states


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

@dataclass
class GroundStateResult:
    energy: float
    states: list
    coeffs: list
    n_dets: int
    history: list = field(default_factory=list)   # list[(n_dets, energy)]
    converged: bool = False                       # met the relative-dE criterion
    stop_reason: str = ""                         # "converged" | "capped" | "max_rounds"
    # Tier-2 array-native path: the final core as compact arrays, so io can save
    # the wavefunction without ever materializing a MixedState per state (the
    # boson tuple alone is ~n_bos*8 B/state). When set, `states` may be left empty
    # and io.`_states_arrays` reads these directly. ferm is (N, W) uint64 words.
    ferm_arr: object = None      # np.ndarray (N, W) uint64 or None
    bos_arr: object = None       # np.ndarray (N, n_bos) uint16 or None


def halving_drop(history):
    """Robust convergence signal: the RELATIVE energy drop over the last core
    DOUBLING, |E(N/2) - E(N)| / |E(N)|, read off the (core_size, energy) ramp.
    Smoothing over a full doubling (vs a single round) beats the stochastic
    round-to-round noise and the fragile 1/N extrapolation — it directly answers
    "does adding ~2x more determinants still move the energy?". Returns None if
    there is no point near N/2 yet."""
    pts = sorted(set((int(n), float(e)) for n, e in history if isinstance(n, int)))
    if len(pts) < 3:
        return None
    n_f, e_f = pts[-1]
    half = n_f / 2.0
    prev = [p for p in pts[:-1] if p[0] <= 0.75 * n_f]     # genuinely smaller core
    if not prev:
        return None
    n_h, e_h = min(prev, key=lambda p: abs(p[0] - half))   # closest to N/2
    return abs(e_f - e_h) / max(abs(e_f), 1e-12)


def ground_state(H, n_elec, n_dets=200, n_init=20, pool_factor=3,
                 num_groups=5, local_keep_ratio=4, max_rounds=12,
                 tol=1e-9, seed=None, boson_init_mean=0.5, diag_fn=None,
                 expand_fn=None, verbose=False,
                 max_n_dets=None, conv_tol_rel=None, conv_patience=2,
                 target_gs_rel=None):
    """Run the TrimCI expansion-trim cycle on the mixed Hamiltonian.

    Two stopping modes:
      * FIXED (default): the core ramps geometrically toward `n_dets` and stops on
        the absolute `tol` once it has reached that size.
      * ADAPTIVE (set `target_gs_rel` or `conv_tol_rel`): the core keeps growing
        each round (ceiling `max_n_dets`, default `n_dets`) until the convergence
        signal holds for `conv_patience` consecutive rounds — or the core hits the
        ceiling. The signal is either the relative energy drop over the last core
        DOUBLING (|E(N/2)-E(N)|/|E(N)|) falling below `target_gs_rel` (preferred:
        robust to the stochastic round-to-round noise, and it makes the core grow
        with system size), or the per-round relative drop |dE|/|E| below
        `conv_tol_rel`. The final core size is data-driven; `result.converged` /
        `result.stop_reason` record why.

    Args:
        n_dets: target final core size (FIXED mode) / default ceiling (ADAPTIVE).
        max_n_dets: hard ceiling on the core size in ADAPTIVE mode (the laptop/HPC
            "true max" switch); defaults to `n_dets`.
        target_gs_rel: target for the last-core-doubling relative energy drop
            (turns on ADAPTIVE mode; the preferred, robust criterion).
        conv_tol_rel: relative per-round dE threshold (alternative ADAPTIVE signal).
        conv_patience: consecutive sub-threshold rounds required to call it converged.
        n_init, pool_factor, num_groups, local_keep_ratio, max_rounds, tol, seed,
        boson_init_mean: as before.

    Returns:
        GroundStateResult (with .converged / .stop_reason).
    """
    import math
    dfn = diag_fn or diagonalize
    efn = expand_fn or expand
    adaptive = conv_tol_rel is not None or target_gs_rel is not None
    ceiling = (max_n_dets if max_n_dets is not None else n_dets)
    if adaptive:
        # ensure enough rounds to ramp n_init -> ceiling (x1.5/round) + settle.
        need = int(math.ceil(math.log(max(ceiling, 2) / max(n_init, 1))
                             / math.log(1.5))) + conv_patience + 3
        max_rounds = max(max_rounds, need)
    rng = np.random.default_rng(seed)
    core = random_core(H, n_elec, n_init, rng, boson_init_mean=boson_init_mean)
    energy, coeffs = dfn(H, core)
    core_coeffs = coeffs
    history = [(len(core), energy)]
    target = max(n_init, 1)
    sector = _sector_size(H, n_elec)   # hoisted: avoid bigint pow each round
    below = 0
    cap_round = None
    converged = False
    stop_reason = "max_rounds"

    for rnd in range(max_rounds):
        # grow the target det count toward the ceiling (gentle ramp).
        target = min(ceiling, max(target + n_init, int(np.ceil(target * 1.5))))

        pool = efn(H, core_coeffs, pool_factor)
        keep_per_group = max(1, (target * local_keep_ratio) // num_groups)
        # Skip local_trim when it would keep every state anyway (keep_per_group
        # >= max group size) — a pure no-op that still costs ~num_groups
        # diagonalizations of the whole pool. That is exactly the regime at the
        # default pool_factor=3 (holds when 1+pool_factor <= local_keep_ratio).
        # global_trim then sees the same survivor set, so the result is identical;
        # we only drop the wasted prefilter (~halving per-round cost). local_trim
        # still runs when the pool is genuinely large (e.g. pool_factor=10).
        group_max = -(-len(pool) // num_groups)   # ceil(len(pool) / num_groups)
        if keep_per_group >= group_max:
            survivors = set(pool)
        else:
            survivors = local_trim(H, pool, num_groups, keep_per_group, rng,
                                   diag_fn=diag_fn)
        # always keep the current core in the survivor set (monotone safety).
        survivors |= set(core_coeffs)
        core_coeffs, energy = global_trim(H, survivors, target, diag_fn=diag_fn)

        history.append((len(core_coeffs), energy))
        dE = abs(history[-1][1] - history[-2][1])
        n_now = len(core_coeffs)
        at_ceiling = n_now >= min(ceiling, sector)
        if verbose:
            print(f"  round {rnd:2d}: n_dets={n_now:6d}  E={energy:.8f}  dE={dE:.2e}")

        if adaptive:
            # convergence signal: extrapolated remaining energy to the true GS
            # (preferred — it targets "close to the GS" and grows the core with
            # system size), else the marginal per-round relative drop.
            if target_gs_rel is not None:
                drop = halving_drop(history)
                signal = (drop is not None and n_now >= 600 and
                          drop < target_gs_rel)
                last_metric = drop
            else:
                rel = dE / max(abs(history[-1][1]), 1e-12)
                signal = rel < conv_tol_rel
                last_metric = rel
            below = below + 1 if signal else 0
            if below >= conv_patience:
                converged, stop_reason = True, "converged"
                break
            if at_ceiling:
                # can't grow further; allow a few settle rounds, then stop.
                if cap_round is None:
                    cap_round = rnd
                elif rnd - cap_round >= conv_patience:
                    converged, stop_reason = bool(signal), "capped"
                    break
        else:
            if len(history) >= 2 and dE < tol and n_now >= min(n_dets, sector):
                converged, stop_reason = True, "converged"
                break

    states = list(core_coeffs)
    return GroundStateResult(
        energy=energy,
        states=states,
        coeffs=[core_coeffs[s] for s in states],
        n_dets=len(states),
        history=history,
        converged=converged,
        stop_reason=stop_reason,
    )


def ground_state_ensemble(H, n_elec, n_runs=8, seed=None, **kwargs):
    """Ensemble TrimCI: run `n_runs` independent random inits, keep the best.

    Faithful to Zhang & Otten 2025: "execute with several random
    initializations (in parallel) ... then the best-performing run is chosen."
    Single-run TrimCI can get trapped in a local basin (seed-dependent); the
    ensemble's min-over-runs escapes it. Returns the lowest-energy
    GroundStateResult, with `.history` augmented by an `ensemble` field.

    seed sets the base; run k uses seed+k (reproducible when seed is set).
    """
    base = 0 if seed is None else int(seed)
    best = None
    per_run = []
    for k in range(n_runs):
        s = None if seed is None else base + k
        res = ground_state(H, n_elec, seed=s, **kwargs)
        per_run.append((s, res.energy, res.n_dets))
        if best is None or res.energy < best.energy:
            best = res
    best.history = list(best.history) + [("ensemble", per_run)]
    return best


def exact_ground_state(H, n_elec):
    """Full ED over the mixed sector (toy). Returns (E0, coeffs_dict).

    The validation truth for `ground_state`. Size = |A sector| * N_f**n_modes.
    """
    from .state import enumerate_basis
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec)
    return diagonalize(H, basis)


def _sector_size(H, n_elec):
    from math import comb
    return comb(H.n_ferm_modes, n_elec) * (H.N_f ** H.n_bos_modes)
