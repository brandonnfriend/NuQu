"""
Ground-state observables read off a solved TrimCI wavefunction — the pion
occupation, which is what the Fock-cutoff (n_b / N_f) convergence study needs.

Two quantities:

  * mean_occupation(res) -> the mean pion number <N> (total and per mode). The
    Tong/SCS analysis (`claude/research/bosonic-encodings/02_tong_fock_cutoff.md`
    §2) predicts <N>_per_mode ~ 0.045 at (L=2, d=3) — "the GS is essentially pion
    vacuum". Measuring <N> directly from the wavefunction tests that prediction and
    is the physical explanation for WHY a small n_b suffices.

  * occupation_tail(res, N_f) -> the GS weight a per-mode cutoff N_f would DROP,
    i.e. the total |amplitude|^2 on determinants where SOME pion mode has occupation
    >= N_f. This is the EMPIRICAL analogue of the squeezed-vacuum tail delta(N_f) in
    the spectral bound (§3.3): the truncation error is controlled by exactly this
    leaked weight. Computing it from a well-converged (large-N_f) solve gives the
    measured delta(N_f) curve to compare against the analytic prediction.

Both work on either solver path (object `res.states` or array `res.ferm_arr/bos_arr`)
and renormalize the retained core amplitudes (the array path keeps the top-k of the
pool eigenvector, so the saved coeffs are not exactly normalized).
"""

from __future__ import annotations

import numpy as np


def _bos_and_weights(res):
    """(bos occupation matrix (N, n_bos) int, normalized weights w_i = |c_i|^2).

    Reads the array path (bos_arr) directly; falls back to the object path
    (states[i].bos). Weights are renormalized so sum(w) = 1.
    """
    c = np.abs(np.asarray(list(res.coeffs), dtype=complex)) ** 2
    s = c.sum()
    if s <= 0:
        raise ValueError("degenerate/zero coefficient vector")
    w = c / s
    if getattr(res, "bos_arr", None) is not None:
        bos = np.asarray(res.bos_arr, dtype=np.int64)
    elif getattr(res, "states", None):
        bos = np.asarray([s.bos for s in res.states], dtype=np.int64)
    else:
        raise ValueError("result carries neither bos_arr nor states")
    if bos.shape[0] != w.shape[0]:
        raise ValueError(f"bos rows {bos.shape[0]} != coeffs {w.shape[0]}")
    return bos, w


def mean_occupation(res):
    """Mean pion occupation of the ground state.

    Returns dict:
      N_total     : <sum_m n_m>              (mean total pion number)
      N_per_mode  : N_total / n_bos_modes    (mean per pion mode)
      N_max_mode  : <max_m n_m>              (mean of the most-occupied mode)
      n_bos_modes : number of pion modes
    """
    bos, w = _bos_and_weights(res)
    tot = bos.sum(axis=1)                       # total bosons per determinant
    mx = bos.max(axis=1) if bos.shape[1] else np.zeros(bos.shape[0])
    n_modes = bos.shape[1]
    N_total = float((w * tot).sum())
    return {
        "N_total": N_total,
        "N_per_mode": N_total / n_modes if n_modes else 0.0,
        "N_max_mode": float((w * mx).sum()),
        "n_bos_modes": n_modes,
    }


def occupation_from_coeffs(cmap):
    """Mean pion occupation from an exact-diagonalization eigenvector given as a
    {MixedState: amplitude} dict (the `lanczos_ground_state(..., return_vec=True)`
    / `exact_ground_state` output). Same return dict as `mean_occupation`, but for
    the exact path — no core-convergence caveat, so this is the clean truth for the
    occupation-vs-A study at ED-reachable sizes (e.g. L=2 d=1)."""
    states = list(cmap.keys())
    c = np.abs(np.asarray([cmap[s] for s in states], dtype=complex)) ** 2
    s = c.sum()
    if s <= 0:
        raise ValueError("degenerate/zero coefficient vector")
    w = c / s
    bos = np.asarray([st.bos for st in states], dtype=np.int64)
    tot = bos.sum(axis=1)
    mx = bos.max(axis=1) if bos.shape[1] else np.zeros(bos.shape[0])
    n_modes = bos.shape[1]
    N_total = float((w * tot).sum())
    return {
        "N_total": N_total,
        "N_per_mode": N_total / n_modes if n_modes else 0.0,
        "N_max_mode": float((w * mx).sum()),
        "n_bos_modes": n_modes,
    }


def occupation_tail(res, N_f):
    """Empirical leaked weight delta(N_f): total GS probability on determinants
    where SOME pion mode has occupation >= N_f (the weight a cutoff-N_f Fock box
    would drop). This is the measured analogue of the spectral-bound tail.

    N_f may be an int or an iterable; returns a float or a dict {N_f: delta}.
    """
    bos, w = _bos_and_weights(res)
    maxocc = bos.max(axis=1) if bos.shape[1] else np.zeros(bos.shape[0])

    def _one(nf):
        return float(w[maxocc >= nf].sum())

    if np.isscalar(N_f):
        return _one(int(N_f))
    return {int(nf): _one(int(nf)) for nf in N_f}


def occupation_histogram(res):
    """Per-mode occupation distribution p(n) = fraction of (weight x modes) at
    occupation n, aggregated over all pion modes. Compare to the squeezed-vacuum
    p_sq(n) from the SCS analysis. Returns a 1-D array indexed by occupation."""
    bos, w = _bos_and_weights(res)
    n_modes = bos.shape[1]
    if n_modes == 0:
        return np.array([1.0])
    nmax = int(bos.max()) if bos.size else 0
    hist = np.zeros(nmax + 1)
    # each determinant contributes weight w_i spread over its n_modes occupations
    for occ_row, wi in zip(bos, w):
        counts = np.bincount(occ_row, minlength=nmax + 1)
        hist += wi * counts
    return hist / n_modes
