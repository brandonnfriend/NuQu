"""
Rigorous per-site Fock cutoff via the exact Bogoliubov ground state of the
free+gradient pion sector (Workstream B / task 25).

This is the *rigorous* replacement for the first-draft `tong_bound.py` estimate.
The derivation is in `claude/research/bosonic-encodings/05_rigorous_cutoff_persite_number.md`.
Headline: choosing the per-site-total pion occupation `n_x = Σ_a n^a_x` as the
Tong local quantum number puts H_WT into H_R (number-preserving; verified
`[H_WT, n̂_x] = 0`), which dissolves the Watson-2026 obstruction. The only
number-changing degree-2 term left is the gradient squeezing, which is
nucleon-independent and *exactly* Bogoliubov-solvable — so its occupation tail
is exact (no factorized-SCS approximation), cross-site correlations included.

The certificate combines:
  * the exact Gaussian per-mode occupation tail  δ_Gauss(N_f)      (§4 of the note)
  * the exact variational eigenvalue bound
        0 ≤ E0(H̃) − E0(H) ≤ ⟨Qψ|(H−E0)|Qψ⟩/(1−δ)                  (§5)
    bounded conservatively by  [m_π N_f + ‖V‖_eff]·(3L^d)·δ_Gauss(N_f).

CONVENTION: everything is in Fock LEVELS N_f (a mode keeps occupations
0..N_f−1). The register uses n_q = ceil(log2(N_f)) qubits. `dim`-general by
construction: the gradient Laplacian is built for any lattice dimension.

STATUS (post adversarial audit — see note §0/§8): this is a rigorous-*modulo*-
approximation ESTIMATE, not yet a certified upper bound. Two gaps:
  (i)  `‖V‖_eff` uses the conservative gradient-inclusive triangle-inequality
       norm from note §3.2 (overcounts → pushes N_f up, the safe direction);
  (ii) `δ` is evaluated on the free+grad *Gaussian* GS, not the true dressed GS
       — `|δ_true − δ_Gauss|` is physically small (H_AV displacement tiny, H_WT
       number-conserving) but not yet bounded, so `δ_Gauss` could *under*count
       (the unsafe direction). Net: a well-motivated estimate comparable to the
       first-draft 'tong', sharpened by the exact Gaussian tail + exact `(★)`.
Kept as the `'tong_rigorous'` switch; `'heuristic'` stays the default until the
gaps are discharged and an ED cross-check confirms it.
"""

from __future__ import annotations

import functools
from math import ceil, log2

import numpy as np
from scipy.linalg import expm


# --------------------------------------------------------------------------- #
# Lattice + Bogoliubov ground state                                            #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=64)
def _laplacian(L: int, dim: int) -> np.ndarray:
    """OBC nearest-neighbour graph Laplacian on an L^dim cubic lattice.

    Σ_{<x,y>} (φ_x − φ_y)² = φ^T · Lap · φ.
    """
    N = L ** dim

    def coord(i):
        c = []
        for _ in range(dim):
            c.append(i % L)
            i //= L
        return c

    def idx(c):
        s = 0
        for k in range(dim - 1, -1, -1):
            s = s * L + c[k]
        return s

    Lap = np.zeros((N, N))
    for i in range(N):
        c = coord(i)
        for d in range(dim):
            if c[d] < L - 1:
                cj = c[:]
                cj[d] += 1
                j = idx(cj)
                Lap[i, i] += 1.0
                Lap[j, j] += 1.0
                Lap[i, j] -= 1.0
                Lap[j, i] -= 1.0
    return Lap


@functools.lru_cache(maxsize=64)
def _worst_site_variances(L: int, dim: int, m_pi: float, a_L: float):
    """Exact Bogoliubov reduced-mode quadrature variances (vacuum = 1/2) at the
    max-coordination (worst-case occupation) site.

    H_free+grad = ½ Σ p² + ½ φ^T M² φ,  M² = m_π² I + Lap/a_L².  Ground state
    covariances ⟨φφ⟩ = ½ M^{-1}, ⟨pp⟩ = ½ M.  Returns (σ_φ², σ_p²) in
    ω_0 = m_π units for the busiest single mode.
    """
    Lap = _laplacian(L, dim)
    M2 = m_pi ** 2 * np.eye(L ** dim) + Lap / a_L ** 2
    w, V = np.linalg.eigh(M2)
    sqrt_w = np.sqrt(w)
    Minv_diag = (V * (1.0 / sqrt_w)) @ V.T            # (M^{-1})_xx via rows
    M_diag = (V * sqrt_w) @ V.T
    x = int(np.argmax(np.diag(Lap)))                   # max-coordination site
    sig_phi2 = m_pi * 0.5 * Minv_diag[x, x]
    sig_p2 = (0.5 * M_diag[x, x]) / m_pi
    return float(sig_phi2), float(sig_p2)


@functools.lru_cache(maxsize=256)
def _gaussian_pn(L: int, dim: int, m_pi: float, a_L: float, n_max: int = 96):
    """Exact Fock-occupation distribution p(n) of the reduced worst-case mode.

    The reduced single-mode state is a squeezed-thermal Gaussian; build its
    density matrix in the Fock basis and read the diagonal. Returns a tuple
    (immutable, cacheable) of length n_max.
    """
    sig_phi2, sig_p2 = _worst_site_variances(L, dim, m_pi, a_L)
    nu = 2.0 * np.sqrt(sig_phi2 * sig_p2)              # symplectic eigenvalue (vac=1)
    r = 0.25 * np.log(sig_p2 / sig_phi2)               # squeeze (φ squeezed if σφ²<σp²)
    n_th = max(0.0, (nu - 1.0) / 2.0)

    a = np.array(np.diag(np.sqrt(np.arange(1, n_max)), 1))   # writable annihilation
    S = expm(0.5 * (r * (a @ a) - r * (a.conj().T @ a.conj().T)))
    if n_th < 1e-12:
        rho_th = np.zeros((n_max, n_max))
        rho_th[0, 0] = 1.0
    else:
        n = np.arange(n_max)
        d = (n_th / (1.0 + n_th)) ** n / (1.0 + n_th)
        rho_th = np.diag(d / d.sum())
    rho = S @ rho_th @ S.conj().T
    pn = np.real(np.diag(rho))
    pn = np.clip(pn, 0.0, None)
    pn = pn / pn.sum()
    return tuple(float(v) for v in pn)


def gaussian_tail(L: int, dim: int, N_f: int, params) -> float:
    """Exact per-mode occupation tail P(n ≥ N_f) at the worst-case site."""
    m_pi, a_L = float(params['m_pi']), float(params['a_L'])
    pn = _gaussian_pn(L, dim, m_pi, a_L)
    if N_f >= len(pn):
        return float(sum(pn[len(pn):]))  # ~0
    return float(sum(pn[N_f:]))


def gaussian_energy_weighted_tail(L: int, dim: int, N_f: int, params) -> float:
    """Σ_{n≥N_f} m_π·n·p(n) per mode — the diagonal part of ⟨Qψ|(H−E0)|Qψ⟩."""
    m_pi = float(params['m_pi'])
    pn = _gaussian_pn(L, dim, params['m_pi'], params['a_L'])
    return float(m_pi * sum(n * pn[n] for n in range(N_f, len(pn))))


# --------------------------------------------------------------------------- #
# The rigorous cutoff                                                          #
# --------------------------------------------------------------------------- #

def _z_eff(L: int, dim: int) -> float:
    """Average OBC coordination number (note §1.2)."""
    return 2.0 * dim * (1.0 - 1.0 / L)


def tong_rigorous_predictions(L, dim, A, params, eps=1e-3, dE_QPE=None,
                              n_f_max=64):
    """Rigorous Fock cutoff from the exact Bogoliubov tail + variational bound.

    Returns a dict with the certified N_f (levels), n_q (qubits), and diagnostics.

    Parameters
    ----------
    eps : target relative eigenvalue error (dimensionless).
    dE_QPE : QPE energy resolution [MeV]; defaults to 0.1·m_π.
    """
    m_pi, a_L = float(params['m_pi']), float(params['a_L'])
    if dE_QPE is None:
        dE_QPE = 0.1 * m_pi
    target = eps * dE_QPE

    n_modes = 3 * (L ** dim)
    C_V = 3.0 * _z_eff(L, dim) / (4.0 * a_L ** 2 * m_pi)   # note §3.2 (conservative)

    def error_bound(N_f):
        # ⟨Qψ|(H−E0)|Qψ⟩/(1−δ), conservative:
        #   numerator ≤ [ energy-weighted tail (diagonal) + ‖V‖_eff·δ ] (union over modes)
        delta_pm = gaussian_tail(L, dim, N_f, params)          # per mode
        delta = min(0.999, n_modes * delta_pm)                  # union bound
        ew_tail = n_modes * gaussian_energy_weighted_tail(L, dim, N_f, params)
        V_eff = C_V * (L ** dim) * (N_f + 2)
        numerator = ew_tail + V_eff * delta
        return numerator / (1.0 - delta)

    N_f = None
    for cand in range(2, n_f_max + 1):
        if error_bound(cand) <= target:
            N_f = cand
            break
    if N_f is None:
        N_f = n_f_max  # did not converge within cap; return the cap (flagged below)

    n_q = max(2, int(ceil(log2(max(2, N_f)))))
    return {
        'N_f': N_f,
        'n_q': n_q,
        'converged': error_bound(N_f) <= target if N_f < n_f_max else False,
        'error_bound_MeV': error_bound(N_f),
        'target_MeV': target,
        'delta_per_mode': gaussian_tail(L, dim, N_f, params),
        'eps': eps,
        'dE_QPE_MeV': dE_QPE,
    }


def tong_rigorous_cutoff(L, dim, A, params, eps=1e-3, dE_QPE=None):
    """n_q (qubits) for the rigorous exact-Bogoliubov Fock cutoff."""
    return tong_rigorous_predictions(L, dim, A, params, eps=eps, dE_QPE=dE_QPE)['n_q']
