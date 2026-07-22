"""
Analytic Fock-cutoff predictions for the dynamical-pion chiral EFT — the
theoretical curve the classical n_b/N_f convergence study is compared against.

This implements the derivation in
`claude/research/bosonic-encodings/02_tong_fock_cutoff.md` (§2 SCS prediction,
§3 spectral bound, §7.2 pseudocode). It is the first-draft rigorous replacement
for the heuristic `estimate_boson_cutoff` (CLAUDE.md open homework); here it
serves to bracket the empirically-measured cutoff.

CONVENTION. Everything is in NUMBER OF FOCK LEVELS `N_f` (our solver's
convention: a mode keeps occupations 0..N_f-1, so N_f = 2**n_b levels needs n_b
qubits). The doc states some formulas in the max-occupation convention
(N_f_maxocc = N_f_levels - 1); differences are O(1) in N_f and do not move
n_b = ceil(log2(N_f)) by more than 1. The leaked-weight tail is defined to match
`observables.occupation_tail` exactly: delta(N_f) = P(n >= N_f) per mode.

Three predictions, in decreasing directness of the comparison to the study:
  * mean_occupation_scs(L,dim,A)  <-> measured mean_occupation(res)  [tightest test]
  * squeezed_tail(N_f)            <-> measured occupation_tail(res, N_f)
  * cutoff_predictions(...)       -> N_eng / N_spec (1st,2nd) as n_b brackets
"""

from __future__ import annotations

from math import ceil, comb, cosh, log, log2, sinh, sqrt, tanh


def _params(params):
    if params is None:
        from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
        params = get_physical_parameters()
    return params


def z_eff(L, dim):
    """Effective per-site coordination number for an L^dim OBC cubic lattice
    (doc §1.2): Z = 2*dim*(1 - 1/L). L=2,d=3 -> 3; L->inf -> 2*dim."""
    return 2 * dim * (1 - 1.0 / L)


def squeeze_r_star(L, dim, params=None):
    """Exact SCS squeezing parameter (doc Eq. in §2.4):
    r* = 1/4 ln(1 + Z_eff / (a_L^2 m_pi^2)). This is the gradient-vs-free-pion
    competition optimum; it drives the GS occupation."""
    p = _params(params)
    a_L, m_pi = p["a_L"], p["m_pi"]
    return 0.25 * log(1.0 + z_eff(L, dim) / (a_L ** 2 * m_pi ** 2))


def mean_occupation_scs(L, dim, A, params=None):
    """Predicted per-mode GS occupation <N> = sinh^2(r*) [squeeze] + disp [H_AV]
    (doc Eq. §2.5). The AV displacement piece is ~1e-7*A^2/L^(2d) — negligible.
    Returns dict {N_per_mode, r_star, N_sq, N_disp, sigma_N}."""
    p = _params(params)
    a_L, m_pi, f_pi, g_A = p["a_L"], p["m_pi"], p["f_pi"], p["g_A"]
    r = squeeze_r_star(L, dim, p)
    N_sq = sinh(r) ** 2
    N_disp = (3 * g_A ** 2 * A ** 2
              / (2 * f_pi ** 2 * m_pi ** 3 * a_L ** 2 * L ** (2 * dim)))
    sigma_N = sqrt(2 * sinh(r) ** 4 + sinh(r) ** 2 * cosh(2 * r))
    return {"N_per_mode": N_sq + N_disp, "r_star": r,
            "N_sq": N_sq, "N_disp": N_disp, "sigma_N": sigma_N}


def squeezed_tail_per_mode(N_f, r, terms=60):
    """Leaked weight of a single squeezed-vacuum mode past a cutoff that keeps
    occupations 0..N_f-1: delta = sum_{n>=N_f, n even} p_sq(n), with
    p_sq(n) = sech(r) C(n, n/2) (tanh(r)/2)^n (doc §3.3). Matches the empirical
    `occupation_tail` convention (P(n >= N_f))."""
    if r <= 0:
        return 0.0
    sech_r = 1.0 / cosh(r)
    half_tanh = tanh(r) / 2.0
    kmin = (N_f + 1) // 2                        # smallest k with 2k >= N_f
    return sum(sech_r * comb(2 * k, k) * half_tanh ** (2 * k)
               for k in range(kmin, kmin + terms))


def squeezed_tail(N_f, L, dim, params=None):
    """Multi-mode union-bound leaked weight delta(N_f) ~ n_modes * per-mode tail
    (doc §3.3). n_modes = 3 L^dim pion modes."""
    r = squeeze_r_star(L, dim, params)
    n_modes = 3 * L ** dim
    return n_modes * squeezed_tail_per_mode(N_f, r)


def cutoff_predictions(L, dim, A, params=None, eps=1e-3, dE_QPE=None,
                       k_safety=5, N_f_max=64):
    """Predicted Fock cutoffs (doc §2-3), all as NUMBER OF LEVELS N_f and the
    corresponding n_b = ceil(log2(N_f)).

    Returns dict with:
      N_eng    : engineering cutoff, ceil(<N> + k_safety*sigma_N) (doc §2.5)
      N_spec1  : 1st-order (Cauchy-Schwarz) spectral bound (doc §3.4)
      N_spec2  : 2nd-order (Rayleigh-Schrodinger) spectral bound
      n_b_*    : ceil(log2(N_f_*)) for each
      r_star, N_per_mode, ...
    Criteria use ||V|| ~ C_V L^dim (N_f+1) and delta(N_f)=P(n>=N_f); the certified
    rigorous choice in the doc is max(N_eng, N_spec1)."""
    p = _params(params)
    a_L, m_pi = p["a_L"], p["m_pi"]
    if dE_QPE is None:
        dE_QPE = 0.1 * m_pi                      # doc working number: 0.1 m_pi
    scs = mean_occupation_scs(L, dim, A, p)
    r = scs["r_star"]
    target = eps * dE_QPE                        # MeV
    N_eng = max(1, ceil(scs["N_per_mode"] + k_safety * scs["sigma_N"]))

    C_V = 3 * z_eff(L, dim) / (4 * a_L ** 2 * m_pi)    # doc §3.2

    def _delta(N_f):
        return 3 * L ** dim * squeezed_tail_per_mode(N_f, r)

    def _passes(N_f, order):
        V = C_V * L ** dim * (N_f + 1)           # ||V||_op, levels convention
        d = _delta(N_f)
        if order == 1:
            return 2 * V * sqrt(d) <= target
        return V * V * d / m_pi <= target        # 2nd order, gap ~ m_pi

    def _first_pass(order):
        for N_f in range(2, N_f_max + 1):
            if _passes(N_f, order):
                return N_f
        return N_f_max

    N_spec1 = _first_pass(1)
    N_spec2 = _first_pass(2)

    def nb(N_f):
        return max(1, ceil(log2(N_f)))

    return {
        "L": L, "dim": dim, "A": A, "eps": eps, "dE_QPE": dE_QPE,
        "r_star": r, "N_per_mode": scs["N_per_mode"], "sigma_N": scs["sigma_N"],
        "N_eng": N_eng, "n_b_eng": nb(N_eng),
        "N_spec1": N_spec1, "n_b_spec1": nb(N_spec1),
        "N_spec2": N_spec2, "n_b_spec2": nb(N_spec2),
    }
