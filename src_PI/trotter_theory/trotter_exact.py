"""Exact Watson Theorem-64 Trotter resource model for the dynamical-pion EFT
(task 11). Reproduces Watson Table IX (1.3e42 T-gates at L=10, A=40) with NO free
coefficient — replacing the loose informal-`Cp` model in `trotter_theory.py`
(kept behind a flag for comparison).

Corrections vs the old model (all confirmed against the primary source, Watson
arXiv:2312.05344; v1=v3 for all Trotter content):
  1. **No free `Cp`.** The step count is exact: `r = t²·Ξ / (2·ε_prod)`, where Ξ is
     the fully-explicit Theorem-64 commutator sum (Table XIX, Lemmas 40/49–78).
     The old `Cp` was the paper's *informal* C_p (Eq 34/35), which they call "very
     loose"; their reported numbers use the exact bound.
  2. **`ε_cut` from the error budget, not a fixed 0.1.** The cutoff error enters as
     `2·√(2·ε_cut) = budget` (Lemma 30, a square root) ⇒ `ε_cut = budget²/8`. This
     is what drives Watson's `n_b = 33–39` at A=40 (the old ε_cut=0.1 gave ~31).
  3. **`log₂`, not `log₁₀`,** in the per-step cost (the 1.15/9.2 are base-2 RUS
     constants). ~3.3× per step.
  4. **No spurious explicit `·L³`.** It is already inside `π_max·Π_max` (∝ L³).

**Sensitivity trap:** Ξ ∝ π_max²·Π_max² (WT–WT dominates), and n_b enters
logarithmically, so **each +1 in n_b is ≈4× in T-count.** Watson's 1.3e42 sits at
n_b=39; reproduction is inherently ±1-n_b sensitive — respect it, don't over-tune.
"""

import numpy as np

from src_PI.hamiltonians.core.EFTParameters import (
    T_cross_MeV, calculate_dynamic_cutoffs, get_physical_parameters,
)

DEFAULT_E_MAX_MEV = 140.0   # ‖H_Dπ‖ ceiling for QPE (≈ m_π; "no pion production"), Watson p49


# ---------------------------------------------------------------------------
#  Ξ — the exact Theorem-64 commutator sum (Table XIX: Lemmas 40, 49–78)
# ---------------------------------------------------------------------------

def dynamical_pion_xi(L, A, pi_max, Pi_max, params, return_breakdown=False):
    """Ξ (units MeV²) = Σ of the 20 Table-XIX pairwise commutator bounds. Each
    lemma bound is the *full block sum* (its Table-XX layer multiplicity is already
    inside), EXCEPT the free–free block, which is `C(6,2)=15 × Lemma-40 (2h²η)`.

    At dynamical-pion scale Ξ is ≈100% the WT–WT term (Lemma 78, ∝ η·π²Π²); the
    rest are included for rigor but are ~10⁵–10⁶× smaller.
    """
    a_L = params['a_L']; m_pi = params['m_pi']; f_pi = params['f_pi']
    g_A = params['g_A']; M_N = params['M_N']; C = params['C']; CI = params['CI']
    eta = float(A)
    h = 1.0 / (2.0 * M_N * a_L ** 2)          # hopping coefficient
    gA2 = g_A / (2.0 * f_pi)
    pp = pi_max * Pi_max

    terms = {
        'free_free': 30.0 * h ** 2 * eta,                                  # L40 ×15
        'free_C':    18.0 * h * abs(C) * eta,                              # L49
        'free_CI2':  528.0 * h * abs(CI) * eta,                           # L51
        'free_AV':   2592.0 * gA2 / a_L * h * pi_max * eta,               # L66
        'free_WT':   432.0 * (h / f_pi ** 2) * pp * eta,                  # L72
        'CI2_CI2':   60.0 * CI ** 2 * eta,                                # L53
        'CI2_AV':    6048.0 * gA2 / a_L * abs(CI) * pi_max * eta,         # L68
        'CI2_WT':    504.0 * (abs(CI) / f_pi ** 2) * pp * eta,            # L74
        'pi1_pi2':   (36.0 / a_L ** 2 + 3.0 * m_pi ** 2) * a_L ** 3 * pp * L,  # L65 (no η, L¹)
        'pi1_AV':    36.0 * gA2 / a_L * Pi_max * eta,                     # L69
        'pi2_WT':    72.0 / (f_pi ** 2 * a_L ** 2) * pi_max ** 2 * eta,   # L76
        'AV_AV':     20736.0 * gA2 ** 2 / a_L ** 2 * pi_max ** 2 * eta,   # L71
        'AV_WT':     (g_A / (f_pi ** 3 * a_L)) * (72.0 / a_L ** 3 + 216.0 * pp) * pi_max * eta,  # L77
        'WT_WT':     384.0 * (1.0 / (4.0 * f_pi ** 2)) ** 2 * (3.0 * pp + 2.0 / a_L ** 3) * pp * eta,  # L78
        # zero blocks (L50,52,67,73,75,70): C–C, C–CI2, C–AV, C–WT, π1–WT, π2–AV
    }
    Xi = sum(terms.values())
    return (Xi, terms) if return_breakdown else Xi


# ---------------------------------------------------------------------------
#  Per-Trotter-step T-cost (Lemma 23) — log₂ fixed; g coefficient switchable
# ---------------------------------------------------------------------------

_G_COEFFS = {'statement': (45.0, 114.0, 76.0),   # Lemma-23 statement (conservative)
             'proof':     (33.0, 90.0, 64.0)}    # Lemma-23 proof Eq (109)


def rz_count_g(L, n_b, coeff='statement'):
    """g(L,n_b) = (a·n_b² + b·n_b + c)·L³ — the Rz-rotation count per Trotter step."""
    a, b, c = _G_COEFFS[coeff]
    return (a * n_b ** 2 + b * n_b + c) * L ** 3


def per_step_tcost(L, n_b, delta, coeff='statement'):
    """Lemma-23 expected T-count for one Trotter step to per-step synthesis error
    `delta`: `g·(1.15·log₂(2g/δ) + 9.2)`. Returns (T_step, g)."""
    g = rz_count_g(L, n_b, coeff)
    return g * (1.15 * np.log2(2.0 * g / delta) + 9.2), g


def eps_cut_from_budget(budget):
    """Cutoff error budget → ε_cut. The truncation error enters as `2·√(2·ε_cut)`
    (Lemma 30), so `2·√(2·ε_cut) = budget ⟹ ε_cut = budget²/8`."""
    return budget ** 2 / 8.0


# ---------------------------------------------------------------------------
#  The two tasks
# ---------------------------------------------------------------------------

def cutoffs_for_nb(n_b, params, ratio):
    """(π_max, Π_max) for a *chosen* n_b, from the digitization relation (Eq 78):
    `2^n_b − 1 = (2·a_L³/π)·π_max·Π_max` ⟹ `π_max·Π_max = (2^n_b − 1)·π/(2·a_L³)`.
    The product is fixed by n_b; `ratio = Π_max/π_max` (a physical shape, from the
    Lemma-5 cutoffs at that point) splits it. Since Ξ is ~100% WT–WT ∝ (π_max·Π_max)²,
    the result depends on n_b essentially through the product alone.
    """
    a_L = params['a_L']
    product = (2.0 ** n_b - 1.0) * np.pi / (2.0 * a_L ** 3)
    return np.sqrt(product / ratio), np.sqrt(product * ratio)


def _assemble(L, A, params, eps_cut, eps_prod, t, coeff, dim, E_bound, n_b_override=None):
    """Shared assembly: cutoffs → Ξ → r → per-step → total. Returns intermediates.

    `n_b_override`: if given, evaluate the Trotter cost in a Hilbert space with THAT
    many bits/pion instead of Watson's Lemma-5 choice — the lever for costing Trotter
    in the same reduced (low-pion-occupation) space we use for qubitization. The
    Lemma-5 n_b is still reported as `n_b_lemma`.
    """
    n_b_lemma, pi_L, Pi_L = calculate_dynamic_cutoffs(
        L, dim, A, params, epsilon_cut=eps_cut, E_bound=E_bound)
    if n_b_override is None:
        n_b, pi_max, Pi_max = n_b_lemma, pi_L, Pi_L
    else:
        n_b = int(n_b_override)
        pi_max, Pi_max = cutoffs_for_nb(n_b, params, ratio=Pi_L / pi_L)
    Xi = dynamical_pion_xi(L, A, pi_max, Pi_max, params)
    r = t ** 2 * Xi / (2.0 * eps_prod)
    delta = eps_prod / r                       # per-step synthesis budget (total ε_syn / r)
    t_step, g = per_step_tcost(L, n_b, delta, coeff)
    total_T = r * t_step
    return dict(n_b=n_b, n_b_lemma=int(n_b_lemma), pi_max=float(pi_max), Pi_max=float(Pi_max),
                Xi=float(Xi), t=float(t), r=float(r), per_step_T=float(t_step), g=float(g),
                total_T=float(total_T), eps_cut=float(eps_cut), eps_prod=float(eps_prod))


def crossing_time_cost(L, A, params=None, eps_total=0.1, E_kin=10.0,
                       coeff='statement', dim=3, budget_split=3, n_b_override=None):
    """Total Trotter T-cost for the crossing-time task (Watson Table IX), p=1.

    Fault-tolerant budget splits ε into `budget_split` equal parts (3 → prod/cut/syn).
    `t = T_cross = a_L·L·√(M_N/2E_kin)` (Eq 136). Validation target at L=10, A=40,
    eps_total=0.1: Watson reports 1.3e42 T (n_b=39). `n_b_override` costs Trotter in a
    reduced n_b Hilbert space (see `_assemble`).
    """
    if params is None:
        params = get_physical_parameters()
    budget = eps_total / budget_split
    eps_cut = eps_cut_from_budget(budget)
    t = T_cross_MeV(params['a_L'], L, E_kin, params['M_N'])
    E_bound = E_kin * A                         # irrelevant at A≳few (contact term dominates)
    out = _assemble(L, A, params, eps_cut, budget, t, coeff, dim, E_bound, n_b_override)
    out['task'] = 'crossing_time'
    return out


def qpe_bits(dE, E_max=DEFAULT_E_MAX_MEV):
    """QPE bit accuracy m from Eq 135 with the Table-VIII equal split: the quadrature
    collapses to `ΔE/‖H‖ = 1/2^m`, so `m = ⌈log₂(E_max/ΔE)⌉`."""
    return int(np.ceil(np.log2(E_max / dE)))


def qpe_cost(L, A, params=None, dE=1.0, E_max=DEFAULT_E_MAX_MEV,
             coeff='statement', dim=3, n_b_override=None):
    """Total Trotter T-cost for the QPE spectroscopy task (Watson Fig 14), p=1 —
    the headline comparison. `m = ⌈log₂(E_max/ΔE)⌉`; per-source budget
    `√3·π/(3·2^m)`; `ε_cut = budget²/8`; `t = 2π/‖H‖` with ‖H‖=E_max.
    `n_b_override` costs Trotter in a reduced n_b Hilbert space (the low-occupation
    space we justify for qubitization) — pass our Fock n_b to compare like-for-like.
    """
    if params is None:
        params = get_physical_parameters()
    m = qpe_bits(dE, E_max)
    budget = np.sqrt(3.0) * np.pi / (3.0 * 2 ** m)
    eps_cut = eps_cut_from_budget(budget)
    t = 2.0 * np.pi / E_max
    out = _assemble(L, A, params, eps_cut, budget, t, coeff, dim, E_max, n_b_override)
    out.update(task='qpe', m=m, dE=dE, E_max=E_max)
    return out
