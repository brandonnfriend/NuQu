"""Per-term source-map + validation for Watson's Theorem-64 commutator sum Ξ
(`src_PI.trotter_theory.trotter_exact.dynamical_pion_xi`).

Codex trotter_comparison_status_2026-08-24 §1 required this: one aggregate reproduction of Watson's
final T-count is not a source-level validation (different transcription errors can cancel). This
suite pins EACH of the 14 nonzero Ξ terms with:
  * its Watson lemma / Table-XIX block + block multiplicity (the source map, `XI_PROVENANCE`);
  * INDEPENDENT structural scaling checks — the powers of η(=A), L, π_max, Π_max are fixed by WHICH
    operators the commutator is between (physics), so they are verifiable without Watson's constants;
  * a per-term value regression (so a coefficient can never silently drift);
  * the six documented ZERO blocks stay absent.

NOTE (flagged for the next audit): the numerical COEFFICIENTS (30, 2592, 384, …) are Watson's,
transcribed from the lemmas cited here. They are regression-pinned + lemma-mapped, but eyeball
verification of each coefficient against arXiv:2312.05344 (Lemmas 40, 49–78) is a remaining
human/domain check — see docs/trotter_xi_sourcemap.md.

Run: python -m pytest -q tests/test_trotter_xi_sourcemap.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.trotter_theory.trotter_exact import dynamical_pion_xi, cutoffs_for_nb
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs

# --- the SOURCE MAP: term -> Watson provenance + structural exponents ---------------------------- #
# exponents: eta (=A), L, pi_max, Pi_max. `None` = the term is NOT a single monomial in that cutoff
# (an additive const+pp or pp+pp² structure); those are covered by the value regression + η/L tests.
XI_PROVENANCE = {
    # term        lemma    block multiplicity note                     eta  L  pimax  Pimax
    "free_free": ("L40",  "C(6,2)=15 free-free blocks x 2h^2 eta",     1,   0, 0,     0),
    "free_C":    ("L49",  "free x contact-C",                          1,   0, 0,     0),
    "free_CI2":  ("L51",  "free x contact-CI2",                        1,   0, 0,     0),
    "free_AV":   ("L66",  "free x axial-vector (H_AV ~ pi_max)",       1,   0, 1,     0),
    "free_WT":   ("L72",  "free x Weinberg-Tomozawa (H_WT ~ pp)",      1,   0, 1,     1),
    "CI2_CI2":   ("L53",  "contact-CI2 self",                          1,   0, 0,     0),
    "CI2_AV":    ("L68",  "contact-CI2 x AV",                          1,   0, 1,     0),
    "CI2_WT":    ("L74",  "contact-CI2 x WT",                          1,   0, 1,     1),
    "pi1_pi2":   ("L65",  "pion kinetic x gradient (no eta; ~L)",      0,   1, 1,     1),
    "pi1_AV":    ("L69",  "pion-momentum x AV (~Pi_max)",              1,   0, 0,     1),
    "pi2_WT":    ("L76",  "pion-field x WT (~pi_max^2)",               1,   0, 2,     0),
    "AV_AV":     ("L71",  "AV self (~pi_max^2)",                       1,   0, 2,     0),
    "AV_WT":     ("L77",  "AV x WT (pi_max*(const+pp)); mixed cutoff", 1,   0, None,  None),
    "WT_WT":     ("L78",  "WT self (pp*(3pp+2/a_L^3)); mixed cutoff",  1,   0, None,  None),
}
# documented ZERO blocks (must NOT appear in the breakdown): C-C, C-CI2, C-AV, C-WT, pi1-WT, pi2-AV
XI_ZERO_BLOCKS = {"C_C": "L50", "C_CI2": "L52", "C_AV": "L67", "C_WT": "L73",
                  "pi1_WT": "L75", "pi2_AV": "L70"}

# per-term value regression at the reference config (L=4, A=1, n_b=2 cutoffs) — locks the transcription
_REF_TERMS = {
    "AV_AV": 8.415940e7, "AV_WT": 1.158568e7, "CI2_AV": 6.675347e5, "CI2_CI2": 1.800934e2,
    "CI2_WT": 3.432974e5, "WT_WT": 1.270372e7, "free_AV": 7.081392e5, "free_C": 4.009500e3,
    "free_CI2": 3.922853e3, "free_WT": 7.283587e5, "free_free": 5.517101e2, "pi1_AV": 7.093967e5,
    "pi1_pi2": 6.489828e6, "pi2_WT": 7.362564e5,
}
_P = get_physical_parameters()


def _cutoffs(n_b=2, L=4, A=1):
    _, piL, PiL = calculate_dynamic_cutoffs(L, 3, A, _P, epsilon_cut=0.001, E_bound=140.0)
    return cutoffs_for_nb(n_b, _P, ratio=PiL / piL)


def _terms(L=4, A=1, pi_max=None, Pi_max=None):
    if pi_max is None:
        pi_max, Pi_max = _cutoffs()
    return dynamical_pion_xi(L, A, pi_max, Pi_max, _P, return_breakdown=True)[1]


def test_breakdown_matches_source_map():
    """Ξ has EXACTLY the 14 documented terms; every zero block is absent."""
    terms = _terms()
    assert set(terms) == set(XI_PROVENANCE), (set(terms) ^ set(XI_PROVENANCE))
    assert not (set(terms) & set(XI_ZERO_BLOCKS)), "a documented zero block leaked into Ξ"


def test_per_term_value_regression():
    """Each term matches its pinned value — no coefficient can silently drift."""
    terms = _terms()
    for k, ref in _REF_TERMS.items():
        assert abs(terms[k] - ref) <= 1e-4 * ref, f"{k}: {terms[k]:.6e} != {ref:.6e}"


def test_eta_scaling_is_number_restriction():
    """η=A exponent per the source map — the key claim that Trotter cost ∝ A (except pi1_pi2)."""
    pim, Pim = _cutoffs()
    t1 = dynamical_pion_xi(4, 1, pim, Pim, _P, return_breakdown=True)[1]
    t3 = dynamical_pion_xi(4, 3, pim, Pim, _P, return_breakdown=True)[1]
    for k, (_lem, _blk, eta, _L, _pi, _Pi) in XI_PROVENANCE.items():
        assert abs(t3[k] / t1[k] - 3 ** eta) < 1e-9, f"{k}: A-scaling {t3[k]/t1[k]} != 3^{eta}"


def test_L_scaling_only_pion_gradient():
    """Only pi1_pi2 carries an explicit L (Ξ is otherwise L-independent — the crux of A·L³ vs L^6.5).
    Cutoffs held fixed so L enters only where the source map says it does."""
    pim, Pim = _cutoffs()
    t4 = dynamical_pion_xi(4, 1, pim, Pim, _P, return_breakdown=True)[1]
    t8 = dynamical_pion_xi(8, 1, pim, Pim, _P, return_breakdown=True)[1]
    for k, (_lem, _blk, _eta, Lexp, _pi, _Pi) in XI_PROVENANCE.items():
        assert abs(t8[k] / t4[k] - 2 ** Lexp) < 1e-9, f"{k}: L-scaling {t8[k]/t4[k]} != 2^{Lexp}"


def test_cutoff_monomial_exponents():
    """π_max / Π_max exponents for the pure-monomial terms (mixed terms handled below)."""
    pim, Pim = _cutoffs()
    base = dynamical_pion_xi(4, 1, pim, Pim, _P, return_breakdown=True)[1]
    hi_pi = dynamical_pion_xi(4, 1, 2 * pim, Pim, _P, return_breakdown=True)[1]
    hi_Pi = dynamical_pion_xi(4, 1, pim, 2 * Pim, _P, return_breakdown=True)[1]
    for k, (_lem, _blk, _eta, _L, pexp, Pexp) in XI_PROVENANCE.items():
        if pexp is None:
            continue
        assert abs(hi_pi[k] / base[k] - 2 ** pexp) < 1e-9, f"{k}: pi_max^{pexp} check"
        assert abs(hi_Pi[k] / base[k] - 2 ** Pexp) < 1e-9, f"{k}: Pi_max^{Pexp} check"


def test_mixed_terms_are_eta_linear_and_pp_monotone():
    """AV_WT, WT_WT are additive (const+pp / pp+pp²) — verify η-linearity + monotone growth in the
    product pp=π_max·Π_max (both factors scaled together)."""
    pim, Pim = _cutoffs()
    for k in ("AV_WT", "WT_WT"):
        lo = dynamical_pion_xi(4, 1, pim, Pim, _P, return_breakdown=True)[1][k]
        hi = dynamical_pion_xi(4, 1, 1.5 * pim, 1.5 * Pim, _P, return_breakdown=True)[1][k]
        assert hi > lo, f"{k} not monotone in pp"
        # η-linear already covered by test_eta_scaling; re-assert for these two explicitly:
        e1 = dynamical_pion_xi(4, 1, pim, Pim, _P, return_breakdown=True)[1][k]
        e2 = dynamical_pion_xi(4, 2, pim, Pim, _P, return_breakdown=True)[1][k]
        assert abs(e2 / e1 - 2.0) < 1e-9, f"{k} not η-linear"


def test_dominance_regime_flips_with_cutoff():
    """AV_AV (∝π_max²) dominates at the LOW n_b=2 cutoff; WT_WT (∝pp², pp huge) dominates at Watson's
    high cutoff — the documented sensitivity trap. Independent structural check on the two big terms."""
    lo = _terms(pi_max=_cutoffs(2)[0], Pi_max=_cutoffs(2)[1])
    assert max(lo, key=lo.get) == "AV_AV", max(lo, key=lo.get)
    hi_pi, hi_Pi = _cutoffs(39)
    hi = dynamical_pion_xi(10, 40, hi_pi, hi_Pi, _P, return_breakdown=True)[1]
    assert max(hi, key=hi.get) == "WT_WT", max(hi, key=hi.get)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("all Ξ source-map checks passed")
