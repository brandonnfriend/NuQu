"""Unit tests for misc/make_nb3_headline.build — the n_b=3 compiled/projected headline (re-audit P0-6).

build() now delegates the L=8..10 projection to the quantization-aware padding model
(misc/nb3_padding_model): walk_T is a step function of L (pyLIQTR pads PREPARE to 2^ceil(log2 terms)),
so a smooth per-step ratio is wrong at bin boundaries. These tests check the contract: exact-vs-
projected classification, the display ratios, the bin-uncertainty band, missing-L handling, and —
the load-bearing scientific check — that the padding model back-predicts the compiled walk_T.

Fixtures are generated FROM the padding law (clean n_terms ∝ L³, constant b/P, log-linear a/P) so the
model recovers them; each row carries the full schema build() consumes (lam, walkT, q, terms, eps, a, b).
Run: python -m pytest -q tests/test_nb3_headline.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.make_nb3_headline import build
from misc.nb3_padding_model import fit_model, walk_T_at_bin, _bin
from src_PI.estimation.qpe_cost import (walk_queries, WALK_QUERY_CONSTANT_HEISENBERG as PI,
                                        qpe_phase_register_qubits_from_nwalk as _m_from_nwalk)
from src_PI.estimation.total_t_optimizer import optimize_qpe_fraction

BP, M, K = 6.0206, 6.0, 15.0          # b/P constant; a/P = M·log2(P)+K  (the padding law)
CPS3, CPS2 = 5400, 675                # terms/site: n_b=3 ~8× n_b=2 → q,λ ratios land ~1.3, ~3.75
DE = 1.0


def _row(terms, lam, q):
    """A stand-in for one `nb3_padding_model._load` record.

    Must mirror that schema, including the phase register: `q` is the WALK/block-encoding
    register and `q_total = q + m` is what the headline reports as "total logical qubits"
    (CONVENTIONS.md section 4). Omitting q_total here would let the fixture drift from the
    loader and hide a real break -- which is exactly what happened when the headline switched
    to the total on 2026-09-09."""
    P = _bin(terms)
    b = BP * P
    a = (M * math.log2(P) + K) * P
    opt = optimize_qpe_fraction(a, b, lam, DE)
    n_walk = walk_queries(lam, opt["eps_qpe"], PI)
    T = n_walk * opt["walk_T"]
    m = _m_from_nwalk(n_walk)
    return dict(lam=lam, walkT=opt["walk_T"], q=q, m=m, q_total=q + m, terms=terms,
                eps=opt["eps_qpe"], a=a, bb=b, dE=DE, T=T)


NB2 = {L: _row(CPS2 * L ** 3, 1.0 * L, 10 * L) for L in range(1, 11)}          # L=1..10 compiled
NB3 = {L: _row(CPS3 * L ** 3, 3.75 * L, 13 * L) for L in range(1, 8)}          # L=1..7 compiled


def test_exact_vs_projected_classification():
    rows, _ = build(NB2, NB3)
    assert rows[4]["exact"] is True and rows[7]["exact"] is True      # compiled where n_b=3 exists
    assert rows[10]["exact"] is False                                 # projected where it doesn't


def test_display_ratios_from_smooth_regime():
    rows, sc = build(NB2, NB3)
    assert abs(sc["lamr"] - 3.75) < 1e-9                              # λ ratio exact by construction
    # The qubit ratio is on the REPORTED total (walk + QPE phase register), not the walk
    # register alone. The walk registers are 13L vs 10L = exactly 1.3; adding m (which differs
    # by only a couple of qubits between the cutoffs) shifts the total ratio slightly off 1.3.
    # Pinning it at exactly 1.3 would silently re-assert the walk-only convention.
    walk_ratio = sum(NB3[L]["q"] / NB2[L]["q"] for L in (4, 5, 6)) / 3
    assert abs(walk_ratio - 1.3) < 1e-9                               # 13L / 10L, walk register
    assert abs(sc["qr"] - 1.3) < 0.05 and sc["qr"] != walk_ratio      # total: near it, not it
    assert sc["fitL"] == [4, 5, 6]                                    # L=7 excluded from the fit
    assert 25 < sc["Tr"] < 45                                         # smooth-regime total-T ratio


def test_reported_qubits_include_the_phase_register():
    """The headline reports walk + m. Until 2026-09-09 it reported the walk register while
    calling it "total logical qubits"; this pins the fix so it cannot regress."""
    rows, _ = build(NB2, NB3)
    for L in (4, 7, 10):
        r = rows[L]
        assert r["m"] and r["m"] > 0, f"L={L}: no phase register recorded"
        assert abs(r["q"] - (r["qwalk"] + r["m"])) < 1e-9, f"L={L}: reported q != walk + m"
        assert r["q"] > r["qwalk"], f"L={L}: reported qubits not above the walk register"
        assert r["qlo"] <= r["q"] <= r["qhi"], f"L={L}: band does not bracket the value"


def test_projected_is_positive_and_banded():
    rows, _ = build(NB2, NB3)
    for L in (8, 9, 10):
        r = rows[L]
        assert r["T"] > 0 and r["Tlo"] <= r["T"] <= r["Thi"]         # central inside its band
        assert r["Thi"] / r["Tlo"] < 3.0                             # band spans at most a neighbour bin


def test_exact_rows_have_degenerate_band():
    rows, _ = build(NB2, NB3)
    assert rows[4]["Tlo"] == rows[4]["Thi"] == rows[4]["T"]          # compiled point: no band
    assert rows[7]["Tlo"] == rows[7]["Thi"] == rows[7]["T"]          # incl. the (compiled) L=7 point


def test_missing_L_only_iterates_nb2():
    # an L present in n_b=3 but NOT n_b=2 must not appear (build iterates n_b=2 keys)
    nb3 = dict(NB3); nb3[11] = _row(CPS3 * 11 ** 3, 3.75 * 11, 13 * 11)
    rows, _ = build(NB2, nb3)
    assert 11 not in rows and set(rows) == set(NB2)


def test_padding_model_backtests_compiled_walk_T():
    """The load-bearing check: the padding model reproduces the compiled walk_T on L=4..7 (<3%)."""
    mp = fit_model(NB3)
    for L in (4, 5, 6, 7):
        P = _bin(NB3[L]["terms"])
        wT, _T, _a, _b = walk_T_at_bin(mp, P, NB3[L]["lam"], DE)
        err = abs(wT - NB3[L]["walkT"]) / NB3[L]["walkT"]
        assert err < 0.03, f"L={L} back-test error {err:.1%}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("all nb3_headline tests passed")
