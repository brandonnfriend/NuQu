"""Unit tests for misc/make_nb3_headline.build — the n_b=3 compiled/projected headline (re-audit P0-6).

Checks: exact-vs-projected classification, missing-L handling, the ratio calc, and the projection band.
Run: python -m pytest -q tests/test_nb3_headline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.make_nb3_headline import build


def _mk(T, q, lam):
    return dict(T=T, q=q, lam=lam, walkT=T / 100.0)


# n_b=2 anchor L=1..6,10; n_b=3 compiled only L=1..6, with clean per-L ratios (T×33, q×1.3, lam×3.75)
NB2 = {L: _mk(1e3 * 10 ** L, 10 * L, 1.0 * L) for L in (1, 2, 3, 4, 5, 6, 10)}
NB3 = {L: _mk(NB2[L]["T"] * 33.0, NB2[L]["q"] * 1.3, NB2[L]["lam"] * 3.75) for L in (1, 2, 3, 4, 5, 6)}


def test_exact_vs_projected_classification():
    rows, _ = build(NB2, NB3)
    assert rows[4]["exact"] is True and rows[6]["exact"] is True     # compiled where n_b=3 exists
    assert rows[10]["exact"] is False                                # projected where it doesn't


def test_projected_value_and_ratio():
    rows, sc = build(NB2, NB3)
    assert abs(sc["Tr"] - 33.0) < 1e-9 and abs(sc["qr"] - 1.3) < 1e-9 and abs(sc["lamr"] - 3.75) < 1e-9
    # projected T = n_b=2 T × mean ratio
    assert abs(rows[10]["T"] - NB2[10]["T"] * 33.0) < 1e-6
    # exact rows carry the compiled value
    assert abs(rows[4]["T"] - NB3[4]["T"]) < 1e-6


def test_projection_band_from_ratio_spread():
    # perturb one ratio so the band is nonzero
    nb3 = dict(NB3); nb3[6] = _mk(NB2[6]["T"] * 30.0, NB2[6]["q"] * 1.29, NB2[6]["lam"] * 3.70)
    rows, sc = build(NB2, nb3)
    lo, hi = sc["Trange"]
    assert lo < sc["Tr"] < hi                                        # band brackets the mean
    assert rows[10]["Tlo"] == NB2[10]["T"] * lo and rows[10]["Thi"] == NB2[10]["T"] * hi
    assert rows[10]["Tlo"] < rows[10]["T"] < rows[10]["Thi"]         # projected point inside its band


def test_missing_L_only_iterates_nb2():
    # an L present in n_b=3 but NOT n_b=2 must not appear (build iterates n_b=2 keys)
    nb3 = dict(NB3); nb3[7] = _mk(1e12, 500, 30.0)
    rows, _ = build(NB2, nb3)
    assert 7 not in rows and set(rows) == set(NB2)


def test_exact_rows_have_degenerate_band():
    rows, _ = build(NB2, NB3)
    assert rows[4]["Tlo"] == rows[4]["Thi"] == rows[4]["T"]          # compiled point: no band


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("all nb3_headline tests passed")
