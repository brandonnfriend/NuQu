"""The gated error convention `nuqu-2026-10` (2026-10-10) and the external-bound registry.

docs/classical_error_convention_2026-10-08.md s11: the block2 DMRG calibration (294249) refuted
`trimci-coo`'s sigma on the identical bare L=2 Hamiltonian. Checks:
  * the registry loads, matches same-operator cells only, and moves between contact conventions;
  * sigma = max(bootstrap, half extrapolation distance, matched-depth seed spread), each recorded;
  * gate 2 refuses a limit above an external same-operator bound, gate 3 a cross-check gap wider
    than sigma, and a shallower seed never enters the matched-depth spread;
  * `trimci-coo` and `nuqu-2026-09` are untouched (same E_inf and sigma as before the switch);
  * the window shift is carried to the pooled record on every convention;
  * ACCEPTANCE on the real 293963 sweep (skipped when the data clone is absent): with the DMRG
    numbers in the registry, bare L=2 A=4 and A=8 are refused and A=2, 6, 10 report limits whose
    sigma (+ DMRG's own extrapolation distance) covers the DMRG deviation.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from classical.trimci import external_bounds as xb  # noqa: E402
from classical.trimci.extrapolation import (GATED_CONVENTION, LEGACY_CONVENTION,  # noqa: E402
                                            LITERATURE_CONVENTION, combine_seeds,
                                            einf_literature, einf_with_uncertainty)

SWEEP = os.path.join(_ROOT, "data/classical/2026-09-25/bare_Asweep_nb3_293963")
DMRG = os.path.join(_ROOT, "data/classical/2026-10-10/dmrg_calib_294249")


def _ladder(offset=0.0, n=10, slope=0.6):
    dE = [-12.0, -10.0, -8.5, -7.2, -6.1, -5.2, -4.5, -3.9, -3.4, -3.0][:n]
    return [{"core": 1000 * 2 ** k, "E_var": 50.0 - slope * d + offset, "dE_pt2": d}
            for k, d in enumerate(dE)]


# ---------------------------------------------------------------------------- registry
def test_registry_loads_and_matches_same_operator_only():
    reg = xb.load_registry()
    assert len(reg) >= 8
    ext = xb.lookup("bare", 3, 2, 4, 3, "wick")
    assert len(ext["upper"]) == 1 and abs(ext["upper"][0]["E_ps"] - 275.573) < 1e-3
    assert ext["exact"] is None
    # a different frame or cutoff is a different operator: no upper bound
    assert xb.lookup("gaussian", 3, 2, 4, 3, "wick")["upper"] == []
    assert xb.lookup("bare", 2, 2, 4, 3, "wick")["upper"] == []
    # the exact vacuum is untruncated: any frame, any n_b, A=0 only
    for frame, nb in (("bare", 3), ("gaussian", 3), ("bare", 4)):
        e = xb.lookup(frame, nb, 3, 0, 3, "wick")
        assert e["upper"] == [] and abs(e["exact"]["E_ps"] - 331.461) < 1e-9
    assert xb.lookup("bare", 3, 5, 0, 3, "wick")["exact"] is None


def test_registry_contact_convention_shift():
    w = xb.lookup("bare", 3, 2, 4, 3, "wick")["upper"][0]
    l = xb.lookup("bare", 3, 2, 4, 3, "legacy")["upper"][0]
    assert abs((w["E"] - l["E"]) - 23.3725 * 4) < 1e-3          # legacy = wick + C_SELF*A
    assert l["contact_convention"] == "legacy"
    z = xb.lookup("bare", 3, 2, 0, 3, "legacy")["exact"]
    assert abs(z["E_ps"] - 304.149) < 1e-9                        # A=0: no contact term


# ---------------------------------------------------------------------------- sigma terms
def test_sigma_is_max_of_three_terms_and_each_is_recorded():
    per = {0: _ladder(), 1: _ladder(0.2), 2: _ladder(0.4)}
    p = combine_seeds(per, sites=8, convention=GATED_CONVENTION)
    assert p["ok"] and p["sigma_convention"] == GATED_CONVENTION
    t = p["sigma_terms"]
    assert set(t) == {"bootstrap", "extrap_half_distance", "seed_spread"}
    assert abs(t["extrap_half_distance"] - 0.5 * (p["E_var_bound"] - p["E_inf"])) < 1e-12
    assert abs(t["seed_spread"] - 0.4) < 1e-9          # exact fits: spread = planted offsets
    assert p["sigma"] == max(v for v in t.values() if v is not None)
    assert p["sigma_source"] in t and t[p["sigma_source"]] == p["sigma"]
    assert p["gate"]["passed"] and p["gate"]["refused_by"] == []
    assert p["seed_robustness"]["n_matched_depth"] == 3 and "PASS" not in p["seed_robustness"]["check"]
    assert "window_shift" in p and "window_shift_by_w" in p


def test_shallower_seed_is_excluded_from_matched_depth_spread():
    per = {0: _ladder(), 1: _ladder(5.0, n=8)}      # seed 1 stops two rungs short, 5 MeV off
    p = combine_seeds(per, sites=8, convention=GATED_CONVENTION)
    assert p["ok"] and p["best_seed"] == 0
    assert p["sigma_terms"]["seed_spread"] is None and p["n_seeds_matched_depth"] == 1
    assert p["sigma_source"] in ("bootstrap", "extrap_half_distance")


# ---------------------------------------------------------------------------- the gate
def test_gate_refuses_a_limit_above_an_external_same_operator_bound():
    per = {0: _ladder()}
    base = combine_seeds(per, sites=8, convention=GATED_CONVENTION)
    assert base["ok"]
    ext = {"upper": [{"E": base["E_inf"] - 1.0, "E_ps": (base["E_inf"] - 1.0) / 8,
                      "method": "planted DMRG", "kind": "variational_upper"}], "exact": None}
    p = combine_seeds(per, sites=8, convention=GATED_CONVENTION, external=ext)
    assert not p["ok"] and p["E_inf"] is None and p["sigma"] is None
    assert p["gate"]["refused_by"] == ["variational"]
    assert abs(p["gate"]["E_inf_refused"] - base["E_inf"]) < 1e-12
    assert "planted DMRG" in p["reason"] and "quote the bound" in p["reason"]
    assert p["E_bound_best_known_source"] == "planted DMRG"
    # an external bound ABOVE the limit does not refuse, and the deviation is recorded
    loose = base["E_var_bound"] + 1.0                   # above OUR bound too: ours stays tightest
    ext2 = {"upper": [{"E": loose, "E_ps": 0, "method": "loose", "kind": "variational_upper"}],
            "exact": None}
    q = combine_seeds(per, sites=8, convention=GATED_CONVENTION, external=ext2)
    assert q["ok"] and abs(q["external_reference"]["upper"][0]["dev"] - (base["E_inf"] - loose)) < 1e-12
    assert q["E_bound_best_known_source"] == "trimci"


def test_gate_refuses_a_limit_below_the_exact_floor_by_more_than_sigma():
    per = {0: _ladder()}
    base = combine_seeds(per, sites=8, convention=GATED_CONVENTION)
    floor = {"upper": [], "exact": {"E": base["E_inf"] + 10 * base["sigma"] + 1.0, "E_ps": 0,
                                    "method": "planted exact", "kind": "exact"}}
    p = combine_seeds(per, sites=8, convention=GATED_CONVENTION, external=floor)
    assert not p["ok"] and p["gate"]["refused_by"] == ["variational"]
    assert "exact untruncated" in p["reason"]
    close = {"upper": [], "exact": {"E": base["E_inf"] + 0.5 * base["sigma"], "E_ps": 0,
                                    "method": "planted exact", "kind": "exact"}}
    assert combine_seeds(per, sites=8, convention=GATED_CONVENTION, external=close)["ok"]


def test_gate_refuses_when_the_cross_check_gap_exceeds_sigma():
    # curve the ladder so the power-law cross-check lands far from the PT2 line
    rungs = _ladder()
    for k, r in enumerate(rungs):
        r["E_var"] += 3.0 * np.exp(-0.9 * k)
    p = combine_seeds({0: rungs}, sites=8, convention=GATED_CONVENTION)
    cc = p["gate"]["checks"]["cross_check"]
    if cc["resolved"] and cc["gap"] > p["gate"]["sigma_if_reported"]:
        assert not p["ok"] and "cross_check" in p["gate"]["refused_by"]
    else:                                           # the fit happened to agree: then it passes
        assert p["ok"] or "cross_check" not in p["gate"]["refused_by"]


# ---------------------------------------------------------------------------- bit-identity
def test_other_conventions_are_untouched():
    per = {0: _ladder(), 1: _ladder(0.3)}
    lit = combine_seeds(per, sites=8, convention=LITERATURE_CONVENTION)
    ref = einf_literature(per[lit["best_seed"]], sites=8)
    assert lit["E_inf"] == ref["E_inf"] and lit["sigma"] == ref["sigma"]
    assert lit["window_shift"] == ref["window_shift"]             # plumbed, not altered
    assert lit["seed_robustness"]["check"].startswith(("PASS", "FAIL", "SINGLE"))
    old = combine_seeds(per, sites=8, convention=LEGACY_CONVENTION)
    ref2 = einf_with_uncertainty(per[old["best_seed"]], sites=8, min_post=3)
    assert old["E_inf"] == ref2["E_inf"] and old["sigma"] == ref2["sigma"]
    assert old["window_shift"] is None
    # an external bound never changes a trimci-coo number, only records the comparison
    ext = {"upper": [{"E": lit["E_inf"] - 1.0, "E_ps": 0, "method": "x", "kind": "variational_upper"}],
           "exact": None}
    lit2 = combine_seeds(per, sites=8, convention=LITERATURE_CONVENTION, external=ext)
    assert lit2["ok"] and lit2["E_inf"] == lit["E_inf"] and lit2["sigma"] == lit["sigma"]
    assert abs(lit2["external_reference"]["upper"][0]["dev"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------- acceptance
@pytest.mark.skipif(not (os.path.isdir(SWEEP) and os.path.isdir(DMRG)),
                    reason="needs the data clone (293963 sweep + 294249 DMRG shards)")
def test_calibration_table_reproduced_on_real_data():
    from misc.aggregate_classical_energies import analyze, load
    from misc.compare_dmrg_trimci import dmrg_extrapolate, load_dmrg
    dm = load_dmrg([DMRG], 2, 3, 8)
    for A, expect_refused in ((2, False), (4, True), (6, False), (8, True), (10, False)):
        g, m = load([SWEEP], A=A, frame="bare")
        r = analyze(g, m, sigma_convention=GATED_CONVENTION)[0]
        assert r["L"] == 2 and r["n_b"] == 3
        if expect_refused:
            assert not r["ok"] and r["gate"]["refused_by"] == ["variational"], (A, r["reason"])
            assert "block2 DMRG" in r["reason"]
            assert "block2" in r["E_bound_best_known_source"]
        else:
            assert r["ok"], (A, r["reason"])
            x = dmrg_extrapolate(dm[A]["results"])
            dev = abs(r["E_inf_ps"] - x["E_inf"] / 8)
            assert dev <= r["sigma_ps"] + x["delta"] / 8 + 1e-6, (A, dev, r["sigma_ps"])
            assert 0.15 <= r["sigma_ps"] <= 0.30, (A, r["sigma_ps"])     # s11e: bare L=2 0.19-0.24
            assert r["sigma_source"] == "extrap_half_distance"
