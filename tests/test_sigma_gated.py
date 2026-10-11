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
from classical.trimci.extrapolation import (DEFAULT_SIGMA_CONVENTION, GATED_CONVENTION,  # noqa: E402
                                            LEGACY_CONVENTION, LITERATURE_CONVENTION,
                                            _cross_check_gate, combine_seeds,
                                            einf_literature, einf_with_uncertainty)

SWEEP = os.path.join(_ROOT, "data/classical/2026-09-25/bare_Asweep_nb3_293963")
SQUEEZE = os.path.join(_ROOT, "data/classical/2026-10-08/Asweep_squeeze_nb3_294246")
DMRG = os.path.join(_ROOT, "data/classical/2026-10-10/dmrg_calib_294249")


def test_gated_convention_is_the_default_since_2026_10_10():
    assert DEFAULT_SIGMA_CONVENTION == GATED_CONVENTION
    p = combine_seeds({0: _ladder()}, sites=8)
    assert p["sigma_convention"] == GATED_CONVENTION and "gate" in p


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
    assert set(t) == {"bootstrap", "extrap_half_distance", "seed_spread", "cross_check_gap"}
    assert abs(t["extrap_half_distance"] - 0.5 * (p["E_var_bound"] - p["E_inf"])) < 1e-12
    assert abs(t["seed_spread"] - 0.4) < 1e-9          # exact fits: spread = planted offsets
    assert p["sigma"] == max(v for v in t.values() if v is not None)
    assert p["sigma_source"] in t and t[p["sigma_source"]] == p["sigma"]
    assert p["sigma_three_term"] == max(t[k] for k in ("bootstrap", "extrap_half_distance", "seed_spread")
                                        if t[k] is not None)
    cc = p["gate"]["checks"]["cross_check"]
    assert {"resolved", "sigma_cc", "tolerance", "gap", "extrap_distance", "reason"} <= set(cc)
    if cc["resolved"]:                                  # a resolved cross-check is a sigma term
        assert t["cross_check_gap"] == cc["gap"] and cc["gap"] <= cc["tolerance"] + 1e-12
    else:
        assert t["cross_check_gap"] is None
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


def test_cross_check_gate_rules():
    """Gate 3, second pass (2026-10-10): resolved = sigma_cc within the extrapolation distance;
    pass = the two 1-sigma intervals overlap (|gap| <= sigma3 + sigma_cc)."""
    # squeeze L=2 A=6 (per site x 8 sites): gap 0.0079, sigma3 0.0058, sigma_cc 0.0043, d 0.0117
    s = 8
    c = _cross_check_gate(0.0079 * s, 0.0043 * s, 0.0058 * s, 0.0117 * s)
    assert c["resolved"] and c["passed"] and abs(c["tolerance"] - 0.0101 * s) < 1e-9
    # the same gap against a cross-check that is twice as sharp: disjoint -> refused
    c2 = _cross_check_gate(0.0079 * s, 0.0010 * s, 0.0058 * s, 0.0117 * s)
    assert c2["resolved"] and not c2["passed"]
    # bare L=3 A=4: the power law's own sigma (145/site) dwarfs the 5.3/site correction -> n/r
    c3 = _cross_check_gate(5.2947 * 27, 145.35 * 27, 2.6477 * 27, 5.2954 * 27)
    assert not c3["resolved"] and c3["passed"] and "unresolved" in c3["reason"]
    # no cross-check at all (L=4): passes, not resolved
    c4 = _cross_check_gate(None, None, 12.9, 25.8)
    assert c4["passed"] and not c4["resolved"]
    # a cross-check without its own sigma is judged strictly (tolerance = sigma3)
    c5 = _cross_check_gate(1.0, None, 0.9, 5.0)
    assert c5["resolved"] and not c5["passed"] and c5["tolerance"] == 0.9


def test_gate_refuses_when_two_resolved_estimators_disagree():
    # curve the ladder so the power-law cross-check lands far from the PT2 line
    rungs = _ladder()
    for k, r in enumerate(rungs):
        r["E_var"] += 3.0 * np.exp(-0.9 * k)
    p = combine_seeds({0: rungs}, sites=8, convention=GATED_CONVENTION)
    cc = p["gate"]["checks"]["cross_check"]
    if cc["resolved"] and cc["gap"] > cc["tolerance"]:
        assert not p["ok"] and "cross_check" in p["gate"]["refused_by"]
        assert "do not overlap" in p["reason"]
    elif cc["resolved"]:                            # overlapping: passes and sigma covers the gap
        assert p["ok"] and p["sigma"] >= cc["gap"] - 1e-12
    else:                                           # unresolved: it does not vote
        assert "cross_check" not in p["gate"]["refused_by"]
        assert p["sigma_terms"]["cross_check_gap"] is None


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
            assert r["gate"]["checks"]["cross_check"]["resolved"]          # L=2: sigma_cc <= 0.04/site


@pytest.mark.skipif(not (os.path.isdir(SWEEP) and os.path.isdir(SQUEEZE)),
                    reason="needs the data clone (293963 + 294246 sweeps)")
def test_cross_check_second_pass_on_real_data():
    """2026-10-10 second pass of gate 3. Squeeze L=2 A=6 / A=10 (gap 0.008 / 0.012 per site
    against sigma3 0.006 / 0.011 and sigma_cc 0.004 / 0.009) are consistent and now report,
    with sigma widened to the gap; bare L=3 reports on the half-distance (s11e expected
    2.4-3.1/site) because the power law's own sigma there is 7-145/site -- unresolved, no vote."""
    from misc.aggregate_classical_energies import analyze, load
    for A, lo, hi in ((6, 0.005, 0.012), (10, 0.010, 0.016)):
        g, m = load([SQUEEZE], A=A, frame="gaussian")
        r = next(r for r in analyze(g, m, sigma_convention=GATED_CONVENTION) if r["L"] == 2)
        cc = r["gate"]["checks"]["cross_check"]
        assert r["ok"], (A, r["reason"])
        assert cc["resolved"] and cc["gap"] > r["sigma_three_term"]        # the near miss
        assert r["sigma_source"] == "cross_check_gap" and lo <= r["sigma_ps"] <= hi, (A, r["sigma_ps"])
        assert abs(r["sigma"] - cc["gap"]) < 1e-12
    g, m = load([SWEEP], A=4, frame="bare")
    r = next(r for r in analyze(g, m, sigma_convention=GATED_CONVENTION) if r["L"] == 3)
    cc = r["gate"]["checks"]["cross_check"]
    assert not cc["resolved"] and cc["sigma_cc_ps"] > 5.0 and cc["gap_ps"] > 5.0
    assert r["ok"] and r["sigma_source"] == "extrap_half_distance" and 2.4 <= r["sigma_ps"] <= 3.1
    assert r["sigma_terms"]["cross_check_gap"] is None
