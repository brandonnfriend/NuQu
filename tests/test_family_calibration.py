"""Gate 5 of nuqu-2026-10, FAMILY CALIBRATION (2026-11-24).

A (frame, dim, n_b, L) family reports an extrapolated limit only if its A=0 cell lands within
sigma of the exact pion vacuum. Checks:
  * `external_bounds.family_calibration` status: failed beats passed; missing when no A=0 limit;
  * the gate refuses a whole family that misses, and one with no check from L=4, and keeps the
    limit and sigma bit-identical when the family passes (or below L=4 with no check);
  * the switch off, or no calibration in the lookup, is bit-identical to the pre-gate pooling;
  * the builder's entry: the exact value's 3-decimal quoting precision is part of the tolerance;
  * the shipped registry: bare n_b=3 L=4 fails (+12.9 MeV/site), squeezed L=2-4 pass, bare L=5
    has no check.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from classical.trimci import external_bounds as xb  # noqa: E402
from classical.trimci import extrapolation as ex  # noqa: E402
from classical.trimci.extrapolation import GATED_CONVENTION, combine_seeds  # noqa: E402
from misc.build_family_calibration import entry  # noqa: E402


def _ladder(offset=0.0):
    dE = [-12.0, -10.0, -8.5, -7.2, -6.1, -5.2, -4.5, -3.9, -3.4, -3.0]
    return [{"core": 1000 * 2 ** k, "E_var": 50.0 + 0.6 * -d + offset, "dE_pt2": d}
            for k, d in enumerate(dE)]


def _e(frame="bare", n_b=3, L=4, ok=True, dev=0.0, sigma=1.0, src="x"):
    return {"frame": frame, "dim": 3, "n_b": n_b, "L": L, "source": src, "ok": ok,
            "E_ps": 300 + dev if ok else None, "sigma_ps": sigma if ok else None,
            "dev_ps": dev if ok else None, "passed": bool(ok and abs(dev) <= sigma)}


def _pool(cal, **kw):
    per = {0: _ladder(), 1: _ladder(0.1), 2: _ladder(0.2)}
    return combine_seeds(per, sites=8, convention=GATED_CONVENTION,
                         external={"upper": [], "exact": None, "calibration": cal}, **kw)


def test_family_status():
    reg = [_e(dev=0.5), _e(dev=3.0, src="y"), _e(L=3, dev=0.5), _e(L=5, ok=False),
           _e(frame="gaussian", dev=0.1)]
    assert xb.family_calibration("bare", 3, 3, 4, registry=reg)["status"] == "failed"
    assert xb.family_calibration("bare", 3, 3, 3, registry=reg)["status"] == "passed"
    assert xb.family_calibration("bare", 3, 3, 5, registry=reg)["status"] == "missing"
    assert xb.family_calibration("bare", 3, 4, 4, registry=reg)["status"] == "missing"
    assert xb.family_calibration("gaussian", 3, 3, 4, registry=reg)["status"] == "passed"
    # a different cutoff or dim is a different family
    assert xb.family_calibration("bare", 3, 2, 3, registry=reg)["status"] == "missing"


def test_gate_refuses_failed_and_missing_families_and_leaves_passing_ones_alone():
    base = _pool(None)
    assert base["ok"] and "family_calibration" not in base["gate"]["checks"]

    ok = _pool(xb.family_calibration("bare", 3, 3, 4, registry=[_e(dev=0.5)]))
    assert ok["ok"] and ok["gate"]["checks"]["family_calibration"]["passed"]
    assert (ok["E_inf"], ok["sigma"]) == (base["E_inf"], base["sigma"])

    bad = _pool(xb.family_calibration("bare", 3, 3, 4, registry=[_e(dev=0.5), _e(dev=3.0)]))
    assert not bad["ok"] and bad["E_inf"] is None and bad["sigma"] is None
    assert bad["gate"]["refused_by"] == ["family_calibration"]
    assert bad["gate"]["E_inf_refused"] == base["E_inf"]
    assert "misses the exact A=0 vacuum" in bad["reason"]

    miss4 = _pool(xb.family_calibration("bare", 3, 3, 4, registry=[]))
    assert not miss4["ok"] and miss4["gate"]["refused_by"] == ["family_calibration"]
    assert "no A=0 calibration" in miss4["reason"]
    miss3 = _pool(xb.family_calibration("bare", 3, 3, 3, registry=[]))
    assert miss3["ok"] and (miss3["E_inf"], miss3["sigma"]) == (base["E_inf"], base["sigma"])
    assert miss3["gate"]["checks"]["family_calibration"]["status"] == "missing"


def test_switch_off_is_the_pre_gate_pooling():
    base = _pool(None)
    try:
        ex.FAMILY_CALIBRATION_GATE = False
        off = _pool(xb.family_calibration("bare", 3, 3, 4, registry=[_e(dev=3.0)]))
    finally:
        ex.FAMILY_CALIBRATION_GATE = True
    assert off["ok"] and (off["E_inf"], off["sigma"]) == (base["E_inf"], base["sigma"])
    assert "family_calibration" not in off["gate"]["checks"]


def test_other_conventions_ignore_the_gate():
    per = {0: _ladder(), 1: _ladder(0.1)}
    cal = xb.family_calibration("bare", 3, 3, 4, registry=[_e(dev=3.0)])
    for conv in ("trimci-coo", "nuqu-2026-09"):
        a = combine_seeds(per, sites=8, convention=conv)
        b = combine_seeds(per, sites=8, convention=conv,
                          external={"upper": [], "exact": None, "calibration": cal})
        assert (a["ok"], a.get("E_inf"), a.get("sigma")) == (b["ok"], b.get("E_inf"), b.get("sigma"))


def test_builder_entry_uses_the_exact_value_precision():
    # squeezed L=2 A=0: 304.14916 +- 0.00005 vs exact 304.149 quoted to 3 decimals
    e = entry("sq", "gaussian", 3, 2, 3, True, 304.1491570, 4.92e-05, 304.1492554)
    assert e["exact_precision_ps"] == 0.0005 and e["passed"]
    e = entry("bare", "bare", 3, 4, 3, True, 357.5404, 5.3395, 368.2194)
    assert not e["passed"] and abs(e["dev_ps"] - 12.944) < 1e-3
    e = entry("bare5", "bare", 3, 5, 3, True, 380.0, 4.8, 385.0)
    assert not e["passed"] and e["exact_ps"] is None and "no exact" in e["reason"]
    e = entry("bo", "bare", 3, 4, 3, False, None, None, 369.07)
    assert not e["passed"] and "not reported" in e["reason"]


def test_shipped_registry():
    st = lambda fr, nb, L: xb.family_calibration(fr, 3, nb, L)["status"]
    assert st("bare", 3, 4) == "failed"
    for L in (2, 3, 4):
        assert st("gaussian", 3, L) == "passed"
    for L in (2, 3):
        assert st("bare", 3, L) == "passed"
    assert st("bare", 3, 5) == "missing" and st("bare", 2, 4) == "missing"
    # the lookup carries it; calibration=False leaves it out
    assert xb.lookup("bare", 3, 4, 6, 3, "wick")["calibration"]["status"] == "failed"
    assert xb.lookup("bare", 3, 4, 6, 3, "wick", calibration=False)["calibration"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_family_calibration: PASS")
