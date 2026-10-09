"""The TrimCI/COO literature sigma convention (2026-10-06) in `classical/trimci/extrapolation.py`.

Matches Zhang & Otten's TrimCI (PT2-linear extrapolation) and COO (arXiv:2605.22977, SM S5.2:
R^2-scan power law + 500-replicate bootstrap sigma) papers. Checks:
  * the R^2-scan recovers a planted power-law limit, below the lowest computed energy, with a
    bootstrap sigma and a 90% c.i. that contains it;
  * the PT2-linear bootstrap recovers a planted intercept; with noise its sigma is the scatter;
  * a ladder with no curvature (pure log decay) is REFUSED rather than given a limit;
  * a pool-energy top rung stays the bound and is kept out of both fits;
  * the primary is PT2-linear when PT2 exists, the power law is the reported cross-check, and
    neither the cross-check gap nor the seed spread enters sigma;
  * `convention="nuqu-2026-09"` still reproduces the previous convention exactly.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from classical.trimci.extrapolation import (combine_seeds, einf_literature,  # noqa: E402
                                            einf_with_uncertainty, fit_einf_power_r2scan,
                                            fit_einf_pt2_bootstrap)

CORES = [1000 * 2 ** k for k in range(9)]


def _power(E_inf=100.0, a=400.0, alpha=0.5):
    return [E_inf + a * N ** -alpha for N in CORES]


def test_r2scan_recovers_planted_limit():
    E = _power()
    f = fit_einf_power_r2scan(CORES, E)
    assert f["ok"], f["reason"]
    assert f["E_inf"] < min(E)
    assert abs(f["E_inf"] - 100.0) < 0.05 and abs(f["alpha"] - 0.5) < 0.01
    lo, hi = f["ci90"]
    assert lo - 0.05 <= 100.0 <= hi + 0.05 and f["sigma"] >= 0   # 0.05 = scan-grid resolution here


def test_no_curvature_is_refused():
    E = [500.0 - 3.0 * np.log(N) for N in CORES]          # log decay: no finite limit
    f = fit_einf_power_r2scan(CORES, E)
    assert not f["ok"] and "curvature" in f["reason"] and f["E_inf"] is None


def test_pt2_bootstrap_recovers_intercept():
    dE = np.array([-12.0, -10.0, -8.5, -7.2, -6.1, -5.2])
    Ev = 50.0 - 0.6 * dE                                   # E_var + dE = 50 + 0.4 dE
    f = fit_einf_pt2_bootstrap(Ev, dE)
    assert f["ok"] and abs(f["E_inf"] - 50.0) < 1e-9 and f["sigma"] < 1e-9
    noisy = Ev + np.random.default_rng(1).normal(0, 0.05, len(Ev))
    g = fit_einf_pt2_bootstrap(noisy, dE)
    assert g["ok"] and 0 < g["sigma"] < 1.0 and g["ci90"][0] < g["E_inf"] < g["ci90"][1]


def _ladder(with_pool=False):
    dE = [-12.0, -10.0, -8.5, -7.2, -6.1, -5.2, -4.5]
    r = [{"core": 1000 * 2 ** k, "E_var": 50.0 - 0.6 * d + 0.01 * k, "dE_pt2": d}
         for k, d in enumerate(dE)]
    if with_pool:
        r.append({"core": r[-1]["core"] * 2, "E_var": r[-1]["E_var"] - 1.0, "dE_pt2": None})
    return r


def test_primary_crosscheck_and_pool_rung():
    v = einf_literature(_ladder(with_pool=True), sites=8)
    assert v["ok"] and v["primary"] == "pt2_linear"
    assert v["pool_rungs_excluded_from_fit"] == [128000]
    assert v["E_var_bound"] == _ladder(True)[-1]["E_var"]
    assert v["cross_check"]["estimator"] == "power_r2scan"
    assert v["sigma"] == v["pt2"]["sigma"], "sigma must be the primary's bootstrap only"


def test_seed_spread_not_in_sigma_and_legacy_switch():
    per = {0: _ladder(), 1: [dict(r, E_var=r["E_var"] + 0.3) for r in _ladder()]}
    lit = combine_seeds(per, sites=8)
    assert lit["sigma_convention"] == "trimci-coo"
    assert lit["sigma"] == lit["per_seed"][lit["best_seed"]]["sigma"]
    old = combine_seeds(per, sites=8, convention="nuqu-2026-09")
    ref = einf_with_uncertainty(per[old["best_seed"]], sites=8, min_post=3)
    assert old["sigma"] == ref["sigma"] and old["E_inf"] == ref["E_inf"]


def test_late_window_and_tie_break():
    """Seeds with IDENTICAL top rungs but different early drops must give the same E_inf
    (late window), and a bound tie picks the lowest extrapolating seed, not float noise
    (squeeze L=2 A=4/8, 2026-10-09)."""
    dE = [-12.0, -10.0, -8.5, -7.2, -6.1, -5.2, -4.5, -3.9, -3.4, -3.0]
    late = [{"core": 1000 * 2 ** k, "E_var": 50.0 - 0.6 * d, "dE_pt2": d} for k, d in enumerate(dE)]
    a = [dict(r) for r in late]
    a[3]["E_var"] += 5.0; a[2]["E_var"] += 5.5; a[1]["E_var"] += 6.0; a[0]["E_var"] += 6.5
    b = [dict(r) for r in late]
    b[0]["E_var"] += 9.0                          # largest drop at the very first rung
    va, vb = einf_literature(a, sites=8), einf_literature(b, sites=8)
    assert va["post_cores"] == vb["post_cores"] == [r["core"] for r in late[-6:]]
    assert va["ok"] and vb["ok"] and abs(va["E_inf"] - vb["E_inf"]) < 1e-12
    assert "window_shift" in va
    pooled = combine_seeds({2: b, 0: a, 1: b}, sites=8)
    assert pooled["best_seed"] == 0


def main():
    test_r2scan_recovers_planted_limit()
    test_no_curvature_is_refused()
    test_pt2_bootstrap_recovers_intercept()
    test_primary_crosscheck_and_pool_rung()
    test_seed_spread_not_in_sigma_and_legacy_switch()
    test_late_window_and_tie_break()
    print("test_sigma_literature: PASS  (R2-scan limit + c.i.; no-curvature refusal; PT2 bootstrap; "
          "PT2 primary / power cross-check; pool rung = bound only; legacy switch intact)")


if __name__ == "__main__":
    main()
