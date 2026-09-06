"""Basin-aware E_infinity with an honest error bar (audit 2026-09-05, P0-2).

`extrapolation.einf_with_uncertainty` replaces the old binary "converged / upper bound"
label with an extrapolated E_infinity carried with an uncertainty. Because that number is
what the paper will quote for EVERY L, its failure modes matter more than its successes,
so this exercises both:

  * a synthetic post-collapse power-law ladder (known E_inf) is recovered, and the basin
    splitter finds the planted collapse rung rather than fitting across it;
  * a synthetic SHCI ladder (E_var + dE_PT2 = E_FCI + c*dE_PT2 exactly) recovers E_FCI
    from the intercept, and PT2 is preferred over the power law when available;
  * the variational guard rejects any extrapolation that lands ABOVE the deepest Ritz
    value it was fitted from;
  * too few post-collapse rungs -> ok=False with only the bound quotable (no invented number);
  * `combine_seeds` turns a seed-to-seed spread into a sigma term that actually reaches the
    per-seed error bars.

Run: python tests/test_einf_uncertainty.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.extrapolation import (combine_seeds, einf_with_uncertainty,
                                            split_at_collapse)

SITES = 8


def _power_ladder(E_inf=1800.0, a=6.0e4, b=0.62, cores=(8000, 16000, 32000, 64000,
                                                        128000, 256000, 512000, 1024000),
                  collapse_at=2, drop=180.0):
    """Geometric ladder on a known power law, with a basin collapse planted before
    index `collapse_at` (rungs below it sit `drop` MeV high, as the real exploration
    basin does)."""
    rungs = []
    for i, N in enumerate(cores):
        E = E_inf + a * N ** (-b)
        if i < collapse_at:
            E += drop + 12.0 * (collapse_at - i)     # the wrong, delocalized basin
        rungs.append({"core": N, "E_var": E})
    return rungs


def _shci_ladder(E_fci=1795.0, c=0.45, dps=(-40.0, -26.0, -17.0, -11.0, -7.0, -4.5)):
    """E_var + dE_PT2 = E_fci + c*dE_PT2 exactly -> the intercept at dE_PT2=0 is E_fci.
    E_var is then decreasing and stays above E_fci, as a real ladder does."""
    cores = [16000 * 2 ** i for i in range(len(dps))]
    return [{"core": N, "dE_pt2": d, "E_var": E_fci + c * d - d}
            for N, d in zip(cores, dps)]


def main():
    fails = []

    # --- 1. power-law recovery + basin split ------------------------------------------
    E_INF = 1800.0
    rungs = _power_ladder(E_inf=E_INF)
    post, basin = split_at_collapse(rungs, sites=SITES)
    if basin["collapse_core"] != rungs[2]["core"]:
        fails.append(f"basin split found core {basin['collapse_core']}, "
                     f"expected {rungs[2]['core']}")
    if len(post) != len(rungs) - 2:
        fails.append(f"post-collapse basin has {len(post)} rungs, expected {len(rungs)-2}")

    r = einf_with_uncertainty(rungs, sites=SITES)
    if not r["ok"]:
        fails.append(f"power-law ladder did not extrapolate: {r['reason']}")
    else:
        if r["primary"] != "power":
            fails.append(f"no PT2 available, primary should be 'power', got {r['primary']}")
        if abs(r["E_inf"] - E_INF) > 2.0:
            fails.append(f"power-law E_inf {r['E_inf']:.2f} != planted {E_INF} (tol 2 MeV)")
        if r["E_inf"] > r["E_var_bound"]:
            fails.append("E_inf above the variational bound (guard failed to fire)")
        if r["sigma"] is None or not np.isfinite(r["sigma"]):
            fails.append("no sigma reported for a successful extrapolation")
        if r["E_inf_ps"] is None or abs(r["E_inf_ps"] - r["E_inf"] / SITES) > 1e-9:
            fails.append("per-site mirror is wrong")

    # --- 2. SHCI/PT2 recovery, and PT2 preferred over power law ------------------------
    E_FCI = 1795.0
    r2 = einf_with_uncertainty(_shci_ladder(E_fci=E_FCI), sites=SITES)
    if not r2["ok"]:
        fails.append(f"SHCI ladder did not extrapolate: {r2['reason']}")
    else:
        if r2["primary"] != "pt2":
            fails.append(f"PT2 available but primary is {r2['primary']}")
        if abs(r2["E_inf"] - E_FCI) > 0.5:
            fails.append(f"SHCI intercept {r2['E_inf']:.3f} != planted {E_FCI} (tol 0.5 MeV)")
        if r2["sigma_terms"].get("method") is None:
            fails.append("both extrapolators ran but no 'method' systematic was reported")
        if r2["sigma_terms"].get("stability") is None:
            fails.append("enough rungs for a leave-one-out refit but no 'stability' term")

    # --- 3. variational guard ----------------------------------------------------------
    # A ladder that RISES after the planted collapse: any downward extrapolation is
    # meaningless, and a fit landing above the deepest Ritz value must be refused.
    bad = [{"core": 16000, "E_var": 2000.0},
           {"core": 32000, "E_var": 1800.0},
           {"core": 64000, "E_var": 1801.0},
           {"core": 128000, "E_var": 1802.0},
           {"core": 256000, "E_var": 1803.0}]
    r3 = einf_with_uncertainty(bad, sites=SITES)
    if r3["ok"]:
        fails.append(f"non-decreasing post-basin ladder extrapolated anyway "
                     f"(E_inf={r3['E_inf']}, should refuse)")
    if r3.get("E_var_bound") is None:
        fails.append("a refused extrapolation must still carry the variational bound")

    # --- 4. too-few post-collapse rungs ------------------------------------------------
    short = _power_ladder(cores=(8000, 16000, 32000), collapse_at=1)
    r4 = einf_with_uncertainty(short, sites=SITES)
    if r4["ok"]:
        fails.append("extrapolated from 2 post-collapse rungs (min_post=3)")
    if r4["E_inf"] is not None:
        fails.append("a failed extrapolation must report E_inf=None, not a guess")

    # --- 5. seed spread reaches the per-seed error bars --------------------------------
    per_seed = {0: _power_ladder(E_inf=1800.0),
                1: _power_ladder(E_inf=1806.0),
                2: _power_ladder(E_inf=1794.0)}
    pooled = combine_seeds(per_seed, sites=SITES)
    if not pooled["ok"]:
        fails.append(f"pooled seeds failed: {pooled['reason']}")
    else:
        if pooled["sigma_seed"] is None or pooled["sigma_seed"] < 1.0:
            fails.append(f"seed spread not captured (sigma_seed={pooled['sigma_seed']})")
        # the seed spread is a property of the POOL: it must enter the pooled sigma ONCE,
        # not be folded into every per-seed sigma and then averaged (that double-counts it).
        for s, v in pooled["per_seed"].items():
            if v["ok"] and v["sigma_terms"].get("seed") is not None:
                fails.append(f"seed {s}: seed spread double-counted inside a per-seed sigma")
        best = pooled["best_seed"]
        if pooled["per_seed"][best]["E_var_bound"] != pooled["E_var_bound"]:
            fails.append("pooled estimate did not come from the tightest-bound seed")
        if abs(pooled["E_inf"] - pooled["per_seed"][best]["E_inf"]) > 1e-9:
            fails.append("pooled E_inf is not the best-ladder seed's estimate")
        want = np.sqrt(pooled["per_seed"][best]["sigma"] ** 2 + pooled["sigma_seed"] ** 2)
        if abs(pooled["sigma"] - want) > 1e-6:
            fails.append(f"pooled sigma {pooled['sigma']:.4f} != quadrature of the best seed's "
                         f"own uncertainty and the between-seed spread ({want:.4f})")
        if pooled["sigma"] < pooled["sigma_seed"] - 1e-9:
            fails.append("pooled sigma smaller than the seed spread it contains")
        if abs(pooled["E_inf"] - 1794.0) > 3.0:
            fails.append(f"pooled E_inf {pooled['E_inf']:.2f} is not the best (lowest-bound) "
                         "seed's planted value 1794")
        if pooled["E_var_bound"] != min(v["E_var_bound"] for v in pooled["per_seed"].values()):
            fails.append("pooled bound is not the tightest (lowest) seed bound")
        if pooled["E_inf"] > pooled["E_var_bound"] + 1e-9:
            fails.append("pooled E_inf above the pooled bound slipped through")
    single = combine_seeds({0: _power_ladder()}, sites=SITES)
    if single["sigma_seed"] is not None:
        fails.append("a single seed must not claim a seed spread")

    # --- 6. the POOLED variational guard --------------------------------------------
    # Each seed can respect its OWN bound while the mean exceeds the TIGHTEST bound, when
    # the seeds reached ladders of unequal quality. Observed on real data (n_b=3, L=2:
    # pooled 1809.25 vs tightest bound 1805.56); it is a contradiction, not a wide error bar.
    mixed = {0: _power_ladder(E_inf=1800.0),
             1: _power_ladder(E_inf=1860.0, drop=400.0)}   # seed 1: much worse ladder
    pm = combine_seeds(mixed, sites=SITES)
    if pm["ok"] and pm["E_inf"] > pm["E_var_bound"] + 1e-9:
        fails.append(f"pooled E_inf {pm['E_inf']:.2f} exceeds the tightest bound "
                     f"{pm['E_var_bound']:.2f} and was still reported")

    # --- 7. a FIT-ONLY uncertainty is refused ----------------------------------------
    # Exactly 3 post-collapse PT2 rungs: the linear fit works, but there is no second
    # extrapolator and no leave-one-out refit, so sigma would be the fit covariance alone.
    # 4 rungs -> the largest drop is the first, so exactly 3 survive the basin split
    short_pt2 = _shci_ladder(dps=(-40.0, -26.0, -17.0, -11.0))
    r7 = einf_with_uncertainty(short_pt2, sites=SITES)
    if r7["ok"]:
        fails.append(f"fit-only uncertainty reported (sigma_terms={r7.get('sigma_terms')}) -- "
                     "needs an independent cross-check")
    elif "FIT-ONLY" not in (r7.get("reason") or ""):
        fails.append(f"refused for the wrong reason: {r7.get('reason')}")

    if fails:
        print("test_einf_uncertainty: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_einf_uncertainty: PASS  (power-law E_inf {r['E_inf']:.2f}±{r['sigma']:.2f} "
          f"vs planted {E_INF}; SHCI intercept {r2['E_inf']:.3f} vs planted {E_FCI}; "
          f"guard + min-rung + fit-only + pooled-bound refusals fire; seed spread "
          f"{pooled['sigma_seed']:.2f} MeV entering the pooled sigma exactly once)")


def test_einf_uncertainty():
    main()


if __name__ == "__main__":
    main()
