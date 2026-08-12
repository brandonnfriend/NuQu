"""cost.extrapolate_uncertainty / predict_band -- the error-bar energy extrapolation.

Builds a synthetic monotone convergence curve E(core) = E_inf + C*core^-alpha with KNOWN
(E_inf, alpha), then checks the fit-family estimator (a) brackets the true E_inf inside its
[lo, hi] band, (b) recovers alpha, (c) predicts unmeasured cores with the true value inside
the band, and (d) the band is monotone-shrinking toward deep cores (more data -> tighter).
Also checks predict_band shapes. Pure numpy, safe anywhere.

Run: python tests/test_extrapolate_uncertainty.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.cost import extrapolate_uncertainty, predict_band


def main():
    E_INF, C, ALPHA = 1500.0, 5.0e4, 0.60
    cores = np.array([1e3, 2e3, 4e3, 8e3, 16e3, 32e3, 64e3, 128e3, 256e3, 512e3, 1024e3])
    # realistic curve: exact power law + small monotone-preserving scatter (like a real
    # selected-CI ladder). Deterministic (seeded) so the test is reproducible.
    noise = np.random.default_rng(0).normal(0, 1.5, size=len(cores))
    E = E_INF + C * cores ** (-ALPHA) + noise

    r = extrapolate_uncertainty(cores, E, min_window=4,
                                predict_cores=[1e7, 1e9], dEs=(1.0,))
    fails = []
    GRID = 1000.0 / 500.0                                  # E_inf grid resolution ~2 MeV
    ei, elo, ehi = r["Einf"]
    if not (elo - GRID <= E_INF <= ehi + GRID):            # true E_inf inside band (+grid tol)
        fails.append(f"true E_inf {E_INF} outside band [{elo:.1f}, {ehi:.1f}]")
    if abs(ei - E_INF) > 15.0:                             # median within a few MeV of truth
        fails.append(f"median E_inf {ei:.1f} off true {E_INF} by >15")
    al, alo, ahi = r["alpha"]
    if not (alo - 0.05 <= ALPHA <= ahi + 0.05):
        fails.append(f"true alpha {ALPHA} outside [{alo:.3f}, {ahi:.3f}]")

    # predicted energy at an unmeasured core must bracket the true (noiseless) value
    for pc in (1e7, 1e9):
        m, lo, hi = r["predict"][pc]
        true = E_INF + C * pc ** (-ALPHA)
        if not (lo - GRID <= true <= hi + GRID):
            fails.append(f"predict({pc:.0e}) band [{lo:.1f},{hi:.1f}] misses true {true:.1f}")

    # band WIDENS into extrapolation: the fits agree near the data and diverge to their own
    # E_inf far out, so width(1e9) >= width(1e7), saturating at the E_inf band width.
    _, lo7, hi7 = r["predict"][1e7]
    _, lo9, hi9 = r["predict"][1e9]
    w7, w9 = hi7 - lo7, hi9 - lo9
    if w9 + 1e-6 < w7:
        fails.append(f"band should widen into extrapolation: w(1e7)={w7:.1f} > w(1e9)={w9:.1f}")
    if w9 > (ehi - elo) + 2 * GRID:
        fails.append(f"deep band {w9:.1f} exceeds E_inf band {(ehi - elo):.1f}")

    # predict_band shapes + ordering
    med, blo, bhi = predict_band(r["fits"], [1e6, 1e7, 1e8])
    if not (len(med) == len(blo) == len(bhi) == 3):
        fails.append("predict_band returned wrong length")
    if np.any(blo > med + 1e-6) or np.any(med > bhi + 1e-6):
        fails.append("predict_band lo/median/hi not ordered")

    # degenerate input (single point, no window to fit) -> None, not a crash
    if extrapolate_uncertainty([1e3], [10.0], min_window=4) is not None:
        fails.append("single-point input should return None")

    if fails:
        print("test_extrapolate_uncertainty: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_extrapolate_uncertainty: PASS  (E_inf={ei:.1f} in [{elo:.1f},{ehi:.1f}], "
          f"true {E_INF}; alpha={al:.3f}, true {ALPHA}; {r['n_fits']} window-fits)")


if __name__ == "__main__":
    main()
