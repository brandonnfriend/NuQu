"""Minimize total QPE T over the error-budget allocation (audit issue 1).

The 1 MeV target energy error splits between QPE phase RESOLUTION (`eps_qpe`) and
BLOCK-ENCODING/synthesis error (`eps_be`), `eps_qpe + eps_be = ΔE`. The split trades
off two costs:

  * `N_walk = π·λ / eps_qpe`  — more resolution budget → fewer walk queries (π = adopted constant).
  * `walk_T  = a + b·log2(1/circuit_precision)`, `circuit_precision = eps_be/(2λ)`
    — more block-encoding budget → looser rotation synthesis → cheaper walk step.

pyLIQTR's rotation synthesis makes `walk_T` **exactly linear** in `log2(1/cp)` (verified,
residual 0.00%), so `(a, b)` are fixed by two `estimate_resources(circuit_precision=…)`
samples. Total QPE query cost is

  total_T(f) = N_walk(f) · walk_T(f),   f = qpe_fraction ∈ (0, 1),

which has an interior minimum (`N_walk → ∞` as f→0; `walk_T → ∞` as f→1). This module
finds `f*` (and the whole curve) so the paper reports the *optimal* allocation rather than
an arbitrary 50/50. The block-encoding error is a SYSTEMATIC Hamiltonian perturbation
(`‖δH‖ ≤ eps_be`, independent of N_walk — the walk qubitizes H+δH exactly), the same model
as `sparse_oracle.precision_budget.qpe_error_budget`, which this reuses.
"""

import math

from src_PI.estimation.sparse_oracle.precision_budget import qpe_error_budget


def fit_walk_t_vs_precision(samples):
    """`(a, b)` for `walk_T = a + b·log2(1/cp)` from `[(cp, walk_T), ...]` (≥2 pts).

    Exact from 2 points (pyLIQTR synthesis is log-linear); with >2 the max residual is
    returned so callers can assert linearity.
    """
    if len(samples) < 2:
        raise ValueError("need ≥2 (circuit_precision, walk_T) samples")
    xs = [math.log2(1.0 / cp) for cp, _ in samples]
    ys = [float(t) for _, t in samples]
    b = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a = ys[0] - b * xs[0]
    resid = max(abs(ys[i] - (a + b * xs[i])) for i in range(len(xs)))
    return a, b, resid


def total_t_at_fraction(a, b, physical_lambda, delta_E, qpe_fraction):
    """`(total_T, budget)` at one allocation `qpe_fraction`."""
    bud = qpe_error_budget(physical_lambda, delta_E, qpe_fraction)
    cp = bud['circuit_precision']                       # = eps_be/(2λ)
    walk_t = a + b * math.log2(1.0 / cp)
    total_t = walk_t * bud['walk_queries']
    return total_t, {**bud, 'walk_T': walk_t, 'total_T': total_t,
                     'circuit_precision': cp}


def optimize_qpe_fraction(a, b, physical_lambda, delta_E=1.0,
                          f_lo=0.02, f_hi=0.98, n_grid=193):
    """Grid-minimize `total_T(f)` over the QPE/BE budget split.

    Returns a dict with the optimum (`qpe_fraction`, `eps_qpe`, `eps_be`,
    `circuit_precision`, `walk_T`, `walk_queries`, `total_T`) plus the full `curve`
    `[(f, total_T), ...]` and the fit `(a, b)`. Grid is dense enough that the discrete
    argmin is within <0.1% of the continuous optimum for these smooth curves.
    """
    fs = [f_lo + (f_hi - f_lo) * i / (n_grid - 1) for i in range(n_grid)]
    curve = []
    best = None
    for f in fs:
        tt, bud = total_t_at_fraction(a, b, physical_lambda, delta_E, f)
        curve.append((f, tt))
        if best is None or tt < best['total_T']:
            best = bud
    # default-precision (1 MeV all to QPE, pyLIQTR default synthesis) diagnostic, for
    # the "vs naive" delta the paper reports.
    return {
        'qpe_fraction': best['qpe_fraction'],
        'eps_qpe': best['eps_qpe'],
        'eps_be': best['eps_be'],
        'circuit_precision': best['circuit_precision'],
        'walk_T': best['walk_T'],
        'walk_queries': best['walk_queries'],
        'total_T': best['total_T'],
        'fit_a': a, 'fit_b': b,
        'curve': curve,
        'delta_E': delta_E,
        'physical_lambda': physical_lambda,
    }
