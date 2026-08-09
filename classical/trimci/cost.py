"""Classical-cost analysis (publication Goal 3): how expensive is selected-CI (TrimCI)
to reach a fixed GSEE accuracy, per (L, A), and how does that scale toward the sizes the
quantum algorithm targets. Three tiers, because no single one is both rigorous AND
applicable at large L (the energy-extrapolation route is dead: E_inf uncertainty spans
>60 orders of magnitude in core*):

  Tier-1  core*(dE)   -- core where E_var reaches within dE of the EXACT E_inf
                         (E_exact from run_frame_shard --exact-ref). The rigorous
                         cost-to-fixed-accuracy; only where ED is feasible (small L). Its
                         SCALING with system size is the defensible extrapolation.
  Tier-2  support     -- the effective # of determinants the state lives on
                         (n(weight), participation ratio). Convergence-free -> works at
                         every (L, A); n(weight) IS the accuracy<->cost curve. Calibrated
                         against the exact support where Tier-1 is available.
  Tier-3  relative    -- core to reach a variational BOUND (DMRG / PT2-extrapolated).
                         A cross-check, labeled relative -- never "to truth".

Both accuracy conventions are reported: total dE, and per-site dE/sites (the primary,
since support grows with volume). Pure analysis -- no solves, safe to run anywhere.
"""
from __future__ import annotations

import numpy as np

WEIGHT_KEYS = [("n90", 0.10), ("n99", 0.01), ("n999", 1e-3), ("n9999", 1e-4)]


def _rung_arrays(rungs):
    r = [x for x in rungs if x.get("E_var") is not None]
    core = np.array([x["core"] for x in r], float)
    E = np.array([x["E_var"] for x in r], float)
    return r, core, E


# --------------------------------------------------------------------------- Tier 1
def core_star(rungs, E_ref, dE):
    """Smallest core where E_var(core) - E_ref <= dE. Linear interp in (log core, log
    gap) between the bracketing rungs. Returns (core_star, reached): reached=False means
    the ladder never got within dE, so core_star is a LOWER bound (= deepest core)."""
    _, core, E = _rung_arrays(rungs)
    if len(core) == 0:
        return np.nan, False
    gap = E - E_ref
    below = np.where(gap <= dE)[0]
    if len(below) == 0:
        return float(core[-1]), False              # not reached -> lower bound
    i = int(below[0])
    if i == 0:
        return float(core[0]), True
    # interpolate in log-core vs log-gap (gap decays ~ power law in core)
    g0, g1 = gap[i - 1], gap[i]
    if g0 <= 0 or g1 <= 0 or g0 == g1:
        return float(core[i]), True
    t = (np.log(g0) - np.log(dE)) / (np.log(g0) - np.log(g1))
    lc = np.log(core[i - 1]) + t * (np.log(core[i]) - np.log(core[i - 1]))
    return float(np.exp(lc)), True


def tier1_costs(rungs, E_exact, sites, dEs=(10.0, 1.0, 0.1)):
    """core*(dE) for TOTAL and PER-SITE accuracy targets against the exact E_inf."""
    out = {"E_exact": E_exact, "sites": sites, "total": {}, "per_site": {}}
    for dE in dEs:
        c, ok = core_star(rungs, E_exact, dE)
        out["total"][dE] = {"core_star": c, "reached": ok}
    for dEs_ in dEs:                                # per-site target dE/site -> total dE = dEs_*sites
        c, ok = core_star(rungs, E_exact, dEs_ * sites)
        out["per_site"][dEs_] = {"core_star": c, "reached": ok, "total_dE": dEs_ * sites}
    return out


# --------------------------------------------------------------------------- Tier 2
def support_vs_core(rungs):
    """{metric: [(core, value), ...]} for participation_ratio + n90/n99/n999/n9999."""
    out = {"participation_ratio": [], "n90": [], "n99": [], "n999": [], "n9999": []}
    for x in rungs:
        s = x.get("support")
        if not s:
            continue
        for k in out:
            if s.get(k) is not None:
                out[k].append((x["core"], s[k]))
    return out


def support_converged(rungs, metric="n999", rtol=0.05):
    """Has `metric` plateaued? True if the last two rungs differ by < rtol (relative) --
    then that plateau IS the classical cost. Else the support is still growing (the core
    is still capturing new weight); use support_weight_exponent to extrapolate."""
    pts = support_vs_core(rungs).get(metric, [])
    if len(pts) < 2:
        return False, None
    (c0, v0), (c1, v1) = pts[-2], pts[-1]
    if v1 <= 0:
        return False, v1
    return abs(v1 - v0) / v1 < rtol, v1


def support_weight_exponent(rung):
    """From n(1-delta) at delta in {0.1,0.01,1e-3,1e-4} fit n ~ C * delta^(-gamma): the
    tail exponent gamma sets how support grows as the weight cutoff tightens. Returns
    (gamma, C) so support at any target weight (1-delta) is C*delta^-gamma -- the
    accuracy<->cost curve. NOTE: measured on a possibly-under-converged core, so a lower
    bound / calibrate against exact_support where available."""
    s = rung.get("support")
    if not s:
        return None
    x, y = [], []
    for k, delta in WEIGHT_KEYS:
        if s.get(k):
            x.append(np.log(delta)); y.append(np.log(s[k]))
    if len(x) < 3:
        return None
    m, b = np.polyfit(x, y, 1)
    return {"gamma": -float(m), "C": float(np.exp(b))}


def deepest_rung(rungs):
    r = [x for x in rungs if x.get("E_var") is not None]
    return max(r, key=lambda x: x["core"]) if r else None
