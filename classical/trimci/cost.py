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

import warnings

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


# ------------------------------------------------ energy extrapolation with error bars
def extrapolate_uncertainty(cores, E, min_window=4, predict_cores=(), dEs=(1.0,)):
    """Power-law fit `E_var(core) = E_inf + C*core^{-alpha}` with a HONEST error bar from
    the fit-range family: fit over every window [start:] ending at the deepest core
    (start = 0 .. N-min_window). The SPREAD across windows is the uncertainty — it captures
    the full-vs-tail-fit disagreement that a single covariance-matrix error bar hides, which
    is the dominant uncertainty here (no exact E_inf anchor at 3D). Each window fit grids
    E_inf and takes the best-R^2 log-linear slope.

    Returns dict with, as (median, lo, hi) triples: `Einf`, `alpha`; `predict[core]` = the
    extrapolated E_var at an unmeasured core; `core_star[dE]` = core to reach within dE of
    that window's own E_inf. lo/hi are the min/max over the fit family = the error band."""
    cores = np.asarray(cores, float); E = np.asarray(E, float)
    n = len(cores)
    if n < 2:                                  # need >=2 points to fit a line
        return None
    fits = []
    for start in range(0, max(1, n - min_window + 1)):
        cc, EE = cores[start:], E[start:]
        best = None
        for Einf in np.linspace(min(EE) - 1000.0, min(EE) - 0.05, 500):
            y = EE - Einf
            if np.any(y <= 0):
                continue
            with warnings.catch_warnings():    # grid hits near-degenerate log-fits; expected
                warnings.simplefilter("ignore", np.RankWarning)
                a, logC = np.polyfit(np.log(cc), np.log(y), 1)
            vy = np.var(np.log(y))
            r2 = 1 - np.var(np.log(y) - (a * np.log(cc) + logC)) / vy if vy > 0 else 0.0
            if best is None or r2 > best[0]:
                best = (r2, Einf, -a, np.exp(logC))
        if best is not None and best[2] > 0:
            fits.append((best[1], best[2], best[3]))          # (E_inf, alpha, C)
    if not fits:
        return None
    F = np.array(fits)
    trip = lambda v: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
    out = {"Einf": trip(F[:, 0]), "alpha": trip(F[:, 1]), "n_fits": len(fits)}
    out["predict"] = {pc: trip(np.array([ei + C * pc ** (-a) for ei, a, C in F]))
                      for pc in predict_cores}
    with np.errstate(over="ignore"):          # tiny alpha -> inf core_star = "unreachable"
        out["core_star"] = {dE: trip(np.array([(C / dE) ** (1.0 / a) for ei, a, C in F]))
                            for dE in dEs}
    out["fits"] = F.tolist()          # (E_inf, alpha, C) per window -- for dense band plots
    return out


def predict_band(fits, cores):
    """Given the raw (E_inf, alpha, C) window fits, return (median, lo, hi) arrays of the
    extrapolated E_var over `cores` -- the shaded error band at every core, measured or not."""
    cores = np.asarray(cores, float)
    M = np.array([[ei + C * c ** (-a) for c in cores] for ei, a, C in fits])
    return np.median(M, 0), M.min(0), M.max(0)
