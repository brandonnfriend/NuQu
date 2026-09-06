"""
Ground-state energy extrapolation + honest reporting for the mixed selected-CI
solver.

WHAT "E_infinity" ACTUALLY IS. The classical baseline reports several energies
that must not be conflated:

  * E_var(N)   -- the variational TrimCI energy at a fixed core of N determinants.
                  A rigorous UPPER BOUND on the ground state OF THE TRUNCATED
                  (fixed-L, fixed-N_f, fixed-frame) Hamiltonian, for THIS N. This
                  is the number we actually solve.
  * E_var+PT2  -- E_var plus the Epstein-Nesbet second-order correction (`pt2.py`):
                  a better estimate at the same N (usually below E_var; can dip
                  slightly below the true truncated GS since PT2 is not variational).
  * E_infinity -- the N -> infinity limit of E_var(N), i.e. Full CI WITHIN the
                  truncated Hamiltonian. We CANNOT solve Full CI at the sizes that
                  matter (the sector is astronomically large), so E_infinity is
                  ESTIMATED BY EXTRAPOLATION, not measured. It is NOT the exact
                  Hamiltonian truth and NOT the experimental binding energy — it
                  still carries the (L, N_f, EFT-truncation, lattice-spacing) error.

PROVENANCE. Earlier runs fit E_infinity from E(N) = E_inf + a * N^(-b) (see
`misc/run_frame_comparison.py`, `misc/plot_L2_loglog.py`). That is centralized
here WITH AN UNCERTAINTY (the fit was previously reported as a bare number) and a
robustness guard. Crucially, `TODO.md` (2026-07-01) found that fitting a SINGLE
growing run's ramp is unreliable for this strongly-bosonic system (the selected-CI
tail is bursty); the fit is only trustworthy over INDEPENDENT solves at a
geometric core ladder. This module assumes that ladder as its input and refuses
to fit too-few / non-decreasing points.

TWO EXTRAPOLATORS.
  * power-law:  E(N) = E_inf + a*N^(-b)   -- needs only (core, energy) rungs.
  * PT2 (SHCI): as the core grows, dE_PT2 -> 0 and E_var+dE_PT2 -> E_FCI. Fit
                E_var + dE_PT2 = E_FCI + c*(dE_PT2) and read the intercept at
                dE_PT2 = 0. The standard selected-CI extrapolation; more defensible
                than the empirical power law when PT2 is available at each rung.
"""

from __future__ import annotations

import numpy as np


def fit_einf_power(cores, energies, min_points=4):
    """Fit E(N) = E_inf + a*N^(-b) over independent-solve rungs.

    Returns a dict {ok, E_inf, sigma, a, b, rms_resid, reason}. `ok=False` (with a
    `reason`) when there are too few points or the ramp is not decreasing enough to
    make the fit meaningful — better to report "no reliable extrapolation" than a
    bogus number.
    """
    from scipy.optimize import curve_fit

    N = np.asarray(cores, dtype=float)
    E = np.asarray(energies, dtype=float)
    order = np.argsort(N)
    N, E = N[order], E[order]
    # de-dup exact-duplicate cores (independent solves can repeat a rung)
    keep = np.concatenate(([True], np.diff(N) > 0))
    N, E = N[keep], E[keep]

    if len(N) < min_points:
        return {"ok": False, "reason": f"need >= {min_points} distinct rungs, got {len(N)}",
                "E_inf": None, "sigma": None, "a": None, "b": None, "rms_resid": None}
    if E[-1] > E[0]:
        return {"ok": False, "reason": "energy not decreasing across the ladder "
                "(non-variational wobble — converge N_f / use more rungs)",
                "E_inf": None, "sigma": None, "a": None, "b": None, "rms_resid": None}

    def model(n, E_inf, a, b):
        return E_inf + a * n ** (-b)

    p0 = [E[-1] - 0.1, max(E[0] - E[-1], 1.0) * N[0] ** 0.7, 0.7]
    try:
        popt, pcov = curve_fit(
            model, N, E, p0=p0,
            bounds=([E.min() - 50.0, 0.0, 0.2], [E[-1] + 1e-9, 1e12, 3.0]),
            maxfev=200000)
    except Exception as e:                       # pragma: no cover - fit failure
        return {"ok": False, "reason": f"curve_fit failed: {e}",
                "E_inf": None, "sigma": None, "a": None, "b": None, "rms_resid": None}

    E_inf, a, b = (float(x) for x in popt)
    sigma = float(np.sqrt(pcov[0, 0])) if np.all(np.isfinite(pcov)) else None
    resid = E - model(N, *popt)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {"ok": True, "reason": "power-law fit E_inf + a*N^-b",
            "E_inf": E_inf, "sigma": sigma, "a": a, "b": b, "rms_resid": rms}


def fit_einf_pt2(E_vars, dE_pt2s, min_points=3):
    """SHCI-style extrapolation: fit (E_var + dE_PT2) = E_FCI + c*(dE_PT2) and
    return the intercept at dE_PT2 = 0.

    As the core -> Full CI, dE_PT2 -> 0 and the total energy -> the FCI energy of
    the truncated Hamiltonian, so the intercept is the extrapolated E_infinity.
    Returns {ok, E_inf, sigma, slope, reason}.
    """
    Ev = np.asarray(E_vars, dtype=float)
    dp = np.asarray(dE_pt2s, dtype=float)
    y = Ev + dp                                  # total energy at each rung
    x = dp                                       # the PT2 correction (-> 0 at FCI)
    if len(x) < min_points:
        return {"ok": False, "reason": f"need >= {min_points} PT2 rungs, got {len(x)}",
                "E_inf": None, "sigma": None, "slope": None}
    if np.ptp(x) < 1e-9:
        return {"ok": False, "reason": "PT2 corrections do not vary across rungs",
                "E_inf": None, "sigma": None, "slope": None}
    # weighted linear fit; intercept = E at dE_PT2 = 0
    coef, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = float(coef[0]), float(coef[1])
    sigma = float(np.sqrt(cov[1, 1])) if np.all(np.isfinite(cov)) else None
    return {"ok": True, "reason": "SHCI PT2 extrapolation (intercept at dE_PT2=0)",
            "E_inf": intercept, "sigma": sigma, "slope": slope}


def report_energies(rungs, exact=None, experiment=None, sites=None, label="",
                    verbose=True):
    """Assemble the honest energy report from a solved core LADDER.

    Args:
        rungs: list of dicts, one per INDEPENDENT solve, each with
               {"core": N, "E_var": E, "dE_pt2": (optional) dE}. Must be the
               independent-ladder rungs, not a single run's ramp.
        exact: optional exact reference energy (Lanczos/ED) when the sector is
               enumerable — the ground truth to validate the extrapolation against.
        experiment: optional experimental value (paper-level; usually None now).
        sites: optional lattice-site count (L**dim). When given, the report also
               carries the SIZE-INTENSIVE (per-site) energies and gaps — the
               quantity to hold fixed across L. Total energy is size-EXTENSIVE
               (~sites), so a total or relative gap silently loosens the per-site
               tolerance as the lattice grows (the extensivity trap); per-site
               numbers are the ones to compare across L.
        label: a tag for the printout.

    Returns a dict with the best variational, variational+PT2, and both
    extrapolations (power-law + PT2 if available), plus the exact/experiment
    references and the extrapolation-vs-exact gap when exact is provided. When
    `sites` is given, mirror keys with a `_per_site` suffix are added.
    """
    rungs = sorted(rungs, key=lambda r: r["core"])
    cores = [r["core"] for r in rungs]
    Evars = [r["E_var"] for r in rungs]
    best = rungs[-1]                             # largest core = best variational
    E_var_best = best["E_var"]
    dE_pt2_best = best.get("dE_pt2")
    E_pt2_best = (E_var_best + dE_pt2_best) if dE_pt2_best is not None else None

    power = fit_einf_power(cores, Evars)
    have_pt2 = all(r.get("dE_pt2") is not None for r in rungs)
    pt2ex = (fit_einf_pt2(Evars, [r["dE_pt2"] for r in rungs])
             if have_pt2 else {"ok": False, "reason": "PT2 not available at every rung",
                               "E_inf": None, "sigma": None})

    out = {
        "label": label,
        "cores": cores,
        "E_var_best": E_var_best,
        "dE_pt2_best": dE_pt2_best,
        "E_var_plus_pt2_best": E_pt2_best,
        "extrap_power": power,
        "extrap_pt2": pt2ex,
        "exact": exact,
        "experiment": experiment,
    }
    # prefer the PT2 (SHCI) extrapolation as "best" when available; else power-law
    best_extrap = pt2ex if pt2ex.get("ok") else power
    out["E_extrap_best"] = best_extrap.get("E_inf")
    out["E_extrap_best_sigma"] = best_extrap.get("sigma")
    out["E_extrap_best_method"] = best_extrap.get("reason")
    if exact is not None and best_extrap.get("E_inf") is not None:
        out["extrap_minus_exact"] = best_extrap["E_inf"] - exact

    # size-intensive (per-site) mirror — the quantities to hold fixed across L.
    if sites:
        out["sites"] = int(sites)
        per = lambda x: (x / sites) if x is not None else None
        out["E_var_best_per_site"] = per(E_var_best)
        out["E_var_plus_pt2_best_per_site"] = per(E_pt2_best)
        out["E_extrap_best_per_site"] = per(out["E_extrap_best"])
        out["E_extrap_best_sigma_per_site"] = per(out["E_extrap_best_sigma"])
        if exact is not None:
            out["exact_per_site"] = per(exact)
        if out.get("extrap_minus_exact") is not None:
            out["extrap_minus_exact_per_site"] = per(out["extrap_minus_exact"])

    if verbose:
        _print_report(out)
    return out


def _fmt(x, s=None, unit="MeV"):
    if x is None:
        return "   n/a"
    return f"{x:.4f} {unit}" + (f"  (± {s:.4f})" if s is not None else "")


def _print_report(out):
    print("=" * 68)
    print(f"  ENERGY REPORT  {out['label']}")
    print("=" * 68)
    print(f"  cores (independent ladder): {out['cores']}")
    print(f"  variational (best, N={out['cores'][-1]}):  {_fmt(out['E_var_best'])}")
    if out["E_var_plus_pt2_best"] is not None:
        print(f"  variational + PT2         :  {_fmt(out['E_var_plus_pt2_best'])}"
              f"   [dE_PT2 = {out['dE_pt2_best']:+.4f}]")
    pw = out["extrap_power"]
    if pw.get("ok"):
        print(f"  extrapolated (power-law)  :  {_fmt(pw['E_inf'], pw['sigma'])}"
              f"   [~N^-{pw['b']:.2f}, rms resid {pw['rms_resid']:.3f}]")
    else:
        print(f"  extrapolated (power-law)  :  unavailable — {pw['reason']}")
    px = out["extrap_pt2"]
    if px.get("ok"):
        print(f"  extrapolated (SHCI/PT2)   :  {_fmt(px['E_inf'], px['sigma'])}"
              f"   [intercept at dE_PT2=0]")
    else:
        print(f"  extrapolated (SHCI/PT2)   :  unavailable — {px['reason']}")
    print("  " + "-" * 64)
    print(f"  >> BEST ESTIMATE          :  {_fmt(out['E_extrap_best'], out['E_extrap_best_sigma'])}")
    if out.get("sites"):
        print(f"  ---- per site ({out['sites']} sites; size-intensive) ----")
        print(f"    variational / site      :  {_fmt(out.get('E_var_best_per_site'))}")
        if out.get("E_var_plus_pt2_best_per_site") is not None:
            print(f"    variational+PT2 / site  :  {_fmt(out.get('E_var_plus_pt2_best_per_site'))}")
        print(f"    BEST estimate / site    :  "
              f"{_fmt(out.get('E_extrap_best_per_site'), out.get('E_extrap_best_sigma_per_site'))}")
        if out.get("extrap_minus_exact_per_site") is not None:
            print(f"    (extrap - exact) / site :  {out['extrap_minus_exact_per_site']:+.4f} MeV/site")
    if out.get("exact") is not None:
        print(f"  exact reference (Lanczos) :  {_fmt(out['exact'])}")
        if out.get("extrap_minus_exact") is not None:
            print(f"    extrapolation - exact   :  {out['extrap_minus_exact']:+.4f} MeV"
                  "   (extrapolation validation)")
    if out.get("experiment") is not None:
        print(f"  experiment                :  {_fmt(out['experiment'])}"
              f"   [gap {out['E_extrap_best'] - out['experiment']:+.3f}]")
    print("=" * 68)


# ---------------------------------------------------------------------------
#  Basin-aware E_infinity WITH an honest error bar  (audit 2026-09-05, P0-2)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. The classical baseline used to report a binary label: "L=2
# converged, L>=3 upper bound", with the last core-doubling's energy change as the
# evidence. A last-doubling change is a convergence DIAGNOSTIC, not an error bar --
# and at L=2 it was 1.6 MeV, larger than the 1 MeV GSEE target it was being used to
# support. The defensible replacement, for EVERY L, is an extrapolated E_infinity
# carried with an uncertainty that states what it does and does not cover. Large
# error bars are the truthful outcome for a variational heuristic that has not
# converged; they are not a failure of the method.
#
# WHAT THE ERROR BAR COVERS (and what it does NOT). The quoted sigma combines four
# terms in quadrature, all of them about the N -> infinity (Full-CI-within-truncation)
# limit of THIS ladder:
#   fit        -- the extrapolator's own parameter uncertainty;
#   method     -- |E_PT2-extrap - E_powerlaw-extrap|, the disagreement between two
#                 independent extrapolators on the SAME rungs (a systematic);
#   stability  -- the shift when the shallowest post-collapse rung is dropped and the
#                 fit repeated (robustness to the fit window, not a free parameter);
#   seed       -- supplied by the caller: the spread over independent solver
#                 trajectories (see `combine_seeds`).
# It does NOT cover, and must never be quoted as covering: the boson cutoff n_b, the
# lattice spacing / finite volume L, the EFT truncation, or the selected-CI SEARCH
# being stuck in the wrong basin altogether. Those are separate budget lines.


def split_at_collapse(rungs, sites=None):
    """Split a warm-grown ladder at its 'basin collapse' rung.

    The greedy selected-CI search sits in a delocalized exploration basin and, once
    the core is wide enough, escapes onto the compact ground-state basin -- a sharp
    single-doubling energy drop. Rungs on opposite sides of that drop describe
    different search basins, so fitting across it mixes two sequences (the pre-collapse
    PT2 extrapolation overshoots the deep answer by ~19 MeV/site at L=2). Everything
    downstream fits the POST-collapse rungs only.

    Returns (post_rungs, info) with info = {collapse_index, collapse_core,
    collapse_drop, collapse_drop_per_site}. A ladder of fewer than 2 rungs is returned
    whole with collapse_index=0.
    """
    rungs = sorted(rungs, key=lambda r: r["core"])
    if len(rungs) < 2:
        return list(rungs), {"collapse_index": 0,
                             "collapse_core": rungs[0]["core"] if rungs else None,
                             "collapse_drop": None, "collapse_drop_per_site": None}
    drops = [(rungs[i - 1]["E_var"] - rungs[i]["E_var"], i) for i in range(1, len(rungs))]
    drop, i_col = max(drops)
    return rungs[i_col:], {
        "collapse_index": i_col,
        "collapse_core": rungs[i_col]["core"],
        "collapse_drop": float(drop),
        "collapse_drop_per_site": (float(drop) / sites) if sites else None,
    }


def einf_with_uncertainty(rungs, sites=None, sigma_seed=None, min_post=3):
    """Basin-aware E_infinity for ONE ladder, with the honest error bar above.

    Args:
        rungs: the full ladder ({"core", "E_var", optional "dE_pt2"} per rung).
        sites: lattice sites, for the per-site mirror keys.
        sigma_seed: seed-spread systematic supplied by the caller (see `combine_seeds`).
        min_post: minimum post-collapse rungs before an extrapolation is attempted.

    Returns a dict. `ok=False` (with `reason`) whenever no defensible extrapolation is
    available -- in which case `E_var_bound` is still a rigorous Ritz upper bound and is
    the only thing that may be quoted. Never invents a number to fill the column.

    The variational guard is load-bearing: E_infinity must lie AT OR BELOW the deepest
    variational energy (the ladder is a decreasing sequence bounded below by the true
    truncated ground state). An extrapolation above it means the fit window is bad, so
    it is rejected rather than reported.
    """
    rungs = sorted([r for r in rungs if r.get("E_var") is not None], key=lambda r: r["core"])
    per = lambda x: (x / sites) if (x is not None and sites) else None
    post, basin = split_at_collapse(rungs, sites=sites)
    E_var_bound = rungs[-1]["E_var"] if rungs else None
    dE_last = (abs(rungs[-1]["E_var"] - rungs[-2]["E_var"]) if len(rungs) >= 2 else None)
    pt2_post = [r for r in post if r.get("dE_pt2") is not None]
    E_pt2_deepest = (pt2_post[-1]["E_var"] + pt2_post[-1]["dE_pt2"]) if pt2_post else None

    out = {
        "n_rungs": len(rungs), "n_post": len(post),
        "cores": [r["core"] for r in rungs],
        "post_cores": [r["core"] for r in post],
        "n_pt2_post": len(pt2_post),
        "E_var_bound": E_var_bound, "E_var_bound_ps": per(E_var_bound),
        "dE_last_doubling": dE_last, "dE_last_doubling_ps": per(dE_last),
        "E_var_plus_pt2": E_pt2_deepest, "E_var_plus_pt2_ps": per(E_pt2_deepest),
        "sites": sites, **basin,
    }

    if len(post) < min_post:
        out.update(ok=False, reason=f"only {len(post)} post-collapse rungs (need {min_post})",
                   E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                   sigma_terms={}, primary=None, power=None, pt2=None)
        return out

    # Degrees of freedom, not just point counts: the power law has 3 parameters, so 3
    # rungs fit it EXACTLY -- zero residual, no covariance, no error bar. The SHCI/PT2
    # fit is linear (2 parameters) and needs 3. Reporting a value whose uncertainty
    # could not be estimated is precisely what audit P0-2 rejects, so the minima differ.
    n_pow = max(4, min_post)
    n_p2 = max(3, min_post)
    power = (fit_einf_power([r["core"] for r in post], [r["E_var"] for r in post],
                            min_points=n_pow) if len(post) >= n_pow else
             {"ok": False, "reason": f"power law needs >= {n_pow} post-collapse rungs for a "
                                     f"fit with a degree of freedom, got {len(post)}",
              "E_inf": None, "sigma": None, "a": None, "b": None, "rms_resid": None})
    pt2 = (fit_einf_pt2([r["E_var"] for r in pt2_post], [r["dE_pt2"] for r in pt2_post],
                        min_points=n_p2) if len(pt2_post) >= n_p2 else
           {"ok": False, "reason": f"PT2 on only {len(pt2_post)} post-collapse rungs "
                                   f"(need {n_p2}) -- run with PT2 on every rung",
            "E_inf": None, "sigma": None, "slope": None})
    out["power"], out["pt2"] = power, pt2

    # PRIMARY = the SHCI/PT2 extrapolation when it is available (it is the standard
    # selected-CI estimator and uses more information per rung); power-law otherwise.
    if pt2.get("ok"):
        primary, E_inf, s_fit = "pt2", pt2["E_inf"], pt2.get("sigma")
    elif power.get("ok"):
        primary, E_inf, s_fit = "power", power["E_inf"], power.get("sigma")
    else:
        out.update(ok=False, reason=f"no extrapolator converged (power: {power.get('reason')}; "
                                    f"pt2: {pt2.get('reason')})",
                   E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                   sigma_terms={}, primary=None)
        return out

    # method systematic: how far apart two independent extrapolators land
    s_method = (abs(pt2["E_inf"] - power["E_inf"])
                if (pt2.get("ok") and power.get("ok")) else None)

    # stability systematic: refit without the shallowest post-collapse rung
    s_stab = None
    drop1 = post[1:]
    if primary == "pt2":
        p1 = [r for r in drop1 if r.get("dE_pt2") is not None]
        alt = (fit_einf_pt2([r["E_var"] for r in p1], [r["dE_pt2"] for r in p1],
                            min_points=n_p2) if len(p1) >= n_p2 else {"ok": False})
    else:
        alt = (fit_einf_power([r["core"] for r in drop1], [r["E_var"] for r in drop1],
                              min_points=n_pow) if len(drop1) >= n_pow else {"ok": False})
    if alt.get("ok"):
        s_stab = abs(alt["E_inf"] - E_inf)

    terms = {"fit": s_fit, "method": s_method, "stability": s_stab, "seed": sigma_seed}
    have = [t for t in terms.values() if t is not None]
    # None means "no uncertainty term could be formed" -- distinct from a genuine 0.0,
    # which only synthetic/exact data produces. Never collapse the two.
    sigma = float(np.sqrt(sum(t ** 2 for t in have))) if have else None

    # A FIT-ONLY uncertainty is not defensible. With the bare minimum number of rungs the
    # fit has no leave-one-out refit and no second extrapolator to disagree with, so sigma
    # collapses to the fit covariance -- which on a short ladder is tiny and says nothing
    # about the extrapolation being right. Seen on real data: L=5, 3 post-collapse rungs,
    # E_inf 1669 MeV BELOW the deepest computed value with a claimed +-16 MeV. Requiring at
    # least one INDEPENDENT term (method or stability) demands >=4 post-collapse rungs, which
    # is the honest floor -- and needs no free threshold.
    if terms["method"] is None and terms["stability"] is None:
        out.update(ok=False,
                   reason=f"{primary} extrapolation has a FIT-ONLY uncertainty ({len(post)} "
                          f"post-collapse rungs, {len(pt2_post)} with PT2): no second "
                          f"extrapolator and no leave-one-out refit to cross-check it. Not "
                          f"reportable -- quote the variational bound.",
                   E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                   sigma_terms=terms, primary=primary)
        return out

    # No error bar -> not reportable. A central value whose uncertainty could not be
    # estimated is exactly what audit P0-2 rejects, so refuse it rather than print it bare.
    if sigma is None:
        out.update(ok=False,
                   reason=f"{primary} extrapolation gave E_inf={E_inf:.3f} but NO uncertainty "
                          f"could be estimated (fit covariance unavailable and no independent "
                          f"systematic) -- not reportable; quote the variational bound",
                   E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                   sigma_terms=terms, primary=primary)
        return out

    # variational guard -- an extrapolation may not sit above the Ritz bound it came from
    if E_var_bound is not None and E_inf > E_var_bound + 1e-9:
        out.update(ok=False,
                   reason=f"extrapolation ({E_inf:.3f}) lies ABOVE the deepest variational "
                          f"energy ({E_var_bound:.3f}) -- the fit window is unreliable",
                   E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                   sigma_terms=terms, primary=primary)
        return out

    out.update(ok=True, reason=f"{primary} extrapolation over {len(post)} post-collapse rungs",
               primary=primary, E_inf=float(E_inf), E_inf_ps=per(E_inf),
               sigma=sigma, sigma_ps=per(sigma),
               sigma_terms={k: (float(v) if v is not None else None) for k, v in terms.items()},
               sigma_terms_ps={k: per(v) for k, v in terms.items()},
               extrap_distance=float(E_var_bound - E_inf),
               extrap_distance_ps=per(E_var_bound - E_inf))
    return out


def combine_seeds(per_seed, sites=None, min_post=3):
    """Aggregate independent solver trajectories (seeds) for one (L, n_b) point.

    Each seed is its own warm-grow ladder, so the seed-to-seed spread of E_infinity is a
    genuine systematic of the SEARCH, not just of the fit -- it is fed back into every
    seed's error bar and the pooled number carries it.

    `per_seed` maps seed -> full rung list. Returns the pooled record: the best (lowest)
    variational bound across seeds -- still rigorous, since every seed's E_var is a valid
    Ritz bound -- and the E_infinity of the seed that reached that bound, with the
    between-seed spread folded into sigma.
    """
    # Each seed is fitted on its OWN ladder only -- the seed spread is a property of the
    # POOL, so folding it into every per-seed sigma and then averaging those sigmas would
    # double-count it. It enters once, below.
    final = {s: einf_with_uncertainty(r, sites=sites, min_post=min_post)
             for s, r in per_seed.items()}
    ok = {s: v for s, v in final.items() if v.get("ok")}
    E_infs = np.array([v["E_inf"] for v in ok.values()], dtype=float)
    sigma_seed = float(E_infs.std(ddof=1)) if len(E_infs) >= 2 else None
    bounds = [v["E_var_bound"] for v in final.values() if v.get("E_var_bound") is not None]
    bound = min(bounds) if bounds else None                      # tightest rigorous bound
    per = lambda x: (x / sites) if (x is not None and sites) else None
    pooled = {
        "seeds": sorted(per_seed), "n_seeds": len(per_seed), "n_seeds_extrapolated": len(ok),
        "sites": sites,
        "E_var_bound": bound, "E_var_bound_ps": per(bound),
        "sigma_seed": sigma_seed, "sigma_seed_ps": per(sigma_seed),
        "per_seed": final,
    }
    if not ok:
        pooled.update(ok=False, E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                      reason="no seed produced a defensible extrapolation; "
                             "only the variational upper bound may be quoted")
        return pooled

    # POOLING: take the extrapolation from the seed with the TIGHTEST variational bound,
    # not the mean over seeds. Independent warm-grow trajectories reach ladders of unequal
    # quality -- a seed stuck in a worse basin carries a worse bound AND a worse
    # extrapolation -- so averaging them mixes a good estimate with a bad one and can even
    # land the mean ABOVE the tightest rigorous bound (observed on real data: n_b=3 L=2,
    # mean 1809.25 vs tightest bound 1805.56). The best-ladder seed's estimate respects that
    # bound by construction. The disagreement between seeds is not discarded -- it becomes
    # the dominant uncertainty term.
    best_seed = min(ok, key=lambda s: ok[s]["E_var_bound"])
    E = float(ok[best_seed]["E_inf"])
    within = ok[best_seed].get("sigma")
    parts = ([within] if within is not None else []) + \
            ([sigma_seed] if sigma_seed is not None else [])
    sig = float(np.sqrt(sum(p ** 2 for p in parts))) if parts else None
    pooled["best_seed"] = best_seed

    # THE POOLED VARIATIONAL GUARD. Each seed's extrapolation respects its OWN bound, but
    # the pooled estimate is a mean while the pooled bound is the tightest (lowest) seed's --
    # so seeds of unequal ladder quality can average to a value ABOVE the best bound. Seen on
    # real data (n_b=3, L=2: pooled 1809.25 vs tightest bound 1805.56). That is a
    # contradiction, not a wide error bar, so it is refused rather than reported.
    if bound is not None and E > bound + 1e-9:
        pooled.update(ok=False, E_inf=None, E_inf_ps=None, sigma=None, sigma_ps=None,
                      reason=f"pooled extrapolation ({E:.3f}) lies ABOVE the tightest "
                             f"variational bound across seeds ({bound:.3f}) -- the seeds "
                             f"disagree too much to average ({len(ok)} extrapolated: "
                             f"{[round(x, 2) for x in E_infs]}); quote the bound")
        return pooled

    pooled.update(ok=True, E_inf=E, E_inf_ps=per(E), sigma=sig, sigma_ps=per(sig),
                  reason=f"best-ladder seed {best_seed} of {len(ok)}/{len(per_seed)} "
                         f"extrapolated; sigma = its own uncertainty (+) between-seed spread")
    return pooled
