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
