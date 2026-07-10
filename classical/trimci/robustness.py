"""
Phase D — honesty & robustness guards for the classical (TrimCI) baseline.

WHY THIS MODULE EXISTS. The dets-vs-L exponent (Phase C) rests on a chain of
estimates, each with a way to lie if left unguarded. This module makes those
failure modes VISIBLE and, where possible, VETOABLE, so a result reads honestly:

  * THE CIRCULARITY. N*(L; eps) is measured against a per-L reference E_inf(L)
    that is itself an EXTRAPOLATION (Phase B), not exact truth (there is no ED at
    L>=2 in 3D). The reference therefore carries a real uncertainty; `pinning`
    (extrapolation.py / converged_reference) already refuses to define an
    eps-accurate N* when that uncertainty is comparable to eps. The STANDING
    VALIDATION that the whole chain is trustworthy where truth exists is the
    L=2 d=1 slice: there the SHCI/PT2 extrapolation reproduces exact Lanczos to
    < 0.001 MeV/site (test [36]). Everywhere else the reference is a best estimate,
    and a bound is reported rather than a fake point.

  * BURSTY / NON-VARIATIONAL-LOOKING LADDERS. Selected-CI reaches important
    determinant classes in jumps, so the independent-ladder energy can momentarily
    RISE as the core grows (an under-converged ensemble at that rung), even though
    each solve is variational within its core. `ladder_monotonicity` surfaces such
    rises so a bursty rung is flagged, not silently trusted. (`_extract_nstar`
    already guards N* against a lone fluke pass; this is the complementary
    diagnostic that quantifies the burstiness.)

  * DETERMINISTIC PT2 DOES NOT SCALE FOREVER. EN-PT2 here sums the FULL connected
    external space (exactly; more accurate than a sampled sum). Its work — and, on
    the pure-Python reference path, its memory — grows ~ core x #terms, i.e. with
    both depth and L. `pt2_memory_report` tracks the per-rung external-space size
    and TRIGGERS when it grows past a budget, flagging that a SEMISTOCHASTIC PT2
    (deterministic core + sampled tail, arXiv:1808.02049) is the right tool at
    deeper cores / larger L. It is a contingency to build when the trigger fires,
    not before — on the laptop L=2-4 run it never did (max n_ext ~ 2.4M).

  * LOAD-BEARING RANDOMNESS. The ensemble uses random, non-warm-started inits by
    design (min over n_runs escapes local basins). `run_cpp.seed_robustness`
    quantifies the residual scatter across independent base seeds so the reader can
    see the result is not an artifact of one random stream.
"""

from __future__ import annotations


def ladder_monotonicity(rungs, energy_key="E_var"):
    """Diagnose bursty / under-converged rungs on an INDEPENDENT core ladder.

    Independent solves at growing cores should be NON-INCREASING in energy (each is
    a variational solve of a strictly larger space). A rise — energy going UP as the
    core grows — signals an under-converged ensemble at that rung (the larger core
    happened to miss determinants the smaller one found), so that point should be
    treated with caution / re-solved with more ensemble runs. Pure function of the
    rung list.

    Returns {monotone, max_rise, n_rungs, offenders:[{from_core,to_core,rise}]}.
    """
    rs = sorted(rungs, key=lambda r: r["core"])
    offenders = []
    max_rise = 0.0
    for a, b in zip(rs, rs[1:]):
        rise = float(b[energy_key] - a[energy_key])   # > 0 == energy rose as core grew
        if rise > 0:
            offenders.append({"from_core": a["core"], "to_core": b["core"],
                              "rise": rise})
            max_rise = max(max_rise, rise)
    return {"monotone": len(offenders) == 0, "max_rise": max_rise,
            "n_rungs": len(rs), "offenders": offenders}


def pt2_memory_report(rungs, n_ext_budget=50_000_000, bytes_per_ext=256):
    """Track the EN-PT2 external-space size and flag when a SEMISTOCHASTIC PT2 is
    the right tool.

    `n_ext` (recorded per rung) is the number of distinct external determinants the
    deterministic EN-PT2 sums. It grows ~ core x #terms, so it is the scaling wall at
    HPC cores / large L. The C++ pass-1 STREAMS (peak memory modest), but the
    pure-Python reference builds an `ext_amp` map ~ n_ext entries; `est_python_mem_gb`
    (= max_n_ext * bytes_per_ext) estimates THAT path's footprint and doubles as a
    proxy for the work. `bytes_per_ext` is a rough per-entry size (MixedState key +
    complex amplitude + dict overhead).

    Returns {max_n_ext, est_python_mem_gb, over_budget, trigger}. `over_budget` /
    `trigger` fire when max_n_ext exceeds `n_ext_budget` — the signal to implement the
    semistochastic split before pushing deeper.
    """
    nexts = [r["n_ext"] for r in rungs if r.get("n_ext") is not None]
    if not nexts:
        return {"max_n_ext": None, "est_python_mem_gb": None, "over_budget": False,
                "trigger": "no PT2 n_ext recorded (pt2 off?)"}
    mx = int(max(nexts))
    over = mx > n_ext_budget
    trigger = (
        f"deterministic PT2 external space large (max n_ext={mx:,} > budget "
        f"{n_ext_budget:,}) — a SEMISTOCHASTIC PT2 (deterministic core + sampled "
        f"tail, arXiv:1808.02049) is the recommended tool at deeper cores / larger L"
        if over else
        f"deterministic PT2 within budget (max n_ext={mx:,} <= {n_ext_budget:,})")
    return {"max_n_ext": mx, "est_python_mem_gb": mx * bytes_per_ext / 1e9,
            "over_budget": over, "trigger": trigger}


def scatter_stats(values):
    """Std / peak-to-peak / mean of a small sample (robustness read-outs). Pure."""
    import numpy as np
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "ptp": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "ptp": float(a.max() - a.min()), "min": float(a.min()),
            "max": float(a.max())}
