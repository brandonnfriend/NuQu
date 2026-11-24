"""External reference energies for the extrapolation gate (nuqu-2026-10, 2026-10-10).

`external_bounds.json` holds, per (frame, dim, n_b, L, A), energies obtained OUTSIDE the
TrimCI ladder:

  * `variational_upper` -- a variational energy from another method (block2 DMRG, 294249) on
    the IDENTICAL truncated Hamiltonian: same frame, same n_b / N_f, same contact convention.
    An extrapolated limit ABOVE such a number is refuted by the data, exactly as a limit above
    our own Ritz bound is. This is what caught bare L=2 A=4 / A=8 (TrimCI's limit 0.97 / 0.68
    MeV/site above DMRG's variational energy, at sigma <= 0.01).
  * `exact` -- the untruncated ground-state energy (the A=0 pion vacuum from the multimode
    Bogoliubov transformation). Every truncated limit, in any frame or cutoff, lies at or
    above it, so a limit BELOW it (by more than sigma) is refuted, and the deviation above it
    is the fit-plus-cutoff error that calibrates sigma.

Only SAME-OPERATOR references may trip the gate: the squeezed truncated H is a different
operator from the bare one, so a squeezed bound below a bare limit says nothing about the
bare extrapolation (the bare truncated limit may sit above the untruncated ground state by
the cutoff error). `lookup` therefore matches `frame` exactly for `variational_upper` and
accepts "*" only for `exact`.
"""
from __future__ import annotations

import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_bounds.json")
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_calibration.json")
_C_SELF = None


def _c_self():
    """-23.3725 MeV: the per-nucleon contact self-interaction the legacy convention carries
    (misc/apply_wick_correction.C_SELF). Imported lazily so this module stays importable
    without the EFT parameter stack."""
    global _C_SELF
    if _C_SELF is None:
        from misc.apply_wick_correction import C_SELF
        _C_SELF = float(C_SELF)
    return _C_SELF


def load_registry(path=None):
    with open(path or REGISTRY_PATH) as f:
        return json.load(f)["entries"]


def _matches(e, frame, n_b, L, A, dim):
    if int(e["L"]) != int(L) or int(e["A"]) != int(A) or int(e["dim"]) != int(dim):
        return False
    if e["kind"] == "exact":
        return True                                  # untruncated: any frame, any cutoff
    return e["frame"] == frame and e["n_b"] != "*" and int(e["n_b"]) == int(n_b)


def _in_convention(e, contact_convention):
    """Return a copy of `e` with E / E_ps moved into `contact_convention` ('wick' | 'legacy')."""
    native = e.get("contact_convention", "wick")
    if contact_convention == native or int(e["A"]) == 0:
        return dict(e)
    sign = {("wick", "legacy"): +1.0, ("legacy", "wick"): -1.0}[(native, contact_convention)]
    shift = sign * _c_self() * int(e["A"])         # legacy = wick + C_SELF*A
    out = dict(e)
    out["E"] = e["E"] + shift
    out["E_ps"] = out["E"] / e["sites"]
    out["contact_convention"] = contact_convention
    return out


def load_calibration(path=None):
    p = path or CALIBRATION_PATH
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)["entries"]


def family_calibration(frame, dim, n_b, L, registry=None):
    """Gate-5 status of the (frame, dim, n_b, L) family (2026-11-24): its A=0 limits vs the exact
    pion vacuum, from `family_calibration.json` (misc/build_family_calibration.py).

      * 'failed'  -- some extrapolated A=0 limit misses the exact vacuum by more than its sigma;
      * 'passed'  -- at least one is within sigma and none misses;
      * 'missing' -- no A=0 limit was reported (bound only, or never run) or no exact value.
    A=0 is the EASIEST point of a family (no nucleons), so a family that misses there cannot be
    trusted at A>0; one that passes there is calibrated only as far as A=0 reaches."""
    reg = load_calibration() if registry is None else registry
    es = [e for e in reg if e["frame"] == frame and int(e["dim"]) == int(dim)
          and int(e["n_b"]) == int(n_b) and int(e["L"]) == int(L)]
    checked = [e for e in es if e.get("ok") and e.get("dev_ps") is not None]
    failed = [e for e in checked if not e.get("passed")]
    status = "failed" if failed else ("passed" if checked else "missing")
    return {"status": status, "frame": frame, "dim": int(dim), "n_b": int(n_b), "L": int(L),
            "entries": es, "failed": failed}


def lookup(frame, n_b, L, A, dim=3, contact_convention="wick", registry=None, calibration=True,
           calibration_registry=None):
    """{'upper': [entries sorted tightest first], 'exact': entry or None, 'calibration': the
    family's gate-5 status (or None with calibration=False)} for one cell."""
    reg = load_registry() if registry is None else registry
    hits = [_in_convention(e, contact_convention) for e in reg if _matches(e, frame, n_b, L, A, dim)]
    upper = sorted((e for e in hits if e["kind"] == "variational_upper"), key=lambda e: e["E"])
    exact = next((e for e in hits if e["kind"] == "exact"), None)
    cal = (family_calibration(frame, dim, n_b, L, registry=calibration_registry)
           if calibration else None)
    return {"upper": upper, "exact": exact, "calibration": cal}


def tightest_upper(ext):
    """(E_total, entry) of the tightest same-operator external bound in a `lookup` result."""
    if not ext or not ext.get("upper"):
        return None
    e = ext["upper"][0]
    return e["E"], e
