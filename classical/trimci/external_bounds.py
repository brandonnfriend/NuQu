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


def lookup(frame, n_b, L, A, dim=3, contact_convention="wick", registry=None):
    """{'upper': [entries sorted tightest first], 'exact': entry or None} for one cell."""
    reg = load_registry() if registry is None else registry
    hits = [_in_convention(e, contact_convention) for e in reg if _matches(e, frame, n_b, L, A, dim)]
    upper = sorted((e for e in hits if e["kind"] == "variational_upper"), key=lambda e: e["E"])
    exact = next((e for e in hits if e["kind"] == "exact"), None)
    return {"upper": upper, "exact": exact}


def tightest_upper(ext):
    """(E_total, entry) of the tightest same-operator external bound in a `lookup` result."""
    if not ext or not ext.get("upper"):
        return None
    e = ext["upper"][0]
    return e["E"], e
