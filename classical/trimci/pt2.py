"""
Epstein-Nesbet second-order perturbation theory (PT2) for the mixed
fermion-boson selected-CI solver.

WHY THIS IS OURS AND NOT A TOGGLE. The *released* (fermionic) TrimCI exposes a
`pt2_correction` flag, but its PT2 kernel is Slater-Condon over an FCIDUMP and
has no bosonic modes — the same reason Route A reimplements the graph algorithm
(see `TODO.md`). So PT2 for the dynamical-pion H is implemented here, over our
generalized mixed H_ij (`hij.connections`).

WHAT PT2 BUYS. TrimCI keeps a compact set of determinants V (the "variational"
space) and returns the lowest eigenpair (E_var, |psi> = sum_i c_i |i>) of H
projected onto V. Everything OUTSIDE V is dropped. EN-PT2 adds back the
leading-order effect of that discarded space perturbatively:

    dE_PT2 = sum_{a not in V}  |<a|H|psi>|^2 / (E_var - H_aa)

with H_aa = <a|H|a> the Epstein-Nesbet denominator. For a variational E_var that
sits ABOVE the true ground state, the external determinants a are higher-lying
(H_aa > E_var), so every term is negative: PT2 pushes the energy DOWN, toward the
true ground state, at fixed determinant count. It is the standard "free" tightening
used in SHCI/heat-bath CI.

COST. One pass over `connections` of the |V| core determinants (same cost as one
expansion round) builds the coherent external amplitudes A_a = <a|H|psi>; then a
cheap diagonal-only evaluation of H_aa per external determinant. No new
diagonalization. This is a POST-PROCESS on a solved wavefunction — it never
changes the variational number (which stays the rigorous upper bound); it is
reported alongside it.

SELF-CONSISTENCY. The same pass also recomputes E_var as the Rayleigh quotient
<psi|H|psi> over the (renormalized) saved core, so (E_var, dE_PT2) are mutually
consistent even when the solver's reported energy came from a slightly larger
pre-trim pool. The recomputed E_var is returned for cross-checking.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .hij import apply_term, connections
from .state import MixedState


# ---------------------------------------------------------------------------
#  Diagonal-element machinery (for the H_aa Epstein-Nesbet denominators)
# ---------------------------------------------------------------------------

def _is_diagonal(term):
    """True iff `term` maps every basis state to a scalar multiple of itself.

    A ladder-operator product is diagonal exactly when the NET occupation change
    on every mode is zero — i.e. each fermion mode and each boson mode carries an
    equal number of creation and annihilation operators. (Necessary: a nonzero
    net changes that mode's occupation, so the term is off-diagonal. Sufficient:
    net-zero-per-mode products only remove-then-restore each mode, mapping
    |state> -> scalar*|state> or 0.) The scalar (n_p for a number operator, the
    boson sqrt product, the fermion sign) is state-dependent and evaluated by
    `apply_term`; this predicate only classifies the OPERATOR structure.
    """
    fnet = Counter()
    for (m, a) in term.ferm_ops:
        fnet[m] += 1 if a == 1 else -1
    if any(v != 0 for v in fnet.values()):
        return False
    bnet = Counter()
    for (m, a) in term.bos_ops:
        bnet[m] += 1 if a == 1 else -1
    return all(v == 0 for v in bnet.values())


def diagonal_terms(H):
    """The diagonal sub-list of H.terms, memoized on H (pure function of H)."""
    dt = getattr(H, "_diag_terms", None)
    if dt is None:
        dt = [t for t in H.terms if _is_diagonal(t)]
        H._diag_terms = dt
    return dt


def diagonal_element(H, state, diag_terms=None):
    """<state|H|state>, evaluated from the diagonal terms only (cheap).

    O(#diagonal terms) — far cheaper than a full `connections(H, state)` call,
    which is what makes the per-external H_aa loop affordable.
    """
    if diag_terms is None:
        diag_terms = diagonal_terms(H)
    val = 0.0 + 0.0j
    for t in diag_terms:
        res = apply_term(t, state, H.N_f)
        if res is None:
            continue
        amp, s2 = res
        if s2 == state:          # always true for diagonal terms; a safety guard
            val += amp
    return val.real


# ---------------------------------------------------------------------------
#  Epstein-Nesbet PT2
# ---------------------------------------------------------------------------

def _cpp_pt2_available(H):
    """True iff the C++ PT2 pass-1 (backend.cpp_pt2_external) can run for this H.

    Needs the standalone `mixed_ci` module built AND H's Fock cutoff within the
    uint16 boson representation the C++ boundary uses (N_f <= 65536, i.e.
    n_b <= 16 — far above the physical n_b ~ 5-11). Note this is independent of
    the official-TrimCI Davidson: the pass-1 port only uses `mixed_ci`, so it
    runs even when the re-diagonalizer falls back to pure Python.
    """
    if getattr(H, "N_f", 0) > 65536:
        return False
    try:
        from .backend import _load_cpp
        _load_cpp()
        return True
    except Exception:
        return False


def epstein_nesbet_pt2(H, states, coeffs=None, E_var=None, diag_fn=None,
                       trust_coeffs=False, intruder_eps=1e-6, amp_tol=1e-12,
                       use_cpp=None):
    """Epstein-Nesbet PT2 correction for a selected-CI ground state.

    CRITICAL — SELF-CONSISTENCY. EN-PT2 is only valid when |psi> is the EXACT
    eigenvector of the projected P_V H P_V and E_var is its eigenvalue; otherwise
    H|psi> - E_var|psi> has spurious components on the variational space itself and
    the correction blows up (a 10x-100x overshoot). The TrimCI solver reports
    `res.energy` from the PRE-trim survivor pool but saves the POST-trim top-k
    determinants — so the saved (states, coeffs) are NOT a self-consistent
    eigenpair. Therefore this routine RE-DIAGONALIZES H over `states` by default to
    obtain the true (E_var, coeffs) of the saved space before perturbing.

    Args:
        H: MixedH.
        states: iterable of MixedState — the variational determinant set V.
        coeffs: optional amplitudes. Ignored unless `trust_coeffs=True`.
        E_var: optional variational energy. Ignored unless `trust_coeffs=True`.
        diag_fn: (H, states) -> (E0, {state: amp}) diagonalizer used to re-solve V.
                Defaults to `graph.diagonalize` (dense; fine for cores up to a few
                thousand — pass a sparse/C++ diagonalizer for larger V).
        trust_coeffs: if True, use the supplied (coeffs, E_var) as-is and SKIP the
                re-diagonalization (only correct when the caller already has the
                exact eigenpair of V).
        intruder_eps: skip external determinants with |E_var - H_aa| < this
                (EN intruder-state guard; near-degenerate denominators blow up).
                Counted and returned so the skip is never silent.
        amp_tol: ignore external determinants whose coherent amplitude |A_a| is
                below this (numerical cancellation / negligible coupling).
        use_cpp: engine for pass-1 (the connections sweep + H_aa evaluation, the
                bottleneck at large #terms / large core). None (default) =
                auto-detect the C++ port (`_cpp_pt2_available`); True/False force
                it. The pure-Python path stays the fallback and the reference the
                C++ port is validated against — both give the same dE_pt2 to
                floating-point round-off.

    Returns:
        dict with:
          dE_pt2      : the correction (<= 0 in the normal case)
          E_var       : the self-consistent variational energy of V used
          E_pt2       : E_var + dE_pt2
          n_ext       : number of distinct external determinants summed
          n_intruder  : number skipped by the intruder guard
          E_var_rayleigh : Rayleigh quotient over V (== E_var to numerical
                           precision when re-diagonalized; a self-consistency check)
    """
    states = list(states)
    if len(states) == 0:
        return {"dE_pt2": 0.0, "E_var": E_var, "E_pt2": E_var,
                "n_ext": 0, "n_intruder": 0, "E_var_rayleigh": None}

    if trust_coeffs:
        if coeffs is None or E_var is None:
            raise ValueError("trust_coeffs=True requires both coeffs and E_var")
        c = np.asarray(list(coeffs), dtype=complex)
        cmap = {s: ci for s, ci in zip(states, c)}
        E_var = float(E_var)
    else:
        # Re-diagonalize the saved space so (E_var, |psi>) is a true eigenpair of
        # P_V H P_V — the precondition for a valid PT2 (see docstring).
        if diag_fn is None:
            from .graph import diagonalize as diag_fn
        E_var, cmap = diag_fn(H, states)
        E_var = float(E_var)
        c = np.asarray([cmap[s] for s in states], dtype=complex)

    nrm = np.linalg.norm(c)
    if nrm == 0:
        return {"dE_pt2": 0.0, "E_var": E_var, "E_pt2": E_var,
                "n_ext": 0, "n_intruder": 0, "E_var_rayleigh": None}
    c = c / nrm
    cmap = {s: ci / nrm for s, ci in cmap.items()}
    Vset = set(states)
    E0 = float(E_var)

    if use_cpp is None:
        use_cpp = _cpp_pt2_available(H)

    if use_cpp:
        # C++ pass-1: one connections sweep + diagonal H_aa, all in C++. Returns
        # per-external-determinant arrays; the reduction below is vectorized.
        from .backend import cpp_pt2_external
        amp_re, amp_im, H_aa, e_ray_re, _e_ray_im = cpp_pt2_external(H, states, c)
        E_ray = e_ray_re
        num = amp_re * amp_re + amp_im * amp_im             # |A_a|^2
        denom = E0 - H_aa
        amp_ok = num >= amp_tol                             # skip negligible A_a
        intruder = np.abs(denom) < intruder_eps             # EN intruder guard
        good = amp_ok & ~intruder
        dE = float(np.sum(num[good] / denom[good]))
        n_ext = int(good.sum())
        n_intruder = int((amp_ok & intruder).sum())
    else:
        # Pure-Python reference path. ONE pass over the core's connections:
        # split each connected row into the internal part (accumulates the
        # Rayleigh quotient <psi|H|psi>) and the external part (accumulates the
        # first-order amplitude A_a = <a|H|psi>).
        ext_amp = {}                      # external MixedState -> A_a
        E_ray_acc = 0.0 + 0.0j
        for j, cj in cmap.items():
            for a, h_aj in connections(H, j).items():     # h_aj = <a|H|j>
                if a in Vset:
                    E_ray_acc += np.conj(cmap[a]) * h_aj * cj
                else:
                    ext_amp[a] = ext_amp.get(a, 0.0 + 0.0j) + h_aj * cj
        E_ray = float(E_ray_acc.real)

        diag_t = diagonal_terms(H)
        dE = 0.0
        n_ext = 0
        n_intruder = 0
        for a, Aa in ext_amp.items():
            num = (Aa.real * Aa.real + Aa.imag * Aa.imag)      # |A_a|^2
            if num < amp_tol:
                continue
            H_aa = diagonal_element(H, a, diag_t)
            denom = E0 - H_aa
            if abs(denom) < intruder_eps:
                n_intruder += 1
                continue
            dE += num / denom
            n_ext += 1

    return {"dE_pt2": dE, "E_var": E0, "E_pt2": E0 + dE,
            "n_ext": n_ext, "n_intruder": n_intruder,
            "E_var_rayleigh": E_ray}


# ---------------------------------------------------------------------------
#  Convenience: PT2 straight off a GroundStateResult (object OR array path)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _arrays_to_states(ferm_arr, bos_arr):
    """Reassemble MixedState objects from the compact (ferm words, bos) arrays
    the array-native solver returns. `ferm_arr` is (N, W) uint64 little-endian
    words; `bos_arr` is (N, n_bos) uint16."""
    states = []
    ferm_arr = np.asarray(ferm_arr)
    bos_arr = np.asarray(bos_arr)
    for words, brow in zip(ferm_arr.tolist(), bos_arr.tolist()):
        f = 0
        for w, word in enumerate(words):
            f |= int(word) << (64 * w)
        states.append(MixedState(f, tuple(int(x) for x in brow)))
    return states


def pt2_from_result(H, res, **kwargs):
    """EN-PT2 for a `GroundStateResult`, handling both the object path
    (`res.states`/`res.coeffs`) and the array-native path
    (`res.ferm_arr`/`res.bos_arr`/`res.coeffs`).

    The saved core (states + amplitudes) is the POST-trim top-k, which is NOT a
    self-consistent eigenpair of its own span, so by default `epstein_nesbet_pt2`
    re-diagonalizes V. Pass `diag_fn=...` for a scalable diagonalizer on large
    cores (the default dense path caps at a few thousand determinants).
    """
    if getattr(res, "states", None):
        states = list(res.states)
    elif getattr(res, "ferm_arr", None) is not None:
        states = _arrays_to_states(res.ferm_arr, res.bos_arr)
    else:
        raise ValueError("GroundStateResult carries neither states nor arrays")
    return epstein_nesbet_pt2(H, states, **kwargs)
