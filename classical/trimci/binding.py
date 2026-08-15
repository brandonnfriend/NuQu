"""Fixed-A box-convergence BINDING ENERGIES from the dynamical-pion lattice EFT.

The raw TrimCI eigenvalue E(A, L) is NOT a binding energy: it is dominated by the
extensive pion-field vacuum energy (exactly 3*m_pi/2 per site = 202.5 MeV/site, an
un-normal-ordered zero-point the classical path never subtracts) plus the extensive
nucleon energy. The PHYSICAL binding energy is the intensive difference against the
A-free-nucleon threshold, in which the pion vacuum cancels:

    BE(A, L) = A*E(1, L) - (A-1)*E(0, L) - E(A, L)
             = A*(E(1)-E(0))  -  (E(A)-E(0))

i.e. A single-nucleon energies (each measured above the pion vacuum E(0)) minus the
bound A-nucleon energy (above the same vacuum). BE > 0 <=> bound. Because every term
carries the same +202.5*sites vacuum constant, it drops out identically -- BE depends
only on the small nucleon/interaction differences.

Increasing the box (side L*a_L) at FIXED A, BE(A, L) converges once the box contains the
nucleus: the finite-volume correction of a bound state decays ~ exp(-kappa * L * a_L).
THAT convergence-to-a-plateau -- not growth -- is how a lattice EFT predicts a binding
energy. (Our filling=1.0 dets-vs-L runs instead grow A with L => nuclear matter => the
extensive energy that motivated this module.)

Fixed parameters (Watson Table I), a_L = 2.2 fm, NO low-energy-constant fitting: we
observe the model at its given couplings, not fit to data -- so the converged BE carries
a coarse-lattice / unfit-LEC systematic and is not expected to reproduce experiment.
"""
from __future__ import annotations

import numpy as np

A_L_FM = 2.2                      # Watson lattice spacing (EFTParameters.py a_L)
ZERO_POINT_PER_SITE = 202.5       # 3 pion species * m_pi/2, m_pi=135 MeV (fock_native.py)

# experimental total binding energies (MeV) -- sanity reference lines only
EXPT_BE = {2: 2.224, 3: 8.482, 4: 28.296}     # deuteron, triton, alpha


def binding_energy(E0, E1, EA, A):
    """BE(A) = A*E(1) - (A-1)*E(0) - E(A). Positive => bound. All E are raw eigenvalues
    (the +202.5*sites vacuum cancels)."""
    return A * E1 - (A - 1) * E0 - EA


def binding_energy_sigma(s0, s1, sA, A):
    """Propagate independent per-sector convergence uncertainties into BE (uncorrelated
    upper bound; the shared vacuum error is actually correlated and cancels more, so this
    is conservative)."""
    return float(np.sqrt((A * s1) ** 2 + ((A - 1) * s0) ** 2 + sA ** 2))


def box_size_fm(L, a_L=A_L_FM):
    """Physical box side length L * a_L in fm."""
    return L * a_L


def vacuum_constant(L, dim=3, per_site=ZERO_POINT_PER_SITE):
    """The exact extensive pion zero-point that dominates the raw E(A,L) and cancels in
    BE: 202.5 MeV * L**dim. (Cross-check vs build_from_eft(L,dim).constant().)"""
    return per_site * (L ** dim)


def box_convergence(sector_energies, A, dim=3, a_L=A_L_FM):
    """Assemble BE(A, L) across box sizes.

    sector_energies: {L: {0: (E, sigma), 1: (E, sigma), A: (E, sigma)}} -- each entry the
    converged (or extrapolated) sector energy and its uncertainty. L missing any of the
    three required sectors (0, 1, A) is skipped.

    Returns a list of dicts sorted by L: {L, box_fm, BE, BE_sigma, E0, E1, EA}. The
    deepest-box BE is the finite-volume estimate; watch it plateau across L."""
    rows = []
    for L in sorted(sector_energies):
        s = sector_energies[L]
        if not all(k in s for k in (0, 1, A)):
            continue
        (E0, s0), (E1, s1), (EA, sA) = s[0], s[1], s[A]
        rows.append({
            "L": L, "box_fm": box_size_fm(L, a_L),
            "BE": binding_energy(E0, E1, EA, A),
            "BE_sigma": binding_energy_sigma(s0, s1, sA, A),
            "E0": E0, "E1": E1, "EA": EA,
        })
    return rows


def finite_volume_extrapolate(rows, kappa=None):
    """Estimate the infinite-volume BE from BE(L) rows. A bound-state finite-volume
    correction decays ~ exp(-kappa * box), so fit BE(box) = BE_inf - C*exp(-kappa*box).
    If kappa is None, grid-search it (like the energy extrapolator). Needs >=3 boxes.
    Returns {BE_inf, kappa, C, r2} or None. The plateau of the deepest boxes is the
    honest headline; this just quantifies the tail."""
    rows = [r for r in rows if np.isfinite(r["BE"])]
    if len(rows) < 3:
        return None
    x = np.array([r["box_fm"] for r in rows], float)
    y = np.array([r["BE"] for r in rows], float)
    grid = [kappa] if kappa else np.linspace(0.05, 2.0, 400)
    best = None
    for k in grid:
        basis = np.exp(-k * x)
        # y = BE_inf - C*basis  ->  linear in (1, basis)
        Aels = np.vstack([np.ones_like(basis), -basis]).T
        coef, *_ = np.linalg.lstsq(Aels, y, rcond=None)
        pred = Aels @ coef
        ss = np.var(y - pred)
        r2 = 1 - ss / np.var(y) if np.var(y) > 0 else 0.0
        if best is None or r2 > best[0]:
            best = (r2, k, coef[0], coef[1])
    r2, k, BE_inf, C = best
    return {"BE_inf": float(BE_inf), "kappa": float(k), "C": float(C), "r2": float(r2)}
