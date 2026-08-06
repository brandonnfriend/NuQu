"""Architecture-C admissibility gate — the LF transition-vertex residual ‖R_trans‖.

Companion to `docs/frame_on_quantum_side.md` §3/§4-step-3 and `frame_arch_study.py`
(Architecture A). Architecture C would qubitize the *walk* of the squeeze+LF frame
`H̃ = squeeze ∘ LF_leadingorder(H)`, so QPE returns `spec(H̃)` — which is only
admissible if `H̃` is close enough to `H` in spectrum. The truncated LF is NOT a
similarity transform for our σ⊗τ transition vertex, so `spec(H̃) ≠ spec(H)`; the
Weyl bound gives `|E0(H̃) − E0(H)| ≤ ‖R_trans‖`, and C is admissible iff
`‖R_trans‖ (per site) × production_sites < ε_budget` (≈1 MeV) at the production
coupling `λ`.

**What we measure and why this shape (settled empirically, 2026-08-05):**

1. **Leading-order only (`fc_dress=None`).** The Franck-Condon-dressed LF frame that
   would restore exactness is CATASTROPHICALLY term-explosive — even at the smallest
   real system (L=2, d=1) an order-1 dressing is already ~37k terms and higher orders
   blow up (matches `frame.fc_dress_from_entries`'s warning). Production LF is
   therefore leading-order (see [[project_lf_leading_order]]), and that is exactly the
   frame whose residual the gate must bound. So there is no "order sweep to a floor" —
   the leading-order residual IS ‖R_trans‖.

2. **Squeeze-referenced.** On a finite Fock cutoff the squeeze itself leaks a small
   truncation error (`squeeze_iso` ~ 20 MeV at N_f=2, ~1 MeV at N_f=4 for L=2 d=1) —
   comparing `E0(H̃)` to `E0(bare)` is dominated by THAT, not the LF residual. Since
   the squeeze is (up to truncation) isospectral, the LF residual is isolated as the
   EXTRA shift LF adds on top of the squeeze:

       ‖R_trans‖(λ)  ≈  | E0(squeeze ∘ LF(λ)) − E0(squeeze) |

   which cancels the common squeeze-truncation floor. (We still report the raw
   vs-bare error and `squeeze_iso` as truncation-health diagnostics.)

3. **Lanczos, converged N_f.** We only need `E0`, so the sparse Lanczos path
   (`lanczos.lanczos_ground_state`) lets us reach the N_f where `squeeze_iso` is small
   without the dense-ED wall (~6000 states).

4. **λ-sweep, read off at production λ.** ‖R_trans‖ grows with the frame amplitude λ
   (measured: ~λ² at small λ, steepening above — higher BCH orders). The small ED
   system's own optimized λ is tiny (~0.004), unrepresentative of the production
   λ≈0.28 (the real L=3 system). So we sweep λ, fit a power law `R = c·λ^p`, and
   evaluate at the production λ — characterizing ‖R_trans‖ as a property of the
   TRANSFORM, independent of any one system's variational optimum (the agreed
   methodology).

**Load-bearing caveat.** ED reaches only L=2 d=1 (2 sites, 1D). The per-site
‖R_trans‖ is extrapolated to the production geometry (L=3 d=3, 27 sites) under a
per-site-extensivity assumption that a 2-site 1D measurement cannot itself verify;
the 3D vertex has more neighbors. The gate is a SCREEN, and its verdict is reported
with this caveat attached — never as a certified bound.
"""

from __future__ import annotations

import math


# Default production coupling: max |λ_{m,p}| at the real L=3 d=3 system (the value
# `frame.fc_dress_from_entries` quotes as needing order>=3 for FC accuracy).
DEFAULT_PRODUCTION_LAMBDA = 0.28
DEFAULT_BUDGET_MEV = 1.0


def _build_frame(L, dim, n_b):
    """Return `(H, Hsq, gen, seed_lambda, sites)` for the squeeze∘projector-LF frame:
    the bare EFT H, its Gaussian-squeezed image, the projector-LF generator, the
    analytic seed amplitude `max|λ_{m,p}|`, and the site count."""
    import numpy as np
    from classical.trimci import frame
    from classical.trimci.hamiltonian import build_from_eft
    H = build_from_eft(L, dim, n_b)
    r, phi = frame.analytic_squeeze(H)
    Hsq = frame.squeeze_terms(H, np.atleast_1d(r), np.atleast_1d(phi))
    entries, _, info = frame.analytic_displacement(Hsq)
    gen = frame.projector_generator(entries)
    return H, Hsq, gen, float(info['max_abs_lambda']), L ** dim


def rtrans_lambda_sweep(L, dim, n_b, A, lambdas, *, max_states=200_000,
                        verbose=False):
    """Squeeze-referenced ‖R_trans‖(λ) for the leading-order squeeze∘LF frame.

    For each target effective amplitude `λ` in `lambdas`, applies the projector-LF at
    the global scale `s = λ / seed_lambda` (so `s·seed = λ`) with `fc_dress=None`, and
    records `‖R_trans‖ = |E0(squeeze∘LF) − E0(squeeze)|` via Lanczos.

    Returns a dict:
        L, dim, n_b, N_f, A, sites, seed_lambda,
        E0_bare, E0_squeeze, squeeze_iso_vs_bare (truncation-health diagnostic),
        points: [{lam, scale, R_trans, R_per_site, R_vs_bare, E0_lf}],
        fit: {c, p, r2} (power-law R = c·λ^p on the positive points), or None.
    """
    from classical.trimci import frame
    from classical.trimci.lanczos import lanczos_ground_state

    def E0(H):
        e, _ = lanczos_ground_state(H, A, k=1, max_states=max_states)
        return float(e)

    H, Hsq, gen, seed, sites = _build_frame(L, dim, n_b)
    if seed <= 0:
        raise ValueError(
            f"L={L} d={dim} has no LF density coupling to displace (seed λ=0) — "
            "not a usable ‖R_trans‖ test point (e.g. L=1 has no transition vertex).")
    e_bare = E0(H)
    e_sq = E0(Hsq)
    squeeze_iso = abs(e_bare - e_sq)
    if verbose:
        print(f"[cgate] L={L}d{dim} N_f={H.N_f} A={A} sites={sites} "
              f"seed λ={seed:.5f} squeeze_iso={squeeze_iso:.2e} MeV")

    points = []
    for lam in lambdas:
        s = lam / seed
        Hlf = frame.displace_terms(Hsq, lambdas=float(s), gen=gen, fc_dress=None)
        e_lf = E0(Hlf)
        R = abs(e_lf - e_sq)
        points.append({'lam': float(lam), 'scale': float(s), 'R_trans': R,
                       'R_per_site': R / sites, 'R_vs_bare': abs(e_lf - e_bare),
                       'E0_lf': e_lf})
        if verbose:
            print(f"[cgate]   λ={lam:.3f}  ‖R_trans‖={R:.4e} MeV  "
                  f"({R / sites:.4e}/site)")

    fit = _powerlaw_fit([(p['lam'], p['R_trans']) for p in points])
    return {'L': L, 'dim': dim, 'n_b': n_b, 'N_f': H.N_f, 'A': A, 'sites': sites,
            'seed_lambda': seed, 'E0_bare': e_bare, 'E0_squeeze': e_sq,
            'squeeze_iso_vs_bare': squeeze_iso, 'points': points, 'fit': fit}


def _powerlaw_fit(pairs, r_floor=1e-6):
    """Least-squares `R = c·λ^p` on `[(λ, R), ...]` in log-log space. Drops points
    with `R < r_floor` (Lanczos noise near zero). Returns `{c, p, r2}` or None if
    fewer than two usable points."""
    pts = [(lam, R) for (lam, R) in pairs if lam > 0 and R > r_floor]
    if len(pts) < 2:
        return None
    xs = [math.log(lam) for lam, _ in pts]
    ys = [math.log(R) for _, R in pts]
    n = len(pts)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    p = sxy / sxx
    ln_c = my - p * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (ln_c + p * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {'c': math.exp(ln_c), 'p': p, 'r2': r2, 'n_points': n}


def rtrans_at_lambda(sweep, production_lambda):
    """Estimate `(R_trans, R_per_site)` at `production_lambda` from a sweep.

    Uses the power-law fit when available (robust extrapolation to a λ outside the
    swept range); otherwise falls back to linear interpolation between the two
    nearest swept λ (or the nearest point if extrapolating with no fit)."""
    sites = sweep['sites']
    fit = sweep.get('fit')
    if fit is not None:
        R = fit['c'] * production_lambda ** fit['p']
        return R, R / sites, 'powerlaw'
    pts = sorted(sweep['points'], key=lambda q: q['lam'])
    # nearest / linear interp fallback
    below = [q for q in pts if q['lam'] <= production_lambda]
    above = [q for q in pts if q['lam'] >= production_lambda]
    if below and above and below[-1]['lam'] != above[0]['lam']:
        lo, hi = below[-1], above[0]
        t = (production_lambda - lo['lam']) / (hi['lam'] - lo['lam'])
        R = lo['R_trans'] + t * (hi['R_trans'] - lo['R_trans'])
        return R, R / sites, 'interp'
    nearest = min(pts, key=lambda q: abs(q['lam'] - production_lambda))
    return nearest['R_trans'], nearest['R_per_site'], 'nearest'


def c_gate_verdict(sweep, *, production_lambda=DEFAULT_PRODUCTION_LAMBDA,
                   production_sites, budget_mev=DEFAULT_BUDGET_MEV,
                   production_label=''):
    """Admissibility verdict for Architecture C at a target production system.

    Reads ‖R_trans‖/site at `production_lambda` off the ED sweep, scales it to
    `production_sites`, and compares to `budget_mev` (Weyl: `|ΔE0| ≤ ‖R_trans‖`).

    Returns a dict with `admissible` (bool), the extrapolated `rtrans_total_mev`,
    per-site value, the scaling exponent `p`, and an explicit `caveat` string (the
    ED geometry is L=2 d=1; the per-site extrapolation to a 3D production system is
    unverified — this is a screen, not a certificate)."""
    R, R_per_site, method = rtrans_at_lambda(sweep, production_lambda)
    total = R_per_site * production_sites
    admissible = total < budget_mev
    p = sweep['fit']['p'] if sweep.get('fit') else None
    geom = '1D' if sweep['dim'] == 1 else f"{sweep['dim']}D"
    target = production_label or f"{production_sites} sites"
    caveat = (f"‖R_trans‖/site measured on L={sweep['L']} d={sweep['dim']} "
              f"({sweep['sites']} sites, {geom}) and extrapolated per-site to {target}; "
              f"per-site extensivity + 3D-vertex geometry NOT verified. "
              f"squeeze truncation floor at this N_f={sweep['N_f']}: "
              f"{sweep['squeeze_iso_vs_bare']:.2e} MeV (a lower bound on resolvable R).")
    return {
        'admissible': admissible,
        'production_lambda': production_lambda,
        'production_sites': production_sites,
        'production_label': production_label,
        'budget_mev': budget_mev,
        'rtrans_per_site_mev': R_per_site,
        'rtrans_total_mev': total,
        'scaling_exponent_p': p,
        'extrapolation_method': method,
        'ed_system': {'L': sweep['L'], 'dim': sweep['dim'], 'N_f': sweep['N_f'],
                      'sites': sweep['sites'], 'squeeze_iso': sweep['squeeze_iso_vs_bare']},
        'verdict': ('ADMISSIBLE' if admissible else 'INADMISSIBLE') +
                   f" @ {production_label or f'{production_sites} sites'}: "
                   f"‖R_trans‖≈{total:.2f} MeV vs {budget_mev} MeV budget",
        'caveat': caveat,
    }
