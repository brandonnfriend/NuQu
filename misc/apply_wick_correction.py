"""Convert published NuQu numbers between the two contact-term conventions.

The 2026-09-14 Wick-ordering fix (`StaticTerms.wick_ordered`) changes the contact
terms from the un-prescribed `rho^2` / `sum_I rho_I^2` to Watson's normal-ordered
`:rho^2:` / `:sum_I rho_I^2:` (his Eqs. 54/55). The two Hamiltonians differ by the
one-body operator

    H_legacy - H_wick = c * N_hat ,   c = C/2 + 3*C_I2/2 = -23.3725 MeV

so EVERY published number converts in closed form -- nothing needs re-running:

  * CLASSICAL (fixed-A sector, N_hat = A): a pure energy shift
        E_wick = E_legacy - c*A = E_legacy + 23.3725*A   [MeV]
    Eigenvectors, determinant selection, PT2, sigma, compactness, every energy
    DIFFERENCE and the binding energy are EXACTLY invariant.

  * QUANTUM (whole register, N_hat is an operator): an exact lambda reduction
        d_lambda = |C + 3*C_I2| * L^dim = 46.745 * L^dim   MeV
    independent of n_b, with the Pauli term count unchanged (verified over
    dim=1,2,3 x L=1..8 x n_b=1..6 in tests/test_contact_normal_ordering.py).
    N_walk scales with lambda; walk_T moves only through
    circuit_precision = eps_be/(2*lambda), i.e. logarithmically.

Usage:
    python -m misc.apply_wick_correction                 # both anchor tables
    python -m misc.apply_wick_correction --classical-only
"""
import argparse
import glob
import json
import math
import os

from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.estimation.qpe_cost import qpe_phase_register_qubits_from_nwalk
from src_PI.estimation.total_t_optimizer import optimize_qpe_fraction

PARAMS = get_physical_parameters()
C_SELF = PARAMS['C'] / 2.0 + 3.0 * PARAMS['CI'] / 2.0        # -23.3725 MeV
DLAMBDA_PER_SITE = abs(PARAMS['C'] + 3.0 * PARAMS['CI'])     # 46.745 MeV


# --------------------------------------------------------------------------- #
#  Convention plumbing shared by every figure script
# --------------------------------------------------------------------------- #

CONVENTIONS = ('wick', 'legacy')
DEFAULT_CONVENTION = 'wick'

_NOTE_SHIFT = ("contact terms: Wick-ordered (Watson Eqs. 54/55) — "
               "energies +{shift:.4f}*A MeV vs the as-run legacy build")
_NOTE_LEGACY = ("contact terms: LEGACY un-Wick-ordered (as run) — carries a "
                "-{shift:.4f}*A MeV nucleon self-interaction")
_NOTE_LAMBDA = ("contact terms: Wick-ordered (Watson Eqs. 54/55) — "
                "lambda -{dl:.3f}*L^3 MeV vs the as-run legacy build")
_NOTE_INVARIANT = ("contact-term convention: EXACTLY INVARIANT under the 2026-09-14 "
                   "Wick fix (differences at matched A)")


def add_convention_arg(parser):
    """Standard `--convention {wick,legacy}` flag for a figure script."""
    parser.add_argument('--convention', choices=CONVENTIONS, default=DEFAULT_CONVENTION,
                        help="contact-term ordering. 'wick' (default) = Watson Eqs. 54/55; "
                             "'legacy' = the as-run pre-2026-09-14 build (a c*N_hat "
                             "self-interaction, c = -23.3725 MeV). See CONVENTIONS.md section 10.")
    return parser


def rung_shift(A, convention=DEFAULT_CONVENTION):
    """MeV to ADD to a fixed-A absolute energy to move it into `convention`.

    Shards are stored in the legacy convention, so 'legacy' is a no-op.
    """
    if convention == 'legacy':
        return 0.0
    if convention == 'wick':
        return -C_SELF * A                       # +23.3725 * A
    raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")


def apply_to_rungs(rungs, A, convention=DEFAULT_CONVENTION):
    """Shift a ladder's absolute energies in place. `dE_pt2` is NOT shifted —
    it is a difference and is exactly invariant (E_var and H_aa move together)."""
    d = rung_shift(A, convention)
    if d == 0.0:
        return rungs
    for r in rungs:
        for k in ('E_var', 'E_pt2'):
            if r.get(k) is not None:
                r[k] = r[k] + d
    return rungs


def figure_note(convention=DEFAULT_CONVENTION, kind='energy'):
    """One-line provenance string to stamp on a figure.

    kind: 'energy' (classical absolute energies), 'lambda' (quantum lambda/T),
          'invariant' (nothing on the figure changes).
    """
    if kind == 'invariant':
        return _NOTE_INVARIANT
    if convention == 'legacy':
        return _NOTE_LEGACY.format(shift=abs(C_SELF))
    if kind == 'lambda':
        return _NOTE_LAMBDA.format(dl=DLAMBDA_PER_SITE)
    return _NOTE_SHIFT.format(shift=abs(C_SELF))


def annotate(fig, convention=DEFAULT_CONVENTION, kind='energy', color='#898781'):
    """Stamp the convention flag along the bottom of a matplotlib figure."""
    fig.text(0.005, 0.005, figure_note(convention, kind), fontsize=6.5,
             color=color, ha='left', va='bottom')


# --------------------------------------------------------------------------- #
#  Classical
# --------------------------------------------------------------------------- #

def correct_energy(E_legacy, A):
    """Legacy total energy -> Wick-ordered convention (MeV)."""
    return E_legacy - C_SELF * A


def correct_energy_per_site(E_ps_legacy, A, sites):
    """Legacy per-site energy -> Wick-ordered convention (MeV/site)."""
    return E_ps_legacy - C_SELF * A / sites


# --------------------------------------------------------------------------- #
#  Quantum
# --------------------------------------------------------------------------- #

def correct_lambda(lam_legacy, L, dim=3):
    """Legacy Pauli one-norm -> Wick-ordered convention (MeV). Exact, n_b-independent."""
    return lam_legacy - DLAMBDA_PER_SITE * L ** dim


def recost(lam, fit_a, fit_b, delta_E=1.0):
    """Re-run the total-T optimizer at a corrected lambda, reusing the shard's own
    walk_T(circuit_precision) fit. Returns the full budget dict plus `m`."""
    out = optimize_qpe_fraction(fit_a, fit_b, lam, delta_E=delta_E)
    out['m'] = qpe_phase_register_qubits_from_nwalk(out['walk_queries'])
    return out


def correct_quantum_record(r, L, convention=DEFAULT_CONVENTION, dim=3):
    """Move one loaded shard record into `convention`, IN PLACE.

    Corrects `lam` by the exact closed form and then RE-RUNS the same total-T optimizer
    the shards themselves used, so `eps`, `walkT`, `T` and `m` all stay mutually
    consistent (walk_T moves only through circuit_precision = eps_be/(2*lambda), i.e.
    logarithmically -- about 0.01%). `q`/`q_total`/`terms` are untouched: the Pauli term
    count and every register width are exactly invariant.

    Expects the `nb3_padding_model._load` schema: lam, walkT, q, q_total, m, terms,
    eps, a, bb, dE (+ optional T). Records missing the walk_T fit are only lambda-shifted.
    """
    if convention == 'legacy' or r.get('lam') is None:
        return r
    r['lam'] = correct_lambda(r['lam'], L, dim)
    a, b = r.get('a'), r.get('bb', r.get('b'))
    if a is None or b is None:
        return r
    out = recost(r['lam'], a, b, r.get('dE', 1.0))
    r['eps'] = out['eps_qpe']
    r['walkT'] = out['walk_T']
    r['T'] = out['total_T']
    r['m'] = out['m']
    if r.get('q') is not None:
        r['q_total'] = r['q'] + out['m']
    return r


def correct_quantum_records(recs, convention=DEFAULT_CONVENTION, dim=3):
    """`{L: record}` -> same dict, moved into `convention` in place."""
    for L, r in recs.items():
        correct_quantum_record(r, L, convention, dim)
    return recs


def _shards(pattern):
    recs = {}
    for f in sorted(glob.glob(pattern)):
        j = json.load(open(f))
        if not (j.get('done') and j.get('results')):
            continue
        r = j['results'][0]
        b = r.get('QPE_Budget') or {}
        fit = b.get('walk_T_fit') or {}
        if fit.get('a') is None:
            continue
        recs[r['L']] = dict(lam=r['Physical_Lambda'], a=fit['a'], b=fit['b'],
                            dE=b.get('delta_E', 1.0), n_b=r['n_b'],
                            q=r['Logical_Qubits'], nw=r.get('QPE_Walk_Queries'),
                            T=b.get('total_T'), walkT=b.get('walk_T'))
    return recs


def anchor_table(pattern, title, dim=3):
    recs = _shards(pattern)
    if not recs:
        print(f"\n{title}: no shards found at {pattern}")
        return
    print(f"\n{title}")
    print(f"{'L':>2} {'lam_legacy':>13} {'lam_wick':>13} {'d%':>7} | "
          f"{'T_legacy':>11} {'T_wick':>11} {'d%':>7} | {'m_leg':>5} {'m_wick':>6} {'Q_tot':>7}")
    print("-" * 104)
    for L in sorted(recs):
        r = recs[L]
        lam_w = correct_lambda(r['lam'], L, dim)
        leg = recost(r['lam'], r['a'], r['b'], r['dE'])
        wic = recost(lam_w, r['a'], r['b'], r['dE'])
        dlam = 100 * (lam_w - r['lam']) / r['lam']
        dT = 100 * (wic['total_T'] - leg['total_T']) / leg['total_T']
        flag = '' if leg['m'] == wic['m'] else '  <-- m MOVED'
        print(f"{L:>2} {r['lam']:13.1f} {lam_w:13.1f} {dlam:7.3f} | "
              f"{leg['total_T']:11.3e} {wic['total_T']:11.3e} {dT:7.3f} | "
              f"{leg['m']:5d} {wic['m']:6d} {r['q'] + wic['m']:7d}{flag}")


def projection_table(dim=3):
    """L=8..10 n_b=3, via the same ratio route the padding model uses:
    lambda3(L) = <lambda3/lambda2 over L=4..6> * lambda2(L), then corrected.

    The padding model's walk_T depends on the term count (UNCHANGED by the fix) and
    on lambda only through circuit_precision, so the projection inherits the same
    relative lambda reduction and nothing about the bin assignment moves.
    """
    nb3 = _shards('data/quantum/nb3_anchor/L*_fock_pauli_nb3_opt.json')
    nb2 = _shards('data/quantum/2026-08-21/vertexfix_r3_290826/*L*_nb2*.json')
    fit_ls = [L for L in (4, 5, 6) if L in nb3 and L in nb2]
    if not fit_ls or not any(L in nb2 for L in (8, 9, 10)):
        print("\nPROJECTION n_b=3 L=8..10: source shards unavailable")
        return
    ratio = sum(nb3[L]['lam'] / nb2[L]['lam'] for L in fit_ls) / len(fit_ls)
    print(f"\nPROJECTION n_b=3, L=8..10 (lambda3/lambda2 ratio = {ratio:.4f} over L={fit_ls})")
    print(f"{'L':>2} {'lam_legacy':>13} {'lam_wick':>13} {'d%':>7}")
    print("-" * 40)
    for L in (8, 9, 10):
        if L not in nb2:
            continue
        lam_leg = ratio * nb2[L]['lam']
        lam_w = correct_lambda(lam_leg, L, dim)
        print(f"{L:>2} {lam_leg:13.1f} {lam_w:13.1f} "
              f"{100 * (lam_w - lam_leg) / lam_leg:7.3f}")
    print("  -> the published projected T (3.05e17 at L=10) inherits this ~0.14% reduction;\n"
          "     the padding bin, the term count and the qubit total are unchanged.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--classical-only', action='store_true')
    args = ap.parse_args()

    print(f"c  = C/2 + 3*C_I2/2      = {C_SELF:+.4f} MeV/nucleon   "
          f"(E_wick = E_legacy + {-C_SELF:.4f}*A)")
    print(f"dl = |C + 3*C_I2|        = {DLAMBDA_PER_SITE:.3f} MeV/site "
          f"(lambda_wick = lambda_legacy - dl*L^dim)")

    print("\nCLASSICAL BASELINE (dim=3, filling 1.0 => A = L^3 => +23.3725 MeV/site)")
    print(f"{'L':>2} {'sites':>6} {'E_var/site':>11} -> {'wick':>9} | "
          f"{'E_inf/site':>11} -> {'wick':>9}  sigma (unchanged)")
    print("-" * 78)
    for L, sites, e_var, e_inf, sig in [(2, 8, 225.7, 225.5, 0.2),
                                        (3, 27, 306.0, 291.2, 13.0),
                                        (4, 64, 345.9, 320.1, 25.1),
                                        (5, 125, 370.3, None, None)]:
        A = sites
        ev = correct_energy_per_site(e_var, A, sites)
        if e_inf is None:
            print(f"{L:>2} {sites:>6} {e_var:11.1f} -> {ev:9.1f} | "
                  f"{'bound only':>11}    {'—':>9}")
        else:
            ei = correct_energy_per_site(e_inf, A, sites)
            print(f"{L:>2} {sites:>6} {e_var:11.1f} -> {ev:9.1f} | "
                  f"{e_inf:11.1f} -> {ei:9.1f}  +/- {sig}")

    if args.classical_only:
        return
    anchor_table('data/quantum/nb3_anchor/L*_fock_pauli_nb3_opt.json',
                 'QUANTUM ANCHOR n_b=3 (compiled L=1..7)')
    anchor_table('data/quantum/2026-08-21/vertexfix_r3_290826/*L*_nb2*.json',
                 'QUANTUM ANCHOR n_b=2 (compiled)')
    projection_table()


if __name__ == '__main__':
    main()
