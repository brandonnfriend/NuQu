"""
Verifies the first-draft Tong-2022 per-site boson-cutoff ESTIMATE (NOT a certificate;
codex_audit/03_cutoff) wired into estimate_boson_cutoff via boson_cutoff_method='tong':

  - 'tong' returns max(n_b_eng, n_b_spec1) from tong_bound.cutoff_predictions
    (the doc's first-draft estimate choice),
  - that value lands at n_q = 4-5 and is essentially A-independent (Tong's
    polylog scaling), in contrast to the heuristic's log2(1+A) growth,
  - 'heuristic' is unchanged (the default) and still grows with A,
  - the NS amplitude path forwards the method (n_b = n_q + 1),
  - an unknown method raises, and
  - Config accepts/validates the new boson_cutoff_method axis and round-trips
    it through to_dict/from_dict.

Run from the project root:
    python -m tests.test_tong_cutoff
"""

import pytest

from src_PI.hamiltonians.core.EFTParameters import (
    calculate_ns_cutoffs,
    estimate_boson_cutoff,
    get_physical_parameters,
)
from src_PI.utils.Config import Config
from classical.trimci.tong_bound import cutoff_predictions


def _check(cond, msg, failures):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    params = get_physical_parameters()
    dim = 3
    failures = []

    print("\n" + "=" * 62)
    print("        TONG-2022 BOSON CUTOFF VERIFICATION")
    print("=" * 62)

    # --- 'tong' == doc's first-draft estimate choice max(n_b_eng, n_b_spec1) ------
    print("\n  method='tong' matches tong_bound.cutoff_predictions:")
    for (L, A) in [(2, 1), (2, 10), (2, 100), (3, 2)]:
        pred = cutoff_predictions(L, dim, A, params=params)
        expect = max(pred["n_b_eng"], pred["n_b_spec1"])
        n_q, _, _ = estimate_boson_cutoff(
            L, dim, A, params, boson_cutoff_method='tong'
        )
        _check(n_q == expect,
               f"L={L} A={A:>3}: tong n_q={n_q} == max(n_b_eng={pred['n_b_eng']}, "
               f"n_b_spec1={pred['n_b_spec1']})={expect}", failures)

    # --- tong is small (4-5) and A-flat; heuristic grows with A -----------
    print("\n  tong is A-flat (4-5); heuristic grows with A:")
    tong_vals = [estimate_boson_cutoff(2, dim, A, params,
                                       boson_cutoff_method='tong')[0]
                 for A in (1, 10, 100)]
    heur_vals = [estimate_boson_cutoff(2, dim, A, params,
                                       boson_cutoff_method='heuristic')[0]
                 for A in (1, 10, 100)]
    _check(all(4 <= v <= 5 for v in tong_vals),
           f"tong n_q in [4,5] for A=1,10,100: {tong_vals}", failures)
    _check(len(set(tong_vals)) == 1,
           f"tong n_q constant across A: {tong_vals}", failures)
    _check(heur_vals[0] < heur_vals[-1],
           f"heuristic n_q grows with A: {heur_vals}", failures)
    _check(tong_vals[-1] < heur_vals[-1],
           f"tong ({tong_vals[-1]}) < heuristic ({heur_vals[-1]}) at A=100",
           failures)

    # --- default is 'heuristic' (backward compatible) ---------------------
    print("\n  default preserves the heuristic:")
    n_default, _, _ = estimate_boson_cutoff(2, dim, 10, params)
    n_heur, _, _ = estimate_boson_cutoff(2, dim, 10, params,
                                         boson_cutoff_method='heuristic')
    _check(n_default == n_heur == 8,
           f"default == heuristic == 8 (A=10): default={n_default}, heur={n_heur}",
           failures)

    # --- NS path forwards the method (n_b = n_q + 1) ----------------------
    print("\n  calculate_ns_cutoffs forwards boson_cutoff_method:")
    for method in ('heuristic', 'tong'):
        n_q, _, _ = estimate_boson_cutoff(2, dim, 50, params,
                                          boson_cutoff_method=method)
        n_b_ns, _, _ = calculate_ns_cutoffs(2, dim, 50, params,
                                            boson_cutoff_method=method)
        _check(n_b_ns == n_q + 1,
               f"ns[{method}] n_b={n_b_ns} == n_q+1={n_q + 1}", failures)

    # --- unknown method raises --------------------------------------------
    print("\n  guards:")
    try:
        estimate_boson_cutoff(2, dim, 1, params, boson_cutoff_method='bogus')
        _check(False, "unknown method should raise ValueError", failures)
    except ValueError:
        _check(True, "unknown method raises ValueError", failures)

    # --- Config validates + round-trips the axis --------------------------
    print("\n  Config axis:")
    c = Config(pion_basis='fock', boson_cutoff_method='tong')
    _check(c.boson_cutoff_method == 'tong', "Config accepts 'tong'", failures)
    _check(Config().boson_cutoff_method == 'heuristic',
           "Config default is 'heuristic'", failures)
    _check(Config.from_dict(c.to_dict()).boson_cutoff_method == 'tong',
           "Config round-trips through to_dict/from_dict", failures)
    try:
        Config(boson_cutoff_method='nope')
        _check(False, "Config should reject invalid method", failures)
    except ValueError:
        _check(True, "Config rejects invalid boson_cutoff_method", failures)

    print("\n" + "=" * 62)
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        print("=" * 62)
        raise SystemExit(1)
    print("  ALL TONG-CUTOFF CHECKS PASSED")
    print("=" * 62)


# --------------------------------------------------------------------------- #
# pytest coverage for the exact-Bogoliubov 'tong_rigorous' method (task 25)     #
# --------------------------------------------------------------------------- #

def test_tong_rigorous_lands_in_single_digit_nq():
    """Gaussian-reference ESTIMATE brackets n_q = 4-5 across the physical L range (not a certificate)."""
    p = get_physical_parameters()
    for L in (2, 4, 6, 10):
        n_q, _, _ = estimate_boson_cutoff(
            L, 3, 4, p, epsilon_cut=1e-3, boson_cutoff_method='tong_rigorous')
        assert 4 <= n_q <= 5, f"L={L}: n_q={n_q} outside [4,5]"


def test_tong_rigorous_monotone_in_precision():
    """Tighter ε (smaller) never decreases the required N_f (polylog growth)."""
    from classical.trimci.gaussian_cutoff import tong_rigorous_predictions
    p = get_physical_parameters()
    N_prev = 0
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        N_f = tong_rigorous_predictions(4, 3, 4, p, eps=eps)['N_f']
        assert N_f >= N_prev, f"N_f dropped as ε tightened: {N_f} < {N_prev}"
        N_prev = N_f


def _cutoff_expecting_dim_warning(L, dim, A, p, **kw):
    """`estimate_boson_cutoff` at dim != 3 emits a RuntimeWarning. It is DELIBERATE and
    it is about the DIAGNOSTIC return values only: `n_q` comes from the requested
    (dim-general) method, but the `pi_max`/`Pi_max` returned alongside it for
    return-shape consistency come from `calculate_dynamic_cutoffs` (Watson Lemma 5),
    whose a_L powers are 3D-specific and are NOT used in Fock operator construction.

    The tests assert the warning instead of filtering it, so it keeps working as a real
    signal for any caller that does consume pi_max/Pi_max, while no longer showing up as
    unexplained noise in the suite (audit 2026-09-05, lower-priority findings)."""
    if dim == 3:
        return estimate_boson_cutoff(L, dim, A, p, **kw)
    with pytest.warns(RuntimeWarning, match="Watson Lemma 5"):
        return estimate_boson_cutoff(L, dim, A, p, **kw)


def test_tong_rigorous_is_dim_general():
    """Unlike the Watson-3D baseline, the Gaussian-reference-estimate path runs for dim != 3."""
    p = get_physical_parameters()
    for dim in (1, 2, 3):
        n_q, _, _ = _cutoff_expecting_dim_warning(
            2, dim, 2, p, epsilon_cut=1e-3, boson_cutoff_method='tong_rigorous')
        assert n_q >= 2


def test_gaussian_reference_estimate_is_canonical_alias():
    """The canonical 'gaussian_reference_estimate' name works and == the deprecated
    'tong_rigorous' alias (audit 03_cutoff rename)."""
    p = get_physical_parameters()
    for (L, dim) in ((2, 3), (2, 1)):
        canon = _cutoff_expecting_dim_warning(
            L, dim, 4, p, epsilon_cut=1e-3,
            boson_cutoff_method='gaussian_reference_estimate')[0]
        alias = _cutoff_expecting_dim_warning(
            L, dim, 4, p, epsilon_cut=1e-3, boson_cutoff_method='tong_rigorous')[0]
        assert canon == alias
    assert Config(boson_cutoff_method='gaussian_reference_estimate').boson_cutoff_method \
        == 'gaussian_reference_estimate'


def test_tong_rigorous_gaussian_tail_decreasing():
    """The exact per-mode occupation tail is monotone decreasing in N_f."""
    from classical.trimci.gaussian_cutoff import gaussian_tail
    p = get_physical_parameters()
    tails = [gaussian_tail(4, 3, N_f, p) for N_f in (2, 4, 6, 8, 10)]
    assert all(tails[i] > tails[i + 1] for i in range(len(tails) - 1))


def test_config_accepts_tong_rigorous():
    c = Config(pion_basis='fock', boson_cutoff_method='tong_rigorous')
    assert c.boson_cutoff_method == 'tong_rigorous'
    assert Config.from_dict(c.to_dict()).boson_cutoff_method == 'tong_rigorous'


if __name__ == "__main__":
    main()
