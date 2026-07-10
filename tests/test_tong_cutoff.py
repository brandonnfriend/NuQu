"""
Verifies the Tong-2022 rigorous per-site boson cutoff wired into
estimate_boson_cutoff via boson_cutoff_method='tong':

  - 'tong' returns max(n_b_eng, n_b_spec1) from tong_bound.cutoff_predictions
    (the doc's "certified rigorous choice"),
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

    # --- 'tong' == doc's certified choice max(n_b_eng, n_b_spec1) ----------
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


if __name__ == "__main__":
    main()
