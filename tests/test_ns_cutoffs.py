"""
Verifies the Nyquist-Shannon optimal cutoff (Path B / Reading B) for the
amplitude basis:

  - calculate_ns_cutoffs returns one qubit more than the shared boson
    cutoff (n_b = n_q + 1, since N_phi = 2 * N_b),
  - the Macridin Eq. 87 windows satisfy the matching ratio Pi_max/pi_max
    = omega_0 and the NS register relation (2 a_L^d / pi) pi_max Pi_max
    = 2^n_b,
  - the Pi_max returned by calculate_ns_cutoffs matches the Pi_max the
    operator builder actually realizes via get_Pp_Qp's conjugate grid
    (exactly (2^n_b - 1)/2^n_b, i.e. one grid cell),
  - n_b is independent of L at fixed A and undercuts the energy-bound path,
  - Config accepts/validates the new cutoff_method axis, and a saved sweep
    records it in metadata.

Run from the project root:
    python -m tests.test_ns_cutoffs
"""

import json
import math
import os

from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    calculate_ns_cutoffs,
    estimate_boson_cutoff,
    get_physical_parameters,
)
from src_PI.utils.Config import Config
from src_PI.utils.DataIO import save_sweep_data
from src_PI.utils.utils import get_Pi_max


def _check(cond, msg, failures):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    params = get_physical_parameters()
    a_L = params['a_L']
    omega_0 = params['m_0']
    L, dim, A = 2, 3, 1
    failures = []

    print("\n" + "=" * 60)
    print("       NYQUIST-SHANNON CUTOFF VERIFICATION (Path B)")
    print("=" * 60)

    # --- Core: cutoff and the n_b = n_q + 1 relation -------------------
    n_q, _, _ = estimate_boson_cutoff(L, dim, A, params, epsilon_cut=0.1, E_bound=10.0)
    n_b, pi_max, Pi_max = calculate_ns_cutoffs(
        L, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )
    N_b = 2 ** n_q
    N_phi = 2 ** n_b
    print(f"L={L}, dim={dim}, A={A}:  n_q={n_q}  ->  N_b={N_b}, "
          f"N_phi={N_phi}, n_b={n_b}")
    print(f"  pi_max={pi_max:.6g} MeV,  Pi_max={Pi_max:.6g} MeV^2")

    _check(n_b == n_q + 1,
           f"n_b == n_q + 1  ({n_b} == {n_q + 1})", failures)

    # --- Matching ratio K/F = omega_0 ----------------------------------
    ratio = Pi_max / pi_max
    _check(math.isclose(ratio, omega_0, rel_tol=1e-9),
           f"Pi_max/pi_max == omega_0  ({ratio:.6g} == {omega_0})", failures)

    # --- NS register relation (2 a_L^d / pi) F K = N_phi ---------------
    ns_count = (2.0 * a_L**dim / math.pi) * pi_max * Pi_max
    _check(math.isclose(ns_count, N_phi, rel_tol=1e-9),
           f"(2 a_L^d/pi) pi_max Pi_max == 2^n_b  ({ns_count:.6g} == {N_phi})",
           failures)

    # --- Pi_max consistency with the realized conjugate grid -----------
    # get_Pp_Qp recomputes Pi_max from (pi_max, n_b); the grid uses
    # spacing over (2^n_b - 1) points, so the realized window is exactly
    # (2^n_b - 1)/2^n_b of the NS target (one grid cell).
    Pi_max_realized = get_Pi_max(pi_max, n_b, a_L, dim)
    expected = Pi_max * (2**n_b - 1) / 2**n_b
    rel_err = abs(Pi_max_realized - Pi_max) / Pi_max
    print(f"  realized Pi_max={Pi_max_realized:.6g}  "
          f"(rel. diff {rel_err:.3%}, one grid cell = {1/2**n_b:.3%})")
    _check(math.isclose(Pi_max_realized, expected, rel_tol=1e-9),
           "realized Pi_max == NS Pi_max * (2^n_b - 1)/2^n_b (exact)", failures)
    _check(rel_err < 2.0 / 2**n_b,
           f"realized vs NS Pi_max within one grid cell ({rel_err:.3%})", failures)

    # The realized grid also reproduces the register count N_phi - 1.
    realized_count = (2.0 * a_L**dim / math.pi) * pi_max * Pi_max_realized
    _check(math.isclose(realized_count, N_phi - 1, rel_tol=1e-9),
           f"realized grid count == 2^n_b - 1  ({realized_count:.6g} "
           f"== {N_phi - 1})", failures)

    # --- L-independence at fixed A -------------------------------------
    n_b_L4, pi_L4, Pi_L4 = calculate_ns_cutoffs(
        4, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )
    _check(n_b_L4 == n_b and math.isclose(pi_L4, pi_max, rel_tol=1e-12),
           f"n_b/pi_max independent of L  (L=2 -> {n_b}, L=4 -> {n_b_L4})",
           failures)

    # --- NS undercuts the energy-bound cutoff --------------------------
    n_b_eb, _, _ = calculate_dynamic_cutoffs(
        L, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )
    print(f"  qubit savings: energy_bound n_b={n_b_eb}  vs  NS n_b={n_b}")
    _check(n_b < n_b_eb,
           f"NS n_b < energy_bound n_b  ({n_b} < {n_b_eb})", failures)

    # --- Config validation ---------------------------------------------
    try:
        Config(pion_basis='amplitude', cutoff_method='ns')
        _check(True, "Config accepts cutoff_method='ns'", failures)
    except Exception as e:  # noqa: BLE001
        _check(False, f"Config rejected cutoff_method='ns': {e}", failures)
    try:
        Config(cutoff_method='bogus')
        _check(False, "Config should reject cutoff_method='bogus'", failures)
    except ValueError:
        _check(True, "Config rejects unknown cutoff_method", failures)

    # --- Save round-trip records cutoff_method in metadata -------------
    config = Config(pion_basis='amplitude', cutoff_method='ns')
    fake_results = [{
        'A': A, 'L': L, 'dim': dim, 'n_b': n_b,
        'pi_max': pi_max, 'Pi_max': Pi_max,
        'Physical_Lambda': 1.0, 'Logical_Qubits': 1,
        'Walk_Clifford_Count': 0, 'Walk_T_Count': 0,
        'QFT_T_Count': 0, 'Total_T_Count': 1,
    }]
    saved_path = save_sweep_data(L, dim, params, fake_results, config=config)
    try:
        with open(saved_path) as f:
            saved = json.load(f)
        cm = saved['metadata'].get('config', {}).get('cutoff_method')
        _check(cm == 'ns',
               f"saved metadata records cutoff_method='ns' (got {cm!r})", failures)
        _check('_ns' in os.path.basename(saved_path),
               f"filename tags the cutoff method ({os.path.basename(saved_path)})",
               failures)
    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)

    print("=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
