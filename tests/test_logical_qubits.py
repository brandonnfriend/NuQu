"""
Verifies logical-qubit accounting for the amplitude split-oracle under BOTH walk
compositions:

  * split_sum   (legacy, invalid-for-QPE): peak = max(pos_walk, mom_walk) — the
    original bug (wrong pyLIQTR key / halved sum) is caught here.
  * combined_lcu (default, QPE-valid controlled-sum LCU): the single combined walk's
    register = system + reused sub-ancilla + LCU control + QFT workspace + reflection,
    which is ABOVE max(pos, mom) (it adds the control/workspace a real walk needs) and
    BELOW pos+mom (the sub-ancillas are reused, not summed).

Updated 2026-08-20 for the combined-walk fix (codex audit P0-4). Run from the root:
    python -m tests.test_logical_qubits
"""

import sys

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    get_physical_parameters,
)
from src_PI.utils.Config import Config


def _run(walk_composition, L, dim, n_b, pi_max, params):
    config = Config(pion_basis='amplitude', walk_mode='series',
                    walk_composition=walk_composition)
    return evaluate_resources(L, dim, n_b, pi_max, params, config)


def main():
    # Small problem so the test runs in seconds.
    L, dim, A = 2, 2, 1
    params = get_physical_parameters()
    n_b, pi_max, _ = calculate_dynamic_cutoffs(
        L, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )

    failures = []

    # --- legacy split_sum: peak = max(pos, mom) (not sum, not half-sum) ----------
    legacy = _run('split_sum', L, dim, n_b, pi_max, params)
    per_sub = legacy.get('Per_Sub_Walk') or []
    if len(per_sub) != 2:
        print(f"FAIL: expected 2 sub-walks for amplitude basis, got {len(per_sub)}")
        return 1
    walk_q = {e['name']: e['LogicalQubits'] for e in per_sub}
    names = list(walk_q)
    pos_w, mom_w = walk_q[names[0]], walk_q[names[1]]
    peak = max(pos_w, mom_w)
    legacy_total = legacy['Logical_Qubits']

    if legacy_total <= 0 or pos_w <= 0 or mom_w <= 0:
        failures.append(f"per-walk / total counts must be > 0 (got {pos_w}, {mom_w}, {legacy_total})")
    if legacy_total != peak:
        failures.append(
            f"split_sum Logical_Qubits={legacy_total} should equal max(pos, mom)={peak}, "
            f"not sum ({pos_w + mom_w}) or half-sum ({(pos_w + mom_w) // 2})")
    if legacy.get('walk_composition') != 'split_sum':
        failures.append(f"expected walk_composition='split_sum', got {legacy.get('walk_composition')!r}")

    # --- combined_lcu: single valid walk, corrected register --------------------
    combined = _run('combined_lcu', L, dim, n_b, pi_max, params)
    comb_total = combined['Logical_Qubits']
    comp = combined.get('composition_components', {})
    if combined.get('walk_composition') != 'combined_lcu':
        failures.append(f"expected walk_composition='combined_lcu', got {combined.get('walk_composition')!r}")
    # ABOVE the legacy peak (adds LCU control + QFT workspace + reflection qubit)…
    if not (comb_total > peak):
        failures.append(f"combined qubits {comb_total} should exceed the legacy peak {peak}")
    # …but BELOW the naive sum of the two full walks (sub-ancillas are reused).
    if not (comb_total < pos_w + mom_w):
        failures.append(f"combined qubits {comb_total} should be below the full-walk sum {pos_w + mom_w}")
    # exact structure: system + reused sub-ancilla + LCU control + QFT workspace + 1
    expect = (comp.get('n_system', 0) + comp.get('n_ancilla_sub_max', 0)
              + comp.get('n_lcu_control', 0) + comp.get('n_qft_workspace', 0) + 1)
    if comb_total != expect:
        failures.append(f"combined qubits {comb_total} != system+ancilla+LCU+QFT+1 = {expect}")
    if comp.get('n_lcu_control') != 1 or comp.get('n_qft_workspace') != n_b:
        failures.append(f"expected n_lcu_control=1, n_qft_workspace={n_b}; got {comp.get('n_lcu_control')}, {comp.get('n_qft_workspace')}")

    print("\n" + "=" * 56)
    print("       LOGICAL-QUBIT VERIFICATION (split vs combined)")
    print("=" * 56)
    print(f"L={L}, dim={dim}, A={A}, n_b={n_b}")
    print(f"per-walk qubits: {names[0]}={pos_w}  {names[1]}={mom_w}  -> peak={peak}")
    print(f"split_sum   Logical_Qubits: {legacy_total}   (== peak)")
    print(f"combined_lcu Logical_Qubits: {comb_total}   (system+ancilla+LCU+QFT+1={expect})")
    print(f"  LCU control={comp.get('n_lcu_control')}  QFT workspace={comp.get('n_qft_workspace')}")
    print("=" * 56)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: split_sum reports the peak; combined_lcu reports the valid single-walk register.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
