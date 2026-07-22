"""
Verifies that the logical-qubit count is correctly extracted from pyLIQTR
and reported as the peak (sequential) hardware requirement, not the sum
of the per-walk counts nor a halved sum. Updated 2026-05-22 for the
HamiltonianBundle / Config pipeline introduced by the Fock-basis refactor.

Run from the project root:
    python -m tests.test_logical_qubits
"""

import sys

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    get_physical_parameters,
)
from src_PI.utils.Config import Config


def main():
    # Small problem so the test runs in seconds.
    L, dim, A = 2, 2, 1
    params = get_physical_parameters()
    n_b, pi_max, _ = calculate_dynamic_cutoffs(
        L, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )

    # Use the amplitude basis with series walk-mode: that produces 2 sub-walks
    # (pos_dyn, mom) so the max-vs-sum distinction is exercised. This is the
    # configuration the original bug was hiding in.
    config = Config(pion_basis='amplitude', walk_mode='series')
    norm_data = evaluate_resources(L, dim, n_b, pi_max, params, config)

    if 'Logical_Qubits' not in norm_data:
        print("FAIL: 'Logical_Qubits' missing from norm_data")
        return 1

    per_sub = norm_data.get('Per_Sub_Walk') or []
    if len(per_sub) != 2:
        print(f"FAIL: expected 2 sub-walks for amplitude basis, got {len(per_sub)}")
        return 1

    total = norm_data['Logical_Qubits']
    by_name = {e['name']: e['LogicalQubits'] for e in per_sub}
    pos = by_name.get('pos_dyn')
    mom = by_name.get('mom')
    if pos is None or mom is None:
        print(f"FAIL: expected sub-walks 'pos_dyn' and 'mom', got {list(by_name)}")
        return 1

    print("\n" + "=" * 50)
    print("       LOGICAL-QUBIT VERIFICATION")
    print("=" * 50)
    print(f"L={L}, dim={dim}, A={A}, n_b={n_b}")
    print(f"basis={config.pion_basis}, walk_mode={config.walk_mode}")
    print(f"pos_dyn walk logical qubits: {pos}")
    print(f"mom     walk logical qubits: {mom}")
    print(f"Reported Logical_Qubits:     {total}")
    print(f"Expected (max of two):       {max(pos, mom)}")
    print("=" * 50)

    failures = []
    if total <= 0:
        failures.append(f"Logical_Qubits={total}, expected > 0 (the original bug)")
    if pos <= 0 or mom <= 0:
        failures.append(f"per-walk counts must be > 0 (got pos={pos}, mom={mom})")
    if total != max(pos, mom):
        failures.append(
            f"Logical_Qubits={total} should equal max(pos, mom)={max(pos, mom)}, "
            f"not the sum ({pos + mom}) or half-sum ({(pos + mom) // 2})"
        )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS: logical-qubit count is saved correctly and equals max(pos, mom).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
