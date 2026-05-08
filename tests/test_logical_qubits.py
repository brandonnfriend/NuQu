"""
Verifies that the logical-qubit count is correctly extracted from pyLIQTR
and reported as the peak (sequential) hardware requirement, not the sum
of the pos/mom walks nor a halved sum.

Run from the project root:
    python -m tests.test_logical_qubits
"""

import sys
from src_PI.hamiltonians.core.EFTParameters import (
    get_physical_parameters,
    calculate_dynamic_cutoffs,
)
from src_PI.estimation.EstimateResources import evaluate_resources


def main():
    # Small problem so the test runs in seconds.
    L, dim, A = 2, 2, 1
    params = get_physical_parameters()
    n_b, pi_max, _ = calculate_dynamic_cutoffs(
        L, dim, A, params, epsilon_cut=0.1, E_bound=10.0
    )

    norm_data = evaluate_resources(L, dim, n_b, pi_max, params)

    expected_keys = (
        "Logical_Qubits",
        "Pos_Walk_Logical_Qubits",
        "Mom_Walk_Logical_Qubits",
    )
    missing = [k for k in expected_keys if k not in norm_data]
    if missing:
        print(f"FAIL: missing keys in norm_data: {missing}")
        return 1

    total = norm_data["Logical_Qubits"]
    pos = norm_data["Pos_Walk_Logical_Qubits"]
    mom = norm_data["Mom_Walk_Logical_Qubits"]

    print("\n" + "=" * 50)
    print("       LOGICAL-QUBIT VERIFICATION")
    print("=" * 50)
    print(f"L={L}, dim={dim}, A={A}, n_b={n_b}")
    print(f"Pos walk logical qubits:  {pos}")
    print(f"Mom walk logical qubits:  {mom}")
    print(f"Reported Logical_Qubits:  {total}")
    print(f"Expected (max of two):    {max(pos, mom)}")
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
