"""Unit test for the amplitude combined-walk COST BOOKKEEPING (combined_walk.py).

Scope note (codex amplitude_combined_walk_audit_2026-08-20): this checks only the numerical
roll-up (branch sums, ancilla reuse, LCU/QFT overhead) — it does NOT construct any unitary,
verify the projected block equals the target amplitude Hamiltonian, or validate H_WT's
species-selective basis. The amplitude path is EXPERIMENTAL and not publication-grade; the
paper anchor is Fock/PauliLCU. Run from the root:
    python -m tests.test_combined_walk
"""
import sys

from src_PI.estimation.combined_walk import (
    compose_combined_walk, wt_basis_change_t, _reflection_t, _rotation_synth_t,
    _qft_t_per_register,
)


def main():
    L, dim, n_b = 2, 3, 2
    sites = L ** dim
    n_system = sites * (4 + 3 * n_b)            # 8 * 10 = 80
    # two synthetic sub-oracle block encodings (pos larger than mom)
    per_sub = [
        {'name': 'pos_dyn', 'T_enc': 1_000_000, 'Clifford_enc': 2_000_000,
         'qubits_enc': n_system + 30, 'n_system': n_system, 'alpha': 100.0},
        {'name': 'mom', 'T_enc': 400_000, 'Clifford_enc': 800_000,
         'qubits_enc': n_system + 22, 'n_system': n_system, 'alpha': 40.0},
    ]
    momentum_qft = 2 * (3 * sites) * _qft_t_per_register(n_b)   # calculate_qft_cost form

    out = compose_combined_walk(per_sub, momentum_qft, L, dim, n_b)
    c = out['composition_components']
    fails = []

    # --- λ-independent structural checks --------------------------------------
    # walk T = Σ SELECT+PREPARE + momentum QFT + WT basis change + LCU prepare + reflection
    n_anc = 30 + 1 + n_b                                    # max sub-ancilla + LCU + QFT ws
    expect_walk_t = (1_000_000 + 400_000
                     + momentum_qft + wt_basis_change_t(L, dim, n_b)
                     + (2 - 1) * _rotation_synth_t()
                     + _reflection_t(n_anc))
    if out['Walk_T_Count'] != expect_walk_t:
        fails.append(f"Walk_T {out['Walk_T_Count']} != expected {expect_walk_t}")

    # It is a SINGLE walk, not the sum of two full independent walks.
    if out['Walk_T_Count'] >= 1_400_000 + 2 * expect_walk_t:
        fails.append("combined walk should not exceed the naive doubled sum")

    # qubits = system + reused max sub-ancilla + LCU control + QFT workspace + reflection
    expect_q = n_system + 30 + 1 + n_b + 1
    if out['Logical_Qubits'] != expect_q:
        fails.append(f"Logical_Qubits {out['Logical_Qubits']} != expected {expect_q}")

    # ancilla is REUSED (max), not summed: 30, not 30+22
    if c['n_ancilla_sub_max'] != 30:
        fails.append(f"n_ancilla_sub_max {c['n_ancilla_sub_max']} != 30 (should be max, reused)")
    if c['n_lcu_control'] != 1 or c['n_qft_workspace'] != n_b:
        fails.append(f"LCU/QFT workspace wrong: {c['n_lcu_control']}, {c['n_qft_workspace']}")

    # WT species-selective basis change is present and nonzero at n_b>1
    if c['wt_species_basis_change_T'] <= 0:
        fails.append("WT species-selective basis change should be > 0 at n_b=2")

    # --- k<2 must refuse (single-walk Fock path never routes here) --------------
    try:
        compose_combined_walk(per_sub[:1], 0, L, dim, n_b)
        fails.append("compose_combined_walk should raise for k<2")
    except ValueError:
        pass

    print("=" * 56)
    print("   COMBINED-WALK COMPOSITION UNIT TEST")
    print("=" * 56)
    print(f"Walk_T={out['Walk_T_Count']}  qubits={out['Logical_Qubits']}")
    print(f"components: {c}")
    print("=" * 56)
    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: cost bookkeeping consistent (ancilla reused, LCU+QFT added). "
          "NOTE: bookkeeping only — not a validated block encoding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
