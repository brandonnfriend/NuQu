"""
Tests for the D-NQS baseline demo (classical/baselines/nqs_netket.py):

  1. OPERATOR CORRECTNESS — the mixed EFT H wrapped as a custom NetKet operator
     (via the project's `connections` oracle) reproduces the exact Lanczos ground
     energy under NetKet's own exact diagonalization. This is the gate: it proves
     the NQS optimizes the *same* H as block2 / TrimCI / Lanczos.
  2. VARIATIONAL BOUND — a short FullSum VMC run returns a finite energy that is a
     variational upper bound (>= exact) and that the optimizer lowers from its
     random-init value. (The *headline* D-NQS result — that off-the-shelf NQS
     stalls ~11 MeV above the ground state while TrimCI is near-exact — is recorded
     in `claude/research/classical_baselines/03_nqs_results.md`, not asserted here,
     since the exact stall value is ansatz/seed dependent.)

Run from the project root:
    python -m tests.test_nqs
"""

import numpy as np

from classical.baselines.nqs_netket import validate_vs_lanczos, run_vmc, build


def test_operator_matches_lanczos():
    ok = validate_vs_lanczos(L=2, dim=1, A=2, N_f=2, tol=1e-5)
    assert ok, "NetKet-wrapped operator does not match exact Lanczos"
    print("  PASS: custom NetKet operator == exact Lanczos (same H)\n")


def test_nqs_variational_bound():
    from classical.trimci import build_from_eft
    from classical.trimci.lanczos import lanczos_ground_state
    E_lanc, _ = lanczos_ground_state(build_from_eft(L=2, dim=1, n_b=1, N_f=2), 2)
    r = run_vmc(2, 1, 2, N_f=2, alpha=2, n_iter=120, lr=0.05, verbose=False)
    print(f"  Lanczos={E_lanc:.4f}  NQS E_best={r['E_best']:.4f}  "
          f"init->best drop, mode={r['mode']}, n_params={r['n_params']}")
    assert np.isfinite(r['E_best']), "NQS energy must be finite"
    # variational: cannot go below exact (small numerical slack)
    assert r['E_best'] >= E_lanc - 1e-2, "NQS below exact -> operator/gradient bug"
    # optimizer actually lowered the energy from its (high) random init
    assert r['history'][0] - r['E_best'] > 1.0, "VMC did not optimize"
    print("  PASS: NQS returns a finite variational upper bound and optimizes\n")


if __name__ == '__main__':
    print("=== D-NQS tests ===\n")
    test_operator_matches_lanczos()
    test_nqs_variational_bound()
    print("ALL TESTS PASSED")
