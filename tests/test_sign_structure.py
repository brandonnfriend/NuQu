"""
Tests for the D-AFQMC-sign diagnostics (classical/baselines/sign_structure.py):

  1. H_WT IS THE PHASE SOURCE — the fraction of COMPLEX off-diagonal matrix
     elements is > 0 for the full model and EXACTLY 0 when H_WT is toggled off.
     This is the exact, basis-invariant fingerprint: the imaginary eps_abc . Pi
     Weinberg-Tomozawa term injects complex weights no real gauge can remove.
  2. PERRON-FROBENIUS — the Troyer-Wiese severity Delta_sign = E0(H) - E0(H_stoq)
     is >= 0 for every term configuration (the stoquastic comparison is a lower
     bound), and > 0 for the full model at a generic filling (a genuine sign
     problem for worldline QMC).
  3. AVERAGE SIGN — <s>(beta) is <= 1 and monotonically non-increasing (decays).
  4. SPARSE == DENSE — the sparse path reproduces the dense complex-fraction and
     Delta_sign (so the size-scaling numbers are trustworthy).

Run from the project root:
    python -m tests.test_sign_structure
"""

import numpy as np

from classical.baselines.sign_structure import (
    dense_H, offdiag_report, sign_severity, average_sign_curve,
    sparse_H, sign_diagnostics_sparse,
)


def test_wt_is_the_phase_source():
    M_full, _ = dense_H(2, 1, 2, N_f=2)
    M_nowt, _ = dense_H(2, 1, 2, N_f=2, wt=0.0)
    r_full = offdiag_report(M_full)
    r_nowt = offdiag_report(M_nowt)
    print(f"  full : complex-offdiag = {r_full['frac_complex']*100:.1f}%")
    print(f"  no-WT: complex-offdiag = {r_nowt['frac_complex']*100:.1f}%")
    assert r_full['frac_complex'] > 0.01, "full model must have complex off-diagonals"
    assert r_nowt['frac_complex'] < 1e-9, "removing H_WT must remove all complex elements"
    print("  PASS: H_WT is exactly the source of the phase (complex-weight) problem\n")


def test_perron_frobenius_and_positive_severity():
    sites = 2
    ok = True
    for label, sc in [('full', {}), ('no-WT', {'wt': 0.0}),
                      ('no-Yukawa', {'av': 0.0}), ('no-pion', {'av': 0.0, 'wt': 0.0})]:
        M, _ = dense_H(2, 1, 2, N_f=2, **sc)
        sv = sign_severity(M, sites)
        ok = ok and (sv['Delta_sign'] >= -1e-8)
        print(f"  {label:10s}  Delta_sign = {sv['Delta_sign']:.4f} (>= 0 required)")
    assert ok, "Delta_sign must be >= 0 (Perron-Frobenius) for all configs"
    # full model at this filling is a genuine (nonzero) sign problem
    M, _ = dense_H(2, 1, 2, N_f=2)
    assert sign_severity(M, sites)['Delta_sign'] > 1e-4, \
        "full model should have a nonzero worldline sign gap at A=2"
    print("  PASS: Delta_sign >= 0 everywhere; > 0 for the full model\n")


def test_average_sign_decays():
    M, _ = dense_H(2, 1, 2, N_f=2)
    betas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    s = average_sign_curve(M, betas)
    print(f"  <s>(beta) = {np.round(s, 4)}")
    assert np.all(s <= 1.0 + 1e-9), "average sign must be <= 1"
    assert np.all(np.diff(s) <= 1e-9), "average sign must be non-increasing in beta"
    assert s[-1] < s[0], "average sign must actually decay"
    print("  PASS: <s>(beta) <= 1 and decays exponentially\n")


def test_sparse_matches_dense():
    M_dense, _ = dense_H(2, 1, 2, N_f=2)
    rd = offdiag_report(M_dense)
    sd = sign_severity(M_dense, 2)
    M_sp, sites, N = sparse_H(2, 1, 2, N_f=2)
    ss = sign_diagnostics_sparse(M_sp, sites)
    print(f"  complex-frac  dense={rd['frac_complex']:.4f}  sparse={ss['frac_complex']:.4f}")
    print(f"  Delta_sign    dense={sd['Delta_sign']:.5f}  sparse={ss['Delta_sign']:.5f}")
    assert abs(rd['frac_complex'] - ss['frac_complex']) < 1e-6
    assert abs(sd['Delta_sign'] - ss['Delta_sign']) < 1e-4
    print("  PASS: sparse path reproduces the dense diagnostics\n")


if __name__ == '__main__':
    print("=== D-AFQMC-sign diagnostics tests ===\n")
    test_wt_is_the_phase_source()
    test_perron_frobenius_and_positive_severity()
    test_average_sign_decays()
    test_sparse_matches_dense()
    print("ALL TESTS PASSED")
