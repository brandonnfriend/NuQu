"""Tests for the Architecture-C admissibility gate (`frame_arch_cgate.py`).

The verdict/fit logic is unit-tested on synthetic data (fast, no ED). One small ED
integration slice (L=2 d=1 A=1 N_f=2, dense fallback ~512 states) checks the sweep
runs end-to-end and ‖R_trans‖ grows with λ.

Run:  python -m classical.trimci.tests.test_frame_arch_cgate
"""

from classical.trimci import frame_arch_cgate as cg


def test_powerlaw_fit_and_readoff():
    """_powerlaw_fit recovers a known exponent; rtrans_at_lambda extrapolates on it,
    and drops sub-floor noise points."""
    # R = 3·λ^2 exactly on clean points
    pairs = [(0.1, 3 * 0.1**2), (0.2, 3 * 0.2**2), (0.3, 3 * 0.3**2), (0.4, 3 * 0.4**2)]
    fit = cg._powerlaw_fit(pairs)
    assert abs(fit['p'] - 2.0) < 1e-9 and abs(fit['c'] - 3.0) < 1e-9
    assert fit['r2'] > 0.999

    sweep = {'sites': 2, 'fit': fit,
             'points': [{'lam': l, 'R_trans': R, 'R_per_site': R / 2} for l, R in pairs]}
    R, R_ps, method = cg.rtrans_at_lambda(sweep, 0.28)
    assert method == 'powerlaw'
    assert abs(R - 3 * 0.28**2) < 1e-9 and abs(R_ps - R / 2) < 1e-12

    # sub-floor points are ignored (Lanczos noise)
    noisy = cg._powerlaw_fit([(0.05, 1e-9), (0.1, 3e-3), (0.2, 1.2e-2), (0.3, 2.7e-2)])
    assert noisy['n_points'] == 3, "the 1e-9 point must be dropped below r_floor"


def test_verdict_admissible_boundary():
    """c_gate_verdict flags INADMISSIBLE exactly when R/site × sites >= budget."""
    # R = 1.3·λ^2  -> at λ=0.28, R=0.1019 MeV over 2 sites -> 0.051/site
    fit = cg._powerlaw_fit([(0.1, 1.3 * 0.1**2), (0.2, 1.3 * 0.2**2),
                            (0.3, 1.3 * 0.3**2), (0.4, 1.3 * 0.4**2)])
    sweep = {'L': 2, 'dim': 1, 'N_f': 4, 'sites': 2, 'squeeze_iso_vs_bare': 1e-3,
             'fit': fit, 'points': []}
    # L=3 (27 sites): 0.051/site × 27 = 1.38 MeV > 1 MeV -> inadmissible
    v3 = cg.c_gate_verdict(sweep, production_lambda=0.28, production_sites=27,
                           budget_mev=1.0, production_label='L=3 d=3')
    assert not v3['admissible'] and 'INADMISSIBLE' in v3['verdict']
    assert v3['scaling_exponent_p'] is not None and abs(v3['scaling_exponent_p'] - 2) < 1e-6
    # L=2 (8 sites): 0.051/site × 8 = 0.41 MeV < 1 MeV -> admissible
    v2 = cg.c_gate_verdict(sweep, production_lambda=0.28, production_sites=8,
                           budget_mev=1.0, production_label='L=2 d=3')
    assert v2['admissible'] and 'ADMISSIBLE' in v2['verdict']
    assert 'NOT verified' in v2['caveat'], "verdict must carry the extrapolation caveat"
    print(f"[2] verdict: L=3 {v3['rtrans_total_mev']:.2f}MeV INADMISSIBLE, "
          f"L=2 {v2['rtrans_total_mev']:.2f}MeV ADMISSIBLE OK")


def test_no_density_coupling_raises():
    """A frame with no LF density coupling (seed λ=0) is not a usable test point."""
    # L=1 d=1 has no transition vertex -> analytic_displacement seeds nothing.
    try:
        cg.rtrans_lambda_sweep(1, 1, 2, 1, [0.1, 0.2])
        raise AssertionError("expected ValueError for seed λ=0")
    except ValueError as e:
        assert 'seed' in str(e).lower()
    print("[3] L=1 (no density coupling) correctly rejected OK")


def test_ed_integration_small():
    """End-to-end ED slice: ‖R_trans‖ is squeeze-referenced, finite, and grows with λ."""
    sweep = cg.rtrans_lambda_sweep(2, 1, 1, 1, [0.05, 0.2, 0.4])  # N_f=2, ~512 states
    assert sweep['seed_lambda'] > 0 and sweep['sites'] == 2
    Rs = [p['R_trans'] for p in sweep['points']]
    assert Rs[-1] > Rs[0], "‖R_trans‖ should grow from small to large λ"
    assert all(R >= 0 for R in Rs)
    print(f"[4] ED slice N_f={sweep['N_f']}: ‖R_trans‖ {Rs[0]:.2e}->{Rs[-1]:.2e} MeV "
          f"(squeeze_iso={sweep['squeeze_iso_vs_bare']:.1e}) OK")


if __name__ == '__main__':
    test_powerlaw_fit_and_readoff()
    print("[1] power-law fit + read-off OK")
    test_verdict_admissible_boundary()
    test_no_density_coupling_raises()
    test_ed_integration_small()
    print("\nall Architecture-C gate tests passed")
