"""Original-H back-evaluation for the approximate LF / composed frames (docs/lf_backevaluation.md).

pytest-collected (audit gap 5). TINY ED systems only (memory-bounded): L=2 d=1 n_b=1 A=1 is
~512 states. Covers: the non-isospectrality trap (E_frame below E_bare), the restored Ritz
bound (E_orig >= E_bare), the sparse map-back == exact dense expm for BOTH the pure
projector-LF generator AND the composed gaussian+lf frame (the production frames), the
real-GroundStateResult interface, and enforced Taylor convergence.

Run: python -m pytest -q tests/test_back_evaluate.py   (or python -m tests.test_back_evaluate)
"""
import numpy as np
from scipy.linalg import expm

from classical.trimci.hamiltonian import build_from_eft
from classical.trimci.lf import displacement_generator, _ground_vector
from classical.trimci.frame import displace_terms, squeeze_generator_terms
from classical.trimci.frame_workflow import initial_frame_state
from classical.trimci.hij import build_dense
from classical.trimci.back_evaluate import (
    back_evaluate, back_evaluate_ed, back_evaluate_frame, generator_from_disp_gen,
    exp_generator_apply,
)

L, DIM, N_B, A = 2, 1, 1, 1                              # ~512 states — tiny


def test_pure_lf_ritz_and_dense_match():
    """Physical coupling: E_orig >= E_bare, sparse map-back == dense expm, convergence certified."""
    saw_trap = False
    for lam in (0.1, 0.2, 0.3):
        r = back_evaluate_ed(L, DIM, N_B, lam=lam, A=A, coupling_scale=1.0)
        assert r['gap_orig'] >= -1e-6, f"Ritz violated (gap {r['gap_orig']})"
        assert r['sparse_vs_dense'] < 1e-6, f"sparse!=dense ({r['sparse_vs_dense']})"
        assert abs(r['norm_ratio'] - 1.0) < 1e-9
        if r['frame_shift'] < -1e-4:
            saw_trap = True
    assert saw_trap, "expected E_frame < E_bare somewhere (the non-isospectral trap)"


def test_composed_gaussian_lf_matches_dense():
    """The PRODUCTION gaussian+lf frame: composed sparse map-back == exact dense U_sq·U_lf,
    E_orig >= E_bare, convergence certified. Exercises the projector-LF generator (audit gap 2)
    and the squeeze∘LF composition (audit gap 3)."""
    H_bare = build_from_eft(L, DIM, N_B)
    state, res, H_frame, info = initial_frame_state(
        H_bare, A, has_gaussian=True, has_lf=True, core=120, num_runs=8, seed=0)
    basis, gf, ef = _ground_vector(H_frame, A)
    sd = {basis[i]: complex(gf[i]) for i in range(len(basis)) if abs(gf[i]) > 1e-14}
    r = back_evaluate_frame(H_bare, state, sd, strict=True)

    # dense exact composed unitary  U_sq U_lf
    S_lf = generator_from_disp_gen(state['disp_gen'], H_bare.n_ferm_modes,
                                   H_bare.n_bos_modes, H_bare.N_f)
    U_lf = expm(state['disp_scale'] * build_dense(S_lf, basis))
    U_sq = expm(build_dense(squeeze_generator_terms(H_bare, state['r'], state.get('phi', 0.0)),
                            basis))
    psi_d = U_sq @ (U_lf @ gf)
    Hd = build_dense(H_bare, basis)
    E_dense = float((psi_d.conj() @ Hd @ psi_d).real / (psi_d.conj() @ psi_d).real)
    _, _, e0 = _ground_vector(H_bare, A)

    assert abs(r['E_orig'] - E_dense) < 1e-9, f"composed sparse!=dense ({abs(r['E_orig']-E_dense)})"
    assert r['E_orig'] >= e0 - 1e-6, "Ritz violated for composed frame"
    assert r['converged'], "composed map-back did not certify convergence"
    # the composed frame is strongly non-isospectral -> E_frame well below E_bare (the trap)
    assert ef < e0 - 0.1, f"expected composed E_frame << E_bare, got {ef-e0:+.3f}"


def test_production_result_interface():
    """back_evaluate_frame consumes a real GroundStateResult (selected-CI solve)."""
    H_bare = build_from_eft(L, DIM, N_B)
    state, res, H_frame, info = initial_frame_state(
        H_bare, A, has_gaussian=True, has_lf=True, core=120, num_runs=8, seed=0)
    r = back_evaluate_frame(H_bare, state, res)          # res is a GroundStateResult
    _, _, e0 = _ground_vector(H_bare, A)
    assert r['E_orig'] >= e0 - 1e-6
    assert r['E_frame'] == res.energy
    assert r['converged']


def test_convergence_is_enforced():
    """A too-small order cap must NOT be silently trusted: converged=False, and strict raises."""
    H = build_from_eft(L, DIM, N_B)
    S = displacement_generator(H)
    Hf = displace_terms(H, 0.3)
    basis, gf, ef = _ground_vector(Hf, A)
    sd = {basis[i]: complex(gf[i]) for i in range(len(basis)) if abs(gf[i]) > 1e-14}
    _, info = exp_generator_apply(S, 0.3, sd, H.N_f, max_order=1, tol=1e-14)
    assert not info['converged'], "order cap 1 must not certify convergence"
    try:
        back_evaluate(H, S, 0.3, sd, max_order=1, tol=1e-14, strict=True)
        assert False, "strict=True should raise on non-convergence"
    except RuntimeError:
        pass


def test_eps_leak_reported_and_shrinks_with_reference_cutoff():
    """eps_leak = 1 - norm_ratio^2 is in [0,1], matches the identity, and does not GROW when
    the map-back reference H uses a larger Fock cutoff (the silent-ceiling-leak fix): a higher
    ceiling captures more of U|psi~>, so the leaked norm can only shrink."""
    H_solve = build_from_eft(L, DIM, N_B)                       # solve at n_b=1 (N_f=2)
    state, res, H_frame, info = initial_frame_state(
        H_solve, A, has_gaussian=True, has_lf=True, core=120, num_runs=8, seed=0)
    r_small = back_evaluate_frame(H_solve, state, res)          # reference N_f=2
    r_big = back_evaluate_frame(build_from_eft(L, DIM, 3), state, res)   # reference N_f=8
    for rr in (r_small, r_big):
        assert 0.0 <= rr['eps_leak'] <= 1.0
        assert abs((1.0 - rr['norm_ratio'] ** 2) - rr['eps_leak']) < 1e-12
    assert r_big['eps_leak'] <= r_small['eps_leak'] + 1e-9, "bigger ceiling should not leak more"


def test_coo_is_isospectral_operator_identity():
    """COO (fermion orbital rotation) is an EXACT operator identity: at full ED the framed
    spectrum equals the bare one, so E_frame is DIRECTLY variational (no back-eval needed) —
    the reason the benchmark routes COO around the map-back. Also confirms back_evaluate_frame
    refuses a COO (R) map-back."""
    from classical.trimci.frame import natural_orbital_terms
    H_bare = build_from_eft(L, DIM, N_B)
    _, _, e0 = _ground_vector(H_bare, A)
    _, _, e_coo = _ground_vector(natural_orbital_terms(H_bare, A), A)
    assert abs(e_coo - e0) < 1e-6, f"COO not isospectral at full ED: {e_coo - e0:+.2e}"
    try:
        back_evaluate_frame(H_bare, {'R': np.eye(2)}, {})
        assert False, "expected NotImplementedError for COO map-back"
    except NotImplementedError:
        pass


def test_kato_temple_brackets_ground_energy():
    """Ritz upper bound (E_orig) + Kato-Temple lower bound bracket the true E_0 on a tiny ED
    system (LF-framed state back-evaluated)."""
    from classical.trimci.back_evaluate import kato_temple_lower
    from classical.trimci.frame import _low_spectrum
    H = build_from_eft(L, DIM, N_B)
    ev = [float(e) for e in _low_spectrum(H, A, k=12)]
    e0 = ev[0]
    above = [e for e in ev if e > e0 + 1e-6]                    # skip degenerate ground multiplet
    e1 = above[0] if above else None
    state, res, H_frame, info = initial_frame_state(
        H, A, has_gaussian=True, has_lf=True, core=120, num_runs=8, seed=0)
    r = back_evaluate_frame(H, state, res)
    assert r['E_orig'] >= e0 - 1e-6, "Ritz upper bound below true E_0"
    if e1 is not None and r['E_orig'] < e1:                     # KT hypothesis beta > E
        kt = kato_temple_lower(r['E_orig'], r['residual'], e1)
        assert kt is not None and kt <= e0 + 1e-6, f"KT lower bound above true E_0 ({kt} > {e0})"


def main():
    for fn in (test_pure_lf_ritz_and_dense_match, test_composed_gaussian_lf_matches_dense,
               test_production_result_interface, test_convergence_is_enforced,
               test_eps_leak_reported_and_shrinks_with_reference_cutoff,
               test_coo_is_isospectral_operator_identity,
               test_kato_temple_brackets_ground_energy):
        fn()
        print(f"  PASS {fn.__name__}")
    # headline demo
    d = back_evaluate_ed(L, DIM, N_B, lam=0.2, A=A, coupling_scale=1.0)
    print(f"demo: E_bare={d['E_bare']:.4f}  E_frame={d['E_frame']:.4f}(shift{d['frame_shift']:+.3f})"
          f"  E_orig={d['E_orig']:.4f}(gap{d['gap_orig']:+.4f})  sparse-vs-dense={d['sparse_vs_dense']:.1e}")
    print("PASS: all back-evaluation tests green.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
