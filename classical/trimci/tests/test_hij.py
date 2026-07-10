"""
Validation tests for the mixed-state H_ij evaluator and TrimCI core.

Strategy: spectra are basis-relabeling invariant, so we cross-check the
*sorted eigenvalues* of our blocks against independent OpenFermion sparse
operators:

  1. Fermion sector  -> openfermion.get_sparse_operator(fermion_part)
     (validates fermion ladder signs — the trickiest part).
  2. Boson sector    -> openfermion.boson_operator_sparse(boson_part, N_f)
     (validates ladder sqrt factors + cutoff).
  3. Full mixed H    -> Hermitian; ground state via dense ED is the truth
     for the TrimCI solver, which (full-space, no trimming) must reproduce
     it, and (trimmed) must approach it from above (variational).

Run: python -m classical.trimci.tests.test_hij
"""

import numpy as np
import openfermion as of

from classical.trimci import (
    build_from_eft, build_dense, enumerate_basis,
    ground_state, ground_state_ensemble, exact_ground_state,
    lanczos_ground_state, write_dump, read_dump, summarize,
)
from classical.trimci.hamiltonian import from_mixed_hamiltonian
from classical.trimci.state import MixedState, fermion_determinants


def _mixed_hamiltonian_object(L, dim, n_b):
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.utils.Config import Config
    params = get_physical_parameters()
    cfg = Config(pion_basis="fock", block_encoder="sparse")
    bundle, _q, _ns = build_eft_hamiltonian(L, dim, n_b, 0.0, params, cfg)
    return bundle.sub_hamiltonians[0].operator


def test_fermion_sector_spectrum(L=1, dim=1, n_b=2, tol=1e-8):
    """Fermion-only block spectrum == OpenFermion get_sparse_operator."""
    H = build_from_eft(L, dim, n_b)
    mh = _mixed_hamiltonian_object(L, dim, n_b)

    # our fermion-only dense over the full fermion Fock space (bos = ()).
    # Exclude the global constant (zero-point) — OpenFermion keeps it in
    # boson_part, not fermion_part.
    ferm_terms = [t for t in H.terms if t.ferm_ops and not t.bos_ops]
    from classical.trimci.hamiltonian import MixedH
    Hf = MixedH(ferm_terms, H.n_ferm_modes, 0, H.N_f)
    basis = [MixedState(occ, ()) for occ in range(1 << H.n_ferm_modes)]
    M = build_dense(Hf, basis)
    ours = np.sort(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))

    ref = of.get_sparse_operator(mh.fermion_part, n_qubits=H.n_ferm_modes).toarray()
    theirs = np.sort(np.linalg.eigvalsh(0.5 * (ref + ref.conj().T)))

    err = np.max(np.abs(ours - theirs))
    assert err < tol, f"fermion spectrum mismatch {err:.2e}"
    print(f"[1] fermion spectrum: max|dE| = {err:.2e}  (n_states={len(basis)})  OK")
    return err


def test_boson_sector_spectrum(L=1, dim=1, n_b=2, tol=1e-8):
    """Boson-only block spectrum == OpenFermion boson_operator_sparse."""
    H = build_from_eft(L, dim, n_b)
    mh = _mixed_hamiltonian_object(L, dim, n_b)
    N_f = H.N_f

    bos_terms = [t for t in H.terms if not t.ferm_ops]
    from classical.trimci.hamiltonian import MixedH
    import itertools
    Hb = MixedH(bos_terms, 0, H.n_bos_modes, N_f)
    basis = [MixedState(0, b)
             for b in itertools.product(range(N_f), repeat=H.n_bos_modes)]
    M = build_dense(Hb, basis)
    ours = np.sort(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))

    ref = of.boson_operator_sparse(mh.boson_part, N_f).toarray()
    theirs = np.sort(np.linalg.eigvalsh(0.5 * (ref + ref.conj().T)))

    err = np.max(np.abs(ours - theirs))
    assert err < tol, f"boson spectrum mismatch {err:.2e}"
    print(f"[2] boson spectrum:   max|dE| = {err:.2e}  (n_states={len(basis)})  OK")
    return err


def test_full_mixed_hermitian_and_solver(L=1, dim=1, n_b=2, A=1, tol=1e-8):
    """Full mixed H is Hermitian; TrimCI (full space) reproduces ED."""
    H = build_from_eft(L, dim, n_b)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A)
    M = build_dense(H, basis)
    herm_err = np.max(np.abs(M - M.conj().T))
    assert herm_err < tol, f"H not Hermitian: {herm_err:.2e}"

    E_ed, _ = exact_ground_state(H, n_elec=A)

    # TrimCI driven to the full sector size must reproduce ED to numerical tol.
    full = len(basis)
    res = ground_state(H, n_elec=A, n_dets=full, n_init=min(8, full),
                       num_groups=3, max_rounds=20, seed=0)
    solver_err = abs(res.energy - E_ed)
    assert solver_err < 1e-6, f"solver vs ED: {solver_err:.2e}"

    # A genuinely trimmed run is variational (E >= E_ed - tol).
    res_trim = ground_state(H, n_elec=A, n_dets=max(4, full // 3),
                            n_init=min(6, full), num_groups=3,
                            max_rounds=12, seed=1)
    assert res_trim.energy >= E_ed - 1e-9, "trimmed energy below ED (non-variational!)"

    print(f"[3] mixed: hermiticity={herm_err:.2e}  E_ED={E_ed:.6f}  "
          f"E_solver(full)={res.energy:.6f} (err {solver_err:.1e})  "
          f"E_trim({res_trim.n_dets}/{full})={res_trim.energy:.6f}  OK")
    return E_ed


def test_dump_roundtrip(L=1, dim=1, n_b=2, A=1, tol=1e-10):
    """write_dump -> read_dump reproduces the same spectrum."""
    import tempfile, os
    H = build_from_eft(L, dim, n_b)
    fd, path = tempfile.mkstemp(suffix=".mixedfci")
    os.close(fd)
    info = write_dump(H, path, n_elec=A)
    H2 = read_dump(path)
    os.unlink(path)

    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A)
    e1 = np.sort(np.linalg.eigvalsh(build_dense(H, basis)))
    basis2 = enumerate_basis(H2.n_ferm_modes, H2.n_bos_modes, H2.N_f, n_elec=A)
    e2 = np.sort(np.linalg.eigvalsh(build_dense(H2, basis2)))
    err = np.max(np.abs(e1 - e2))
    assert err < tol, f"dump round-trip spectrum mismatch {err:.2e}"
    print(f"[4] dump round-trip: {info['n_records']} records, max|dE|={err:.2e}  OK")


def test_ensemble_reaches_ed(L=2, dim=1, n_b=1, A=2):
    """The compact-core claim: ensemble TrimCI reaches ED with few dets.

    A=2/L=2 is a basin-trapping case for single-run TrimCI (seed-dependent);
    the ensemble's min-over-runs escapes it. We assert the ensemble reaches
    the exact GS within tolerance using a small fraction of the sector.
    """
    from math import comb
    H = build_from_eft(L, dim, n_b)
    E_ed, _ = exact_ground_state(H, n_elec=A)
    full = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)

    target = max(8, full // 20)   # ~5% of the sector
    res = ground_state_ensemble(H, n_elec=A, n_runs=8, seed=0,
                                n_dets=target, n_init=8, num_groups=5,
                                max_rounds=30)
    assert res.energy >= E_ed - 1e-7, "ensemble energy below ED (non-variational!)"
    assert abs(res.energy - E_ed) < 1e-5, \
        f"ensemble did not reach ED: dE={res.energy - E_ed:.2e}"
    frac = res.n_dets / full
    print(f"[5] ensemble: E={res.energy:.6f} (dE={res.energy - E_ed:.1e}) "
          f"with {res.n_dets}/{full} dets ({frac:.1%})  OK")


def test_lanczos_matches_dense_ed(tol=1e-8):
    """Sparse Lanczos E0 == dense ED E0 on a system big enough to use eigsh.

    L=2,n_b=1,A=2 has 1792 states (> the dense-fallback threshold), so this
    exercises the real scipy.sparse + eigsh path, not the tiny-N dense branch.
    """
    H = build_from_eft(2, 1, 1)
    E_dense, _ = exact_ground_state(H, n_elec=2)
    E_lan, info = lanczos_ground_state(H, n_elec=2)
    assert info["method"] == "lanczos", "expected the sparse path on 1792 states"
    err = abs(E_lan - E_dense)
    assert err < tol, f"lanczos vs dense ED mismatch {err:.2e}"
    print(f"[6] lanczos: E={E_lan:.6f} vs dense {E_dense:.6f} (dE={err:.1e}), "
          f"N={info['n_states']} nnz={info['nnz']:,}  OK")


def test_lanczos_oom_guard():
    """Lanczos refuses an over-budget sector cleanly (no allocation)."""
    H = build_from_eft(1, 1, 2)
    try:
        lanczos_ground_state(H, n_elec=1, max_states=10)   # 256 > 10
        assert False, "expected MemoryError"
    except MemoryError:
        print("[7] lanczos OOM guard fires before allocation  OK")


def test_nf_convergence_runs():
    """N_f-convergence study runs and returns finite, variational-consistent rows.

    Tiny (single site, N_f<=4 -> <=256 states) so it stays fast; just exercises
    the sweep machinery + the growable Lanczos build end to end.
    """
    from classical.trimci.nf_convergence import nf_convergence
    rows = nf_convergence(L=1, dim=1, A=1, n_b_max=2, verbose=False)
    assert len(rows) == 2, f"expected 2 N_f points, got {len(rows)}"
    assert all(abs(r["E0"]) < 1e6 for r in rows), "energies look unphysical"
    # single-site E0 is N_f-converged (H_WT negligible): the two agree closely.
    assert abs(rows[1]["E0"] - rows[0]["E0"]) < 1e-3, "unexpected N_f jump at L=1"
    print(f"[8] nf_convergence: N_f={rows[0]['N_f']}->{rows[1]['N_f']} "
          f"E0={rows[1]['E0']:.6f} (single-site, converged)  OK")


def test_official_backend():
    """Official C++ TrimCI Davidson reproduces ED on our complex H, and the
    hybrid pipeline (our selection + C++ Davidson) reaches ED. Skipped cleanly
    if the compiled backend isn't installed."""
    from classical.trimci.backend import backend_available
    if not backend_available():
        print("[10] official backend not built — SKIPPED "
              "(build: python -m pip install /path/to/TrimCI)")
        return
    from classical.trimci.backend import (davidson_lowest, backend_diagonalize,
                                          backend_diagonalize_sparse)
    from classical.trimci import ground_state_ensemble, lanczos_ground_state

    # (a) complex Hermitian L=1 toy (all Weinberg-Tomozawa) via C++ Davidson
    H = build_from_eft(1, 1, 2)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=1)
    M = build_dense(H, basis)
    M = 0.5 * (M + M.conj().T)
    E_ed = float(np.linalg.eigvalsh(M)[0])
    E_dav, v = davidson_lowest(M)
    assert abs(E_dav - E_ed) < 1e-8, f"backend Davidson vs ED: {abs(E_dav-E_ed):.2e}"
    assert np.linalg.norm(M @ v - E_dav * v) < 1e-6, "bad eigenvector"

    # (b) hybrid pipeline reaches ED on the L=2 toy
    H2 = build_from_eft(2, 1, 1)
    E_ref, _ = lanczos_ground_state(H2, n_elec=1)
    res = ground_state_ensemble(H2, n_elec=1, n_runs=4, n_dets=120, seed=0,
                                diag_fn=backend_diagonalize)
    assert abs(res.energy - E_ref) < 1e-6, f"hybrid vs ED: {abs(res.energy-E_ref):.2e}"

    # (c) Tier-1 fork: sparse C++ Davidson (skipped if fork not built)
    from classical.trimci.backend import has_sparse_davidson, davidson_lowest_sparse
    import scipy.sparse as sp
    sparse_msg = "dense-only (sparse fork not built)"
    if has_sparse_davidson():
        E_sp, vsp = davidson_lowest_sparse(sp.csr_matrix(M))
        assert abs(E_sp - E_ed) < 1e-8, f"sparse Davidson vs ED: {abs(E_sp-E_ed):.2e}"
        res2 = ground_state_ensemble(H2, n_elec=1, n_runs=4, n_dets=120, seed=0,
                                     diag_fn=backend_diagonalize_sparse)
        assert abs(res2.energy - E_ref) < 1e-6, "sparse hybrid vs ED"
        sparse_msg = f"sparse fork dE={abs(E_sp-E_ed):.1e}, hybrid reaches ED"
    print(f"[10] official backend: dense C++ Davidson dE={abs(E_dav-E_ed):.1e} "
          f"(N={M.shape[0]}), hybrid reaches ED; {sparse_msg}  OK")


def test_mixed_ci_cpp_port():
    """Tier-2 C++ connections port matches the Python reference bit-for-bit.
    Skipped cleanly if the standalone module isn't compiled."""
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    fork = _os.path.join(_os.path.dirname(__file__), "..", "backend_fork")
    fork = _os.path.abspath(fork)
    so = [f for f in _os.listdir(fork) if f.startswith("mixed_ci") and f.endswith(".so")] \
        if _os.path.isdir(fork) else []
    if not so:
        print("[11] Tier-2 C++ connections port not built — SKIPPED "
              "(bash classical/trimci/backend_fork/build_mixed_ci.sh)")
        return
    if fork not in _sys.path:
        _sys.path.insert(0, fork)
    import mixed_ci
    from classical.trimci.hij import connections as pyc

    H = build_from_eft(2, 1, 1)   # full coupling (gradient + H_AV + H_WT)
    terms = [(complex(t.coeff),
              [(int(m), int(a)) for (m, a) in t.ferm_ops],
              [(int(m), int(a)) for (m, a) in t.bos_ops]) for t in H.terms]
    prov = mixed_ci.MixedProvider(terms, H.N_f)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=1)
    max_err = 0.0
    for st in basis:
        py = {(s.ferm, tuple(s.bos)): v for s, v in pyc(H, st).items()}
        cpp = {(int(f), tuple(b)): complex(v)
               for ((f, b), v) in prov.connections(int(st.ferm), list(st.bos))}
        assert set(py) == set(cpp), "neighbor-set mismatch"
        for k in py:
            max_err = max(max_err, abs(py[k] - cpp[k]))
    assert max_err < 1e-12, f"value mismatch {max_err:.2e}"
    print(f"[11] Tier-2 C++ port: {len(basis)} states, neighbor sets identical, "
          f"max|dValue|={max_err:.1e}  OK")


def test_io_roundtrip():
    """Data pipeline: save a run -> load it -> metadata/arrays/transform axis match."""
    import tempfile
    import os as _os
    from classical.trimci import ground_state
    from classical.io import (save_classical_run, load_classical_run,
                              TRANSFORM_BARE)
    H = build_from_eft(1, 1, 2)
    res = ground_state(H, n_elec=1, n_dets=30, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_classical_run(
            res, H, A=1, runtime_s=1.23, method="TrimCI",
            transform=TRANSFORM_BARE, data_root=tmp,
            solver_params={"n_dets": 30, "seed": 0, "backend": "python"},
            convergence=[(10, res.energy + 1.0), (30, res.energy)])
        r = load_classical_run(run_dir)
        assert _os.path.exists(_os.path.join(run_dir, "hamiltonian.mixedfci"))
    m = r["metadata"]
    assert m["method"] == "TrimCI" and m["transform"] == "bare"
    assert abs(m["runtime_s"] - 1.23) < 1e-9
    assert r["ferm"].shape[0] == res.n_dets == r["coeffs"].shape[0]
    assert r["bos"].shape == (res.n_dets, H.n_bos_modes)
    assert abs(r["energy"] - res.energy) < 1e-12
    assert len(m["convergence"]) == 2
    print(f"[13] io round-trip: method={m['method']} transform={m['transform']} "
          f"runtime={m['runtime_s']}s, {r['ferm'].shape[0]} dets saved/loaded  OK")


def test_arbitrary_width_fermion():
    """The C++ fermion mask is arbitrary-width: relabel an 8-mode Hamiltonian's
    fermion indices by +SHIFT (an isospectral permutation that pushes masks into
    high 64-bit words) and check the C++ matrix build matches the arbitrary-
    precision Python `connections` reference exactly, plus expand round-trips the
    wide masks. This is what unblocks L>=3 in 3d (108+ fermion modes).
    Skipped cleanly if the standalone module isn't compiled."""
    import os as _os
    fork = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..",
                                          "backend_fork"))
    so = [f for f in _os.listdir(fork)
          if f.startswith("mixed_ci") and f.endswith(".so")] \
        if _os.path.isdir(fork) else []
    if not so:
        print("[14] arbitrary-width fermion path — SKIPPED (build mixed_ci)")
        return
    import itertools
    import scipy.sparse as sp
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci.backend import (_cpp_provider, _states_to_arrays,
                                          _ferm_words, cpp_expand)
    H = build_from_eft(2, 1, 1)   # 8 fermion modes, full coupling
    bos_cfgs = list(itertools.islice(
        itertools.product(range(H.N_f), repeat=H.n_bos_modes), 12))
    worst = 0.0
    for SHIFT in (60, 130, 300):   # W = 2, 3, 5
        so_ = lambda ops: tuple((m + SHIFT, a) for (m, a) in ops)
        st = [OperatorTerm(t.coeff, so_(t.ferm_ops), t.bos_ops) for t in H.terms]
        Hs = MixedH(st, H.n_ferm_modes + SHIFT, H.n_bos_modes, H.N_f)
        Hs.meta = dict(H.meta); Hs.meta["n_ferm_modes"] = H.n_ferm_modes + SHIFT
        W = _ferm_words(Hs)
        states = [MixedState(1 << p, tuple(b))
                  for p in range(SHIFT, SHIFT + H.n_ferm_modes) for b in bos_cfgs]
        Mpy = build_dense(Hs, states)
        prov = _cpp_provider(Hs)
        ferm, bos = _states_to_arrays(states, W)
        assert ferm.shape == (len(states), W)
        rows, cols, re, im = prov.build_coo(ferm, bos)
        N = len(states)
        Mcpp = sp.csr_matrix((np.asarray(re) + 1j * np.asarray(im),
                              (np.asarray(rows), np.asarray(cols))),
                             shape=(N, N)).toarray()
        err = np.max(np.abs(Mpy - Mcpp))
        worst = max(worst, err)
        assert err < 1e-12, f"shift {SHIFT} (W={W}): build mismatch {err:.2e}"
        pool = cpp_expand(Hs, {s: 1.0 + 0j for s in states[:8]}, pool_factor=5)
        maxbit = max((s.ferm.bit_length() for s in pool), default=0)
        assert maxbit > SHIFT, f"expand lost high bits at shift {SHIFT}"
    print(f"[14] arbitrary-width fermion: W=2,3,5 builds match Python reference "
          f"(max|dH|={worst:.1e}), expand preserves high-word masks  OK")


def test_lf_machinery():
    """Lang-Firsov (transform='LF') Phase-1 machinery: the displacement generator is
    exactly anti-Hermitian, U(λ)=exp(λS) is unitary, λ=0 reproduces COO, and L=1 (no
    inter-site gradient ⇒ no linear coupling) is a clean no-op."""
    from scipy.linalg import expm
    from classical.trimci.lf import (displacement_generator, _ground_vector,
                                     compactness, lf_compactness_scan)
    # L=1: no linear coupling ⇒ empty generator ⇒ LF no-op
    assert len(displacement_generator(build_from_eft(1, 1, 2)).terms) == 0
    # L=2: real generator; check anti-Hermiticity + unitarity + COO reproduction
    H = build_from_eft(2, 1, 1)
    basis, g, E0 = _ground_vector(H, 1)
    S0 = build_dense(displacement_generator(H), basis)
    anti = np.max(np.abs(S0 + S0.conj().T))
    assert anti < 1e-12, f"generator not anti-Hermitian: {anti:.2e}"
    U = expm(0.7 * S0)
    uerr = np.max(np.abs(U.conj().T @ U - np.eye(len(basis))))
    assert uerr < 1e-10, f"U(λ) not unitary: {uerr:.2e}"
    recs = lf_compactness_scan(2, 1, 1, lambdas=[0.5], verbose=False)
    assert recs[0]["lam"] == 0.0 and recs[0]["n999"] == compactness(g)["n999"]
    print(f"[15] Lang-Firsov machinery: L=1 no-op, generator anti-Herm={anti:.0e}, "
          f"U unitary={uerr:.0e}, λ=0≡COO  OK")


def test_frame_gaussian_squeeze():
    """task 33 STEP 0/1: frame infrastructure + Gaussian (Bogoliubov) squeeze.
    (a) identity r=0 is exactly isospectral and leaves the bare path unchanged;
    (b) squeeze_terms == the generator's U†HU (cross-check on a degeneracy-free toy);
    (c) on a quadratic boson sector, squeezing DIAGONALIZES it -> GS compacts to the
        vacuum with the ground energy preserved; (d) on our EFT H the ground energy is
        frame-invariant to cutoff precision. Pure Python + ED — never skipped."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    # (a) STEP 0 identity: r=0 -> exact isospectral; bare path unchanged by the new arg
    H = build_from_eft(1, 1, 3)
    H0 = build_from_eft(1, 1, 3, transform="gaussian", frame_params={"r": 0.0})
    assert frame.isospectral_check(H, H0, 1) < 1e-12
    assert len(build_from_eft(1, 1, 3, transform="bare").terms) == len(H.terms)
    # (b) generator <-> substitution agree (single-mode quadratic toy, no degeneracy)
    toy = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                  OperatorTerm(0.3, (), ((0, 1), (0, 1))),
                  OperatorTerm(0.3, (), ((0, 0), (0, 0)))], 0, 1, 32)
    _, gg = frame.squeeze_state(toy, None, 0.25)
    _, gt, _ = frame._ground_vector(frame.squeeze_terms(toy, 0.25), None)
    assert abs(np.vdot(gg, gt)) > 1 - 1e-6
    # (c) squeezing diagonalizes the quadratic sector -> GS compacts to |0>
    best = None
    for r in np.linspace(-0.6, 0.0, 25):
        c = frame.frame_compactness(toy, frame.squeeze_terms(toy, r), None)
        if best is None or c["frame"]["n999"] < best["frame"]["n999"]:
            best = c
    assert best["bare"]["n999"] > 1 and best["frame"]["n999"] == 1, \
        f"squeeze failed to compact: {best['bare']['n999']}->{best['frame']['n999']}"
    assert abs(best["dE"]) < 1e-6, f"ground energy not preserved: {best['dE']:.1e}"
    # (d) EFT H: ground energy frame-invariant to cutoff precision at small r
    dE = frame.isospectral_check(H, frame.squeeze_terms(H, 0.05), 1, k=1, tol=1e-4)
    # (e) STEP 2: analytic squeeze extracts a non-trivial r* on a coupled system
    #     (L>=2 has the inter-site quadratic coupling), and optimize_squeeze runs
    #     while staying ground-energy isospectral.
    H2 = build_from_eft(2, 1, 1)
    r_seed, _ph = frame.analytic_squeeze(H2)
    assert r_seed.max() > 0.05, f"analytic r* trivial at L=2: {r_seed.max():.2e}"
    opt = frame.optimize_squeeze(H2, 1)
    dE2 = frame.isospectral_check(H2, frame.squeeze_terms(H2, opt["r"], opt["phi"]),
                                  1, k=1, tol=1e-3)
    print(f"[18] frame squeeze: identity exact, generator≡substitution, toy GS compacts "
          f"{best['bare']['n999']}->1 (dE={best['dE']:.0e}), EFT ground iso={dE:.0e}; "
          f"STEP2 analytic r*={r_seed.max():.2f} iso={dE2:.0e}  OK")


def test_frame_multimode_bogoliubov():
    """task 33 STEP 2b: the FULL multi-mode Bogoliubov (also kills the cross-mode
    pair terms b_i†b_j†, i≠j, that the per-mode seed leaves in place).
    (a) with a DIAGONAL (α,β) it reproduces `squeeze_terms` exactly (same machinery);
    (b) the analytic transform is canonical (αα†−ββ†=I, αβ^T=βα^T) and ALGEBRAICALLY
        removes both same- and cross-mode pairs (framed |B|→0) on a cross-coupled toy;
    (c) it is isospectral, the error converging with the Fock cutoff (a truncation
        leak, exact in the limit), and the cross-mode-entangled GS compacts to |0>.
    Pure Python + ED — never skipped."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    # (a) diagonal Bogoliubov == per-mode squeeze (validates the substitution engine)
    H = MixedH([OperatorTerm(1.3, (), ((0, 1), (0, 0))),
                OperatorTerm(0.25, (), ((0, 1), (0, 1))),
                OperatorTerm(0.5, ((0, 1), (0, 0)), ((1, 1),))], 2, 2, 6)
    r = 0.3
    dsq = {(t.ferm_ops, t.bos_ops): t.coeff for t in frame.squeeze_terms(H, r).terms}
    al = np.diag([np.cosh(r)] * 2).astype(complex)
    be = np.diag([np.sinh(r)] * 2).astype(complex)
    dbg = {(t.ferm_ops, t.bos_ops): t.coeff for t in frame.bogoliubov_terms(H, al, be).terms}
    dmax = max(abs(dsq.get(k, 0) - dbg.get(k, 0)) for k in set(dsq) | set(dbg))
    assert dmax < 1e-12, f"diagonal Bogoliubov != squeeze_terms: {dmax:.1e}"
    # cross-coupled 2-mode boson toy: same-mode (ks) AND cross-mode (kc) pairs
    w0, w1, ks, kc = 1.5, 1.2, 0.20, 0.28
    tt = [OperatorTerm(w0, (), ((0, 1), (0, 0))), OperatorTerm(w1, (), ((1, 1), (1, 0))),
          OperatorTerm(ks, (), ((0, 1), (0, 1))), OperatorTerm(ks, (), ((0, 0), (0, 0))),
          OperatorTerm(ks, (), ((1, 1), (1, 1))), OperatorTerm(ks, (), ((1, 0), (1, 0))),
          OperatorTerm(kc, (), ((0, 1), (1, 1))), OperatorTerm(kc, (), ((1, 0), (0, 0)))]
    alc, bec = frame.analytic_bogoliubov(MixedH(tt, 0, 2, None))
    a2, b2 = alc[:2, :2], bec[:2, :2]
    # (b) canonical + pairs algebraically eliminated
    assert np.max(np.abs(a2 @ a2.conj().T - b2 @ b2.conj().T - np.eye(2))) < 1e-10
    assert np.max(np.abs(a2 @ b2.T - b2 @ a2.T)) < 1e-10, "Bogoliubov not canonical (αβ^T≠βα^T)"
    tf16 = frame.bogoliubov_terms(MixedH(tt, 0, 2, 16), alc, bec)
    _, Bres, _ = frame.boson_quadratic_matrices(tf16)
    assert np.max(np.abs(Bres)) < 1e-9, f"pairs not removed: |B|={np.max(np.abs(Bres)):.1e}"
    # (c) isospectral, converging with N_f; cross-entangled GS -> vacuum
    dEs = [frame.isospectral_check(MixedH(tt, 0, 2, nf),
                                   frame.bogoliubov_terms(MixedH(tt, 0, 2, nf), alc, bec),
                                   0, k=6, tol=1e-2) for nf in (8, 16)]
    assert dEs[1] < dEs[0] and dEs[1] < 1e-6, f"not converging in N_f: {dEs}"
    fc = frame.frame_compactness(MixedH(tt, 0, 2, 16), tf16, 0)
    assert fc["bare"]["n999"] > 1 and fc["frame"]["n999"] == 1, \
        f"cross-toy not compacted: {fc['bare']['n999']}->{fc['frame']['n999']}"
    print(f"[19] multimode Bogoliubov: diagonal≡squeeze, canonical, cross+same pairs "
          f"|B|={np.max(np.abs(Bres)):.0e}, iso {dEs[0]:.0e}->{dEs[1]:.0e} (N_f 8->16), "
          f"cross-GS compacts {fc['bare']['n999']}->1 (dE={fc['dE']:.0e})  OK")


def test_frame_lang_firsov():
    """task 33 STEP 3: Lang-Firsov displacement (polaron frame), term-list transform.
    (a) DENSITY Holstein, STATIC fermion sector: the boson substitution b→b+λD is
        EXACT & finite — isospectral to cutoff precision for any λ, and the polaron GS
        compacts to the vacuum; (b) density Holstein WITH fermion hopping: the boson
        substitution alone leaves a defect (undressed hopping), which the Franck-Condon
        dressing (`fc_dress`, order→N_f) drives to zero — validating the FC expansion;
        (c) `transform="LF"` dispatch. Pure Python + ED — never skipped."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    n0 = ((0, 1), (0, 0))
    # (a) static-fermion density Holstein: w b†b + g n0(b†+b) + e0 n0 + e1 n1  (2F, 1B, A=1)
    Hs = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                 OperatorTerm(0.8, n0, ((0, 1),)), OperatorTerm(0.8, n0, ((0, 0),)),
                 OperatorTerm(0.5, n0, ()), OperatorTerm(1.3, ((1, 1), (1, 0)), ())], 2, 1, 24)
    dE_any = max(frame.isospectral_check(Hs, frame.displace_terms(Hs, lam), 1, k=4, tol=1e-6)
                 for lam in (-1.0, 0.4))
    fc = frame.frame_compactness(Hs, frame.displace_terms(Hs, -1.0), 1)   # λ=-1/ω
    assert fc["bare"]["n999"] > 1 and fc["frame"]["n999"] == 1, "polaron not compacted to vacuum"
    assert abs(fc["dE"]) < 1e-9, f"LF ground energy not preserved: {fc['dE']:.1e}"
    # (b) density Holstein WITH hopping: FC dressing convergence in truncation order
    Hh = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))), OperatorTerm(1.2, (), ((1, 1), (1, 0))),
                 OperatorTerm(0.6, n0, ((0, 1),)), OperatorTerm(0.6, n0, ((0, 0),)),
                 OperatorTerm(0.5, ((1, 1), (1, 0)), ((1, 1),)),
                 OperatorTerm(0.5, ((1, 1), (1, 0)), ((1, 0),)),
                 OperatorTerm(0.4, ((0, 1), (1, 0)), ()), OperatorTerm(0.4, ((1, 1), (0, 0)), ()),
                 OperatorTerm(0.5, n0, ()), OperatorTerm(0.9, ((1, 1), (1, 0)), ())], 2, 2, 16)
    lam = [-1.0, -1.0 / 1.2]
    fcmap = {0: (0, 0.6), 1: (1, 0.5)}
    e_bare = frame._low_spectrum(Hh, 1, 4)
    d_lo = np.max(np.abs(e_bare - frame._low_spectrum(frame.displace_terms(Hh, lam, fc_dress=fcmap, order=2), 1, 4)))
    d_hi = np.max(np.abs(e_bare - frame._low_spectrum(frame.displace_terms(Hh, lam, fc_dress=fcmap, order=8), 1, 4)))
    assert d_hi < 1e-5 and d_hi < d_lo, f"FC dressing did not converge: order2={d_lo:.1e} order8={d_hi:.1e}"
    # (c) transform="LF" dispatch reaches displace_terms
    HLF = build_from_eft(1, 1, 3, transform="LF", frame_params={"lambdas": 0.1})
    assert HLF.meta["frame"]["transform"] == "LF"
    print(f"[20] Lang-Firsov: density-Holstein substitution exact (iso={dE_any:.0e}, polaron "
          f"{fc['bare']['n999']}->1), FC dressing converges (order2={d_lo:.0e}->order8={d_hi:.0e}), "
          f"transform='LF' wired  OK")


def test_frame_combined_step4():
    """task 33 STEP 4: the layered squeeze ∘ displace frame + projector-conditioned
    displacement (the recommended frame).
    (a) on a toy with BOTH a quadratic pair (squeeze target) and a density linear
        coupling (LF target), the combined frame is isospectral and compacts the GS to
        the VACUUM — beating either single layer (the layered win);
    (b) the projector-conditioned generator (commuting matter projectors) is an EXACT
        finite frame (state-dependent displacement) — isospectral to machine precision;
    (c) `transform="gaussian+LF"` dispatch. Pure Python + ED — never skipped."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    n0 = ((0, 1), (0, 0))
    # (a) w b†b + k(b†²+b²) + g n0(b†+b) + energies, static fermion sector
    toy = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                  OperatorTerm(0.25, (), ((0, 1), (0, 1))), OperatorTerm(0.25, (), ((0, 0), (0, 0))),
                  OperatorTerm(0.5, n0, ((0, 1),)), OperatorTerm(0.5, n0, ((0, 0),)),
                  OperatorTerm(0.7, n0, ()), OperatorTerm(1.3, ((1, 1), (1, 0)), ())], 2, 1, 24)
    _phi = frame.analytic_squeeze(toy)[1]
    r_opt = frame.optimize_squeeze(toy, 1)["r"]                 # compacting squeeze sign
    lam = -1.0 / frame.boson_quadratic_form(frame.squeeze_terms(toy, r_opt, _phi))[0][0]
    cmp = frame.frame_comparison(toy, 1, r_opt, _phi, lam)
    assert cmp["gaussian+LF"]["n999"] == 1, f"combined did not reach vacuum: {cmp['gaussian+LF']['n999']}"
    assert cmp["gaussian+LF"]["n999"] < min(cmp["gaussian"]["n999"], cmp["LF"]["n999"]), \
        "layered frame not better than each single layer"
    assert abs(cmp["gaussian+LF"]["dE_vs_bare"]) < 1e-9, "combined frame not isospectral"
    # (b) projector-conditioned (commuting occupation projectors) = exact finite frame
    g0, g1 = 0.7, 0.4
    ptoy = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                   OperatorTerm(g0, n0, ((0, 1),)), OperatorTerm(g0, n0, ((0, 0),)),
                   OperatorTerm(g1, ((1, 1), (1, 0)), ((0, 1),)), OperatorTerm(g1, ((1, 1), (1, 0)), ((0, 0),)),
                   OperatorTerm(0.6, n0, ()), OperatorTerm(1.1, ((1, 1), (1, 0)), ())], 2, 1, 20)
    gen = frame.projector_generator([(0, 0, -g0), (0, 1, -g1)])
    dEp = frame.isospectral_check(ptoy, frame.displace_terms(ptoy, 1.0, gen=gen), 1, k=6, tol=1e-6)
    fcp = frame.frame_compactness(ptoy, frame.displace_terms(ptoy, 1.0, gen=gen), 1)
    assert fcp["frame"]["n999"] == 1, "projector-conditioned LF did not compact"
    # (c) transform="gaussian+LF" dispatch
    HC = build_from_eft(1, 1, 3, transform="gaussian+LF", frame_params={"r": 0.05, "lambdas": 0.05})
    assert HC.meta["frame"]["transform"] == "gaussian+LF"
    print(f"[21] STEP4 combined: gaussian+LF -> vacuum (bare {cmp['bare']['n999']}, gauss "
          f"{cmp['gaussian']['n999']}, LF {cmp['LF']['n999']}, combined 1); projector-cond "
          f"exact (iso={dEp:.0e}, n999 {fcp['bare']['n999']}->1); dispatch wired  OK")


def test_frame_orbital_rotation():
    """task 33 STEP 5: fermion orbital rotation (U(n) CS) = Core-Optimized Orbitals.
    (a) a one-body fermion ground state (a single Slater determinant that SPREADS in
        the bare basis) collapses to ONE determinant in its natural orbitals, isospectral;
    (b) a generator-based (antisymmetric κ) rotation is isospectral;
    (c) COO compacts the fermion factor of a MIXED fermion-boson system, isospectral;
    (d) `transform="COO"` dispatch (explicit κ + natural-orbital). Pure Python + ED."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    # (a) one-body H = Σ h_pq a_p†a_q (n=4, A=2): GS is 1 determinant in h's eigenbasis
    h = np.array([[0.2, 0.5, 0.1, 0.0], [0.5, 0.9, 0.3, 0.2],
                  [0.1, 0.3, 1.4, 0.6], [0.0, 0.2, 0.6, 2.1]])
    Hf = MixedH([OperatorTerm(h[p, q], ((p, 1), (q, 0)), ())
                 for p in range(4) for q in range(4)], 4, 0, 2)
    R, occ = frame.natural_orbitals(Hf, 2)
    Hc = frame.rotate_orbitals_terms(Hf, R=R)
    dE_a = frame.isospectral_check(Hf, Hc, 2, tol=1e-9)
    fc = frame.frame_compactness(Hf, Hc, 2)
    assert fc["bare"]["n999"] > 1 and fc["frame"]["n999"] == 1, \
        f"COO failed to compact one-body GS: {fc['bare']['n999']}->{fc['frame']['n999']}"
    # (b) generator-based rotation is isospectral
    kap = np.array([[0, 0.3, -0.2, 0.1], [-0.3, 0, 0.4, 0.0],
                    [0.2, -0.4, 0, 0.25], [-0.1, 0.0, -0.25, 0]])
    dE_b = frame.isospectral_check(Hf, frame.rotate_orbitals_terms(Hf, kappa=kap), 2, tol=1e-9)
    # (c) mixed fermion-boson: COO compacts the fermion factor, isospectral
    hm = np.array([[0.3, 0.4, 0.1], [0.4, 1.0, 0.5], [0.1, 0.5, 1.8]])
    Hm = MixedH([OperatorTerm(hm[p, q], ((p, 1), (q, 0)), ()) for p in range(3) for q in range(3)]
                + [OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                   OperatorTerm(0.3, ((0, 1), (0, 0)), ((0, 1),)),
                   OperatorTerm(0.3, ((0, 1), (0, 0)), ((0, 0),))], 3, 1, 6)
    Hmc = frame.natural_orbital_terms(Hm, 2)
    dE_c = frame.isospectral_check(Hm, Hmc, 2, k=6, tol=1e-6)
    fcm = frame.frame_compactness(Hm, Hmc, 2)
    assert fcm["frame"]["n999"] < fcm["bare"]["n999"], "COO did not compact the mixed system"
    # (d) transform="COO" dispatch (explicit κ works at any size)
    kbig = 0.05 * np.ones((8, 8)); kbig = kbig - kbig.T
    HD = build_from_eft(2, 1, 2, transform="COO", frame_params={"kappa": kbig})
    assert HD.meta["frame"]["transform"] == "COO"
    print(f"[22] orbital rotation (COO): one-body GS compacts {fc['bare']['n999']}->1 "
          f"(occ={np.round(occ, 2)}, iso={dE_a:.0e}); κ-rotation iso={dE_b:.0e}; mixed f-b "
          f"{fcm['bare']['n999']}->{fcm['frame']['n999']} (iso={dE_c:.0e}); COO wired  OK")


def test_frame_non_gaussianity():
    """task 33 STEP 6 gate: the non-Gaussianity diagnostic (decide whether the heavy
    non-Gaussian layer is warranted). Validates BOTH the ED measure (validation-only)
    and the TrimCI-CORE measure (the scalable go/no-go), against analytic references:
    vacuum/squeezed → δ≈0 (Gaussian), Fock |1⟩ → δ=2 bits (non-Gaussian).
    The go/no-go itself is the TrimCI L=2/L=3 d=3 measurement (task 33: NO-GO)."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    a = frame._boson_ladder_matrices(1, 16)

    def dED(rho):
        _, s = frame._covariance(rho, a)
        w = np.clip(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real, 0, None)
        w = w[w > 1e-14]
        return frame._gaussian_entropy(s) + float(np.sum(w * np.log2(w)))
    r0 = np.zeros((16, 16), complex); r0[0, 0] = 1                 # vacuum
    r1 = np.zeros((16, 16), complex); r1[1, 1] = 1                 # Fock |1>
    assert abs(dED(r0)) < 1e-9, f"vacuum not Gaussian: {dED(r0):.1e}"
    assert abs(dED(r1) - 2.0) < 1e-6, f"Fock|1> non-Gaussianity != 2 bits: {dED(r1):.4f}"
    # squeezed vacuum (quadratic-boson-H ground state) is Gaussian
    sq = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                 OperatorTerm(0.3, (), ((0, 1), (0, 1))), OperatorTerm(0.3, (), ((0, 0), (0, 0)))],
                0, 1, 16)
    assert frame.non_gaussianity(sq, None)["delta"] < 1e-4
    # core-based (scalable) path: synthetic 1-state cores, Fock|1> -> δ=2, vacuum -> 0
    ferm = np.array([[1]], dtype=np.uint64)
    ng1 = frame.core_non_gaussianity([1.0], ferm, np.array([[1]]), 16)
    ng0 = frame.core_non_gaussianity([1.0], ferm, np.array([[0]]), 16)
    assert abs(ng1["max"] - 2.0) < 1e-6 and ng0["max"] < 1e-9
    print(f"[23] non-Gaussianity gate: ED vacuum=0 Fock|1>={dED(r1):.2f} squeezed~0; "
          f"core-based Fock|1>={ng1['max']:.2f} vac={ng0['max']:.0e}  OK")


def test_frame_qpe_payoff():
    """task 33 STEP 7: quantum-side payoff of the frame as a QPE warm start.
    (a) warm-start overlap p₀ = dominant |c|² of a core; (b) mean boson number ⟨n⟩;
    (c) state-prep T-count sums the enabled layers and is TINY vs the walk; (d) the
    combined payoff — a p₀ gain gives <1 repetition factor and prep ≪ walk. Pure Python."""
    from classical.trimci import frame_qpe as fq
    # (a) p0 = weight on the dominant determinant
    assert abs(fq.warmstart_overlap([0.8, 0.6]) - 0.64) < 1e-12          # 0.64/(0.64+0.36)
    assert abs(fq.warmstart_overlap([1.0]) - 1.0) < 1e-12                 # single determinant
    # (b) mean boson number: weighted total occupation
    assert abs(fq.mean_boson_number([1.0, 1.0], np.array([[0, 0], [1, 2]])) - 1.5) < 1e-12
    # (c) state-prep sums enabled layers; adding orbital increases it; all one-time
    sp = fq.stateprep_tcount(24, 32, 3, squeeze=True, displace=True, orbital=False)
    sp_orb = fq.stateprep_tcount(24, 32, 3, squeeze=True, displace=True, orbital=True)
    assert sp["T_orbital"] == 0 and sp_orb["T_orbital"] > 0
    assert abs(sp["T_total"] - (sp["T_squeeze"] + sp["T_displace"])) < 1e-6
    # (d) payoff: a p0 gain -> fewer repetitions; prep negligible vs walk
    pay = fq.qpe_payoff(p0_bare=0.5, p0_frame=0.667, mean_n_bare=1.05, mean_n_frame=0.48,
                        t_step=1e6, physical_lambda=6.5e5, n_bos=81, n_ferm=108, n_b=3)
    assert pay["p0_gain"] > 1 and pay["repetition_factor"] < 1
    assert abs(pay["qpe_T_ratio"] - pay["repetition_factor"]) < 1e-9
    assert pay["prep_vs_walk"] < 1e-6, "state-prep should be negligible vs the walk"
    print(f"[24] QPE payoff: p0-overlap {fq.warmstart_overlap([0.8,0.6]):.2f}; state-prep "
          f"{sp['T_total']:.1e} T; payoff p0_gain={pay['p0_gain']:.2f}x -> "
          f"{pay['repetition_factor']:.2f}x reps, prep/walk={pay['prep_vs_walk']:.0e}  OK")


def test_frame_projector_lf_variational():
    """task 33 STEP 4 (projector-conditioned LF, variational amplitude) — go/no-go.
    (a) analytic_displacement reads the polaron seed λ=−g/ω off the diagonal density
        coupling; (b) with a SINGLE boson mode + uniform λ, leading-order (undressed)
        projector-LF is exactly isospectral even with hopping (the generator depends only
        on total N, which hopping conserves); (c) when hopping CROSSES boson modes (our
        real vertex), leading-order is NOT isospectral, and only the multi-mode
        Franck-Condon dressing (`fc_dress_from_entries`) restores it — the "then full"
        escalation (which is term-explosive on the real system). Pure Python + ED."""
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci import frame
    def nh(p): return ((p, 1), (p, 0))
    # (a) seed: H = ω b†b + g n0(b†+b) -> λ = -g/ω
    w, g = 2.0, 0.6
    Ha = MixedH([OperatorTerm(w, (), ((0, 1), (0, 0))),
                 OperatorTerm(g, nh(0), ((0, 1),)), OperatorTerm(g, nh(0), ((0, 0),)),
                 OperatorTerm(0.9, ((1, 1), (1, 0)), ())], 2, 1, 8)
    ent, _, info = frame.analytic_displacement(Ha)
    assert info["n_seeded"] == 1 and abs(ent[0][2] - (-g / w)) < 1e-12, f"bad seed {ent}"
    # (b) single boson mode, uniform λ, WITH hopping: leading-order exact (N conserved)
    wg = 0.3
    Hb = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))),
                 OperatorTerm(wg, nh(0), ((0, 1),)), OperatorTerm(wg, nh(0), ((0, 0),)),
                 OperatorTerm(wg, nh(1), ((0, 1),)), OperatorTerm(wg, nh(1), ((0, 0),)),
                 OperatorTerm(0.5, ((0, 1), (1, 0)), ()),
                 OperatorTerm(0.5, ((1, 1), (0, 0)), ())], 2, 1, 8)
    eb = frame.analytic_displacement(Hb)[0]
    dE_single = frame.isospectral_check(
        Hb, frame.displace_terms(Hb, 1.0, gen=frame.projector_generator(eb)), 1, k=4, tol=1e-9)
    # (c) hopping crosses TWO boson modes: leading-order breaks; multi-mode dressing fixes
    gg = 0.3
    Hc = MixedH([OperatorTerm(1.0, (), ((0, 1), (0, 0))), OperatorTerm(1.0, (), ((1, 1), (1, 0))),
                 OperatorTerm(+gg, nh(0), ((0, 1),)), OperatorTerm(+gg, nh(0), ((0, 0),)),
                 OperatorTerm(-gg, nh(0), ((1, 1),)), OperatorTerm(-gg, nh(0), ((1, 0),)),
                 OperatorTerm(-gg, nh(1), ((0, 1),)), OperatorTerm(-gg, nh(1), ((0, 0),)),
                 OperatorTerm(+gg, nh(1), ((1, 1),)), OperatorTerm(+gg, nh(1), ((1, 0),)),
                 OperatorTerm(0.4, ((0, 1), (1, 0)), ()),
                 OperatorTerm(0.4, ((1, 1), (0, 0)), ())], 2, 2, 6)
    ec = frame.analytic_displacement(Hc)[0]
    genc = frame.projector_generator(ec)
    e0 = frame._low_spectrum(Hc, 1, 6)
    dE_lead = float(np.max(np.abs(
        frame._low_spectrum(frame.displace_terms(Hc, 1.0, gen=genc, fc_dress=None), 1, 6) - e0)))
    dE_full = float(np.max(np.abs(
        frame._low_spectrum(frame.displace_terms(
            Hc, 1.0, gen=genc, fc_dress=frame.fc_dress_from_entries(ec), order=4), 1, 6) - e0)))
    assert dE_lead > 1e-2, f"expected leading-order to break, got {dE_lead:.1e}"
    assert dE_full < 1e-4, f"multi-mode dressing did not restore isospectrality: {dE_full:.1e}"
    print(f"[25] STEP4 projector-LF: seed λ=−g/ω OK; single-mode leading-order exact "
          f"(iso={dE_single:.0e}); cross-mode hopping breaks leading-order (dE={dE_lead:.2f}) "
          f"-> multi-mode FC dress restores (dE={dE_full:.0e})  OK")


def test_frame_coo_from_core(tol=1e-8):
    """task 33 STEP 5 (scalable COO seed): the ED-free 1-RDM read off a selected-CI core.
    (a) one_rdm_from_core fed the SAME ED ground vector reproduces the ED one_rdm to
        machine precision (sign-consistent via the solver's _apply_fermion_ops);
    (b) natural_orbitals_from_core yields a unitary R whose rotation is isospectral;
    (c) natural_orbitals_smallcutoff + transform='COO' natural_smallcutoff dispatch is
        isospectral. Pure Python + ED."""
    from classical.trimci import frame
    from classical.trimci.lf import _ground_vector
    Hs = build_from_eft(1, 3, 2); A = 2
    basis, g, _ = _ground_vector(Hs, A)
    ferm_arr = np.array([[s.ferm] for s in basis], dtype=np.uint64)
    bos_arr = np.array([list(s.bos) for s in basis], dtype=np.uint16)
    coeffs = np.asarray(g, dtype=complex)
    # (a) core 1-RDM (fed the ED vector) == ED 1-RDM
    g_ed = frame.one_rdm(Hs, A)
    g_core = frame.one_rdm_from_core(Hs, coeffs, ferm_arr, bos_arr)
    dgamma = float(np.max(np.abs(0.5 * (g_ed + g_ed.conj().T)
                                 - 0.5 * (g_core + g_core.conj().T))))
    assert dgamma < tol, f"core 1-RDM != ED 1-RDM: {dgamma:.2e}"
    # (b) natural_orbitals_from_core -> unitary R, isospectral rotation
    R, _ = frame.natural_orbitals_from_core(Hs, coeffs, ferm_arr, bos_arr)
    assert np.max(np.abs(R.conj().T @ R - np.eye(R.shape[0]))) < 1e-10, "R not unitary"
    dE_rot = frame.isospectral_check(Hs, frame.rotate_orbitals_terms(Hs, R=R), A, k=6, tol=1e-6)
    # (c) small-cutoff seed + dispatch
    R2, _ = frame.natural_orbitals_smallcutoff(1, 3, A, seed_n_b=1)
    assert np.max(np.abs(R2.conj().T @ R2 - np.eye(R2.shape[0]))) < 1e-10, "smallcutoff R not unitary"
    HC = build_from_eft(1, 3, 2, transform="COO",
                        frame_params={"natural_smallcutoff": True, "n_elec": A, "seed_n_b": 1})
    dE_disp = frame.isospectral_check(Hs, HC, A, k=6, tol=1e-6)
    print(f"[26] STEP5 scalable COO: core 1-RDM == ED (d={dgamma:.0e}); "
          f"natural_orbitals_from_core R unitary & isospectral (iso={dE_rot:.0e}); "
          f"small-cutoff seed + dispatch isospectral (iso={dE_disp:.0e})  OK")


def test_frame_workflow():
    """TrimCI-aligned frame WORKFLOW (frame_workflow.py; TrimCI_skill.py three-phase method).
    (a) probe_frame = Phase-0 stochastic best-of-N at a small core (best ≤ mean);
    (b) coo_orbopt = iterative COO orbopt loop — accepted cycles are energy NON-increasing,
        the net rotation R is unitary, and the framed H is ISOSPECTRAL (correctness);
    (c) three_phase_run returns a Phase-0 frame + Phase-2 growth curve.
    Requires the C++ backend (drives the solver) — skipped if absent."""
    from classical.trimci.backend import cpp_available
    if not cpp_available():
        print("[27] frame workflow — SKIPPED (build mixed_ci + sparse fork)")
        return
    from classical.trimci import frame, frame_workflow as fw
    Hs = build_from_eft(1, 3, 2); A = 2                    # small enumerable mixed system
    # (a) probe_frame
    p = fw.probe_frame(Hs, A, n_probe=60, num_runs=8, seed=0)
    assert p["best"] <= p["mean"] + 1e-9 and p["num_runs"] == 8, f"bad probe {p}"
    # (b) coo_orbopt: non-increasing on accepted cycles, R unitary, framed H isospectral
    oo = fw.coo_orbopt(Hs, A, core=120, num_runs=8, cycles=3, seed=0)
    acc = [h["energy"] for h in oo["history"] if h.get("accepted")]
    assert all(acc[i + 1] <= acc[i] + 1e-6 for i in range(len(acc) - 1)), \
        f"orbopt energy not monotone: {acc}"
    R = oo["R"]
    assert np.max(np.abs(R.conj().T @ R - np.eye(R.shape[0]))) < 1e-9, "net R not unitary"
    dE = frame.isospectral_check(Hs, oo["H_frame"], A, k=6, tol=1e-6)
    # (c) three_phase_run structure
    tp = fw.three_phase_run(Hs, A, frame_spec="squeeze", phase2_cores=(80, 160),
                            phase2_runs=2, seed=0)
    assert tp["phase2"] and "energy" in tp and tp["frame_spec"] == "squeeze"
    print(f"[27] frame workflow: probe best={p['best']:.3f}≤mean={p['mean']:.3f}; "
          f"coo_orbopt {oo['cycles_run']}cyc monotone, R unitary, framed iso={dE:.0e}; "
          f"3-phase Phase2 {len(tp['phase2'])} rungs  OK")


def test_cpp_full_path():
    """Full C++ hot path (build_coo + expand + sparse Davidson) reaches ED.
    Skipped if the standalone mixed_ci module or the sparse-Davidson fork is absent."""
    from classical.trimci.backend import cpp_available
    if not cpp_available():
        print("[12] full C++ path unavailable — SKIPPED (build mixed_ci + sparse fork)")
        return
    from classical.trimci.backend import cpp_ground_state_ensemble
    from classical.trimci import lanczos_ground_state
    H = build_from_eft(2, 1, 2)   # 32768-state sector, full coupling
    E_ref, _ = lanczos_ground_state(H, n_elec=1)
    res = cpp_ground_state_ensemble(H, n_elec=1, n_runs=4, n_dets=400, seed=0)
    dE = res.energy - E_ref
    assert dE >= -1e-7, "non-variational (below ED)"
    assert abs(dE) < 1e-3, f"full C++ path far from ED: dE={dE:.2e}"
    print(f"[12] full C++ path: E={res.energy:.6f} vs ED {E_ref:.6f} "
          f"(dE={dE:.1e}, {res.n_dets} dets)  OK")


def test_near_vacuum_init():
    """random_core seeds near vacuum: low mean occupation, monotone weights."""
    import numpy as np
    from classical.trimci.graph import random_core, boson_occupation_weights

    # weights strictly decreasing (higher occupation => lower probability)
    w = boson_occupation_weights(8, 0.5)
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1)), \
        "boson weights not monotone decreasing"

    # near-vacuum init hugs low occupation; compare against a ~uniform draw
    # (large mean_occ -> p~1 -> ~uniform) at the same modest n_init.
    H = build_from_eft(1, 1, 3)   # N_f=8, 3 boson modes (uniform mean = 3.5)
    core_nv = random_core(H, 1, 24, np.random.default_rng(0), boson_init_mean=0.5)
    core_un = random_core(H, 1, 24, np.random.default_rng(0), boson_init_mean=1e6)
    mean_nv = np.mean([n for s in core_nv for n in s.bos])
    mean_un = np.mean([n for s in core_un for n in s.bos])
    assert mean_nv < mean_un, f"near-vacuum ({mean_nv:.2f}) not below uniform ({mean_un:.2f})"
    assert mean_nv < 1.5, f"init not near-vacuum: mean occ {mean_nv:.2f}"
    assert any(s.total_bosons() == 0 for s in core_nv), "no boson-vacuum anchor"
    print(f"[9] near-vacuum init: mean occ {mean_nv:.2f} vs uniform {mean_un:.2f} "
          f"(N_f=8)  OK")


def test_matfree_diagonalize():
    """cpp_diagonalize_matfree (C++ CSC + eigsh) matches sparse Davidson energy.

    Runs the matfree path on a state set drawn from the L=2 dim=1 system (32768
    states, well past the _MATFREE_N threshold when including pool/local sets).
    Also tests SubspaceContext.build_context + .matvec directly.  Skipped if the
    C++ backend is unavailable."""
    from classical.trimci.backend import cpp_available
    if not cpp_available():
        print("[16] matfree path unavailable — SKIPPED (build mixed_ci + sparse fork)")
        return

    from classical.trimci.backend import (
        cpp_diagonalize, cpp_diagonalize_matfree, _cpp_provider,
        _states_to_arrays, _ferm_words, _MATFREE_N,
    )
    from classical.trimci import lanczos_ground_state
    from classical.trimci.state import enumerate_basis

    H = build_from_eft(2, 1, 2)   # 32768 states
    E_ref, _ = lanczos_ground_state(H, n_elec=1)

    # Use a fixed state set larger than _MATFREE_N to exercise the matfree branch.
    # enumerate_basis is capped; use a slice of the full basis instead.
    all_states = list(enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f,
                                      n_elec=1))
    # Pick _MATFREE_N + 500 states so matfree is always triggered.
    n_test = min(_MATFREE_N + 500, len(all_states))
    import random; random.seed(7)
    test_states = random.sample(all_states, n_test)

    # (a) sparse Davidson energy on the same states
    E_sparse, _ = cpp_diagonalize(H, test_states)

    # (b) matfree energy — must agree to 1e-8
    E_mf, _ = cpp_diagonalize_matfree(H, test_states)
    delta = abs(E_mf - E_sparse)
    assert delta < 1e-6, f"matfree vs sparse Davidson disagreement: {delta:.2e}"

    # (c) SubspaceContext.matvec agrees with the sparse Davidson matvec
    prov = _cpp_provider(H)
    ferm, bos = _states_to_arrays(test_states, _ferm_words(H))
    ctx = prov.build_context(ferm, bos)
    assert ctx.size() == n_test
    assert ctx.nnz() > 0

    # Build the sparse result via np (build_coo) and compare one matvec output.
    import numpy as np
    v = np.random.default_rng(42).random(n_test) + 1j * np.random.default_rng(43).random(n_test)
    vr = np.ascontiguousarray(v.real)
    vi = np.ascontiguousarray(v.imag)
    or_, oi_ = ctx.matvec(vr, vi)
    Hv_mf = np.asarray(or_) + 1j * np.asarray(oi_)

    # Reference: build complex COO and compute H@v directly.
    rows, cols, re, im = prov.build_coo(ferm, bos)
    import scipy.sparse as sp
    Hs = sp.csr_matrix((np.asarray(re) + 1j * np.asarray(im),
                        (np.asarray(rows), np.asarray(cols))),
                       shape=(n_test, n_test), dtype=complex)
    Hv_ref = Hs @ v
    max_diff = float(np.abs(Hv_mf - Hv_ref).max())
    assert max_diff < 1e-10, f"SubspaceContext.matvec vs build_coo mismatch: {max_diff:.2e}"

    print(f"[16] matfree: E_sparse={E_sparse:.6f} E_mf={E_mf:.6f} "
          f"dE={delta:.1e}; matvec max_diff={max_diff:.1e}; "
          f"N={n_test} nnz={ctx.nnz()}  OK")


def test_ground_state_arrays():
    """Tier-2 array-native TrimCI matches ED and the object path, and carries a
    compact-array result (no MixedState materialization). Skipped if the C++
    backend is unavailable."""
    from classical.trimci.backend import cpp_available
    if not cpp_available():
        print("[17] array-native path unavailable — SKIPPED (build mixed_ci + sparse fork)")
        return
    import numpy as np
    from classical.trimci.backend import (cpp_ground_state_ensemble,
                                          cpp_ground_state_ensemble_arrays)
    from classical.trimci import lanczos_ground_state

    # (a) ED-reachable system: array path must be variational and hit ED.
    H = build_from_eft(2, 1, 2)   # 32768-state sector
    E_ref, _ = lanczos_ground_state(H, n_elec=1)
    res_arr = cpp_ground_state_ensemble_arrays(H, n_elec=1, n_runs=4,
                                               n_dets=400, seed=0)
    dE_ed = res_arr.energy - E_ref
    assert dE_ed >= -1e-7, "array path non-variational (below ED)"
    assert abs(dE_ed) < 1e-3, f"array path far from ED: dE={dE_ed:.2e}"

    # (b) array result carries compact arrays + empty states list (io reads arrays).
    assert res_arr.ferm_arr is not None and res_arr.bos_arr is not None
    assert res_arr.ferm_arr.shape[0] == res_arr.n_dets
    assert res_arr.bos_arr.shape[0] == res_arr.n_dets
    assert np.asarray(res_arr.coeffs).shape[0] == res_arr.n_dets
    assert len(res_arr.states) == 0, "array path should not materialize MixedStates"

    # (c) array vs object energy agreement on the same seed.
    res_obj = cpp_ground_state_ensemble(H, n_elec=1, n_runs=4, n_dets=400, seed=0)
    assert abs(res_arr.energy - res_obj.energy) < 1e-6, \
        f"array vs object mismatch: {res_arr.energy - res_obj.energy:.2e}"

    # (d) io round-trips the compact-array result (uses ferm_arr/bos_arr directly).
    import tempfile
    from classical.io import save_classical_run, load_classical_run, TRANSFORM_BARE
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    with tempfile.TemporaryDirectory() as d:
        rd = save_classical_run(res_arr, H, 1, runtime_s=1.0, method="TrimCI",
                                transform=TRANSFORM_BARE,
                                params=get_physical_parameters(), data_root=d,
                                convergence=[(400, res_arr.energy)])
        loaded = load_classical_run(rd)
        assert abs(loaded["energy"] - res_arr.energy) < 1e-9
        assert loaded["bos"].shape[0] == res_arr.n_dets

    print(f"[17] array-native: E={res_arr.energy:.6f} vs ED {E_ref:.6f} "
          f"(dE={dE_ed:.1e}); vs object {res_arr.energy - res_obj.energy:+.1e}; "
          f"{res_arr.n_dets} dets, compact-array result + io round-trip  OK")


def test_nf_decoupling():
    """N_f decoupled from 2**n_b (a classical freedom — the boson register is an
    occupation-number integer, so the power-of-two box is a quantum-only constraint).
    (a) explicit N_f=4 reproduces the n_b=2 default bit-for-bit (terms + energy);
    (b) a non-power-of-two N_f (=3, =5) builds, sets H.N_f, and diagonalizes;
    (c) enlarging N_f is monotone at L=1 (on-site only ⇒ flat, but never rising).
    Pure Python + ED — never skipped."""
    Ha = build_from_eft(1, 1, 2)                 # default N_f = 2**2 = 4
    Hb = build_from_eft(1, 1, 2, N_f=4)          # explicit N_f = 4
    assert Ha.N_f == Hb.N_f == 4 and len(Ha.terms) == len(Hb.terms)
    Ea, _ = exact_ground_state(Ha, n_elec=1)
    Eb, _ = exact_ground_state(Hb, n_elec=1)
    assert abs(Ea - Eb) < 1e-12, f"explicit N_f=4 != n_b=2 default: {abs(Ea-Eb):.1e}"
    prev = None
    for N_f in (2, 3, 4, 5, 6):
        H = build_from_eft(1, 1, 4, N_f=N_f)     # n_b arg nominal; N_f overrides
        assert H.N_f == N_f and H.meta["N_f"] == N_f
        E0, _ = exact_ground_state(H, n_elec=1)
        if prev is not None:
            assert E0 <= prev + 1e-9, f"energy rose at N_f={N_f} (non-variational within fixed N_f)"
        prev = E0
    print(f"[28] N_f decoupling: explicit N_f=4≡n_b=2 (|dE|={abs(Ea-Eb):.0e}), "
          f"non-power-of-2 N_f∈{{3,5}} build+solve, monotone at L=1  OK")


def test_pt2_epstein_nesbet():
    """Epstein-Nesbet PT2 (`pt2.py`) over the mixed selected-CI solution.
    (a) re-diagonalization makes (E_var,|psi>) a self-consistent eigenpair of V
        (Rayleigh gap ~ 0) — the precondition without which PT2 overshoots wildly;
    (b) on an UNconverged core, E_var+PT2 is closer to the exact Lanczos energy than
        E_var, and the correction is negative (external states lie above E_var);
    (c) at (near-)convergence the correction collapses to ~0.
    Pure Python + Lanczos truth — never skipped."""
    from classical.trimci.pt2 import epstein_nesbet_pt2, pt2_from_result
    L, dim, A, N_f = 2, 1, 2, 2
    H = build_from_eft(L, dim, 1, N_f=N_f)
    E_exact, _ = lanczos_ground_state(H, n_elec=A)
    # unconverged core: PT2 must recover most of the correlation, without overshoot
    res = ground_state_ensemble(H, n_elec=A, n_runs=4, n_dets=30, seed=0)
    r = pt2_from_result(H, res)
    assert abs(r["E_var_rayleigh"] - r["E_var"]) < 1e-9, \
        f"eigenpair not self-consistent (ray gap {abs(r['E_var_rayleigh']-r['E_var']):.1e})"
    assert r["dE_pt2"] <= 1e-9, f"PT2 correction not <= 0: {r['dE_pt2']:.4f}"
    err_var = abs(r["E_var"] - E_exact)
    err_pt2 = abs(r["E_pt2"] - E_exact)
    assert err_var > 1e-3, "core unexpectedly already exact — pick a smaller core"
    assert err_pt2 < err_var, f"PT2 did not improve: {err_var:.4f} -> {err_pt2:.4f}"
    recovered = 1.0 - err_pt2 / err_var
    # near-converged core: correction collapses
    resb = ground_state_ensemble(H, n_elec=A, n_runs=4, n_dets=200, seed=0)
    rb = pt2_from_result(H, resb)
    assert abs(rb["dE_pt2"]) < 0.05, f"PT2 not ~0 at convergence: {rb['dE_pt2']:.4f}"
    # trust_coeffs=True with a NON-eigenvector must NOT be used silently: sanity that
    # the self-consistent path is the default (coeffs re-solved from V)
    print(f"[29] EN-PT2: self-consistent (ray gap<1e-9), recovers {recovered:.0%} of "
          f"correlation at N=30 (err {err_var:.4f}->{err_pt2:.4f}), ~0 at N=200  OK")


def test_energy_extrapolation():
    """Honest E_infinity extrapolation + reporting (`extrapolation.py`).
    (a) the SHCI/PT2 extrapolation over an independent ladder lands within a keV-scale
        tolerance of the exact Lanczos energy (extrapolation validated against truth);
    (b) the power-law fit refuses (<4 rungs) rather than emitting a bogus number;
    (c) `report_energies` returns the three headline numbers + the exact cross-check.
    Pure Python + Lanczos truth — never skipped."""
    from classical.trimci.extrapolation import (fit_einf_power, fit_einf_pt2,
                                                report_energies)
    from classical.trimci.pt2 import pt2_from_result
    L, dim, A, N_f = 2, 1, 2, 2
    H = build_from_eft(L, dim, 1, N_f=N_f)
    E_exact, _ = lanczos_ground_state(H, n_elec=A)
    rungs = []
    for c in (20, 40, 80, 160):
        res = ground_state_ensemble(H, n_elec=A, n_runs=4, n_dets=c, seed=0)
        pr = pt2_from_result(H, res)
        rungs.append({"core": c, "E_var": pr["E_var"], "dE_pt2": pr["dE_pt2"]})
    # (a) SHCI/PT2 extrapolation vs exact
    px = fit_einf_pt2([r["E_var"] for r in rungs], [r["dE_pt2"] for r in rungs])
    assert px["ok"] and abs(px["E_inf"] - E_exact) < 0.05, \
        f"PT2 extrapolation off exact: {px['E_inf']:.4f} vs {E_exact:.4f}"
    # (b) power-law guard fires on too-few rungs
    g = fit_einf_power([r["core"] for r in rungs[:3]], [r["E_var"] for r in rungs[:3]])
    assert not g["ok"] and "rungs" in g["reason"], "power-law guard did not fire"
    # power-law with the full ladder should fit
    gp = fit_einf_power([r["core"] for r in rungs], [r["E_var"] for r in rungs])
    assert gp["ok"] and gp["sigma"] is not None
    # (c) the report bundles the three numbers + exact cross-check
    out = report_energies(rungs, exact=E_exact, label="test", verbose=False)
    assert out["E_var_best"] is not None and out["E_var_plus_pt2_best"] is not None
    assert out["E_extrap_best"] is not None
    assert abs(out["extrap_minus_exact"]) < 0.05
    print(f"[30] extrapolation: SHCI/PT2 E∞={px['E_inf']:.4f} vs exact {E_exact:.4f} "
          f"(Δ={px['E_inf']-E_exact:+.4f}), power-law guard fires <4 rungs  OK")


def test_observables():
    """<N> / occupation-tail observables (`observables.py`) on a hand-built state.
    Deterministic: 2 equal-weight determinants, bos = [0,0,0] and [2,1,0]."""
    from types import SimpleNamespace
    from classical.trimci.observables import (mean_occupation, occupation_tail,
                                              occupation_histogram)
    res = SimpleNamespace(
        bos_arr=np.array([[0, 0, 0], [2, 1, 0]], dtype=np.uint16),
        coeffs=np.array([1.0, 1.0], dtype=complex) / np.sqrt(2), states=None)
    mo = mean_occupation(res)
    assert abs(mo["N_total"] - 1.5) < 1e-12         # 0.5*0 + 0.5*3
    assert abs(mo["N_per_mode"] - 0.5) < 1e-12
    assert abs(mo["N_max_mode"] - 1.0) < 1e-12      # 0.5*0 + 0.5*2
    assert abs(occupation_tail(res, 2) - 0.5) < 1e-12   # state 2 has a mode at occ 2
    assert abs(occupation_tail(res, 3) - 0.0) < 1e-12   # nothing at occ >= 3
    d = occupation_tail(res, [1, 2, 3])
    assert d[1] == 0.5 and d[2] == 0.5 and d[3] == 0.0
    hist = occupation_histogram(res)                # sums to 1 over occupations
    assert abs(hist.sum() - 1.0) < 1e-12
    print(f"[31] observables: <N>/mode=0.5, <N_max>=1.0, tail(2)=0.5 tail(3)=0.0  OK")


def test_tong_bound():
    """Analytic SCS / spectral predictions (`tong_bound.py`) reproduce the numbers
    in 02_tong_fock_cutoff.md §2.4/§6.1 at the reference point L=2 d=3 A=4."""
    from classical.trimci.tong_bound import (mean_occupation_scs, squeeze_r_star,
                                             squeezed_tail, cutoff_predictions)
    scs = mean_occupation_scs(2, 3, 4)
    assert abs(scs["r_star"] - 0.210) < 0.01, f"r* {scs['r_star']:.3f} != doc 0.210"
    assert abs(scs["N_per_mode"] - 0.045) < 0.005, f"<N> {scs['N_per_mode']:.3f} != 0.045"
    assert abs(scs["sigma_N"] - 0.23) < 0.02
    # r* grows with L (doc §2.4: 0.210, 0.272, 0.303 at L=2,4,10)
    assert squeeze_r_star(2, 3) < squeeze_r_star(4, 3) < squeeze_r_star(10, 3)
    # spectral bound brackets: eng n_b ~1, spectral n_b ~4-5 (doc §6.1)
    pred = cutoff_predictions(2, 3, 4)
    assert pred["n_b_eng"] <= 2
    assert 3 <= pred["n_b_spec2"] <= 5 and pred["n_b_spec1"] >= pred["n_b_spec2"]
    # tail decreases monotonically in N_f
    tails = [squeezed_tail(nf, 2, 3) for nf in (2, 4, 8, 16)]
    assert all(tails[i] > tails[i + 1] for i in range(len(tails) - 1))
    print(f"[32] tong_bound: r*={scs['r_star']:.3f} <N>={scs['N_per_mode']:.3f} "
          f"n_b(eng/spec2/spec1)={pred['n_b_eng']}/{pred['n_b_spec2']}/{pred['n_b_spec1']}  OK")


def test_nb_convergence_smoke():
    """`nb_convergence_sweep` runs end-to-end and returns well-formed rows with the
    truncation error monotone-ish and the exact ref matched where enumerable
    (L=2 d=1, N_f<=4 is exact). Tiny (core=120) — just the plumbing + PT2 hookup."""
    from classical.trimci.run_cpp import nb_convergence_sweep
    from classical.trimci.backend import cpp_available
    if not cpp_available():
        print("[33] nb_convergence_sweep — SKIPPED (C++ path not built)")
        return
    out = nb_convergence_sweep(L=2, dim=1, A=1, N_f_list=(2, 4), core=120,
                               n_runs=2, seed=0, verbose=False)
    rows = out["rows"]
    assert len(rows) == 2
    for r in rows:
        assert {"n_b", "N_f", "E_var", "dE_pt2", "E_pt2", "N_per_mode",
                "trunc_err", "runtime_s", "exact"} <= set(r)
        assert r["exact"] is not None            # both N_f=2,4 are enumerable here
        assert r["E_var"] >= r["exact"] - 1e-4   # variational (within PT2 tol)
    assert abs(rows[-1]["trunc_err"]) < 1e-9     # ref point vs itself
    assert "predictions" in out
    print(f"[33] nb_convergence_sweep: 2 rows, exact-matched, E>=E_ED, ref trunc=0  OK")


def test_occupation_A_independence():
    """The A-dependence study's core claim, at an EXACT (ED) size: the GS pion
    occupation <N> is A-INDEPENDENT (vacuum-squeezing dominated), so a low n_b is
    A-robust. `exact_occupation_vs_A` at L=2 d=1 N_f=3 must return <N>(A) flat
    across A (spread << the value) and nonzero. Also exercises the
    `occupation_from_coeffs` primitive (its output must match a direct recompute)."""
    import numpy as np
    from classical.trimci.run_cpp import exact_occupation_vs_A
    from classical.trimci.observables import occupation_from_coeffs
    from classical.trimci.lanczos import lanczos_ground_state
    from classical.trimci.hamiltonian import build_from_eft

    out = exact_occupation_vs_A(L=2, dim=1, A_list=(1, 2, 3), N_f=3, verbose=False)
    rows = out["rows"]
    assert len(rows) == 3
    Nm = np.array([r["N_per_mode"] for r in rows])
    assert (Nm > 1e-4).all()                         # nonzero occupation
    spread = Nm.max() - Nm.min()
    assert spread < 1e-3, f"<N>(A) not flat: spread={spread:.2e}"  # A-independent
    # occupation_from_coeffs matches a direct recompute on a fresh eigenvector
    H = build_from_eft(2, 1, 2, N_f=3)
    E0, cmap, info = lanczos_ground_state(H, n_elec=2, return_vec=True)
    mo = occupation_from_coeffs(cmap)
    w = np.abs(np.array(list(cmap.values()), dtype=complex)) ** 2
    w /= w.sum()
    bos = np.array([s.bos for s in cmap], dtype=float)
    assert abs(mo["N_total"] - float((w * bos.sum(1)).sum())) < 1e-9
    assert abs(mo["N_per_mode"] - Nm[1]) < 1e-9      # same system as A=2 row
    print(f"[34] occupation A-independence: <N>/mode flat over A=1,2,3 "
          f"(spread={spread:.1e}, <N>={Nm.mean():.4f})  OK")


def test_per_site_reporting():
    """Phase A — size-intensive (per-site) reporting in `report_energies`.
    (a) the per-site keys are total/sites exactly (the size-intensive mirror);
    (b) on the L=2 d=1 A=2 exact system, the SHCI/PT2 extrapolation is within a
        fixed PER-SITE tolerance of exact — the metric to hold constant across L,
        not the total or relative gap that the extensivity trap flatters at large L.
    Pure Python + Lanczos truth — never skipped."""
    from classical.trimci.extrapolation import report_energies
    from classical.trimci.pt2 import pt2_from_result
    L, dim, A, N_f = 2, 1, 2, 2
    sites = L ** dim
    H = build_from_eft(L, dim, 1, N_f=N_f)
    E_exact, _ = lanczos_ground_state(H, n_elec=A)
    rungs = []
    for c in (20, 40, 80, 160):
        res = ground_state_ensemble(H, n_elec=A, n_runs=4, n_dets=c, seed=0)
        pr = pt2_from_result(H, res)
        rungs.append({"core": c, "E_var": pr["E_var"], "dE_pt2": pr["dE_pt2"]})
    out = report_energies(rungs, exact=E_exact, sites=sites, label="per-site",
                          verbose=False)
    # (a) per-site == total / sites, exactly
    assert out["sites"] == sites
    for tot, ps in (("E_var_best", "E_var_best_per_site"),
                    ("E_var_plus_pt2_best", "E_var_plus_pt2_best_per_site"),
                    ("E_extrap_best", "E_extrap_best_per_site")):
        assert abs(out[ps] - out[tot] / sites) < 1e-12, f"{ps} != {tot}/sites"
    # a report WITHOUT sites carries no per-site keys (opt-in)
    bare = report_energies(rungs, exact=E_exact, verbose=False)
    assert "sites" not in bare and "E_extrap_best_per_site" not in bare
    # (b) per-site extrapolation error is small (pipeline proven vs truth)
    assert abs(out["extrap_minus_exact_per_site"]) < 0.02, \
        f"per-site extrap error too large: {out['extrap_minus_exact_per_site']:.4f}"
    print(f"[35] per-site reporting: E/site={out['E_extrap_best_per_site']:.4f} "
          f"(±{out['E_extrap_best_sigma_per_site']:.4f}), (extrap-exact)/site="
          f"{out['extrap_minus_exact_per_site']:+.4f} MeV/site  OK")


def test_converged_reference():
    """Phase B — per-L converged reference E_inf(L) +/- sigma + per-target pinning.
    On the L=2 d=1 A=2 exact system: (a) the extrapolated E_inf validates vs Lanczos
    to a fixed per-site tolerance and the loose eps=1 target PINS as a 'point';
    (b) an absurdly tight eps=1e-5 target that the reference cannot resolve is
    correctly reported as a 'bound' (NOT pinned) — the honest refusal that keeps the
    dets-vs-L exponent from inventing false precision; (c) the returned rungs carry
    the per-core (E_var, dE_pt2, E_pt2) the Phase-C N* extraction consumes.
    Pure Python + Lanczos truth — never skipped."""
    from classical.trimci.run_cpp import converged_reference
    out = converged_reference(L=2, dim=1, A=2, n_b=1, N_f=2,
                              cores=(20, 40, 80, 160),
                              eps_persite_targets=(1.0, 1e-5),
                              n_runs=4, seed=0, verbose=False)
    # exact reference available at this (enumerable) size
    assert out["exact"] is not None
    assert abs(out["extrap_minus_exact_per_site"]) < 0.02
    # (a) loose target pins as a validated point
    t_loose = out["targets"][1.0]
    assert t_loose["pinned"] and t_loose["kind"] == "point"
    # (b) unresolvable tight target is an honest bound, not a fake point
    t_tight = out["targets"][1e-5]
    assert (not t_tight["pinned"]) and t_tight["kind"] == "bound"
    # (c) rungs carry what Phase C needs
    assert len(out["rungs"]) == 4
    for r in out["rungs"]:
        assert {"core", "E_var", "dE_pt2", "E_pt2"} <= set(r)
        assert abs(r["E_pt2"] - (r["E_var"] + r["dE_pt2"])) < 1e-9
    print(f"[36] converged reference: E_inf/site={out['E_inf_per_site']:.4f}"
          f"±{out['sigma_per_site']:.4f}, eps=1 POINT / eps=1e-5 BOUND  OK")


def test_dets_vs_L():
    """Phase C — N* extraction, exponent fit, and the end-to-end driver.
    (a) `_extract_nstar` brackets N* on the STABLE (all-larger-rungs-hold) criterion,
        so a fluke early pass is not taken, and returns honest upper/lower bounds at
        the ladder edges; (b) `_fit_exponent` recovers the slope of exp-in-V and
        poly-in-V synthetic data exactly; (c) the driver runs end-to-end, writes its
        JSON/PNG, and LOGS non-fit-worthy L's as bounds rather than dropping them."""
    import math
    import os
    import tempfile
    from classical.trimci.run_cpp import (_extract_nstar, _fit_exponent,
                                          dets_vs_L_at_fixed_accuracy)
    sites, E_inf = 8, 100.0
    cores = [250, 500, 1000, 2000, 4000]
    gaps = [5.0, 2.0, 0.8, 0.3, 0.05]        # MeV/site, decreasing
    rungs = [{"core": c, "E_pt2": E_inf + g * sites, "E_var": E_inf + g * sites}
             for c, g in zip(cores, gaps)]
    # (a) bracketing + bounds
    assert _extract_nstar(rungs, E_inf, sites, 1.0)["n_lo"] == 500      # gap<1 at 1000
    assert _extract_nstar(rungs, E_inf, sites, 0.1)["n_lo"] == 2000     # gap<0.1 at 4000
    assert _extract_nstar(rungs, E_inf, sites, 10.0)["status"] == "upper_bound"
    assert _extract_nstar(rungs, E_inf, sites, 1e-3)["status"] == "lower_bound"
    # stability guard: a fluke early pass then a later fail must NOT be taken as N*
    nm = [{"core": 250, "E_pt2": E_inf + 0.5 * sites, "E_var": 0.0},
          {"core": 500, "E_pt2": E_inf + 2.0 * sites, "E_var": 0.0},
          {"core": 1000, "E_pt2": E_inf + 0.3 * sites, "E_var": 0.0}]
    r = _extract_nstar(nm, E_inf, sites, 1.0)
    assert r["n_lo"] == 500 and r["n_hi"] == 1000, "stability guard failed"
    # (b) fit recovers exact slopes
    V = [8, 16, 24, 32]
    fe = _fit_exponent(V, [math.exp(0.5 * v) for v in V])
    assert abs(fe["exponential_in_V"]["slope"] - 0.5) < 1e-6
    fp = _fit_exponent(V, [v ** 3 for v in V])
    assert abs(fp["polynomial_in_V"]["slope"] - 3.0) < 1e-6
    # (c) driver end-to-end (tiny 1D, temp out_dir — no data/ pollution)
    tmp = tempfile.mkdtemp()
    res = dets_vs_L_at_fixed_accuracy(
        dim=1, L_values=(2, 3), A=1, n_b=1, eps_persite_targets=(5.0,),
        ladder_start=20, n_rungs=4, max_core=160, n_runs=2, seed=0,
        out_dir=tmp, label="detsvsL_test", verbose=False)
    assert [p["L"] for p in res["per_L"]] == [2, 3]
    assert "5.0" in res["fits"]
    # every L that isn't fit-worthy must be accounted for in `dropped` (no silent loss)
    nfw = [p["L"] for p in res["per_L"] if not p["eps"]["5.0"]["fit_worthy"]]
    logged = [d["L"] for d in res["fits"]["5.0"]["dropped"]]
    assert set(nfw) == set(logged)
    assert os.path.exists(os.path.join(tmp, "detsvsL_test.json"))
    print(f"[37] dets-vs-L: bracket/bounds + stability guard, exp slope 0.5 & poly "
          f"slope 3.0 recovered, driver writes JSON + logs bounds  OK")


def test_robustness_guards():
    """Phase D — honesty/robustness guards.
    (a) `ladder_monotonicity` flags a bursty rung (energy rises as core grows) and
        passes a monotone ladder; (b) `pt2_memory_report` fires the semistochastic-PT2
        trigger past its n_ext budget and is graceful with no n_ext; (c)
        `converged_reference` now carries the monotonicity + pt2_memory diagnostics;
        (d) `seed_robustness` runs, returns the per-seed scatter, and a bool verdict.
    Pure Python + Lanczos truth — never skipped."""
    from classical.trimci.robustness import (ladder_monotonicity, pt2_memory_report,
                                             scatter_stats)
    from classical.trimci.run_cpp import converged_reference, seed_robustness
    # (a) monotonicity
    good = ladder_monotonicity([{"core": 500, "E_var": 100.0},
                                {"core": 1000, "E_var": 98.0}])
    assert good["monotone"] and good["max_rise"] == 0.0
    bad = ladder_monotonicity([{"core": 500, "E_var": 100.0},
                               {"core": 1000, "E_var": 98.0},
                               {"core": 2000, "E_var": 99.5}])
    assert (not bad["monotone"]) and abs(bad["max_rise"] - 1.5) < 1e-9
    assert bad["offenders"][0]["from_core"] == 1000
    # (b) pt2 memory / semistochastic trigger
    assert not pt2_memory_report([{"n_ext": 2_400_000}], n_ext_budget=50_000_000)["over_budget"]
    hot = pt2_memory_report([{"n_ext": 80_000_000}], n_ext_budget=50_000_000)
    assert hot["over_budget"] and "SEMISTOCHASTIC" in hot["trigger"]
    assert pt2_memory_report([{"core": 1}])["max_n_ext"] is None      # graceful
    assert scatter_stats([1.0, 3.0])["ptp"] == 2.0
    # (c) converged_reference now carries the diagnostics
    ref = converged_reference(L=2, dim=1, A=2, n_b=1, N_f=2, cores=(20, 40, 80, 160),
                              n_runs=4, seed=0, verbose=False)
    assert "monotonicity" in ref and "pt2_memory" in ref
    assert ref["pt2_memory"]["max_n_ext"] is not None
    # (d) seed_robustness end-to-end (tiny, exact-size); verdict is a bool, scatter real
    sr = seed_robustness(L=2, dim=1, A=2, n_b=1, N_f=2, core=40, seeds=(0, 100),
                         n_runs=2, eps_persite=1.0, verbose=False)
    assert len(sr["per_seed"]) == 2 and isinstance(sr["robust"], bool)
    assert sr["E_pt2_ptp_per_site"] is not None
    print(f"[38] robustness guards: monotonicity + PT2 semistochastic-trigger + "
          f"converged_reference diagnostics + seed_robustness (ptp/site="
          f"{sr['E_pt2_ptp_per_site']:.3f})  OK")


def main():
    print("=" * 64)
    print("  classical/trimci validation  (L=1, dim=1, n_b=2, A=1)")
    print("=" * 64)
    H = build_from_eft(1, 1, 2)
    s = summarize(H)
    print(f"  H blocks: ferm={s['n_fermion']} bos={s['n_boson']} "
          f"mixed={s['n_mixed']} const={s['constant'].real:.3f}  "
          f"(modes: {s['n_ferm_modes']}F x {s['n_bos_modes']}B, N_f={s['N_f']})")
    test_fermion_sector_spectrum()
    test_boson_sector_spectrum()
    test_full_mixed_hermitian_and_solver()
    test_dump_roundtrip()
    test_ensemble_reaches_ed()
    test_lanczos_matches_dense_ed()
    test_lanczos_oom_guard()
    test_nf_convergence_runs()
    test_near_vacuum_init()
    test_official_backend()
    test_mixed_ci_cpp_port()
    test_arbitrary_width_fermion()
    test_lf_machinery()
    test_frame_gaussian_squeeze()
    test_frame_multimode_bogoliubov()
    test_frame_lang_firsov()
    test_frame_combined_step4()
    test_frame_orbital_rotation()
    test_frame_non_gaussianity()
    test_frame_qpe_payoff()
    test_frame_projector_lf_variational()
    test_frame_coo_from_core()
    test_frame_workflow()
    test_cpp_full_path()
    test_matfree_diagonalize()
    test_ground_state_arrays()
    test_io_roundtrip()
    test_nf_decoupling()
    test_pt2_epstein_nesbet()
    test_energy_extrapolation()
    test_observables()
    test_tong_bound()
    test_nb_convergence_smoke()
    test_occupation_A_independence()
    test_per_site_reporting()
    test_converged_reference()
    test_dets_vs_L()
    test_robustness_guards()
    print("=" * 64)
    print("  ALL TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
