"""
Tests for the D-CC fermionic-reduction baseline demo
(classical/baselines/cc_reduction.py):

  1. INTEGRAL CONVERSION IS EXACT — the spatial (h1, eri) fed to pyscf reproduce
     the exact OpenFermion full-Fock ground state to <1e-6 across couplings. This
     is the correctness gate: if it passes, the pyscf CCSD numbers are trustworthy.
  2. PHYSICS MATCH — build_fermion_H reproduces the project's own static-nucleon
     sector (same excitation spectrum as ConstructEFT's fermion_part), so the demo
     runs CC on the *actual* model, not a lookalike.
  3. CC-BREAKDOWN SIGNATURE — at weak coupling CC error is small; at physical
     coupling the RHF reference is internally unstable and single-reference CCSD is
     uncontrolled (large / non-variational error or non-convergence), while FCI
     (the selected-CI target) is exact throughout.

Run from the project root:
    python -m tests.test_cc_reduction
"""

import numpy as np

from classical.baselines.cc_reduction import (
    build_fermion_H, exact_E0, spatial_integrals, fci_reference,
    run_pyscf_cc, sweep,
)


def test_integral_conversion_exact():
    """pyscf FCI from the extracted spatial integrals == OpenFermion full-Fock E0."""
    ok = True
    for (L, dim) in [(2, 1)]:
        for s in (0.0, 0.3, 1.0, 2.0):
            H, ns = build_fermion_H(L, dim, s=s)
            e_of, nexp = exact_E0(H, 2)
            const, h1, eri = spatial_integrals(H, ns)
            e_fci = fci_reference(h1, eri, const, 2 * ns, 2)
            match = abs(e_of - e_fci) < 1e-6
            ok = ok and match and abs(nexp - 2.0) < 1e-6
            print(f"  [conv] L={L} d={dim} s={s:.1f}: "
                  f"OF={e_of:10.5f}  pyscfFCI={e_fci:10.5f}  match={match}")
    assert ok, "integral conversion does not reproduce exact FCI"
    print("  PASS: integral conversion exact\n")


def test_physics_match_pipeline():
    """build_fermion_H has the same excitation spectrum as the project's own
    static-nucleon fermion sector (ConstructEFT fermion_part). Compared via
    energy gaps (offset-independent, so pion zero-point constants don't matter)."""
    from classical.trimci import build_from_eft
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    from classical.trimci.hij import build_dense
    from classical.trimci.state import enumerate_basis

    L, dim, A = 2, 1, 2
    # our standalone fermionic H, dense A-particle spectrum
    Hf, ns = build_fermion_H(L, dim, s=1.0)
    from openfermion import get_sparse_operator, number_operator
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    n_so = 4 * ns
    ham = get_sparse_operator(Hf, n_qubits=n_so).toarray()
    Nop = get_sparse_operator(number_operator(n_so), n_qubits=n_so).toarray()
    P = Nop - A * np.eye(ham.shape[0])
    ev_mine = np.linalg.eigvalsh(ham + 1e4 * (P @ P))
    ev_mine = np.sort(ev_mine[ev_mine < 5e3])[:6]

    # project's fermion sector: strip pion (fermion-only terms), rebuild MixedH
    Hmix = build_from_eft(L=L, dim=dim, n_b=1)
    fterms = [t for t in Hmix.terms if not t.bos_ops]
    Hferm_proj = MixedH(terms=fterms, n_ferm_modes=Hmix.n_ferm_modes,
                        n_bos_modes=0, N_f=1, meta={})
    basis = enumerate_basis(Hmix.n_ferm_modes, 0, 1, n_elec=A)
    M = build_dense(Hferm_proj, basis)
    ev_proj = np.sort(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))[:6]

    gaps_mine = ev_mine - ev_mine[0]
    gaps_proj = ev_proj - ev_proj[0]
    close = np.allclose(gaps_mine, gaps_proj, atol=1e-6)
    print(f"  [match] excitation gaps mine={np.round(gaps_mine,4)}")
    print(f"  [match] excitation gaps proj={np.round(gaps_proj,4)}")
    assert close, "build_fermion_H spectrum differs from the project's fermion sector"
    print("  PASS: physics matches the project's static-nucleon sector\n")


def test_cc_breakdown_signature():
    """Weak coupling: CC accurate + RHF stable. Physical coupling: RHF unstable
    and CC uncontrolled (large error / non-variational / non-convergent)."""
    rows = sweep(L=2, dim=3, A=2, s_values=(0.1, 1.0, 2.0), verbose=False)
    r_weak = rows[0]
    r_phys = rows[1]
    r_strong = rows[2]

    # weak coupling: RHF stable and CCSD close to exact
    weak_err = abs(r_weak.get('err_E_CCSD', 1e9))
    print(f"  [weak s=0.1]  RHF_stable={r_weak['RHF_stable']}  |errCCSD|={weak_err:.3f} MeV")
    assert r_weak['RHF_stable'] in (True, None), "expected RHF stable at weak coupling"
    assert weak_err < 0.5, "CCSD should be accurate at weak coupling"

    # physical/strong: static-correlation onset + uncontrolled CC
    phys_unstable = (r_phys['RHF_stable'] is False)
    # 'uncontrolled' = large error, or non-variational (below FCI), or non-converged
    def uncontrolled(r):
        e = r.get('E_CCSD')
        conv = r.get('CCSD_converged', True)
        err = r.get('err_E_CCSD')
        return (e is None) or (not conv) or (err is not None and (err < -1e-6 or abs(err) > 3.0))
    print(f"  [phys s=1.0]  RHF_stable={r_phys['RHF_stable']}  "
          f"CCSDconv={r_phys.get('CCSD_converged')}  errCCSD={r_phys.get('err_E_CCSD')}")
    print(f"  [strong s=2]  errCCSD={r_strong.get('err_E_CCSD')}  "
          f"(non-variational if < 0)")
    assert phys_unstable, "expected RHF instability (static correlation) at physical coupling"
    assert uncontrolled(r_phys) or uncontrolled(r_strong), \
        "expected CC to be uncontrolled at/above physical coupling"
    print("  PASS: CC breaks down at physical coupling; FCI (selected-CI target) exact\n")


if __name__ == '__main__':
    print("=== D-CC fermionic-reduction tests ===\n")
    test_integral_conversion_exact()
    test_physics_match_pipeline()
    test_cc_breakdown_signature()
    print("ALL TESTS PASSED")
