"""
D-AFQMC-sign : exact, term-resolved sign-problem diagnostics for the QMC family.

The auxiliary-field / QMC methods (AFQMC, AFDMC, NLEFT, worldline/SSE, FCIQMC)
are ruled out as *cheaper* classical baselines for this model because it has a
sign problem. The airtight, method-specific argument for AFQMC is analytic (the
determinant-factorization + time-reversal argument in
`claude/research/classical_baselines/afqmc_sign_problem/00_afqmc_sign_problem.md`):
the pion couples to an OFF-DIAGONAL spin-AND-isospin transition bilinear
(chi ~ sigma (x) tau), which destroys the det_up * det_down factorization that
makes Holstein/SSH couplings sign-free, and it fails the Kramers/time-reversal
sign-free condition (sigma is T-odd). It is sign-problematic even at A=Z, because
Wigner-SU(4) protection is a spin-isospin-*independence* symmetry, not a
proton-neutron-*number* symmetry.

This module supplies the cheap, exact NUMERICAL corroboration, and pinpoints the
responsible terms. On a small lattice we form the exact Hamiltonian matrix in the
natural mixed fermion-boson occupation basis and measure two basis-diagnostics:

  * COMPLEX off-diagonal fraction (a PHASE problem). The Weinberg-Tomozawa term
    carries an imaginary, antisymmetric  pi(x) . Pi(x)  ~ eps_abc (b)(i(b^dag - b))
    structure -> genuinely complex off-diagonal matrix elements. Complex phases are
    a basis-invariant obstruction: no real diagonal gauge / basis sign flip removes
    them. Toggling H_WT off collapses the complex fraction to ~0, isolating it as
    THE source of the phase problem.

  * Worldline "average-sign severity" (Troyer-Wiese). Build the stoquastic
    comparison H_stoq (same diagonal; each off-diagonal element replaced by
    -|element|), whose worldline weights are non-negative by construction. Then
        <s>(beta) = Tr e^{-beta H} / Tr e^{-beta H_stoq},   <s> ~ e^{-beta*Delta_sign}
        Delta_sign = E0(H) - E0(H_stoq) >= 0   (Perron-Frobenius).
    Delta_sign > 0 (and extensive) => a worldline/SSE QMC average sign decays
    exponentially. We report Delta_sign and its per-site value, and attribute it to
    the pion Yukawa (H_AV) and WT (H_WT) terms by toggling them.

Caveats stated honestly:
  * These are natural-BASIS diagnostics. The AFQMC (determinantal) sign statement
    is the analytic argument above; the worldline severity here is the SSE/worldline
    family. Both QMC families thus hit a sign wall for this model.
  * Free-boson terms (m_pi n + gradient^2) are non-stoquastic in the number/Fock
    basis but that piece is a known, curable basis artifact (they are trivially
    sign-free in the field/quadrature basis). We therefore attribute via TERM
    TOGGLES and the g_A=0 baseline rather than reading absolute numbers, and we
    lead with the COMPLEX (phase) diagnostic, which is NOT a basis artifact.
"""

from __future__ import annotations

import numpy as np


def scaled_terms(H, av=1.0, wt=1.0, boson=1.0, fermion=1.0):
    """Return a copy of MixedH with term groups scaled.

    Groups: mixed terms with a degree-1 boson factor are H_AV (Yukawa, scale `av`);
    degree-2 boson factor are H_WT (scale `wt`); boson-only terms scale `boson`;
    fermion-only terms scale `fermion`. Constants are untouched.
    """
    from classical.trimci.hamiltonian import MixedH, OperatorTerm
    new = []
    for t in H.terms:
        f = 1.0
        if t.ferm_ops and t.bos_ops:
            f = av if len(t.bos_ops) == 1 else wt
        elif t.bos_ops and not t.ferm_ops:
            f = boson
        elif t.ferm_ops and not t.bos_ops:
            f = fermion
        new.append(OperatorTerm(t.coeff * f, t.ferm_ops, t.bos_ops))
    return MixedH(terms=new, n_ferm_modes=H.n_ferm_modes,
                  n_bos_modes=H.n_bos_modes, N_f=H.N_f, meta=dict(H.meta))


def dense_H(L, dim, A, N_f=2, n_b=1, max_dense=6000, **scales):
    """Exact dense Hamiltonian matrix over the A-nucleon sector (complex)."""
    from classical.trimci import build_from_eft
    from classical.trimci.hij import build_dense
    from classical.trimci.state import enumerate_basis
    H = build_from_eft(L=L, dim=dim, n_b=n_b, N_f=N_f)
    H = scaled_terms(H, **scales)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A,
                            max_states=max_dense)
    M = build_dense(H, basis, max_dense=max_dense)
    M = 0.5 * (M + M.conj().T)          # symmetrize numerically
    return M, len(basis)


def offdiag_report(M, tol=1e-9):
    """Off-diagonal sign/phase structure of a Hermitian matrix M."""
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    vals = M[off]
    mag = np.abs(vals)
    nz = mag > tol
    total = int(nz.sum())
    if total == 0:
        return {'n_offdiag_nonzero': 0}
    im = np.abs(vals.imag)
    re = vals.real
    complex_mask = im[nz] > tol
    posreal_mask = (~complex_mask) & (re[nz] > tol)
    # magnitude-weighted (what actually matters for the weights)
    w = mag[nz]
    return {
        'n_offdiag_nonzero': total,
        'frac_complex': float(complex_mask.mean()),
        'frac_positive_real': float(posreal_mask.mean()),
        'frac_frustrating': float((complex_mask | posreal_mask).mean()),
        'wfrac_complex': float(w[complex_mask].sum() / w.sum()),
        'wfrac_frustrating': float(w[complex_mask | posreal_mask].sum() / w.sum()),
        'max_abs_imag': float(im.max()),
    }


def sign_severity(M, sites):
    """Troyer-Wiese worldline average-sign severity from exact diagonalization.

    Delta_sign = E0(M) - E0(M_stoq), M_stoq = same diagonal, off-diagonals -> -|.|.
    Returns Delta_sign, Delta_sign/site, and the two ground energies.
    """
    n = M.shape[0]
    M_stoq = -np.abs(M)
    np.fill_diagonal(M_stoq, np.real(np.diag(M)))
    e0 = float(np.linalg.eigvalsh(M)[0])
    e0s = float(np.linalg.eigvalsh(M_stoq)[0])
    d = e0 - e0s
    return {'E0': e0, 'E0_stoq': e0s, 'Delta_sign': d,
            'Delta_sign_per_site': d / sites}


def average_sign_curve(M, betas):
    """Exact <s>(beta) = Tr e^{-bH}/Tr e^{-bH_stoq} from the full spectra."""
    ev = np.linalg.eigvalsh(M)
    M_stoq = -np.abs(M)
    np.fill_diagonal(M_stoq, np.real(np.diag(M)))
    evs = np.linalg.eigvalsh(M_stoq)
    out = []
    for b in betas:
        shift = min(ev.min(), evs.min())
        z = np.exp(-b * (ev - shift)).sum()
        zs = np.exp(-b * (evs - shift)).sum()
        out.append(z / zs)
    return np.array(out)


def sparse_H(L, dim, A, N_f=2, n_b=1, max_states=400000, **scales):
    """Sparse complex Hermitian H over the A-nucleon sector (scales past dense)."""
    import scipy.sparse as sp
    from classical.trimci import build_from_eft
    from classical.trimci.hij import connections_nocache
    from classical.trimci.state import enumerate_basis
    H = build_from_eft(L=L, dim=dim, n_b=n_b, N_f=N_f)
    H = scaled_terms(H, **scales)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A,
                            max_states=max_states)
    index = {s: i for i, s in enumerate(basis)}
    rows, cols, data = [], [], []
    for j, sj in enumerate(basis):
        for si, amp in connections_nocache(H, sj).items():
            rows.append(index[si]); cols.append(j); data.append(amp)
    N = len(basis)
    M = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=complex)
    M = 0.5 * (M + M.getH())
    return M, L ** dim, N


def sign_diagnostics_sparse(M, sites):
    """Complex off-diagonal fraction + Troyer-Wiese Delta_sign, via sparse eigsh."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    Mc = M.tocoo()
    off = Mc.row != Mc.col
    vals = Mc.data[off]
    mag = np.abs(vals)
    nz = mag > 1e-9
    frac_complex = float((np.abs(vals.imag)[nz] > 1e-9).mean()) if nz.any() else 0.0
    # stoquastic comparison: off-diagonals -> -|.|, diagonal kept (real)
    r, c, d = Mc.row, Mc.col, Mc.data
    d2 = np.where(r == c, np.real(d), -np.abs(d))
    M_stoq = sp.csr_matrix((d2.astype(float), (r, c)), shape=M.shape)
    M_stoq = 0.5 * (M_stoq + M_stoq.T)
    e0 = float(sla.eigsh(M.tocsc(), k=1, which='SA', return_eigenvectors=False)[0])
    e0s = float(sla.eigsh(M_stoq.tocsc(), k=1, which='SA', return_eigenvectors=False)[0])
    return {'dimH': M.shape[0], 'frac_complex': frac_complex,
            'Delta_sign': e0 - e0s, 'Delta_sign_per_site': (e0 - e0s) / sites}


def scaling_run(points=((2, 1, 2), (3, 1, 2), (2, 1, 4), (3, 1, 4)), N_f=2):
    """Show the sign severity grows with volume (L) and particle number (A)."""
    print("D-AFQMC-sign scaling (full model): Delta_sign extensive in L and A\n")
    print(f"  {'L':>2} {'dim':>3} {'A':>2} {'dimH':>10}  {'%complex':>9}  "
          f"{'Delta_sign':>11}  {'/site':>8}")
    rows = []
    for (L, dim, A) in points:
        M, sites, N = sparse_H(L, dim, A, N_f=N_f)
        d = sign_diagnostics_sparse(M, sites)
        rows.append({'L': L, 'dim': dim, 'A': A, **d})
        print(f"  {L:>2} {dim:>3} {A:>2} {N:>10,}  {d['frac_complex']*100:8.1f}%  "
              f"{d['Delta_sign']:11.4f}  {d['Delta_sign_per_site']:8.4f}")
    return rows


# term-toggle configurations for attribution
CONFIGS = {
    'full model':               dict(),
    'no WT (H_WT off)':         dict(wt=0.0),
    'no Yukawa (H_AV off)':     dict(av=0.0),
    'no pion coupling':         dict(av=0.0, wt=0.0),
    'free + contacts only':     dict(av=0.0, wt=0.0),   # same matrix; kept for label
}


def run(L=2, dim=1, A=2, N_f=2, betas=(0.5, 1.0, 2.0, 4.0, 8.0), save=True):
    """Term-resolved sign diagnostics; prints a table and (optionally) saves."""
    sites = L ** dim
    print(f"D-AFQMC-sign: L={L} dim={dim} A={A} N_f={N_f} ({sites} sites)")
    print(f"(physical g_A=1.26; toggling H_AV/H_WT to attribute the sign source)\n")
    results = {}
    for label, sc in CONFIGS.items():
        M, dimH = dense_H(L, dim, A, N_f=N_f, **sc)
        od = offdiag_report(M)
        sv = sign_severity(M, sites)
        results[label] = {'dimH': dimH, **od, **sv}
        print(f"  {label:24s}  complex-offdiag={od.get('frac_complex',0)*100:5.1f}%  "
              f"frustrating={od.get('frac_frustrating',0)*100:5.1f}%  "
              f"Delta_sign/site={sv['Delta_sign_per_site']:7.3f} MeV")

    # average-sign decay for the full model
    Mfull, _ = dense_H(L, dim, A, N_f=N_f)
    s_curve = average_sign_curve(Mfull, betas)
    print("\n  <sign>(beta) for the FULL model (worldline/SSE):")
    for b, s in zip(betas, s_curve):
        print(f"     beta={b:5.1f}  <s>={s:.3e}")

    if save:
        _save(L, dim, A, N_f, results, list(betas), s_curve.tolist())
    return results, (list(betas), s_curve)


def _save(L, dim, A, N_f, results, betas, s_curve):
    import json, os, datetime as _dt
    date = _dt.date.today().isoformat()
    outdir = f"data/classical/baselines/{date}"
    os.makedirs(outdir, exist_ok=True)
    stamp = f"sign_structure_L{L}_d{dim}_A{A}"
    with open(f"{outdir}/{stamp}.json", 'w') as f:
        json.dump({'L': L, 'dim': dim, 'A': A, 'N_f': N_f,
                   'results': results, 'betas': betas, 's_curve': s_curve},
                  f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
        labels = [k for k in results if k != 'free + contacts only']
        cvals = [results[k].get('frac_complex', 0) * 100 for k in labels]
        fvals = [results[k].get('frac_frustrating', 0) * 100 for k in labels]
        x = np.arange(len(labels))
        ax1.bar(x - 0.2, cvals, 0.4, label='complex (phase problem)', color='C3')
        ax1.bar(x + 0.2, fvals, 0.4, label='frustrating (sign)', color='C1')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax1.set_ylabel('% of off-diagonal elements')
        ax1.set_title('Sign/phase structure by term (exact)\n'
                      'H_WT -> complex phase; H_AV -> sign frustration')
        ax1.legend(fontsize=8)
        ax2.semilogy(betas, s_curve, 'o-', color='C0')
        ax2.set_xlabel(r'projection time  $\beta$  (MeV$^{-1}$)')
        ax2.set_ylabel(r'average sign  $\langle s\rangle$')
        ax2.set_title('Worldline average sign decays exponentially\n'
                      r'$\langle s\rangle \sim e^{-\beta\,\Delta_{\rm sign}}$')
        ax2.grid(True, which='both', alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{outdir}/{stamp}.png", dpi=130)
        plt.close(fig)
        print(f"\nSaved: {outdir}/{stamp}.json and {stamp}.png")
    except Exception as e:
        print(f"\nSaved JSON (plot skipped: {e})")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="D-AFQMC-sign exact diagnostics")
    p.add_argument('--L', type=int, default=2)
    p.add_argument('--dim', type=int, default=1)
    p.add_argument('--A', type=int, default=2)
    p.add_argument('--N_f', type=int, default=2)
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    run(L=args.L, dim=args.dim, A=args.A, N_f=args.N_f, save=args.save)
