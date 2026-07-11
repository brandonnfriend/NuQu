"""
D-CC : coupled cluster on the fermionic reduction of the dynamical-pion EFT.

Why this is the honest, cheap CC comparison
-------------------------------------------
Standard fermionic CC (pyscf CCSD/CCSD(T)) needs a purely fermionic, *2-body*
Hamiltonian (an FCIDUMP: h1 + eri). Our full model has explicit pion Fock modes,
and once you fermionize the bosons the pion-nucleon terms become 3-body (H_AV)
and 4-body (H_WT) — which a 2-body FCIDUMP cannot hold. So a naive "fermionize ->
pyscf CCSD" route is blocked, and the boson fermionization is CC-unfavorable
anyway (it dresses the boson occupation as extra static correlation).

Instead we run CC on the **fermionic reduction**: freeze the pion out and keep
the static-nucleon sector, which is *already* a genuine 2-body fermionic
Hamiltonian — a 4-component (spin x isospin) Hubbard-like model:

    H_F = -h  sum_<ij>,m ( a†_{i,m} a_{j,m} + h.c. )        (nearest-neighbor hop)
          + (2 d h) sum_{i,m} n_{i,m}                       (on-site kinetic)
          + (C /2) sum_i  rho_i^2                           (density contact, C<0)
          + (C_I/2) sum_i sum_{I=1,2,3} rho_{I,i}^2         (isospin contact)

with m=(spin,isospin), rho_i = total nucleon density, rho_{I,i} = isospin
density (tau_I). This is the **most CC-favorable version of our model**: no
explicit bosons, purely 2-body, spin-independent interactions -> a standard
restricted molecular Hamiltonian pyscf ingests directly. The interactions are
*strong and attractive* (C = -51.94 MeV binds the nucleus), i.e. exactly the
large-attractive-pairing / large-U regime where single-reference CC is known to
fail catastrophically (Bulik-Henderson-Scuseria 2015, arXiv:1505.01894).

The demonstration
-----------------
Scale the interaction by s in [0, ..] (s=1 physical; s=0 free fermions) and, at
each s, compare:
  * exact FCI            E0(s)   -- ground truth (OpenFermion sparse; also pyscf FCI)
  * CCSD, CCSD(T)        -- pyscf restricted CC on the RHF reference
Expected: at small s, E_CC ~= E_FCI (and selected CI needs only a few dets, so
CC's cheapness is vacuous); toward/above physical s, the *signed* CC error grows
and the CC amplitude equations / RHF stability degrade -- CC is uncontrolled and
non-variational, while FCI (and the selected-CI upper bound converging to it)
stays a controlled number.

Because the full explicit-pion model is *strictly harder* for CC (it adds the
boson-correlation axis on top of this fermionic static correlation), a CC
breakdown already visible in this reduction is a conservative lower bound on
CC's difficulty for the real model.

Everything here is laptop-scale (n_orbitals = 2*N_sites; A=2 -> tiny FCI).
"""

from __future__ import annotations

import numpy as np

# --- mode / orbital bookkeeping -------------------------------------------
# nucleon internal modes at a site: (spin, isospin), spin,iso in {0,1}.
# spin-orbital convention (matches pyscf "spin-orbital = 2*spatial + spin"):
#     spatial(site, iso) = 2*site + iso            (n_orb = 2 * N_sites)
#     spin_orbital       = 2*spatial + spin        (n_so  = 4 * N_sites)
_TAU = [
    np.array([[0, 1], [1, 0]], dtype=complex),      # tau_1
    np.array([[0, -1j], [1j, 0]], dtype=complex),   # tau_2
    np.array([[1, 0], [0, -1]], dtype=complex),     # tau_3
]


def _spatial(site, iso):
    return 2 * site + iso


def _so(spatial, spin):
    return 2 * spatial + spin


def _mode(site, spin, iso):
    return _so(_spatial(site, iso), spin)


def _sites_and_neighbors(L, dim):
    """OBC nearest-neighbor bonds on an L^dim lattice (forward only)."""
    from src_PI.utils.LatticeGeometry import (
        get_total_sites, index_to_coord,
    )
    num_sites = get_total_sites(L, dim)
    bonds = []
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for d in range(dim):
            if coords[d] < L - 1:
                bonds.append((x, x + L ** d))
    return num_sites, bonds


def build_fermion_H(L, dim, s=1.0, params=None):
    """Fermionic reduction H_F as an OpenFermion FermionOperator.

    `s` scales the interaction (contacts) only; hopping+on-site are the fixed
    non-interacting reference. Physical parameters at s=1.
    """
    from openfermion import FermionOperator, normal_ordered
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    if params is None:
        params = get_physical_parameters()
    h, C, CI = params['h'], params['C'], params['CI']

    num_sites, bonds = _sites_and_neighbors(L, dim)
    H = FermionOperator()

    # hopping (spin- and isospin-diagonal) + on-site kinetic
    for (i, j) in bonds:
        for spin in (0, 1):
            for iso in (0, 1):
                mi, mj = _mode(i, spin, iso), _mode(j, spin, iso)
                H += FermionOperator(f'{mi}^ {mj}', -h)
                H += FermionOperator(f'{mj}^ {mi}', -h)
    onsite = 2 * dim * h
    for site in range(num_sites):
        for spin in (0, 1):
            for iso in (0, 1):
                m = _mode(site, spin, iso)
                H += FermionOperator(f'{m}^ {m}', onsite)

    # contacts (scaled by s)
    for site in range(num_sites):
        # density contact HC = (C/2) rho^2
        rho = FermionOperator()
        for spin in (0, 1):
            for iso in (0, 1):
                m = _mode(site, spin, iso)
                rho += FermionOperator(f'{m}^ {m}')
        H += normal_ordered(rho * rho) * (s * C / 2.0)
        # isospin contact HCI2 = (C_I/2) sum_I rho_I^2
        for I in range(3):
            mat = _TAU[I]
            rho_I = FermionOperator()
            for spin in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        coeff = mat[b, c]
                        if abs(coeff) > 1e-12:
                            mb = _mode(site, spin, b)
                            mc = _mode(site, spin, c)
                            rho_I += FermionOperator(f'{mb}^ {mc}', coeff)
            H += normal_ordered(rho_I * rho_I) * (s * CI / 2.0)
    return normal_ordered(H), num_sites


def exact_E0(H_ferm, A, n_so_cap=22):
    """Ground-state energy in the A-nucleon sector via OpenFermion sparse (truth).

    A convention-free cross-check on pyscf FCI. Full-Fock, so guarded to small
    spin-orbital counts (n_so <= n_so_cap); returns (None, None) above that.
    """
    from openfermion import get_sparse_operator, number_operator
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    n_so = 1 + max(idx for term in H_ferm.terms for (idx, _a) in term) if H_ferm.terms else 0
    if n_so > n_so_cap:
        return None, None
    ham = get_sparse_operator(H_ferm, n_qubits=n_so)
    Nop = get_sparse_operator(number_operator(n_so), n_qubits=n_so)
    P = (Nop - A * sp.identity(ham.shape[0], format='csr'))
    vals, vecs = sla.eigsh(ham + 1e4 * (P @ P), k=1, which='SA')
    v = vecs[:, 0]
    nexp = float((v.conj() @ (Nop @ v)).real)
    return float(vals[0].real), nexp


def _nelec(A):
    """(n_alpha, n_beta) split for A nucleons (spin-independent H -> any split ok)."""
    return (A - A // 2, A // 2)


def fci_reference(h1, eri, const, n_orb, A):
    """Exact FCI energy from the spatial integrals (robust; independent of RHF).

    Validated (d=1) to equal the OpenFermion full-Fock ground state to <1e-5.
    Scales to d=3 (works in the A-particle sector, not the 2^n_so Fock space).
    """
    from pyscf import fci
    return float(fci.direct_spin1.kernel(h1, eri, n_orb, _nelec(A), ecore=const)[0])


def spatial_integrals(H_ferm, num_sites):
    """(const, h1, eri_chem) spatial restricted integrals for a spin-independent H.

    Uses OpenFermion's InteractionOperator (physicist ordering) and reduces to
    spatial orbitals. Validated downstream by pyscf-FCI == OpenFermion-E0.
    """
    from openfermion import get_interaction_operator
    iop = get_interaction_operator(H_ferm)
    const = float(np.real(iop.constant))
    T = iop.one_body_tensor            # [n_so, n_so]
    V = iop.two_body_tensor            # H2 = sum V[p,q,r,s] a†p a†q ar as (physicist)
    n_orb = 2 * num_sites

    # one-body: spin-independent -> take the spin-up block
    h1 = np.zeros((n_orb, n_orb))
    for a in range(n_orb):
        for b in range(n_orb):
            h1[a, b] = np.real(T[_so(a, 0), _so(b, 0)])

    # two-body -> chemist spatial eri, extracted the convention-free way as the
    # OPPOSITE-spin 2-particle matrix element  eri[p,q,r,s] = <p_up r_dn|H2|q_up s_dn>
    # (exchange-free; validated to reproduce exact FCI to 1e-5, factor 1.0). The
    # analytic form below evaluates that matrix element directly from V without
    # ever building the 2^n_so Fock space, so it scales to d=3 / larger lattices.
    #   |q_up s_dn> = a†_Q a†_S|vac>,  <p_up r_dn| = <vac|a_R a_P
    #   H2 = sum V[a,b,g,d] a†a a†b ag ad ;  nonzero when {a,b}={P,R},{g,d}={Q,S}
    #   => eri = V[P,R,S,Q] - V[P,R,Q,S] - V[R,P,S,Q] + V[R,P,Q,S]
    eri = np.zeros((n_orb, n_orb, n_orb, n_orb))
    for p in range(n_orb):
        P = _so(p, 0)
        for r in range(n_orb):
            R = _so(r, 1)
            for q in range(n_orb):
                Q = _so(q, 0)
                for sdx in range(n_orb):
                    S = _so(sdx, 1)
                    eri[p, q, r, sdx] = np.real(
                        V[P, R, S, Q] - V[P, R, Q, S]
                        - V[R, P, S, Q] + V[R, P, Q, S]
                    )
    return const, h1, eri


def _make_mf(mf, h1, eri, n_orb, const):
    import numpy as _np
    from pyscf import ao2mo
    mf.get_hcore = lambda *a: h1
    mf.get_ovlp = lambda *a: _np.eye(n_orb)
    mf._eri = ao2mo.restore(8, eri, n_orb)
    mf.energy_nuc = lambda *a: const
    mf.max_cycle = 400
    return mf


def run_pyscf_cc(const, h1, eri, n_orb, A, verbose=False):
    """CCSD/CCSD(T) on custom restricted integrals, from BOTH an RHF and a
    (broken-symmetry) UHF reference.

    Reporting both preempts the "just use a better reference" objection: RHF
    instability flags the static-correlation onset (the single-reference ansatz
    itself failing, per Bulik-Henderson-Scuseria); UHF-CCSD restores a converging
    reference but is spin-contaminated and still uncontrolled at strong coupling.
    """
    from pyscf import gto, scf, cc
    import numpy as _np

    na, nb = _nelec(A)
    mol = gto.M(verbose=0)
    mol.nelectron = A
    mol.spin = na - nb
    mol.incore_anyway = True
    mol.nao_nr = lambda *a: n_orb

    out = {}

    # --- RHF reference ---
    rmf = _make_mf(scf.RHF(mol), h1, eri, n_orb, const)
    with np.errstate(all='ignore'):
        out['E_RHF'] = float(rmf.kernel())
    out['RHF_converged'] = bool(rmf.converged)
    try:
        mo_stable = rmf.stability()[0]
        out['RHF_stable'] = bool(_np.allclose(mo_stable, rmf.mo_coeff))
    except Exception:
        out['RHF_stable'] = None

    def _cc(mf, tag):
        try:
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 400
            with np.errstate(all='ignore'):
                mycc.kernel()
            e = float(mycc.e_tot)
            out[f'E_CCSD{tag}'] = e if np.isfinite(e) else None
            out[f'CCSD{tag}_converged'] = bool(mycc.converged) and np.isfinite(e)
            try:
                with np.errstate(all='ignore'):
                    et = mycc.ccsd_t()
                out[f'E_CCSD(T){tag}'] = float(e + et) if np.isfinite(et) else None
            except Exception:
                out[f'E_CCSD(T){tag}'] = None
        except Exception as ex:
            out[f'E_CCSD{tag}'] = None
            out[f'CCSD{tag}_converged'] = False
            out[f'CCSD{tag}_error'] = str(ex)[:80]

    _cc(rmf, '')  # RHF-based (primary)

    # --- UHF (broken-symmetry) reference + CCSD ---
    umf = _make_mf(scf.UHF(mol), h1, eri, n_orb, const)
    with np.errstate(all='ignore'):
        # break spin symmetry to let UHF find the lower solution
        dm = umf.get_init_guess()
        try:
            dm[0][0, 0] += 0.1
            dm[1][0, 0] -= 0.1
        except Exception:
            pass
        out['E_UHF'] = float(umf.kernel(dm))
    out['UHF_converged'] = bool(umf.converged)
    _cc(umf, '_U')  # UHF-based
    return out


def sweep(L=2, dim=1, A=2, s_values=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
          verbose=True):
    """Run the coupling sweep; returns a list of per-s result dicts.

    Reference is the exact FCI from the spatial integrals (validated == OpenFermion
    full-Fock ground state for small systems, which is cross-checked when feasible).
    """
    rows = []
    for s in s_values:
        H_ferm, num_sites = build_fermion_H(L, dim, s=s)
        n_orb = 2 * num_sites
        const, h1, eri = spatial_integrals(H_ferm, num_sites)
        e_fci = fci_reference(h1, eri, const, n_orb, A)
        e_of, nexp = exact_E0(H_ferm, A)          # cross-check (small systems only)
        cc_res = run_pyscf_cc(const, h1, eri, n_orb, A)
        row = {
            's': s, 'E_FCI': e_fci,
            'E_OF_xcheck': e_of, 'n_particles': nexp,
            'xcheck_ok': (e_of is None) or (abs(e_of - e_fci) < 1e-5),
            **{k: cc_res[k] for k in cc_res if k.startswith(('E_', 'RHF', 'CCSD'))},
        }
        for key in ('E_CCSD', 'E_CCSD(T)', 'E_CCSD_U', 'E_CCSD(T)_U'):
            if cc_res.get(key) is not None:
                row[f'err_{key}'] = cc_res[key] - e_fci
        rows.append(row)
        if verbose:
            errR = row.get('err_E_CCSD')
            errU = row.get('err_E_CCSD_U')
            print(f"  s={s:4.2f}  E_FCI={e_fci:9.2f}  "
                  f"RHF-CCSD err={('%+8.2f' % errR) if errR is not None else ' DIVERGED':>9} "
                  f"(conv={str(cc_res.get('CCSD_converged'))[0]})   "
                  f"UHF-CCSD err={('%+8.2f' % errU) if errU is not None else ' DIVERGED':>9} "
                  f"(conv={str(cc_res.get('CCSD_U_converged'))[0]})   "
                  f"RHFstable={cc_res.get('RHF_stable')}")
    return rows


def plot_sweep(rows_by_dim, out_png):
    """The D-CC 'no-go' figure: CC error vs coupling for each lattice dim."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ndim = len(rows_by_dim)
    fig, axes = plt.subplots(1, ndim, figsize=(6.2 * ndim, 4.6), squeeze=False)
    for ax, (label, rows) in zip(axes[0], rows_by_dim.items()):
        s = [r['s'] for r in rows]
        def col(k):
            return [abs(r[k]) if r.get(k) is not None else np.nan for r in rows]
        ax.semilogy(s, col('err_E_CCSD'), 'o-', label='RHF-CCSD', color='C3')
        ax.semilogy(s, col('err_E_CCSD(T)'), 's--', label='RHF-CCSD(T)', color='C1')
        ax.semilogy(s, col('err_E_CCSD_U'), '^-', label='UHF-CCSD', color='C0')
        # mark selected CI / FCI as the exact, controlled reference (zero error)
        ax.axhline(1e-3, color='green', lw=1.2, ls=':',
                   label='selected CI / FCI (exact)')
        ax.axvline(1.0, color='gray', lw=1.0, ls='-', alpha=0.6)
        ax.text(1.0, ax.get_ylim()[1], ' physical', color='gray',
                va='top', ha='left', fontsize=8)
        # convergence-failure markers
        for r in rows:
            if r.get('E_CCSD') is None or not _conv(r, 'CCSD_converged'):
                ax.axvspan(r['s'] - 0.03, r['s'] + 0.03, color='red', alpha=0.08)
        ax.set_xlabel('interaction scale  s  (s=1: physical contacts)')
        ax.set_ylabel('|E_method - E_exact|  (MeV)')
        ax.set_title(f'D-CC: coupled-cluster error vs coupling\n{label}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _conv(row, key):
    # convergence flag lives in the cc_res dict, mirrored into the row
    return row.get(key, row.get('CCSD_converged', True))


def run_and_save(L=2, dims=(1, 3), A=2,
                 s_values=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """Run the sweep across lattice dims, save JSON + the no-go figure."""
    import json
    import datetime as _dt   # only for the output folder date stamp
    # date stamp is taken from the filesystem, not Date.now (kept out of logic)
    date = _dt.date.today().isoformat()
    outdir = f"data/classical/baselines/{date}"
    import os
    os.makedirs(outdir, exist_ok=True)

    rows_by_dim = {}
    for dim in dims:
        print(f"\n======== D-CC  L={L}  dim={dim}  A={A} "
              f"(physical s=1.0; C=-51.94, C_I=1.73 MeV) ========")
        rows = sweep(L=L, dim=dim, A=A, s_values=s_values)
        rows_by_dim[f'L={L}, dim={dim} ({L**dim} sites)'] = rows

    stamp = f"cc_reduction_L{L}_A{A}"
    with open(f"{outdir}/{stamp}.json", 'w') as f:
        json.dump({'L': L, 'A': A, 's_values': list(s_values),
                   'rows_by_dim': rows_by_dim}, f, indent=2, default=str)
    png = plot_sweep(rows_by_dim, f"{outdir}/{stamp}.png")
    print(f"\nSaved: {outdir}/{stamp}.json  and  {png}")
    return rows_by_dim, outdir


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="D-CC coupling sweep")
    p.add_argument('--L', type=int, default=2)
    p.add_argument('--dim', type=int, default=None, help='single dim (else 1 and 3)')
    p.add_argument('--A', type=int, default=2)
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    if args.save:
        dims = (args.dim,) if args.dim else (1, 3)
        run_and_save(L=args.L, dims=dims, A=args.A)
    else:
        print(f"D-CC fermionic-reduction sweep: L={args.L} dim={args.dim or 1} A={args.A}")
        print(f"(physical coupling = s=1.0; C=-51.94, C_I=1.73 MeV)")
        sweep(L=args.L, dim=args.dim or 1, A=args.A)
