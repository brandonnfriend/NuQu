"""
D-DMRG (part 2): actual DMRG on the mixed fermion-boson H via block2.

Builds the project's exact mixed Hamiltonian (`classical.trimci.build_from_eft`)
as a block2 custom mixed fermion-boson MPO and runs DMRG, sweeping the bond
dimension chi. Two jobs:
  * reach the 3D points (L=2^3, L=3^3) that exact Lanczos cannot (2^(3*sites)
    boson Fock space) -- the geometry where the wall lives;
  * L=2 3D also serves as the INDEPENDENT cross-check of the TrimCI selected-CI
    reference (the "double duty" / Phase-D stretch item).

Encoding. Each lattice site -> 4 fermion modes (spin x isospin, 2-dim each) + 3
pion modes (N_f-dim each), laid out contiguously: block2 site index
    fermion mode m -> (m//4)*7 + (m%4)
    boson   mode m -> (m//3)*7 + 4 + (m%3)
Single U(1) fermion-number symmetry (`U1Fermi`, carries the JW statistics);
bosons sit at charge 0 (bosonic). Complex coefficients (H_WT) via SAny|CPX.
Fermion signs are handled by block2 (`finalize(fermionic_ops="CD")`).

VALIDATION GATE: at L=2 (1D), block2 DMRG must reproduce the exact Lanczos ground
energy. Only then are the 3D numbers trustworthy.
"""

from __future__ import annotations

import numpy as np

_DRIVER_COUNT = 0     # for unique block2 scratch dirs across runs


def _fermion_ops():
    I = np.eye(2)
    C = np.zeros((2, 2)); C[1, 0] = 1.0     # create: |0> -> |1>
    D = np.zeros((2, 2)); D[0, 1] = 1.0     # annihilate: |1> -> |0>
    return {"": I, "C": C, "D": D}


def _boson_ops(N_f):
    I = np.eye(N_f)
    B = np.zeros((N_f, N_f))                 # b:  |n> -> sqrt(n)|n-1>
    A = np.zeros((N_f, N_f))                 # b†: |n> -> sqrt(n+1)|n+1>
    for n in range(1, N_f):
        B[n - 1, n] = np.sqrt(n)
        A[n, n - 1] = np.sqrt(n)
    return {"": I, "B": B, "A": A}


def build_mpo(H, L, dim, A, N_f, driver=None, iprint=0):
    """Construct (driver, mpo, constant) for the mixed H over the A-nucleon sector."""
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    import os
    num_sites = L ** dim
    n_b2 = 7 * num_sites
    if driver is None:
        # unique scratch per driver so concurrent/repeated block2 runs never collide
        global _DRIVER_COUNT
        _DRIVER_COUNT += 1
        scratch = f"data/.block2_scratch/pid{os.getpid()}_{L}d{dim}_{_DRIVER_COUNT}"
        os.makedirs(scratch, exist_ok=True)
        driver = DMRGDriver(symm_type=SymmetryTypes.SAny | SymmetryTypes.CPX,
                            n_threads=4, scratch=scratch, stack_mem=4 << 30)
    driver.set_symmetry_groups("U1Fermi")
    Q = driver.bw.SX

    site_basis, site_ops = [], []
    for s in range(num_sites):
        for _ in range(4):
            site_basis.append([(Q(0), 1), (Q(1), 1)])
            site_ops.append(_fermion_ops())
        for _ in range(3):
            site_basis.append([(Q(0), N_f)])
            site_ops.append(_boson_ops(N_f))

    driver.initialize_system(n_b2, vacuum=Q(0), target=Q(A), hamil_init=False)
    driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)

    def fidx(m):
        return (m // 4) * 7 + (m % 4)

    def bidx(m):
        return (m // 3) * 7 + 4 + (m % 3)

    b = driver.expr_builder()
    for t in H.terms:
        expr, idx = "", []
        for (m, a) in t.ferm_ops:
            expr += "C" if a == 1 else "D"
            idx.append(fidx(m))
        for (m, a) in t.bos_ops:
            expr += "A" if a == 1 else "B"
            idx.append(bidx(m))
        if expr == "":
            continue                          # constant tracked separately
        b.add_term(expr, idx, complex(t.coeff))

    mpo = driver.get_mpo(b.finalize(fermionic_ops="CD"), iprint=iprint)
    return driver, mpo, H.constant()


def run_dmrg(L, dim, A, N_f=2, n_b=1, bond_dims=(20, 40, 80, 160, 320),
             n_sweeps_per=4, iprint=0):
    """DMRG energy vs bond dimension chi for the mixed H. Returns list of dicts."""
    from classical.trimci import build_from_eft
    H = build_from_eft(L=L, dim=dim, n_b=n_b, N_f=N_f)
    driver, mpo, const = build_mpo(H, L, dim, A, N_f, iprint=iprint)

    out = []
    mps = driver.get_random_mps("KET", bond_dim=bond_dims[0], nroots=1)
    for chi in bond_dims:
        bdims = [chi] * n_sweeps_per
        noises = [1e-4] * (n_sweeps_per - 1) + [0.0]
        thrds = [1e-8] * n_sweeps_per
        e = driver.dmrg(mpo, mps, n_sweeps=n_sweeps_per, bond_dims=bdims,
                        noises=noises, thrds=thrds, iprint=iprint)
        # bond entropy (max over bonds) if available
        try:
            bd = driver.get_bipartite_entanglement()
            smax = float(np.max(bd))
        except Exception:
            smax = None
        out.append({'chi': chi, 'E': float(np.real(e)) + float(np.real(const)),
                    'S_max_bond': smax})
    return out, H


def validate_vs_lanczos(L=2, dim=1, A=2, N_f=2, tol=1e-4):
    """Correctness gate: block2 DMRG == exact Lanczos at a small system."""
    from classical.trimci import build_from_eft
    from classical.trimci.lanczos import lanczos_ground_state
    H = build_from_eft(L=L, dim=dim, n_b=1, N_f=N_f)
    E_lanc, _ = lanczos_ground_state(H, A)
    res, _ = run_dmrg(L, dim, A, N_f=N_f, bond_dims=(50, 100, 200), n_sweeps_per=6)
    E_dmrg = min(r['E'] for r in res)
    ok = abs(E_dmrg - E_lanc) < tol
    print(f"  L={L} d={dim} A={A} N_f={N_f}: Lanczos={E_lanc:.6f}  "
          f"DMRG={E_dmrg:.6f}  diff={abs(E_dmrg-E_lanc):.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok, E_lanc, E_dmrg


def ladder_and_crosscheck(A=2, N_f=2, bond_dims=(20, 40, 80, 160),
                          n_sweeps_per=5, save=True):
    """Run the geometry ladder (1D->2D->3D at L=2), cross-check L=2^3 against
    TrimCI selected CI, and save the no-go figure + data."""
    import time
    from classical.trimci import build_from_eft
    from classical.trimci.run_cpp import cpp_ground_state_ensemble

    geoms = [(2, 1, 'L=2 1D'), (2, 2, 'L=2 2D'), (2, 3, 'L=2 3D')]
    data = {}
    for (L, dim, label) in geoms:
        t = time.time()
        res, _ = run_dmrg(L, dim, A, N_f=N_f, bond_dims=bond_dims,
                          n_sweeps_per=n_sweeps_per)
        data[label] = {'L': L, 'dim': dim, 'sites': L ** dim,
                       'wall_s': time.time() - t, 'sweep': res}
        print(f"{label}: E(chi={bond_dims[-1]})={res[-1]['E']:.4f}  "
              f"({data[label]['wall_s']:.1f}s)", flush=True)

    # independent 3D cross-check: TrimCI selected CI at L=2 3D (needs n_runs>=16)
    H = build_from_eft(L=2, dim=3, n_b=1, N_f=N_f)
    tc = cpp_ground_state_ensemble(H, n_elec=A, n_dets=8000, n_runs=16, seed=0)
    e_dmrg = data['L=2 3D']['sweep'][-1]['E']
    data['crosscheck_L2_3D'] = {
        'block2_dmrg': e_dmrg, 'trimci_sci': float(tc.energy),
        'gap_MeV': float(tc.energy) - e_dmrg,
        'gap_MeV_per_site': (float(tc.energy) - e_dmrg) / 8.0,
    }
    print(f"\nCross-check L=2^3: block2 DMRG={e_dmrg:.3f}  TrimCI={tc.energy:.3f}  "
          f"gap={data['crosscheck_L2_3D']['gap_MeV_per_site']:.2f} MeV/site", flush=True)
    if save:
        _save_ladder(data, N_f)
    return data


def _save_ladder(data, N_f):
    import json, os, datetime as _dt
    date = _dt.date.today().isoformat()
    outdir = f"data/classical/baselines/{date}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/dmrg_ladder.json", 'w') as f:
        json.dump({'N_f': N_f, 'data': data}, f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
        for label in ('L=2 1D', 'L=2 2D', 'L=2 3D'):
            sw = data[label]['sweep']
            ebest = min(r['E'] for r in sw)
            chi = [r['chi'] for r in sw]
            dE = [max(r['E'] - ebest, 1e-12) for r in sw]
            ax1.semilogy(chi, dE, 'o-', label=f"{label} ({data[label]['sites']} sites)")
        ax1.set_xlabel('bond dimension  chi')
        ax1.set_ylabel('E(chi) - E_best  (MeV)')
        ax1.set_title('DMRG convergence slows with dimension\n(the area-law wall onset)')
        ax1.legend(fontsize=9); ax1.grid(True, which='both', alpha=0.25)
        dims = [data[l]['dim'] for l in ('L=2 1D', 'L=2 2D', 'L=2 3D')]
        walls = [data[l]['wall_s'] for l in ('L=2 1D', 'L=2 2D', 'L=2 3D')]
        ax2.semilogy(dims, walls, 's-', color='C3')
        ax2.set_xticks([1, 2, 3])
        ax2.set_xlabel('spatial dimension (L=2 fixed)')
        ax2.set_ylabel('DMRG wall-clock (s)')
        ax2.set_title('Cost explodes with dimension at fixed L\n(L=3,4 3D -> HPC)')
        ax2.grid(True, which='both', alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{outdir}/dmrg_ladder.png", dpi=130)
        plt.close(fig)
        print(f"Saved: {outdir}/dmrg_ladder.json and dmrg_ladder.png")
    except Exception as e:
        print(f"Saved JSON (plot skipped: {e})")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--ladder', action='store_true')
    p.add_argument('--L', type=int, default=2)
    p.add_argument('--dim', type=int, default=1)
    p.add_argument('--A', type=int, default=2)
    p.add_argument('--N_f', type=int, default=2)
    args = p.parse_args()
    if args.validate:
        validate_vs_lanczos(args.L, args.dim, args.A, args.N_f)
    elif args.ladder:
        ladder_and_crosscheck(A=args.A, N_f=args.N_f)
    else:
        res, _ = run_dmrg(args.L, args.dim, args.A, N_f=args.N_f)
        for r in res:
            print(f"  chi={r['chi']:4d}  E={r['E']:.6f}  S_max={r['S_max_bond']}")
