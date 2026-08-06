"""Architecture-B smoke + resource comparison (task 34, Stage 2).

Qubitizes the Gaussian-SQUEEZED Fock Hamiltonian (`config.pion_basis='fock_squeezed'`)
and compares its quantum resources to the bare `fock` walk at matched (L, A, n_b).
Because the squeeze is canonical (isospectral), QPE returns spec(H) exactly while the
walk encodes the more-compact H_sq — the point of Architecture B.

What it does:
  1. `--lambda-sweep`: cheap Λ(r) scan (build + normalize only, no pyLIQTR) — shows how
     the squeeze shifts the block-encoding 1-norm and where Λ is minimized.
  2. resource compare: full `evaluate_resources` (pyLIQTR pauli_lcu) for bare fock vs
     fock_squeezed at r∈{r*_classical, Λ-min} — Λ, T_step, logical qubits, QPE total-T.
  3. `--isospectral-check`: sparse-eigsh low-spectrum of fock vs fock_squeezed(r) at
     L=2 d=1 (the full H incl. gradient+AV) — confirms the walk encodes H_sq=H.
  4. timing per stage -> local (<10 min) vs cluster recommendation.

Run:
  python -m misc.run_frame_arch_B --L 2 --dim 3 --n-b 2 --lambda-sweep
  python -m misc.run_frame_arch_B --isospectral-check     # L=2 d=1 validation
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.utils.Config import Config


def classical_squeeze_r(L, dim, n_b, verbose=True):
    """Representative scalar squeeze amplitude r* from the classical analytic squeeze
    (per-mode); returns the median |r| over squeezed modes (the translation-symmetric
    bulk value). Lazy import — classical may read src_PI, and this driver bridges."""
    from classical.trimci import frame
    from classical.trimci.hamiltonian import build_from_eft
    H = build_from_eft(L, dim, n_b)
    r, _phi = frame.analytic_squeeze(H)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    nz = np.abs(r[np.abs(r) > 1e-9])
    r_star = float(np.median(nz)) if nz.size else 0.0
    if verbose:
        print(f"[B] classical analytic_squeeze: {nz.size}/{r.size} modes squeezed, "
              f"|r| range [{nz.min() if nz.size else 0:.4f}, {nz.max() if nz.size else 0:.4f}], "
              f"median r*={r_star:.4f}")
    return r_star


def lambda_only(L, dim, n_b, params, pion_basis):
    """Physical Λ (block-encoding 1-norm) only — build + normalize, NO pyLIQTR."""
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
    config = Config(pion_basis=pion_basis, block_encoder='pauli_lcu', walk_mode='series')
    bundle, _q, _ns = build_eft_hamiltonian(L, dim, n_b, 0.0, params, config)
    return normalize_for_qpe(bundle)['physical_lambda']


def resource_point(L, dim, n_b, params, pion_basis):
    """Full resource estimate (pyLIQTR pauli_lcu) -> {Λ, T_step, qubits, QPE_T}."""
    from src_PI.estimation.EstimateResources import evaluate_resources
    from src_PI.estimation.qpe_cost import total_qpe_t_count
    config = Config(pion_basis=pion_basis, block_encoder='pauli_lcu', walk_mode='series')
    norm = evaluate_resources(L, dim, n_b, 0.0, params, config)
    lam, t_step = norm['Physical_Lambda'], norm['Total_T_Count']
    return {'physical_lambda': lam, 'total_t_count': t_step,
            'logical_qubits': norm['Logical_Qubits'],
            'qpe_total_t': total_qpe_t_count(t_step, lam)}


def isospectral_check(L, dim, n_b, r, k=4):
    """Low-spectrum |ΔE| of fock vs fock_squeezed(r) over the full qubit space
    (static-nucleon JW + pion sector) via sparse eigsh — the full-H isospectrality."""
    import scipy.sparse.linalg as sla
    from openfermion import get_sparse_operator, jordan_wigner
    from src_PI.hamiltonians.core.pion_basis import fock, fock_squeezed
    from src_PI.hamiltonians.core.StaticTerms import Static_Nucleon_Hamiltonian
    from src_PI.utils.LatticeGeometry import total_qubits

    def full(mod, rr):
        p = dict(get_physical_parameters()); p['squeeze_r'] = rr
        stat = jordan_wigner(Static_Nucleon_Hamiltonian(p['h'], p['C'], p['CI'], L, dim, n_b))
        return stat + mod.Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, 0.0, p)[0][1]

    nq = total_qubits(L, dim, n_b)

    def low(qop):
        M = get_sparse_operator(qop, n_qubits=nq)
        M = (M + M.getH()) * 0.5
        return np.sort(sla.eigsh(M, k=k, which='SA', return_eigenvectors=False))

    e_bare = low(full(fock, 0.0))
    e_sq = low(full(fock_squeezed, r))
    return float(np.max(np.abs(e_bare - e_sq))), e_bare, e_sq, nq


def main():
    ap = argparse.ArgumentParser(description="Architecture-B squeeze-walk resource smoke.")
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--n-b', type=int, default=2)
    ap.add_argument('--lambda-sweep', action='store_true',
                    help="cheap Λ(r) scan (no pyLIQTR)")
    ap.add_argument('--r-values', type=float, nargs='*',
                    default=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    ap.add_argument('--isospectral-check', action='store_true',
                    help="full-H isospectrality at L=2 d=1 (sparse eigsh)")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    t_all = time.time()
    report = {'L': args.L, 'dim': args.dim, 'n_b': args.n_b, 'stages': {}}

    # --- isospectrality (own small system) --------------------------------
    if args.isospectral_check:
        print("\n[B] isospectrality check (L=2 d=1, full H incl. gradient+AV) ...")
        t0 = time.time()
        r_iso = classical_squeeze_r(2, 1, args.n_b, verbose=False) or 0.15
        err, eb, esq, nq = isospectral_check(2, 1, args.n_b, r_iso)
        dt = time.time() - t0
        print(f"[B]  {nq} qubits, r={r_iso:.4f}: E0(bare)={eb[0]:.4f}  E0(sq)={esq[0]:.4f}  "
              f"max|ΔE|_lowk={err:.3e} MeV  ({dt:.0f}s)")
        print(f"[B]  -> {'ISOSPECTRAL' if err < 5e-2 else 'MISMATCH'} "
              f"(canonical squeeze preserves the spectrum)")
        report['stages']['isospectral'] = {'r': r_iso, 'max_dE': err, 'seconds': dt,
                                            'qubits': nq}

    # --- classical r* -----------------------------------------------------
    r_star = classical_squeeze_r(args.L, args.dim, args.n_b)
    report['r_star_classical'] = r_star

    # --- cheap Λ(r) sweep -------------------------------------------------
    r_lambda_min = r_star
    if args.lambda_sweep:
        print(f"\n[B] Λ(r) sweep (L={args.L} d={args.dim} n_b={args.n_b}, no pyLIQTR):")
        t0 = time.time()
        params = dict(get_physical_parameters())
        lam_bare = lambda_only(args.L, args.dim, args.n_b, {**params, 'squeeze_r': 0.0},
                               'fock')
        sweep = []
        for r in args.r_values:
            lam = lambda_only(args.L, args.dim, args.n_b, {**params, 'squeeze_r': r},
                              'fock_squeezed')
            sweep.append({'r': r, 'lambda': lam, 'ratio_vs_bare': lam / lam_bare})
            print(f"[B]   r={r:+.2f}  Λ={lam:.4e}  ({lam/lam_bare:.3f}× bare)")
        best = min(sweep, key=lambda s: s['lambda'])
        r_lambda_min = best['r']
        dt = time.time() - t0
        print(f"[B]  bare fock Λ={lam_bare:.4e}; Λ-min at r={best['r']:+.2f} "
              f"({best['ratio_vs_bare']:.3f}× bare)  ({dt:.0f}s)")
        report['stages']['lambda_sweep'] = {'lambda_bare': lam_bare, 'points': sweep,
                                            'seconds': dt}

    # --- full resource compare (bare vs squeezed) -------------------------
    print(f"\n[B] full resource estimate (pyLIQTR pauli_lcu, L={args.L} d={args.dim} "
          f"n_b={args.n_b}) ...")
    params = dict(get_physical_parameters())
    rows = {}
    for label, basis, r in [('bare_fock', 'fock', 0.0),
                            ('squeezed@r*', 'fock_squeezed', r_star),
                            ('squeezed@Λmin', 'fock_squeezed', r_lambda_min)]:
        t0 = time.time()
        pt = resource_point(args.L, args.dim, args.n_b, {**params, 'squeeze_r': r}, basis)
        pt['seconds'] = time.time() - t0
        pt['r'] = r
        rows[label] = pt
        print(f"[B]   {label:16} r={r:+.2f}  Λ={pt['physical_lambda']:.3e}  "
              f"T_step={pt['total_t_count']:.3e}  qubits={pt['logical_qubits']}  "
              f"QPE_T={pt['qpe_total_t']:.3e}  ({pt['seconds']:.0f}s)")
    report['stages']['resources'] = rows

    # --- verdict + local/cluster --------------------------------------
    b, s = rows['bare_fock'], rows.get('squeezed@Λmin', rows['squeezed@r*'])
    print(f"\n[B] squeeze vs bare (same n_b): Λ {s['physical_lambda']/b['physical_lambda']:.3f}×, "
          f"QPE_T {s['qpe_total_t']/b['qpe_total_t']:.3f}×, "
          f"qubits {s['logical_qubits']}-{b['logical_qubits']}")
    total = time.time() - t_all
    max_stage = max((v.get('seconds', 0) for st in report['stages'].values()
                     for v in ([st] if 'seconds' in st else st.values())
                     if isinstance(v, dict) or 'seconds' in st), default=0) \
        if report['stages'] else 0
    slowest = max([pt['seconds'] for pt in rows.values()] +
                  ([report['stages']['isospectral']['seconds']] if args.isospectral_check else []))
    print(f"\n[B] total wall {total:.0f}s; slowest single estimate {slowest:.0f}s.")
    print(f"[B]  -> {'LOCAL OK (<10 min/point)' if slowest < 600 else 'CLUSTER (slow points)'}")
    report['wall_s'] = total
    report['slowest_point_s'] = slowest

    outdir = 'data/quantum/2026-08-06'
    os.makedirs(outdir, exist_ok=True)
    outpath = args.out or os.path.join(outdir, f'frame_arch_B_L{args.L}d{args.dim}_nb{args.n_b}.json')
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[B] wrote -> {outpath}")


if __name__ == '__main__':
    main()
