"""Classical-side γ hook (total_costs §4 item #4): the TRUE QPE warm-start success probability.

For a small, ED-tractable (L, dim, n_b, A): build the bare H, get the EXACT ground EIGENSPACE |g_k>
(the honest reference), then measure the ground-manifold success probability of three input states:

    p0_cold      = max_det Σ_k |g_k[det]|²           best SINGLE-determinant cold start (no frame)
    p0_bare_core = Σ_k |⟨g_k | ψ_bare_D⟩|²           bare-H core of D dets (no frame, U=I)
    p0_warm      = Σ_k |⟨g_k | U | ψ̃_D⟩|²            frame core of D dets through the squeeze U

QPE succeeds on ANY ground eigenstate, so p0 projects onto the whole degenerate manifold, not one
vector (A=1 is lattice-symmetry degenerate). `U|ψ̃⟩` is applied with `expm_multiply` on the SPARSE
squeeze generator (never the dense N×N U): the truncated per-mode squeeze exp(anti-Hermitian) is
exactly unitary, so ‖U ψ̃‖²=‖ψ̃‖² and `p0 = ‖G†(Uψ̃)‖²/‖ψ̃‖²` (G = orthonormal manifold). Dense eigh
for small bases; sparse eigsh for larger (n_b≥2 multi-site). The one new classical *measurement*
feeding `gsee_total_cost`. Honest-claim: ED-exact here (genuine); larger L is the self-referential
proxy (uses `frame_qpe.warmstart_fidelity`, the sparse bilinear, when g is unavailable).

    python -m misc.run_warmstart_fidelity --L 2 --dim 1 --n_b 2 --A 1 --cores 200,800,3200
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci import build_from_eft, frame, frame_workflow
from classical.trimci.state import enumerate_basis
from classical.trimci.hij import build_dense, connections_nocache
from classical.trimci.back_evaluate import state_dict_from_result


def _sparse_matrix(op, basis, index):
    """CSR of a MixedH `op` over `basis` (works for H or the squeeze generator)."""
    rows, cols, data = [], [], []
    for j, sj in enumerate(basis):
        for si, val in connections_nocache(op, sj).items():
            i = index.get(si)
            if i is not None:
                rows.append(i); cols.append(j); data.append(val)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex)


def _ground_manifold(H, n_elec, index, basis, max_basis, degen_tol=1e-6, k_sparse=16):
    """Exact ground manifold `G` (N×d orthonormal) + per-det weight Σ_k|g_k[i]|² + E0 + degeneracy.
    Dense eigh below `max_basis`; sparse eigsh above."""
    N = len(basis)
    if N <= max_basis:
        w, V = np.linalg.eigh(build_dense(H, basis))
    else:
        M = _sparse_matrix(H, basis, index)
        M = 0.5 * (M + M.getH())
        w, V = eigsh(M, k=min(k_sparse, N - 2), which="SA")
        o = np.argsort(w); w, V = w[o], V[:, o]
    sel = np.abs(w - w[0]) < degen_tol
    if sel.all():
        print(f"  WARNING: all {len(w)} computed vectors are within degen_tol of E0 — the ground "
              f"manifold may exceed k_sparse={k_sparse}; raise it for a correct p0.")
    # Orthonormalize the manifold: eigsh can return near-degenerate vectors that are NOT mutually
    # orthonormal, which would make Σ_k|⟨g_k|x⟩|² over-count (p0>1). QR gives a true orthonormal
    # basis of the ground eigenspace (a harmless no-op for dense eigh's already-orthonormal vectors).
    G, _ = np.linalg.qr(V[:, sel])
    return G, np.sum(np.abs(G) ** 2, axis=1), float(w[0]), int(G.shape[1])


def _vec(psi, index, N):
    v = np.zeros(N, complex)
    for s, c in psi.items():
        v[index[s]] = c
    return v


def _p0(G, vec):
    nv = float(np.vdot(vec, vec).real)
    proj = G.conj().T @ vec
    return float(np.vdot(proj, proj).real / nv) if nv > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Classical-side warm-start fidelity (true p0 = γ²)")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n_b", type=int, default=1)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--filling", type=float, default=None)
    ap.add_argument("--cores", default="200,800,3200")
    ap.add_argument("--frame-runs", type=int, default=16)
    ap.add_argument("--max-basis", type=int, default=6000, help="dense eigh below, sparse eigsh above")
    ap.add_argument("--max-basis-hard", type=int, default=300000, help="refuse ED above this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    cores = [int(c) for c in str(args.cores).split(",") if c]

    Hbare = build_from_eft(args.L, args.dim, args.n_b, transform="bare")
    N_f, n_bos = Hbare.N_f, Hbare.n_bos_modes
    r, phi = frame.analytic_squeeze(Hbare)
    r_norm = float(np.linalg.norm(np.asarray(r, dtype=float)))

    basis = enumerate_basis(Hbare.n_ferm_modes, n_bos, N_f, A)
    N = len(basis)
    if N > args.max_basis_hard:
        raise SystemExit(f"basis {N:,} > hard cap {args.max_basis_hard:,}: not ED-tractable")
    index = {s: i for i, s in enumerate(basis)}
    G, per_det, E0, degen = _ground_manifold(Hbare, A, index, basis, args.max_basis)
    p0_cold = float(per_det.max())
    Gsq = _sparse_matrix(frame.squeeze_generator_terms(Hbare, r, phi), basis, index)  # sparse U gen

    rows = []
    for core in cores:
        _Hf, res_f = frame_workflow.optimize_frame(Hbare, A, core, has_gaussian=True,
                                                   num_runs=args.frame_runs, cycles=1, seed=0)
        _Hb, res_b = frame_workflow.optimize_frame(Hbare, A, core, has_gaussian=False,
                                                   num_runs=args.frame_runs, cycles=1, seed=0)
        pv_f = _vec(state_dict_from_result(res_f), index, N)
        pv_b = _vec(state_dict_from_result(res_b), index, N)
        p0_warm = _p0(G, expm_multiply(Gsq, pv_f))         # U|ψ̃⟩ via sparse expm_multiply
        p0_bare = _p0(G, pv_b)                              # bare core, U=I
        rows.append({"core": core, "D_warm": int(np.count_nonzero(pv_f)),
                     "D_bare": int(np.count_nonzero(pv_b)), "E_frame": float(res_f.energy),
                     "E_bare": float(res_b.energy), "p0_bare_core": p0_bare, "p0_warm": p0_warm,
                     "reps_ratio_vs_cold": (p0_cold / p0_warm) if p0_warm > 0 else None})

    out = {"kind": "warmstart_fidelity", "L": args.L, "dim": args.dim, "n_b": args.n_b, "A": A,
           "sites": sites, "N_f": N_f, "n_bos_modes": n_bos, "basis_dim": N,
           "ground_degeneracy": degen, "E0_exact": E0, "r_norm": r_norm,
           "p0_cold_bestdet": p0_cold, "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)

    print(f"[warmfid] L={args.L} d{args.dim} n_b={args.n_b} A={A} basis={N} "
          f"degeneracy={degen} E0={E0:.3f} r_norm={r_norm:.3f}")
    print(f"  p0_cold (best single det) = {p0_cold:.4e}")
    print(f"  {'D':>7} {'p0_bare_core':>13} {'p0_warm(frame)':>15} {'warm/cold (fewer reps x)':>24}")
    for rw in rows:
        rr = rw["reps_ratio_vs_cold"]
        print(f"  {rw['D_warm']:>7} {rw['p0_bare_core']:>13.4e} {rw['p0_warm']:>15.4e} "
              f"{('%.1fx' % (1.0/rr)) if rr else 'n/a':>24}")


if __name__ == "__main__":
    main()
