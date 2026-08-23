"""Classical-side γ hook (total_costs §4 item #4): the TRUE QPE warm-start success probability.

For a small, ED-tractable (L, dim, n_b, A): build the bare H, get the EXACT ground state |g> by
dense diagonalization (the honest reference the honest-claim policy requires), then measure

    p0_cold  = max_i |g_i|²                         best SINGLE-determinant cold start (no frame)
    p0_warm(D) = |⟨g | U | ψ̃_D⟩|²                    full frame core of D dets, through the squeeze U

as a function of the loaded determinant count D (a compact selected-CI frame solve at growing core).
Reports p0_cold, p0_warm(D), the repetition-reduction ratio p0_warm/p0_cold, and D. This is the one
new classical *measurement* that feeds the quantum-cost assembler (state_prep_cost / repetition
factor / total_gsee_cost). ED-only by design: |g> must be exact, so this runs where ED fits; larger
L is the self-referential proxy (separate), per the honest-claim policy.

    python -m misc.run_warmstart_fidelity --L 2 --dim 3 --n_b 1 --A 1 --cores 200,800,3200
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.linalg import expm

from classical.trimci import build_from_eft, frame, frame_workflow
from classical.trimci.state import enumerate_basis
from classical.trimci.hij import build_dense
from classical.trimci.back_evaluate import state_dict_from_result


def _ground_eigenspace(H, n_elec, max_basis, degen_tol=1e-6):
    """Exact ground EIGENSPACE (columns G) + basis via dense eigh (guarded). QPE succeeds on ANY
    ground eigenstate, so at A=1 the lattice-symmetry degeneracy MUST be projected as a subspace,
    not a single vector (two independent solves pick different degenerate members -> false 0)."""
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec)
    if len(basis) > max_basis:
        raise MemoryError(f"basis {len(basis)} > cap {max_basis}: not ED-tractable, pick a smaller case")
    w, V = np.linalg.eigh(build_dense(H, basis))
    E0 = float(w[0])
    G = V[:, np.abs(w - w[0]) < degen_tol]        # degenerate ground manifold (basis x d)
    return basis, G, E0, len(basis)


def _manifold_weight(G, vec):
    """Ground-eigenspace success probability of a (dense) state: ‖Gᴴ vec‖² / ‖vec‖²."""
    nv = float(np.vdot(vec, vec).real)
    return float(np.vdot(G.conj().T @ vec, G.conj().T @ vec).real / nv) if nv > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Classical-side warm-start fidelity (true p0 = γ²)")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n_b", type=int, default=1)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--filling", type=float, default=None,
                    help="if set, A = round(filling * sites)")
    ap.add_argument("--cores", default="200,800,3200",
                    help="comma-list of frame-solve core sizes (the loaded-D sweep)")
    ap.add_argument("--frame-runs", type=int, default=16)
    ap.add_argument("--max-basis", type=int, default=200000,
                    help="refuse ED above this basis dimension (memory guard)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sites = args.L ** args.dim
    A = max(1, round(args.filling * sites)) if args.filling is not None else args.A
    cores = [int(c) for c in str(args.cores).split(",") if c]

    Hbare = build_from_eft(args.L, args.dim, args.n_b, transform="bare")
    N_f, n_bos = Hbare.N_f, Hbare.n_bos_modes
    r, phi = frame.analytic_squeeze(Hbare)
    r_norm = float(np.linalg.norm(np.asarray(r, dtype=float)))

    basis, G, E0, dim_basis = _ground_eigenspace(Hbare, A, args.max_basis)
    degen = int(G.shape[1])
    idx = {s: i for i, s in enumerate(basis)}
    Udense = expm(build_dense(frame.squeeze_generator_terms(Hbare, r, phi), basis))  # U = exp(G_sq)

    # cold start = the single determinant with the MOST ground-manifold weight (best cold guess).
    per_det = np.sum(np.abs(G) ** 2, axis=1)               # Σ_k |g_k[det]|² per determinant
    p0_cold = float(per_det.max())

    rows = []
    for core in cores:
        _Hf, res = frame_workflow.optimize_frame(
            Hbare, A, core, has_gaussian=True, has_lf=False, has_coo=False,
            num_runs=args.frame_runs, cycles=1, seed=0)
        psi = state_dict_from_result(res)
        D = len(psi)
        pv = np.zeros(len(basis), complex)                 # frame core -> dense vector
        for s, c in psi.items():
            pv[idx[s]] = c
        p0_warm = _manifold_weight(G, Udense @ pv)         # ‖P_gs U|ψ̃⟩‖² — QPE success prob
        p0_bare_core = _manifold_weight(G, pv)             # same core, NO frame (U=I): isolates U's value
        rows.append({"core": core, "D": D, "E_frame": float(res.energy),
                     "p0_warm": p0_warm, "p0_bare_core": p0_bare_core,
                     "reps_ratio_vs_cold": (p0_cold / p0_warm) if p0_warm > 0 else None})

    out = {"kind": "warmstart_fidelity", "L": args.L, "dim": args.dim, "n_b": args.n_b, "A": A,
           "sites": sites, "N_f": N_f, "n_bos_modes": n_bos, "basis_dim": dim_basis,
           "ground_degeneracy": degen, "E0_exact": E0, "r_norm": r_norm,
           "p0_cold_bestdet": p0_cold, "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)

    print(f"[warmfid] L={args.L} d{args.dim} n_b={args.n_b} A={A} basis={dim_basis} "
          f"degeneracy={degen} E0={E0:.3f} r_norm={r_norm:.3f}")
    print(f"  p0_cold (best single det -> ground manifold) = {p0_cold:.4e}")
    print(f"  {'D':>7} {'p0_bare_core':>13} {'p0_warm(frame)':>15} {'warm/cold (fewer reps x)':>24}")
    for rw in rows:
        rr = rw["reps_ratio_vs_cold"]
        print(f"  {rw['D']:>7} {rw['p0_bare_core']:>13.4e} {rw['p0_warm']:>15.4e} "
              f"{('%.1fx' % (1.0/rr)) if rr else 'n/a':>24}")


if __name__ == "__main__":
    main()
