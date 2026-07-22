"""
Exact N_f-convergence study via sparse Lanczos.

Pushes the exact reference up to the *real* per-mode Fock cutoff (the heuristic
`estimate_boson_cutoff`, e.g. N_f=32 for A=1) and reports how the exact
ground-state energy converges as N_f grows. Two payoffs:

  1. It demonstrates Lanczos reaching real N_f (single-site A=1 is ~1.3e5
     states at N_f=32 — far past dense ED's ~6000, still exact).
  2. The E0(N_f) curve is the *empirical* cutoff-convergence check that the
     rigorous Tong bound (open task 25) is meant to certify: it shows how many
     pion Fock levels the energy actually needs, vs the (over-padded) heuristic.

REACH CEILING (be honest about it): at real N_f only a SINGLE lattice site is
exactly diagonalizable. L=1 → 4 fermion modes × 3 pion modes, so A=1 is
4·N_f^3 states (1.3e5 at N_f=32). L=2 explodes to N_f^6·C(8,A) (8.6e9 at
N_f=32) — hopeless for any exact method, which is exactly the regime TrimCI
exists for. So this study runs single-site by default.

Run:
    python -m classical.trimci.nf_convergence                  # L=1,A=1 -> real N_f
    python -m classical.trimci.nf_convergence --A 2 --trimci
"""

from __future__ import annotations

import argparse
import time

from .hamiltonian import build_from_eft
from .lanczos import lanczos_ground_state
from .state import basis_size


def _real_n_b(L, dim, A):
    """The heuristic per-mode cutoff exponent (N_f = 2**n_b)."""
    from src_PI.hamiltonians.core.EFTParameters import (
        get_physical_parameters, estimate_boson_cutoff)
    n_b, _pi, _Pi = estimate_boson_cutoff(L, dim, A, get_physical_parameters())
    return n_b


def nf_convergence(L=1, dim=1, A=1, n_b_max=None, max_states=400_000,
                   trimci=False, verbose=True):
    """Sweep N_f = 2..2**n_b_max, exact E0 via Lanczos at each. Returns rows."""
    real_nb = _real_n_b(L, dim, A)
    if n_b_max is None:
        n_b_max = real_nb

    if verbose:
        n_sites = L ** dim
        print("=" * 74)
        print(f"  N_f CONVERGENCE  (L={L}, dim={dim}, A={A}, {n_sites} site(s))")
        print(f"  real heuristic cutoff: n_b={real_nb} -> N_f={2**real_nb}")
        print("=" * 74)
        print(f"  {'n_b':>3} {'N_f':>4} {'n_states':>10} {'E0 (MeV)':>16} "
              f"{'dE vs prev':>13} {'nnz':>10} {'t(s)':>7}")
        print("  " + "-" * 70)

    rows = []
    prev_E = None
    for n_b in range(1, n_b_max + 1):
        N_f = 2 ** n_b
        size = basis_size(4 if L == 1 else 4 * (L ** dim),
                          3 * (L ** dim), N_f, A)
        if size > max_states:
            if verbose:
                print(f"  {n_b:>3} {N_f:>4} {size:>10,}  -- skipped "
                      f"(> max_states={max_states:,}; exact reach exceeded)")
            continue

        H = build_from_eft(L, dim, n_b)
        t0 = time.time()
        E0, info = lanczos_ground_state(H, n_elec=A, max_states=max_states)
        dt = time.time() - t0
        dE = (E0 - prev_E) if prev_E is not None else None
        rows.append({"n_b": n_b, "N_f": N_f, "n_states": info["n_states"],
                     "E0": E0, "dE": dE, "nnz": info["nnz"], "t": dt,
                     "method": info["method"]})
        if verbose:
            de_s = f"{dE:>13.2e}" if dE is not None else f"{'--':>13}"
            nnz_s = f"{info['nnz']:>10,}" if info["nnz"] else f"{'(dense)':>10}"
            print(f"  {n_b:>3} {N_f:>4} {info['n_states']:>10,} {E0:>16.8f} "
                  f"{de_s} {nnz_s} {dt:>7.1f}")
        prev_E = E0

    if verbose and len(rows) >= 2:
        print("  " + "-" * 70)
        # report where the energy settled below a few thresholds
        for thr in (1.0, 1e-2, 1e-4):
            conv = next((r["N_f"] for r in rows[1:]
                         if r["dE"] is not None and abs(r["dE"]) < thr), None)
            tag = f"N_f={conv}" if conv else "not within sweep"
            print(f"  |dE| < {thr:>7g} MeV first reached at: {tag}")
        print(f"  heuristic cutoff would use N_f={2**real_nb} "
              f"(over-padding visible if convergence is earlier).")

    # optional TrimCI cross-check at the largest N_f reached
    if trimci and rows:
        from .graph import ground_state_ensemble
        top = rows[-1]
        H = build_from_eft(L, dim, top["n_b"])
        if verbose:
            print("  " + "-" * 70)
            print(f"  TrimCI cross-check at N_f={top['N_f']} "
                  f"({top['n_states']:,} states):")
        res = ground_state_ensemble(H, n_elec=A, n_runs=6,
                                    n_dets=min(500, top["n_states"]), seed=0)
        dE = res.energy - top["E0"]
        if verbose:
            print(f"  TrimCI({res.n_dets} dets, "
                  f"{res.n_dets / top['n_states']:.2%} of sector): "
                  f"E={res.energy:.6f}  dE above exact = {dE:+.2e} MeV")
        top["trimci_E"] = res.energy
        top["trimci_dets"] = res.n_dets

    if verbose:
        print("=" * 74)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Exact N_f-convergence via Lanczos")
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--n_b_max", type=int, default=None,
                    help="max cutoff exponent (default: the real heuristic)")
    ap.add_argument("--max_states", type=int, default=400_000)
    ap.add_argument("--trimci", action="store_true",
                    help="cross-check TrimCI at the largest N_f reached")
    args = ap.parse_args()
    nf_convergence(L=args.L, dim=args.dim, A=args.A, n_b_max=args.n_b_max,
                   max_states=args.max_states, trimci=args.trimci)


if __name__ == "__main__":
    main()
