"""
End-to-end toy TrimCI run via Route A.

Walks the full stated pipeline on a small, ED-tractable system:

    choose (L, d, A, n_b)
        -> build mixed H            (hamiltonian.build_from_eft)
        -> write generalized dump   (dump.write_dump)
        -> read it back             (dump.read_dump)         [proves the dump
                                                              IS the artifact
                                                              driving the solve]
        -> H_ij                     (hij.connections)
        -> TrimCI                   (graph.ground_state)
        -> compare to ED            (graph.exact_ground_state)

and reports the compact-core effect: how few determinants TrimCI needs to
reach the exact ground-state energy, vs the full sector size.

Run:  python -m classical.trimci.run_toy
      python -m classical.trimci.run_toy --L 2 --dim 1 --A 1 --n_b 1
"""

from __future__ import annotations

import argparse
import os
import tempfile
from math import comb

import numpy as np

from .hamiltonian import build_from_eft
from .dump import write_dump, read_dump, summarize
from .graph import ground_state_ensemble
from .lanczos import lanczos_ground_state


def sector_size(H, A):
    return comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)


def run_toy(L=1, dim=1, A=1, n_b=2, seed=0, n_runs=8, dump_path=None, verbose=True):
    """Run the full Route-A pipeline on one toy system. Returns a dict."""
    # --- 1. build H -------------------------------------------------------
    H = build_from_eft(L, dim, n_b)
    s = summarize(H)
    full = sector_size(H, A)

    if verbose:
        print("=" * 70)
        print(f"  TOY TrimCI RUN  (L={L}, dim={dim}, A={A}, n_b={n_b} -> N_f={H.N_f})")
        print("=" * 70)
        print(f"  H: {s['n_ferm_modes']}F x {s['n_bos_modes']}B modes,  "
              f"blocks F={s['n_fermion']} B={s['n_boson']} M={s['n_mixed']},  "
              f"const={s['constant'].real:.3f}")
        print(f"  Sector A={A}: C({H.n_ferm_modes},{A}) x {H.N_f}^{H.n_bos_modes} "
              f"= {full:,} basis states")

    # --- 2/3. write dump, read it back -> solve FROM the dump --------------
    tmp = dump_path
    if tmp is None:
        fd, tmp = tempfile.mkstemp(suffix=".mixedfci")
        os.close(fd)
    info = write_dump(H, tmp, n_elec=A)
    Hd = read_dump(tmp)                      # everything below uses Hd
    if dump_path is None:
        os.unlink(tmp)
    if verbose:
        print(f"  Dump: {info['n_records']} records written + read back "
              f"({s['n_terms_total']} terms incl. constant)")

    # --- 4. exact reference (sparse Lanczos; scales past dense ED) ---------
    E_ed, ed_info = lanczos_ground_state(Hd, n_elec=A)
    if verbose:
        print(f"\n  Exact reference (Lanczos):  E0 = {E_ed:.8f} MeV  "
              f"({full:,}-state sector, {ed_info['method']}"
              + (f", nnz={ed_info['nnz']:,}" if ed_info['nnz'] else "") + ")")

    # --- 5. TrimCI convergence sweep --------------------------------------
    # target det counts: a geometric ramp up to the full sector size.
    targets = sorted({int(round(x)) for x in
                      np.unique(np.geomspace(max(2, full // 64), full, num=6))
                      if 2 <= x <= full})
    if full not in targets:
        targets.append(full)

    if verbose:
        print(f"\n  TrimCI (Route A) convergence  "
              f"[ensemble of {n_runs} runs, base seed={seed}]:")
        print(f"  {'n_dets':>8} {'frac sector':>12} {'E_TrimCI':>16} "
              f"{'E-E_ED':>14}")
        print("  " + "-" * 52)

    rows = []
    reached_at = None
    for nd in targets:
        res = ground_state_ensemble(Hd, n_elec=A, n_runs=n_runs, seed=seed,
                                    n_dets=nd,
                                    n_init=min(8, max(2, nd // 4)),
                                    num_groups=min(5, max(1, nd // 4)),
                                    max_rounds=30)
        dE = res.energy - E_ed
        rows.append((res.n_dets, res.energy, dE))
        if reached_at is None and abs(dE) < 1e-6:
            reached_at = res.n_dets
        if verbose:
            print(f"  {res.n_dets:>8} {res.n_dets / full:>12.4f} "
                  f"{res.energy:>16.8f} {dE:>14.2e}")

    # --- variational sanity ----------------------------------------------
    min_dE = min(dE for (_n, _e, dE) in rows)
    variational_ok = min_dE >= -1e-7

    if verbose:
        print("  " + "-" * 52)
        if reached_at is not None:
            print(f"  Reached ED (|dE|<1e-6) at n_dets = {reached_at}  "
                  f"= {reached_at / full:.1%} of the sector")
        print(f"  Variational (all E >= E_ED): {variational_ok}")
        print("=" * 70)

    return {
        "L": L, "dim": dim, "A": A, "n_b": n_b, "N_f": H.N_f,
        "full_sector": full, "E_ed": E_ed,
        "rows": rows, "reached_ed_at": reached_at,
        "variational_ok": variational_ok,
        "blocks": {k: s[k] for k in
                   ("n_fermion", "n_boson", "n_mixed", "n_terms_total")},
    }


def main():
    ap = argparse.ArgumentParser(description="Toy Route-A TrimCI run")
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--n_b", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_runs", type=int, default=8,
                    help="ensemble size (random inits per det target)")
    ap.add_argument("--dump", type=str, default=None,
                    help="keep the generalized dump at this path")
    args = ap.parse_args()
    res = run_toy(L=args.L, dim=args.dim, A=args.A, n_b=args.n_b,
                  seed=args.seed, n_runs=args.n_runs, dump_path=args.dump)
    if not res["variational_ok"]:
        raise SystemExit("NON-VARIATIONAL: a TrimCI energy fell below ED")


if __name__ == "__main__":
    main()
