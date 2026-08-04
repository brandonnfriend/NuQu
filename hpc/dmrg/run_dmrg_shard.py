"""One block2-DMRG isospectrality-reference SHARD.

Runs DMRG on the BARE mixed EFT Hamiltonian at (L, dim, A, N_f) over a bond-dimension
schedule and writes E-vs-chi to JSON, RE-SAVING after every chi (the DMRG cost climbs
steeply with chi, so a shard that times out at the top rung still keeps everything it
finished). block2 on the bare H is variational -> an upper bound on the true E_inf that
every isospectral frame must converge to from above; downstream (isospectrality_check)
compares the extrapolated E_inf to the framed TrimCI energies (gaussian+lf must NOT dip
below it -- the leading-order projector-LF go/no-go).

    python -m hpc.dmrg.run_dmrg_shard --L 3 --A 14 --N_f 4 --bond-dims 100,200,400,800 --out x.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classical.baselines.dmrg_block2 import run_dmrg


def main():
    ap = argparse.ArgumentParser(description="One block2-DMRG (L, A) reference shard")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--A", type=int, required=True)
    ap.add_argument("--N_f", type=int, default=4, help="boson levels (match the frame runs: 4)")
    ap.add_argument("--n_b", type=int, default=2, help="boson bits (match the frame runs: 2)")
    ap.add_argument("--bond-dims", default="100,200,400,800",
                    help="comma-separated chi schedule (warm-started in order)")
    ap.add_argument("--n-sweeps-per", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bond_dims = tuple(int(x) for x in args.bond_dims.split(","))
    sites = args.L ** args.dim
    t0 = time.time()

    out = {
        "kind": "dmrg_shard", "L": args.L, "dim": args.dim, "A": args.A,
        "N_f": args.N_f, "n_b": args.n_b, "sites": sites,
        "bond_dims": list(bond_dims), "n_sweeps_per": args.n_sweeps_per,
        "results": [], "done": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, args.out)          # atomic — reader never sees a half-write

    save()  # header immediately: even a shard that dies at chi[0] leaves a record

    def on_chi(rung):
        out["results"].append(rung)
        out["wall_s"] = time.time() - t0
        save()
        print(f"[dmrg] chi={rung['chi']:>5}  E={rung['E']:.4f}  "
              f"S_max={rung['S_max_bond']}  ({out['wall_s']:.0f}s)", flush=True)

    run_dmrg(args.L, args.dim, args.A, N_f=args.N_f, n_b=args.n_b,
             bond_dims=bond_dims, n_sweeps_per=args.n_sweeps_per, on_chi=on_chi)

    out["done"] = True
    out["wall_s"] = time.time() - t0
    save()
    Es = [r["E"] for r in out["results"]]
    print(f"[dmrgshard] L={args.L} A={args.A} N_f={args.N_f} "
          f"chi={out['bond_dims']} E={Es} wall={out['wall_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
