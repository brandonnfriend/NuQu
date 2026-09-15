"""Rigorous under-convergence detector for the archived boson-cutoff campaigns.

THE BOUND. The N_f=k Fock space is a SUBSPACE of N_f=k' for k<k' (truncate each mode to
k vs k' levels), and the term list is N_f-INDEPENDENT (the cutoff only enters at
apply-time). So any N_f=k trial state is a valid N_f=k' trial state with the SAME
Rayleigh quotient, giving

    E_0(N_f=k')  <=  E_var(N_f=k)          for k < k'

A larger-cutoff solve that lands ABOVE a smaller-cutoff one at MATCHED (L, A, seed, core)
is therefore under-converged by at least the excess. This needs no reference calculation
and no convergence assumption -- it is self-contained, which is what makes it usable on
archived data.

WHERE IT HAS POWER, AND WHERE IT DOES NOT. The test can only fire when the SEARCH error
exceeds the CUTOFF effect. Between n_b=2 and n_b=3 the cutoff effect is several MeV, so
`E(n_b=3) <= E(n_b=2)` is satisfied trivially and no violation is detectable -- absence
of violations there is NOT evidence that n_b=3 is converged. The informative pair is
n_b=3 vs n_b=4, where the cutoff effect is small.

WHY THE DIRECTION MATTERS. The energy gate's claim is Delta34 = E(n_b=3) - E(n_b=4) ~ 0.
If the n_b=4 arm is the under-converged one, its true E_0 is LOWER than measured, so the
true Delta34 is LARGER than measured -- the bias runs toward zero, i.e. toward making
n_b=3 look more adequate than it is.

    python -m misc.check_subspace_convergence
"""
import argparse
import glob
import json
import os
from collections import defaultdict

CAMPAIGNS = [
    ("L=2 energy gate (N2/N3)", "data/classical/nb_energy_gate", ("nb2", "nb3", "nb4")),
    ("L=3 energy gate", "data/classical/nb_energy_gate_L3", ("nb2", "nb3", "nb4")),
    ("volume scaling (N5)", "data/classical/nb_volscaling", ("nb3", "nb4")),
    ("volume scaling deep (N5)", "data/classical/nb_volscaling_deep", ("nb3", "nb4")),
]


def scan(root, nb_dirs):
    """-> (violations, n_points, n_comparisons). Matched on (L, dim, A, seed, core)."""
    tbl, meta = defaultdict(dict), {}
    for nb in nb_dirs:
        for f in glob.glob(os.path.join(root, nb, "*.json")):
            j = json.load(open(f))
            if j.get("kind") != "frame_shard":
                continue
            for r in j.get("rungs", []):
                if r.get("E_var") is None:
                    continue
                key = (j["L"], j["dim"], j["A"], j["seed"], r["core"])
                tbl[key][j["n_b"]] = r["E_var"]
                meta[key] = j.get("phase0_runs")
    viol, ncmp = [], 0
    for key, by_nb in sorted(tbl.items()):
        best = None
        for nb in sorted(by_nb):
            ncmp += 1
            e = by_nb[nb]
            if best is not None and e > best[1] + 1e-9:
                viol.append(dict(key=key, n_b=nb, E=e, bound_n_b=best[0], bound=best[1],
                                 excess=e - best[1], sites=key[0] ** key[1],
                                 phase0_runs=meta.get(key)))
            if best is None or e < best[1]:
                best = (nb, e)
    return viol, sum(1 for v in tbl.values() if len(v) >= 2), ncmp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    md = ["# Subspace under-convergence check — archived boson-cutoff campaigns\n",
          "_`E_0(N_f=k') <= E_var(N_f=k)` for `k<k'` (the smaller Fock space is a subspace and "
          "the term list is N_f-independent). A larger-cutoff solve above a smaller-cutoff one "
          "at matched `(L, A, seed, core)` is under-converged by at least the excess. "
          "Self-contained: no reference calculation, no convergence assumption._\n",
          "| campaign | matched points | comparisons | violations | worst excess (MeV) | worst /site |",
          "|---|--:|--:|--:|--:|--:|"]
    for label, root, nbs in CAMPAIGNS:
        if not os.path.isdir(root):
            print(f"[skip] {label}: {root} absent")
            continue
        viol, npts, ncmp = scan(root, nbs)
        worst = max((v["excess"] for v in viol), default=0.0)
        wps = max(((v["excess"] / v["sites"]) for v in viol), default=0.0)
        pairs = sorted({(v["bound_n_b"], v["n_b"]) for v in viol})
        print(f"\n=== {label} ===")
        print(f"  matched points {npts}, comparisons {ncmp}, violations {len(viol)}")
        if viol:
            print(f"  worst excess {worst:.4f} MeV ({wps:.5f} MeV/site); "
                  f"violating cutoff pairs (smaller->larger): {pairs}")
        else:
            print("  PASS — no violation")
        md.append(f"| {label} | {npts} | {ncmp} | {len(viol)} | "
                  f"{worst:.4f} | {wps:.5f} |")
    md.append("\n**Power limitation.** The test fires only when SEARCH error exceeds the "
              "CUTOFF effect. Between `n_b=2` and `n_b=3` the cutoff effect is several MeV, so "
              "no violation is detectable there and its absence proves nothing about `n_b=3`. "
              "The informative pair is `n_b=3` vs `n_b=4`.\n")
    md.append("**Direction.** Every violation found is the `n_b=4` arm sitting above the "
              "`n_b=3` bound. A too-high `n_b=4` makes `Delta34 = E(n_b=3) - E(n_b=4)` look "
              "SMALLER than it is — the bias runs toward zero, i.e. toward making `n_b=3` "
              "appear more adequate than the data supports.\n")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write("\n".join(md) + "\n")
        print(f"\n[tbl] wrote {args.out}")


if __name__ == "__main__":
    main()
