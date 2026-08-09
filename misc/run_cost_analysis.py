"""Aggregate a campaign's shards into the Goal-3 classical-cost tables.

Per (L, dim, A, frame): Tier-1 core*(dE) against the exact E_inf (where --exact-ref gave
one), and Tier-2 support (converged n(weight) or the decay exponent gamma, calibrated to
the exact support). Pure analysis. Run:

    python -m misc.run_cost_analysis <campaign_shards_dir> [--dEs 1.0 0.1]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classical.trimci import cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards_dir")
    ap.add_argument("--dEs", type=float, nargs="+", default=[1.0, 0.1])
    args = ap.parse_args()

    rows = []
    for fn in sorted(glob.glob(os.path.join(args.shards_dir, "*.json"))):
        d = json.load(open(fn))
        rungs = d.get("rungs", [])
        if not rungs:
            continue
        L, dim, A, sites = d.get("L"), d.get("dim"), d.get("A"), d.get("sites")
        frame, E_exact = d.get("frame"), d.get("E_exact")
        deep = cost.deepest_rung(rungs)
        we = cost.support_weight_exponent(deep) if deep else None
        we_ex = (cost.support_weight_exponent({"support": d["exact_support"]})
                 if d.get("exact_support") else None)
        conv, nconv = cost.support_converged(rungs, "n999")
        row = {"L": L, "dim": dim, "A": A, "sites": sites, "frame": frame,
               "E_exact": E_exact, "top_core": deep["core"] if deep else None,
               "n999_conv": nconv if conv else None,
               "gamma": we["gamma"] if we else None,
               "gamma_exact": we_ex["gamma"] if we_ex else None,
               "t1": cost.tier1_costs(rungs, E_exact, sites, tuple(args.dEs)) if E_exact else None}
        rows.append(row)

    rows.sort(key=lambda r: (r["dim"], r["L"], r["A"], r["frame"]))
    print(f"{'L':>2}{'d':>2}{'A':>4} {'frame':<13} {'E_exact':>9} "
          f"{'core*(tot)':>22} {'n999(conv)':>10} {'gamma':>6}{'/exact':>7}")
    for r in rows:
        cs = ("  ".join(f"{dE}:{v['core_star']:.0f}{'' if v['reached'] else 'LB'}"
                        for dE, v in r["t1"]["total"].items()) if r["t1"] else "--")
        ee = f"{r['E_exact']:.2f}" if r["E_exact"] is not None else "--(ED n/a)"
        g = f"{r['gamma']:.2f}" if r["gamma"] else "--"
        ge = f"{r['gamma_exact']:.2f}" if r["gamma_exact"] else "--"
        nc = str(r["n999_conv"]) if r["n999_conv"] else "growing"
        print(f"{r['L']:>2}{r['dim']:>2}{r['A']:>4} {r['frame']:<13} {ee:>9} "
              f"{cs:>22} {nc:>10} {g:>6}{ge:>7}")
    print("\nTier-1 core*(dE): rigorous cost to true E_inf (ED points only). Extrapolate")
    print("its scaling with sites for L=3. Tier-2 gamma: accuracy<->cost exponent, robust")
    print("from any core; prefactor (n999) calibrated by the exact-ED points.")


if __name__ == "__main__":
    main()
