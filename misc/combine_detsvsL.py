"""
Combine the parallel dets-vs-L campaign shards into the Phase-C result.

Each (L, seed) shard (misc/run_detsvsL_shard.py) wrote its per-rung energies. Here
we reconstruct the n_runs ensemble by taking the MIN over seeds per (L, core) -- the
same min-over-random-inits the in-process ensemble does -- then extrapolate the per-L
reference E_inf +/- sigma, extract N*(eps), and fit the dets-vs-L exponent. Output
matches dets_vs_L_at_fixed_accuracy's JSON so plot_dets_vs_L and any downstream
combine-with-laptop step work unchanged.

    python -m misc.combine_detsvsL --shard-dir <dir> --label detsvsL_hpc_combined
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.extrapolation import report_energies
from classical.trimci.run_cpp import _extract_nstar, _nstar_repr, _fit_exponent


def combine(shard_dir, dim=3, eps_targets=(1.0, 0.1), sigma_frac=0.3, cross_frac=1.0):
    files = sorted(glob.glob(os.path.join(shard_dir, "L*_s*.json")))
    if not files:
        raise SystemExit(f"no shard files (L*_s*.json) in {shard_dir}")
    shards = [json.load(open(f)) for f in files]
    by_L = {}
    for s in shards:
        by_L.setdefault(s["L"], []).append(s)

    per_L = []
    for L in sorted(by_L):
        group = by_L[L]
        sites = group[0]["sites"]
        # ENSEMBLE = min over seeds per core (variational selection: lowest E_var)
        by_core = {}
        for s in group:
            for r in s["rungs"]:
                c = r["core"]
                if c not in by_core or r["E_var"] < by_core[c]["E_var"]:
                    by_core[c] = r
        rungs = [by_core[c] for c in sorted(by_core)]

        rep = report_energies(rungs, exact=None, sites=sites,
                              label=f"L={L} d={dim} combined ({len(group)} seeds)",
                              verbose=False)
        E_inf = rep.get("E_extrap_best")
        sigma_ps = rep.get("E_extrap_best_sigma_per_site")
        power, pt2ex = rep["extrap_power"], rep["extrap_pt2"]
        cross_ps = (abs(power["E_inf"] - pt2ex["E_inf"]) / sites
                    if power.get("E_inf") is not None and pt2ex.get("E_inf") is not None
                    else None)

        eps_out = {}
        for eps in eps_targets:
            sigma_ok = sigma_ps is not None and sigma_ps < sigma_frac * eps
            cross_ok = cross_ps is not None and cross_ps < cross_frac * eps
            pinned = bool(sigma_ok and cross_ok)
            base = _extract_nstar(rungs, E_inf, sites, eps, "E_pt2")  # handles E_inf=None
            eps_out[str(eps)] = {
                "eps_persite": eps, "reference_pinned": pinned,
                "nstar_pt2": base, "nstar_repr": _nstar_repr(base),
                "fit_worthy": bool(pinned and base["status"] == "bracketed"),
            }
        per_L.append({
            "L": L, "sites": sites, "A": group[0]["A"], "N_f": group[0]["N_f"],
            "n_terms": group[0]["n_terms"], "transform": group[0]["transform"],
            "n_seeds": len(group), "E_inf": E_inf, "sigma": rep.get("E_extrap_best_sigma"),
            "E_inf_per_site": rep.get("E_extrap_best_per_site"), "sigma_per_site": sigma_ps,
            "cross_check_per_site": cross_ps,
            "rungs": rungs, "eps": eps_out,
        })

    fits = {}
    for eps in eps_targets:
        pts = [(p["sites"], p["eps"][str(eps)]["nstar_repr"]) for p in per_L
               if p["eps"][str(eps)]["fit_worthy"]]
        dropped = [(p["L"], p["eps"][str(eps)]["nstar_pt2"]["status"],
                    "reference not pinned" if not p["eps"][str(eps)]["reference_pinned"]
                    else p["eps"][str(eps)]["nstar_pt2"]["note"])
                   for p in per_L if not p["eps"][str(eps)]["fit_worthy"]]
        if len(pts) >= 2:
            fit = _fit_exponent([s for s, _ in pts], [n for _, n in pts])
            fit["ok"] = True
            fit["points"] = [{"sites": s, "nstar": n} for s, n in pts]
        else:
            fit = {"ok": False, "n_points": len(pts),
                   "reason": f"only {len(pts)} fit-worthy point(s) — need >= 2; deeper ladders required",
                   "points": [{"sites": s, "nstar": n} for s, n in pts]}
        fit["dropped"] = [{"L": L, "status": st, "why": why} for L, st, why in dropped]
        fits[str(eps)] = fit

    # extensivity signal at the deepest core every L reached (min over seeds)
    extensivity = None
    if per_L:
        common = set.intersection(*[{r["core"] for r in p["rungs"]} for p in per_L])
        if common:
            c = max(common)
            extensivity = {"fixed_core": c, "rows": [
                {"L": p["L"], "sites": p["sites"],
                 "dE_pt2_per_site": next(r["dE_pt2"] for r in p["rungs"] if r["core"] == c) / p["sites"]}
                for p in per_L]}

    return {
        "kind": "dets_vs_L", "dim": dim, "combined_from_shards": True,
        "transform": per_L[0]["transform"] if per_L else None,
        "n_seeds_by_L": {p["L"]: p["n_seeds"] for p in per_L},
        "L_values": [p["L"] for p in per_L],
        "eps_persite_targets": list(eps_targets),
        "per_L": per_L, "fits": fits,
        "robustness": {"extensivity_signal": extensivity},
    }


def main():
    ap = argparse.ArgumentParser(description="Combine dets-vs-L campaign shards")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--out-dir", default=os.path.join("data", "classical"))
    ap.add_argument("--label", default="detsvsL_hpc_combined")
    args = ap.parse_args()

    result = combine(args.shard_dir, dim=args.dim, eps_targets=tuple(args.eps))
    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, f"{args.label}.json")
    with open(jp, "w") as f:
        json.dump(result, f, indent=2)
    try:
        from classical.plotting import plot_dets_vs_L
        plot_dets_vs_L(result, out_path=os.path.join(args.out_dir, f"{args.label}.png"))
    except Exception as e:
        print(f"[plot] skipped ({e})")

    print(f"\n  COMBINED ({args.label}) — frame={result['transform']}, "
          f"seeds/L={result['n_seeds_by_L']}")
    for p in result["per_L"]:
        top = p["rungs"][-1]
        gap = (top["E_pt2"] - p["E_inf"]) / p["sites"] if p["E_inf"] else None
        print(f"  L={p['L']:>2} ({p['sites']:>3} sites): E_inf/site="
              f"{p['E_inf_per_site'] if p['E_inf_per_site'] is None else round(p['E_inf_per_site'],3)}"
              f" +/- {p['sigma_per_site']}  top core {top['core']}"
              + (f" gap {gap:.2f}/site" if gap is not None else ""))
        for eps in args.eps:
            e = p["eps"][str(eps)]
            print(f"       eps={eps:g}: {e['nstar_pt2']['status']} (repr {e['nstar_repr']}), "
                  f"pinned={e['reference_pinned']}")
    for eps in args.eps:
        f = result["fits"][str(eps)]
        if f["ok"]:
            ex, po = f["exponential_in_V"], f["polynomial_in_V"]
            better = "EXPONENTIAL" if (ex["r2"] or -9) > (po["r2"] or -9) else "POLYNOMIAL"
            print(f"  eps={eps:g}: exp gamma={ex['slope']:.4g}/site (R^2={ex['r2']:.3f}); "
                  f"poly gamma={po['slope']:.3g} (R^2={po['r2']:.3f}) -> {better}")
        else:
            print(f"  eps={eps:g}: {f['reason']}")
    print(f"  wrote {jp}")


if __name__ == "__main__":
    main()
