"""
Combine dets-vs-L shards into the Phase-C result — frame-aware.

Each shard (misc/run_frame_shard.py or the older run_detsvsL_shard.py) wrote its per-rung
energies. Here we reconstruct the n_runs ensemble by taking the MIN over seeds per
(frame, L, core) -- the same min-over-random-inits the in-process ensemble does -- then
extrapolate each per-L reference E_inf +/- sigma, extract N*(eps), and fit the exponent.

Shards are grouped by FRAME first: a single-frame dir (e.g. the deep dilute run) writes
one <label>.json; a multi-frame dir (the frame-comparison run) with --by-frame writes one
<label>_<frame>.json per frame so they can be compared. Tolerant of missing/partial shards
(incremental save leaves a valid file even if a shard died at a deep rung).

    python -m misc.combine_detsvsL --shard-dir <dir> --label detsvsL_deep_<id>
    python -m misc.combine_detsvsL --shard-dir <dir> --by-frame --label frame_<id>
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.extrapolation import report_energies
from classical.trimci.run_cpp import _extract_nstar, _nstar_repr, _fit_exponent


def _frame_of(shard):
    return shard.get("frame") or shard.get("transform") or "gaussian"


def _reference_for_group(shards, dim, eps_targets, sigma_frac=0.3, cross_frac=1.0):
    """The per-L dets-vs-L reference + exponent for ONE frame's shards."""
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
        if not rungs:
            continue

        rep = report_energies(rungs, exact=None, sites=sites, verbose=False)
        E_inf = rep.get("E_extrap_best")
        sigma_ps = rep.get("E_extrap_best_sigma_per_site")
        power, pt2ex = rep["extrap_power"], rep["extrap_pt2"]
        cross_ps = (abs(power["E_inf"] - pt2ex["E_inf"]) / sites
                    if power.get("E_inf") is not None and pt2ex.get("E_inf") is not None
                    else None)

        eps_out = {}
        for eps in eps_targets:
            pinned = bool(sigma_ps is not None and sigma_ps < sigma_frac * eps
                          and cross_ps is not None and cross_ps < cross_frac * eps)
            base = _extract_nstar(rungs, E_inf, sites, eps, "E_pt2")
            eps_out[str(eps)] = {
                "eps_persite": eps, "reference_pinned": pinned,
                "nstar_pt2": base, "nstar_repr": _nstar_repr(base),
                "fit_worthy": bool(pinned and base["status"] == "bracketed"),
            }
        per_L.append({
            "L": L, "sites": sites, "A": group[0].get("A"),
            "filling": group[0].get("filling"), "N_f": group[0].get("N_f"),
            "n_terms": group[0].get("n_terms"), "n_seeds": len(group),
            "E_inf": E_inf, "sigma": rep.get("E_extrap_best_sigma"),
            "E_inf_per_site": rep.get("E_extrap_best_per_site"), "sigma_per_site": sigma_ps,
            "cross_check_per_site": cross_ps,
            "mean_occ_top": rungs[-1].get("mean_occ"),
            "top_core": rungs[-1]["core"], "rungs": rungs, "eps": eps_out,
        })

    fits = {}
    for eps in eps_targets:
        pts = [(p["sites"], p["eps"][str(eps)]["nstar_repr"]) for p in per_L
               if p["eps"][str(eps)]["fit_worthy"]]
        if len(pts) >= 2:
            fit = _fit_exponent([s for s, _ in pts], [n for _, n in pts])
            fit["ok"] = True
        else:
            fit = {"ok": False, "n_points": len(pts),
                   "reason": f"only {len(pts)} fit-worthy point(s) — need >= 2"}
        fit["points"] = [{"sites": s, "nstar": n} for s, n in pts]
        fits[str(eps)] = fit

    extensivity = None
    if per_L:
        common = set.intersection(*[{r["core"] for r in p["rungs"]} for p in per_L])
        if common:
            c = max(common)
            extensivity = {"fixed_core": c, "rows": [
                {"L": p["L"], "sites": p["sites"],
                 "dE_pt2_per_site": next(r["dE_pt2"] for r in p["rungs"] if r["core"] == c) / p["sites"]}
                for p in per_L]}
    return {"kind": "dets_vs_L", "dim": dim, "per_L": per_L, "fits": fits,
            "eps_persite_targets": list(eps_targets),
            "L_values": [p["L"] for p in per_L],
            "robustness": {"extensivity_signal": extensivity}}


def combine(shard_dir, dim=3, eps_targets=(1.0, 0.1)):
    files = sorted(glob.glob(os.path.join(shard_dir, "*.json")))
    shards = []
    for f in files:
        try:
            s = json.load(open(f))
        except Exception:
            continue
        if isinstance(s, dict) and s.get("rungs") is not None and "L" in s:
            shards.append(s)
    if not shards:
        raise SystemExit(f"no usable shard files in {shard_dir}")
    by_frame = {}
    for s in shards:
        by_frame.setdefault(_frame_of(s), []).append(s)
    out = {}
    for fr, grp in by_frame.items():
        res = _reference_for_group(grp, dim, eps_targets)
        res["frame"] = fr
        res["n_seeds_by_L"] = {p["L"]: p["n_seeds"] for p in res["per_L"]}
        out[fr] = res
    return out


def _print_frame(res):
    print(f"  [frame={res['frame']}] seeds/L={res['n_seeds_by_L']}")
    for p in res["per_L"]:
        top = p["rungs"][-1]
        gap = (top["E_pt2"] - p["E_inf"]) / p["sites"] if p["E_inf"] else None
        occ = f" occ={p['mean_occ_top']:.4f}" if p.get("mean_occ_top") is not None else ""
        print(f"    L={p['L']:>2} ({p['sites']:>3} sites, A={p['A']}): E_inf/site="
              f"{None if p['E_inf_per_site'] is None else round(p['E_inf_per_site'],3)}"
              f" +/-{p['sigma_per_site']} top {top['core']}"
              + (f" gap {gap:.2f}/site" if gap is not None else "") + occ)
    for eps in res["eps_persite_targets"]:
        f = res["fits"][str(eps)]
        if f["ok"]:
            ex, po = f["exponential_in_V"], f["polynomial_in_V"]
            better = "POLY" if (po["r2"] or -9) > (ex["r2"] or -9) else "EXP"
            print(f"    eps={eps:g}: poly V^{po['slope']:.2g}(R2={po['r2']:.2f}) "
                  f"exp e^{ex['slope']:.3g}V(R2={ex['r2']:.2f}) -> {better}")
        else:
            print(f"    eps={eps:g}: {f['reason']}")


def main():
    ap = argparse.ArgumentParser(description="Combine dets-vs-L shards (frame-aware)")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--by-frame", action="store_true",
                    help="write one JSON+PNG per frame (multi-frame comparison run)")
    ap.add_argument("--out-dir", default=os.path.join("data", "classical"))
    ap.add_argument("--label", default="detsvsL_combined")
    args = ap.parse_args()

    by_frame = combine(args.shard_dir, dim=args.dim, eps_targets=tuple(args.eps))
    os.makedirs(args.out_dir, exist_ok=True)
    multi = args.by_frame or len(by_frame) > 1
    for fr, res in by_frame.items():
        tag = ("_" + fr.replace("+", "_")) if multi else ""
        jp = os.path.join(args.out_dir, f"{args.label}{tag}.json")
        with open(jp, "w") as f:
            json.dump(res, f, indent=2)
        try:
            from classical.plotting import plot_dets_vs_L
            plot_dets_vs_L(res, out_path=os.path.join(args.out_dir, f"{args.label}{tag}.png"))
        except Exception as e:
            print(f"[plot] skipped ({e})")
        _print_frame(res)
        print(f"  wrote {jp}\n")


if __name__ == "__main__":
    main()
