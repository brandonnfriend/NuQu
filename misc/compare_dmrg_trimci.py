"""DMRG calibration of the TrimCI extrapolation (classical_next_studies A1, 2026-10-08).

TrimCI's sigma (TrimCI/COO literature convention) is the bootstrap scatter of ONE ladder about
ONE estimator -- it cannot tell whether the PT2-linear intercept E_inf lands on the true ground
energy of the truncated Hamiltonian. block2 DMRG on the SAME operator (bare, n_b=3 -> N_f=8, the
exact term list the A-sweep shards ran) is variational and independent of the selected-CI
search, so its chi -> inf extrapolation is the anchor. This script puts the two side by side:

    gap = E_inf(TrimCI, PT2-linear) - E_inf(DMRG)        reported in MeV/site and in units of sigma

DMRG EXTRAPOLATION (the DMRG convention): E is linear in the discarded weight dw for small dw,
so E_inf is the intercept of a straight-line fit of E(chi) against dw(chi) over the deepest
`n_fit` rungs, and the quoted uncertainty is the extrapolation DISTANCE |E(chi_max) - E_inf|
(the same "half/whole of the extrapolation" rule TrimCI's legacy sigma used). A shard with a
single rung gives a variational bound only.

Sanity flags the table carries:
  * DMRG_BELOW_TRIMCI_BOUND  -- E(chi_max) < TrimCI's best variational E_var: DMRG found a
    better variational state than the deepest TrimCI core (expected at L=2, cf. flag F-003).
  * TRIMCI_OVERSHOOTS        -- E_inf(TrimCI) < E_inf(DMRG) - delta by more than 2 sigma: the
    PT2-linear intercept is BELOW what DMRG can support -> the extrapolation over-corrects.
  * TRIMCI_UNDERSHOOTS       -- E_inf(TrimCI) > E_inf(DMRG) + delta by more than 2 sigma.

    python -m misc.compare_dmrg_trimci --dmrg data/classical/<date>/dmrg_calib_<cid> \
        --trimci data/classical/2026-09-25/bare_Asweep_nb3_293963 --out-dir docs/presentation
"""
import argparse
import glob
import json
import os

import numpy as np

from misc.aggregate_classical_energies import analyze, load as load_trimci_shards

FLAG_SIGMA = 2.0


def dmrg_extrapolate(rungs, n_fit=3):
    """Linear E-vs-discarded-weight extrapolation over the deepest `n_fit` rungs.

    Returns dict(E_chi_max, chi_max, dw_min, E_inf, delta, n_pts, slope). With fewer than two
    usable rungs E_inf is None (bound only). Rungs without a discarded weight are skipped.
    """
    pts = [(float(r["discarded_weight"]), float(r["E"]), int(r["chi"]))
           for r in rungs if r.get("discarded_weight") is not None and r.get("E") is not None]
    if not pts:
        return {"E_chi_max": None, "chi_max": None, "dw_min": None, "E_inf": None,
                "delta": None, "n_pts": 0, "slope": None}
    pts.sort(key=lambda p: p[2])                  # by chi, deepest last
    last = pts[-1]
    out = {"E_chi_max": last[1], "chi_max": last[2], "dw_min": last[0],
           "E_inf": None, "delta": None, "n_pts": 0, "slope": None}
    fit = pts[-n_fit:] if n_fit else pts
    if len(fit) < 2:
        return out
    dw = np.array([p[0] for p in fit]); E = np.array([p[1] for p in fit])
    slope, intercept = np.polyfit(dw, E, 1)
    out.update(E_inf=float(intercept), delta=float(abs(last[1] - intercept)),
               n_pts=len(fit), slope=float(slope))
    return out


def load_dmrg(dirs, L, dim, N_f):
    """{A: shard dict} for the DMRG shards at (L, dim, N_f); the deepest ladder wins per A."""
    recs = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "dmrg_L*.json"))):
            j = json.load(open(f))
            if (j["L"], j["dim"], j["N_f"]) != (L, dim, N_f) or not j.get("results"):
                continue
            top = max(r["chi"] for r in j["results"])
            prev = recs.get(j["A"])
            if prev is None or top > max(r["chi"] for r in prev["results"]):
                j["path"] = f
                recs[j["A"]] = j
    return recs


def load_trimci(dirs, A, L, n_b):
    """The pooled TrimCI record (wick convention) for one (A, L, n_b), or None."""
    groups, meta = load_trimci_shards(dirs, A=A, frame="bare", convention="wick")
    for rec in analyze(groups, meta):
        if rec["L"] == L and rec["n_b"] == n_b:
            return rec
    return None


def compare_one(A, dmrg, trimci, n_fit=3):
    sites = dmrg["sites"]
    x = dmrg_extrapolate(dmrg["results"], n_fit=n_fit)
    row = {"A": A, "sites": sites, "dmrg": x, "dmrg_done": dmrg.get("done"),
           "trimci_ok": bool(trimci and trimci.get("ok")),
           "trimci_bound": trimci.get("E_var_bound") if trimci else None,
           "trimci_E_inf": trimci.get("E_inf") if trimci else None,
           "trimci_sigma": trimci.get("sigma") if trimci else None,
           "trimci_crosscheck_E_inf": None, "gap": None, "gap_sigma": None, "flags": []}
    if trimci:
        cc = (trimci.get("per_seed", {}).get(trimci.get("best_seed"), {}).get("cross_check")) or {}
        row["trimci_crosscheck_E_inf"] = cc.get("E_inf")
        if x["E_chi_max"] is not None and row["trimci_bound"] is not None \
                and x["E_chi_max"] < row["trimci_bound"]:
            row["flags"].append("DMRG_BELOW_TRIMCI_BOUND")
    if row["trimci_ok"] and x["E_inf"] is not None:
        gap = row["trimci_E_inf"] - x["E_inf"]
        row["gap"] = gap
        if row["trimci_sigma"]:
            row["gap_sigma"] = gap / row["trimci_sigma"]
            # the two extrapolations disagree beyond both stated uncertainties
            if abs(gap) > x["delta"] + FLAG_SIGMA * row["trimci_sigma"]:
                row["flags"].append("TRIMCI_OVERSHOOTS" if gap < 0 else "TRIMCI_UNDERSHOOTS")
    return row


def compare(dmrg_dirs, trimci_dirs, L=2, dim=3, n_b=3, n_fit=3, As=None):
    N_f = 2 ** n_b
    dm = load_dmrg(dmrg_dirs, L, dim, N_f)
    rows = []
    for A in sorted(dm):
        if As and A not in As:
            continue
        rows.append(compare_one(A, dm[A], load_trimci(trimci_dirs, A, L, n_b), n_fit=n_fit))
    return rows


def _ps(x, sites, nd=2):
    return "—" if x is None else f"{x / sites:.{nd}f}"


def table(rows, path, L, n_b):
    md = [f"# DMRG calibration of the TrimCI extrapolation — bare, n_b={n_b} (N_f={2 ** n_b}), L={L}, 3D (MeV/site)",
          "",
          "_TrimCI: tightest variational bound over seeds and the PT2-linear E∞ ± σ (literature convention); "
          "cross-check = the COO power law. DMRG: block2 on the identical truncated H, E∞ = linear "
          "extrapolation in discarded weight over the deepest rungs, δ = extrapolation distance. "
          f"gap = E∞(TrimCI) − E∞(DMRG); a flag fires when |gap| > δ + {FLAG_SIGMA:g}σ._",
          "",
          "| A | TrimCI bound | TrimCI **E∞ ± σ** | COO power law | χ_max | dw(χ_max) | DMRG E(χ_max) | DMRG E∞ ± δ | gap | gap/σ | flags |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in rows:
        s, x = r["sites"], r["dmrg"]
        einf = ("—" if r["trimci_E_inf"] is None else
                f"**{r['trimci_E_inf'] / s:.2f} ± {(r['trimci_sigma'] or 0) / s:.2f}**")
        dinf = "bound only" if x["E_inf"] is None else f"{x['E_inf'] / s:.3f} ± {x['delta'] / s:.3f}"
        dw = "—" if x["dw_min"] is None else f"{x['dw_min']:.1e}"
        md.append(f"| {r['A']} | {_ps(r['trimci_bound'], s)} | {einf} | "
                  f"{_ps(r['trimci_crosscheck_E_inf'], s)} | {x['chi_max'] or '—'} | {dw} | "
                  f"{_ps(x['E_chi_max'], s, 3)} | {dinf} | {_ps(r['gap'], s, 3)} | "
                  f"{'—' if r['gap_sigma'] is None else '%+.1f' % r['gap_sigma']} | "
                  f"{' '.join(r['flags']) or 'ok'} |")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    return "\n".join(md)


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, r in enumerate(rows):
        if r["trimci_E_inf"] is None:
            continue
        s, ref, c = r["sites"], r["trimci_E_inf"], f"C{i}"
        x = r["dmrg"]
        if x["E_inf"] is not None:
            dws = np.array([p[0] for p in r["_pts"]]); Es = np.array([p[1] for p in r["_pts"]])
            ax.plot(dws, (Es - ref) / s, "o", color=c, label=f"A={r['A']}")
            xx = np.linspace(0, dws.max(), 20)
            ax.plot(xx, (x["E_inf"] + x["slope"] * xx - ref) / s, "-", color=c, lw=1)
            ax.errorbar([0], [(x["E_inf"] - ref) / s], yerr=[x["delta"] / s], fmt="s", color=c, ms=5)
        if r["trimci_sigma"]:
            ax.axhspan(-r["trimci_sigma"] / s, r["trimci_sigma"] / s, color=c, alpha=0.08)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("DMRG discarded weight")
    ax.set_ylabel("E(DMRG) − E∞(TrimCI, PT2-linear)  [MeV/site]")
    ax.set_title("DMRG vs TrimCI E∞ (bands: ±σ TrimCI; squares: DMRG χ→∞)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dmrg", nargs="+", required=True, help="DMRG shard directories")
    ap.add_argument("--trimci", nargs="+", required=True, help="TrimCI bare A-sweep shard directories")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n_b", type=int, default=3)
    ap.add_argument("--n-fit", type=int, default=3, help="deepest DMRG rungs in the dw fit")
    ap.add_argument("--As", nargs="*", type=int, default=None)
    ap.add_argument("--out-dir", default="docs/presentation")
    ap.add_argument("--json", default=None, help="also dump the rows here")
    args = ap.parse_args()

    rows = compare(args.dmrg, args.trimci, L=args.L, dim=args.dim, n_b=args.n_b,
                   n_fit=args.n_fit, As=args.As)
    if not rows:
        raise SystemExit(f"no DMRG shards at L={args.L} dim={args.dim} N_f={2 ** args.n_b} in {args.dmrg}")
    dm = load_dmrg(args.dmrg, args.L, args.dim, 2 ** args.n_b)
    for r in rows:   # the fitted points, for the figure
        deepest = sorted(dm[r["A"]]["results"], key=lambda p: p["chi"])[-args.n_fit:]
        r["_pts"] = sorted((p["discarded_weight"], p["E"]) for p in deepest
                           if p.get("discarded_weight") is not None)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"dmrg_calibration_nb{args.n_b}_L{args.L}")
    print(table(rows, stem + "_table.md", args.L, args.n_b))
    figure(rows, stem + ".png")
    for r in rows:
        r.pop("_pts", None)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
    print(f"\nwrote {stem}_table.md and {stem}.png" + (f" and {args.json}" if args.json else ""))


if __name__ == "__main__":
    main()
