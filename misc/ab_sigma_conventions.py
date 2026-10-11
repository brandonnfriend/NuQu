"""Old-versus-new A/B of the E_inf error-bar conventions over every local (dir, frame, A, L) cell.

    python -m misc.ab_sigma_conventions --out docs/sigma_convention_ab_<date>.md

Runs `combine_seeds` under `trimci-coo` (the default 2026-10-06..10), `nuqu-2026-10` (gated;
the default since 2026-10-10) and, for the bit-identity check, `nuqu-2026-09`, on the same loaded
shards, and tabulates per cell: the bound, each convention's E_inf +- sigma (or the refusal), the
nuqu-2026-10 sigma terms, the cross-check gap with the cross-check's own sigma (n/r = unresolved,
does not vote), the gate verdict and the deviation from any registered external reference. The
acceptance criterion in docs/classical_error_convention_2026-10-08.md s11e: the old conventions
do not move, and the gate refuses exactly the cells an external same-operator bound refutes.

Cells are discovered from DATA_DIRS below (one Hamiltonian per (frame, A, L)); the A-sweep and
squeeze directories are loaded one A at a time, the filling-1.0 baselines as they are.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from classical.trimci.extrapolation import (GATED_CONVENTION, LEGACY_CONVENTION,  # noqa: E402
                                            LITERATURE_CONVENTION)
from misc.aggregate_classical_energies import analyze, load  # noqa: E402

DATA_DIRS = [
    # (dir, frames, As)  -- As=None loads the directory as one (n_b, L)-keyed set (filling 1.0)
    ("data/classical/2026-09-25/bare_Asweep_nb3_293963", ["bare"], list(range(2, 11))),
    ("data/classical/2026-10-08/Asweep_squeeze_nb3_294246", ["gaussian", "bare"], list(range(0, 11))),
    ("data/classical/2026-09-06/bare_baseline_nb3_292477", ["bare"], None),
    ("data/classical/2026-09-06/bare_baseline_nb2_292478", ["bare"], None),
    ("data/classical/2026-10-10/bare_baseline_nb3_294250", ["bare"], None),
    ("data/classical/2026-10-10/bare_baseline_nb2_294250", ["bare"], None),
]


def cells(dirs=DATA_DIRS):
    for d, frames, As in dirs:
        if not os.path.isdir(os.path.join(_ROOT, d)):
            continue
        for frame in frames:
            for A in (As if As is not None else [None]):
                try:
                    g, m = load([os.path.join(_ROOT, d)], A=A, frame=frame)
                except ValueError as e:        # a directory mixing A at one (n_b, L)
                    print(f"[ab] skip {d} {frame} A={A}: {e}")
                    continue
                if g:
                    yield d, frame, A, g, m


def run(dirs=DATA_DIRS, conventions=(LITERATURE_CONVENTION, GATED_CONVENTION, LEGACY_CONVENTION)):
    rows = []
    for d, frame, A, g, m in cells(dirs):
        recs = {c: {r["key"]: r for r in analyze(g, m, sigma_convention=c)} for c in conventions}
        for key, r_lit in recs[LITERATURE_CONVENTION].items():
            r_new = recs[GATED_CONVENTION][key]
            r_old = recs[LEGACY_CONVENTION][key] if LEGACY_CONVENTION in recs else None
            gate = r_new.get("gate") or {}
            cc = (gate.get("checks") or {}).get("cross_check") or {}
            ext = r_new.get("external_reference") or {}
            dev = [u["dev_ps"] for u in ext.get("upper") or []] + \
                  ([ext["exact"]["dev_ps"]] if ext.get("exact") else [])
            rows.append({
                "dir": os.path.basename(d), "frame": frame, "A": r_lit["A"], "L": r_lit["L"],
                "n_b": r_lit["n_b"], "n_seeds": r_lit["n_seeds"], "bound": r_lit["E_var_bound_ps"],
                "lit_ok": r_lit["ok"], "lit_E": r_lit.get("E_inf_ps"), "lit_sigma": r_lit.get("sigma_ps"),
                "old_ok": None if r_old is None else r_old["ok"],
                "old_E": None if r_old is None else r_old.get("E_inf_ps"),
                "old_sigma": None if r_old is None else r_old.get("sigma_ps"),
                "new_ok": r_new["ok"], "new_E": r_new.get("E_inf_ps"), "new_sigma": r_new.get("sigma_ps"),
                "new_E_refused": gate.get("E_inf_refused_ps"),
                "new_sigma_if": gate.get("sigma_if_reported_ps"),
                "terms": r_new.get("sigma_terms_ps") or {}, "source": r_new.get("sigma_source"),
                "cc_gap": cc.get("gap_ps"), "cc_sigma": cc.get("sigma_cc_ps"),
                "cc_resolved": cc.get("resolved"),
                "refused_by": gate.get("refused_by") or [], "window_shift": r_new.get("window_shift_ps"),
                "ext_dev": dev[0] if dev else None, "best_known": r_new.get("E_bound_best_known_source"),
            })
    return rows


def table(rows, path):
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    pm = lambda ok, E, s: (f"{f(E)} ± {f(s)}" if ok else "bound only")
    n_lit = sum(r["lit_ok"] for r in rows)
    n_new = sum(r["new_ok"] for r in rows)
    n_ref = sum(1 for r in rows if r["lit_ok"] and not r["new_ok"])
    md = ["# σ-convention A/B: `trimci-coo` (default 2026-10-06..10) vs `nuqu-2026-10` (gated, default "
          "since 2026-10-10) — every local cell\n",
          f"_{len(rows)} (dir, frame, A, L) cells. trimci-coo reports {n_lit} limits; nuqu-2026-10 reports "
          f"{n_new} ({n_ref} of the trimci-coo limits are refused by the gate). Central values never "
          f"move: the gated convention only widens σ (max of the terms) and refuses. `cross-check` is "
          f"the gap to the other estimator with that estimator's own bootstrap σ in parentheses; "
          f"`n/r` = unresolved (σ_cc beyond the extrapolation distance), which does not vote. `vs ext.` is "
          f"the deviation from a registered external reference (block2 DMRG at bare L=2, the exact "
          f"vacuum at A=0), positive = our limit is higher. MeV/site, Wick convention._\n",
          "| dir | frame | A | L | seeds | bound | trimci-coo E∞ ± σ | **nuqu-2026-10 E∞ ± σ** | "
          "σ boot | ½-dist | seed spread | cross-check gap (σ_cc) | source | window shift | vs ext. | gate |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|:--|"]
    for r in sorted(rows, key=lambda r: (r["dir"], r["frame"], r["L"], r["A"], r["n_b"])):
        t = r["terms"]
        new = (f"**{pm(True, r['new_E'], r['new_sigma'])}**" if r["new_ok"] else
               ("bound only" + (f" (refused {f(r['new_E_refused'])} ± {f(r['new_sigma_if'])})"
                                if r["new_E_refused"] is not None else "")))
        gate = ("passed" if r["new_ok"] else (", ".join(r["refused_by"]) or "no limit"))
        src = {"bootstrap": "boot", "extrap_half_distance": "½-dist", "seed_spread": "seed",
               "cross_check_gap": "cross-check"}.get(r["source"], "—")
        cc = ("—" if r["cc_gap"] is None else
              f"{f(r['cc_gap'])} ({f(r['cc_sigma'])})" + ("" if r["cc_resolved"] else " n/r"))
        md.append(f"| {r['dir']} | {r['frame']} | {r['A']} | {r['L']} | {r['n_seeds']} | {f(r['bound'])} | "
                  f"{pm(r['lit_ok'], r['lit_E'], r['lit_sigma'])} | {new} | {f(t.get('bootstrap'))} | "
                  f"{f(t.get('extrap_half_distance'))} | {f(t.get('seed_spread'))} | {cc} | {src} | "
                  f"{f(r['window_shift'])} | {f(r['ext_dev'])} | {gate} |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(md) + "\n")
    return n_lit, n_new, n_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "docs", "sigma_convention_ab.md"))
    args = ap.parse_args()
    rows = run()
    n_lit, n_new, n_ref = table(rows, args.out)
    moved = [r for r in rows if r["lit_ok"] and r["new_ok"] and abs(r["lit_E"] - r["new_E"]) > 1e-9]
    print(f"[ab] {len(rows)} cells: trimci-coo {n_lit} limits, nuqu-2026-10 {n_new} limits, "
          f"{n_ref} refused by the gate, {len(moved)} central values moved (must be 0) -> {args.out}")
    by = {}
    for r in rows:
        for k in r["refused_by"]:
            by[k] = by.get(k, 0) + 1
    print(f"[ab] refusals by check: {by}")


if __name__ == "__main__":
    main()
