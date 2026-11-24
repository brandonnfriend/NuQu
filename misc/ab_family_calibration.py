"""Old-vs-new record for gate 5 of nuqu-2026-10, family calibration (2026-11-24).

Gate 5 is the LAST gate and can only refuse: it never moves E_inf or sigma. So its effect on
every cell follows exactly from an A/B record that already pooled the cells without it
(misc/ab_collapse_rule.py, the 'new' collapse rule = the current default): a cell reported
there stays reported iff its (frame, n_b, L) family passes `_family_calibration_check`. No fit
is re-run.

    python -m misc.ab_family_calibration --ab docs/classical/ab_collapse_rule_20261123/ab_collapse_rule.json \
        --out docs/classical/ab_collapse_rule_20261123/gate5_family_calibration.md
"""
import argparse
import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classical.trimci import external_bounds as xb  # noqa: E402
from classical.trimci.extrapolation import _family_calibration_check  # noqa: E402

CONV = "nuqu-2026-10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.ab))
    rows, stat = [], Counter()
    for k, v in sorted(d.items()):
        p = k.split("|")
        if len(p) != 7 or p[5] != CONV or p[6] != "new" or "error" in v:
            continue
        rel, fr, A, L, nb = p[:5]
        chk = _family_calibration_check(xb.family_calibration(fr, 3, int(nb[2:]), int(L[1:])))
        new_ok = bool(v.get("ok")) and chk["passed"]
        fam = (fr, int(nb[2:]), int(L[1:]))
        stat[fam + ("cells",)] += 1
        stat[fam + ("old",)] += bool(v.get("ok"))
        stat[fam + ("new",)] += new_ok
        stat[fam + ("status:" + chk["status"],)] = 1
        if bool(v.get("ok")) != new_ok:
            rows.append((rel, fr, A, L, nb, v, chk))
    fams = sorted({k[:3] for k in stat})
    md = ["# Gate 5 (family calibration) — before vs after, `nuqu-2026-10`", "",
          f"Source: `{a.ab}` (the outlier collapse rule, i.e. the current default before gate 5). "
          "Gate 5 only refuses, so E∞ and σ of every surviving cell are unchanged. "
          f"Cells that change: **{len(rows)}**, all from extrapolated to bound only.", "",
          "| frame | n_b | L | family A=0 calibration | cells | extrapolated before → after |",
          "|:--|--:|--:|:--|--:|--:|"]
    for f in fams:
        st = next(k[3][7:] for k in stat if k[:3] == f and k[3].startswith("status:"))
        md.append(f"| {f[0]} | {f[1]} | {f[2]} | {st} | {stat[f + ('cells',)]} | "
                  f"{stat[f + ('old',)]} → {stat[f + ('new',)]} |")
    md += ["", "## Cells refused by gate 5", "",
           "| directory | frame | A | L | n_b | was E∞ ± σ (MeV/site) | now: bound | why |",
           "|:--|:--|--:|--:|--:|--:|--:|:--|"]
    for rel, fr, A, L, nb, v, chk in rows:
        md.append(f"| `{rel}` | {fr} | {A[1:]} | {L[1:]} | {nb[2:]} | {v['E']:.3f} ± {v['s']:.3f} | "
                  f"{v['bound']:.3f} | {chk['status']} |")
    open(a.out, "w").write("\n".join(md) + "\n")
    print(f"[gate5] {len(rows)} cells extrapolated -> bound only; {len(fams)} families -> {a.out}")
    for f in fams:
        st = next(k[3][7:] for k in stat if k[:3] == f and k[3].startswith("status:"))
        if stat[f + ("old",)] != stat[f + ("new",)]:
            print(f"  {f[0]} n_b={f[1]} L={f[2]} ({st}): {stat[f + ('old',)]} -> {stat[f + ('new',)]}")


if __name__ == "__main__":
    main()
