"""Build `classical/trimci/family_calibration.json`, the registry gate 5 of nuqu-2026-10 reads.

Gate 5, "family calibration" (2026-11-24): a (frame, dim, n_b, L) family reports an extrapolated
limit only if its A=0 cell -- the one cell with an exact answer, the untruncated pion vacuum --
lands within its own sigma of that answer. Each A=0 cell is pooled exactly as the aggregator
pools it (nuqu-2026-10, gate 5 OFF, so the registry never reads itself) and compared with the
exact vacuum in `external_bounds.json`. One entry per (directory, family): a family is
calibrated by every directory that holds its A=0 shards.

Two sources, same entries:
    # from the shards (runs the A=0 fits; a few minutes per family -- cluster or a quiet laptop)
    python -m misc.build_family_calibration --data-root data/classical [--workers N]
    # from an A/B record that already pooled every cell (misc/ab_collapse_rule.py, the 'new' rule)
    python -m misc.build_family_calibration --from-ab docs/classical/ab_collapse_rule_20261123/ab_collapse_rule.json
"""
import argparse
import datetime
import json
import os
import sys
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classical.trimci import external_bounds as xb  # noqa: E402

CONV = "nuqu-2026-10"
OUT = os.path.join(_ROOT, "classical", "trimci", "family_calibration.json")


def entry(source, frame, n_b, L, dim, ok, E_ps, sigma_ps, bound_ps, reason=None):
    """One A=0 calibration record. passed: extrapolated AND |E - exact| <= sigma + the exact
    value's quoting precision (E_ps_precision; 0.0005/site for the 3-decimal vacua)."""
    ex = xb.lookup(frame, n_b, L, 0, dim, "wick", calibration=False)["exact"]
    e = {"frame": frame, "dim": int(dim), "n_b": int(n_b), "L": int(L), "source": source,
         "ok": bool(ok), "E_ps": E_ps, "sigma_ps": sigma_ps, "bound_ps": bound_ps,
         "exact_ps": ex["E_ps"] if ex else None,
         "exact_precision_ps": float(ex.get("E_ps_precision") or 0.0) if ex else None, "dev_ps": None, "dev_over_sigma": None,
         "passed": False}
    if not ok:
        e["reason"] = f"A=0 limit not reported ({(reason or 'bound only')[:120]}): no calibration"
    elif ex is None:
        e["reason"] = "no exact A=0 vacuum registered at this L: no calibration"
    else:
        dev = E_ps - ex["E_ps"]
        tol = sigma_ps + e["exact_precision_ps"]
        e.update(dev_ps=dev, dev_over_sigma=(abs(dev) / sigma_ps if sigma_ps else None),
                 passed=bool(sigma_ps is not None and abs(dev) <= tol + 1e-12))
        e["reason"] = (f"A=0 limit {E_ps:.4f} ± {sigma_ps:.4f} vs exact {ex['E_ps']:.3f} "
                       f"(± {e['exact_precision_ps']:g}): {dev:+.4f} = {abs(dev) / sigma_ps:.2f} sigma -> "
                       f"{'calibrated' if e['passed'] else 'MISSES'}")
    return e


def from_ab(path):
    d = json.load(open(path))
    out = []
    for k, v in d.items():
        p = k.split("|")
        if len(p) != 7 or p[2] != "A0" or p[5] != CONV or p[6] != "new" or "error" in v:
            continue
        rel, fr, _, L, nb = p[:5]
        out.append(entry(rel, fr, int(nb[2:]), int(L[1:]), 3, v.get("ok"), v.get("E"),
                         v.get("s"), v.get("bound"), v.get("reason")))
    return out


def _one(job):
    d, fr, root = job
    from classical.trimci import extrapolation as ex
    from misc.aggregate_classical_energies import analyze, load
    ex.FAMILY_CALIBRATION_GATE = False
    g, m = load([d], convention="wick", A=0, frame=fr)
    return [entry(os.path.relpath(d, root), fr, r["n_b"], r["L"], r.get("dim", 3), r["ok"],
                  r.get("E_inf_ps"), r.get("sigma_ps"), r.get("E_var_bound_ps"), r.get("reason"))
            for r in analyze(g, m, sigma_convention=CONV)]


def from_data(root, workers):
    from misc.ab_collapse_rule import _groups
    jobs = sorted({(d, fr, root) for d, fr, A in _groups(root) if A == 0})
    print(f"[calib] {len(jobs)} (dir, frame) groups with A=0 shards, {workers} workers", flush=True)
    with Pool(workers) as pool:
        return [e for part in pool.imap_unordered(_one, jobs) for e in part]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root")
    src.add_argument("--from-ab")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    entries = from_ab(a.from_ab) if a.from_ab else from_data(a.data_root, a.workers)
    entries.sort(key=lambda e: (e["frame"], e["dim"], e["n_b"], e["L"], e["source"]))
    doc = {"_doc": ("Gate 5 of nuqu-2026-10 (family calibration), built by "
                    "misc/build_family_calibration.py. One entry per (source directory, frame, dim, "
                    "n_b, L): the pooled A=0 limit vs the exact untruncated pion vacuum. A family "
                    "FAILS if any extrapolated A=0 cell misses by more than its sigma, PASSES if at "
                    "least one is within sigma and none miss, and is MISSING otherwise (no A=0 "
                    "limit, or no exact value). Per-site MeV."),
           "built": datetime.date.today().isoformat(),
           "built_from": a.from_ab or a.data_root, "convention": CONV, "entries": entries}
    json.dump(doc, open(a.out, "w"), indent=1)
    fams = {}
    for e in entries:
        fams.setdefault((e["frame"], e["dim"], e["n_b"], e["L"]), []).append(e)
    print(f"[calib] {len(entries)} entries, {len(fams)} families -> {a.out}")
    for f, es in sorted(fams.items()):
        st = xb.family_calibration(*f, registry=entries)["status"]
        print(f"  {f[0]:9s} n_b={f[2]} L={f[3]}: {st:8s} "
              + "; ".join(f"{e['dev_ps']:+.3f}/{e['sigma_ps']:.3f}" if e["dev_ps"] is not None
                          else "—" for e in es))


if __name__ == "__main__":
    main()
