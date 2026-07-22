"""
Analysis helpers for the classical pipeline — index saved runs into a tidy
table for cross-method / cross-parameter comparison.

Method-agnostic: the `method` and `transform` columns let us compare TrimCI
against future solvers (DMRG, AFQMC, NQS) and the COO vs LF boson frame on the
same axes.
"""

from __future__ import annotations

import glob
import json
import os

from .io import DATA_ROOT


def index_runs(data_root=DATA_ROOT):
    """Scan `data_root` for metadata.json and return a pandas DataFrame —
    one row per run, flattening the key metadata for comparison/plots."""
    import pandas as pd

    rows = []
    for mpath in glob.glob(os.path.join(data_root, "**", "metadata.json"),
                           recursive=True):
        try:
            with open(mpath) as f:
                m = json.load(f)
        except Exception:
            continue
        sys_ = m.get("system", {})
        res = m.get("result", {})
        ref = m.get("exact_reference") or {}
        rows.append({
            "method": m.get("method"),
            "transform": m.get("transform"),
            "label": m.get("label"),
            "L": sys_.get("L"), "dim": sys_.get("dim"), "A": sys_.get("A"),
            "n_b": sys_.get("n_b"), "N_f": sys_.get("N_f"),
            "sector_size": sys_.get("sector_size"),
            "n_terms": sys_.get("n_terms"),
            "n_dets": res.get("n_dets"),
            "energy": res.get("energy"),
            "runtime_s": m.get("runtime_s"),
            "ref_method": ref.get("method"),
            "ref_energy": ref.get("energy"),
            "dE_vs_ref": (res.get("energy") - ref["energy"])
                         if ref.get("energy") is not None else None,
            "timestamp": m.get("timestamp"),
            "run_dir": os.path.dirname(mpath),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["method", "transform", "L", "dim", "A", "N_f",
                             "n_dets"]).reset_index(drop=True)
    return df


def convergence_of(run_dir):
    """Return the (n_dets, energy) convergence history of a saved run."""
    with open(os.path.join(run_dir, "metadata.json")) as f:
        m = json.load(f)
    return m.get("convergence", [])


def summary(data_root=DATA_ROOT):
    """Print a compact table of all runs."""
    df = index_runs(data_root)
    if df.empty:
        print(f"(no runs under {data_root})")
        return df
    cols = ["method", "transform", "L", "dim", "A", "N_f", "n_dets",
            "energy", "runtime_s", "dE_vs_ref"]
    print(df[cols].to_string(index=False))
    return df
