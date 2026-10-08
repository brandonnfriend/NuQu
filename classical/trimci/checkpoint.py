"""Per-rung core CHECKPOINTS for the warm-grow ladder, so a shard restarts from its last
finished rung instead of rung 0 (classical infrastructure C1, 2026-10-08).

Why. Every out-of-memory hold in the 293963 fixed-A sweep restarted the shard from
scratch; one L=4 shard started four times and each restart repeated 15-20 h of finished
work. The ladder is deterministic given the core it grows from, so saving that core after
each rung makes a restart cost one rung, not one shard.

What is saved (`np.savez_compressed`, beside the shard JSON):
  ferm_arr (N, W) uint64, bos_arr (N, n_bos) uint16, coeffs (N,) complex128 -- the
  finished core and its amplitudes -- plus `rung_index`, `core`, `energy`, the full rung
  list, and a `config_key` (the JSON of every solver/physics setting that determines the
  ladder). Only the latest checkpoint and the one before it are kept.

Resume contract (see `run_cpp.growing_ladder(resume=...)`): the saved core at rung index
k is grown into rung k+1 with seed `seed + (k+1)`, exactly as the uninterrupted ladder
would have done, so the resumed ladder is bit-identical. A select-stage rung is only a
valid resume point once the whole select stage has finished (`rung["resumable"]`).

Size: ~ N x (8 W + 2 n_bos + 16) bytes before compression -- a few hundred MB at 512k
dets and L=4 -- written to the shard directory on /nfs_scratch, not the Condor sandbox,
so `request_disk` is unaffected.
"""
from __future__ import annotations

import json
import os

import numpy as np

CKPT_SUFFIX = ".core.npz"
PREV_SUFFIX = ".core.prev.npz"

# settings that do not change the ladder's numbers: a restart may legitimately change them
CONFIG_IGNORE = ("phase0_workers", "max_rung_seconds")


def checkpoint_paths(out_json, ckpt_dir=None):
    """(latest, previous) checkpoint paths for a shard JSON `out_json`. Default: beside
    the JSON, `<stem>.core.npz` / `<stem>.core.prev.npz` (no `.json` in the name, so a
    `*.json` glob never picks them up; rsync shards with `--exclude '*.npz'`)."""
    d = ckpt_dir or os.path.dirname(os.path.abspath(out_json))
    stem = os.path.splitext(os.path.basename(out_json))[0]
    return os.path.join(d, stem + CKPT_SUFFIX), os.path.join(d, stem + PREV_SUFFIX)


def config_key(physical, solver):
    """Canonical string of the settings that determine the ladder (everything in the
    shard manifest's `physical` + `solver` blocks except CONFIG_IGNORE), so a restart
    with different settings is refused instead of silently continuing a different run."""
    def _clean(d):
        out = {}
        for k, v in sorted(d.items()):
            if k in CONFIG_IGNORE:
                continue
            out[k] = _clean(v) if isinstance(v, dict) else v
        return out
    return json.dumps({"physical": _clean(physical), "solver": _clean(solver)},
                      sort_keys=True, default=str)


def save_checkpoint(out_json, res, rung_index, rungs, key, ckpt_dir=None, extra=None):
    """Write the finished core of rung `rung_index` atomically (tmp + rename), rotating
    the previous latest to `.prev`. Returns the path written."""
    latest, prev = checkpoint_paths(out_json, ckpt_dir)
    os.makedirs(os.path.dirname(latest), exist_ok=True)
    tmp = latest + ".tmp.npz"
    meta = {"rung_index": int(rung_index), "core": int(res.n_dets),
            "energy": float(res.energy), "rungs": [int(r) for r in rungs],
            "config_key": key, **(extra or {})}
    np.savez_compressed(tmp, ferm_arr=np.ascontiguousarray(res.ferm_arr, dtype=np.uint64),
                        bos_arr=np.ascontiguousarray(res.bos_arr, dtype=np.uint16),
                        coeffs=np.ascontiguousarray(res.coeffs, dtype=complex),
                        meta=np.array(json.dumps(meta)))
    if os.path.exists(latest):
        os.replace(latest, prev)
    os.replace(tmp, latest)
    return latest


def load_checkpoint(out_json, ckpt_dir=None):
    """Load the latest checkpoint (falling back to `.prev` if the latest is unreadable).
    Returns {"ferm_arr", "bos_arr", "coeffs", **meta, "path"} or None."""
    latest, prev = checkpoint_paths(out_json, ckpt_dir)
    for path in (latest, prev):
        if not os.path.exists(path):
            continue
        try:
            with np.load(path) as z:
                meta = json.loads(str(z["meta"]))
                return {"ferm_arr": z["ferm_arr"], "bos_arr": z["bos_arr"],
                        "coeffs": z["coeffs"], "path": path, **meta}
        except Exception as e:                       # a torn write: try the previous one
            print(f"[checkpoint] unreadable {path}: {e}")
    return None


def drop_previous(out_json, ckpt_dir=None):
    """Remove the `.prev` rotation (done shards keep only their final core)."""
    _, prev = checkpoint_paths(out_json, ckpt_dir)
    if os.path.exists(prev):
        os.remove(prev)


def resume_plan(shard, ckpt, rungs, key):
    """Decide how a restart continues. `shard` = the existing shard JSON (dict) or None,
    `ckpt` = load_checkpoint() or None, `rungs` = the ladder this run would use, `key` =
    config_key() of this run.

    Returns (action, info): action in {"fresh", "done", "resume", "refuse"}; for "resume",
    info = {"start_index", "core", "n_rungs_keep"} where the JSON's rung list must be
    truncated to `n_rungs_keep` (= rung_index + 1) before continuing, so a rung recorded
    after the last checkpoint is redone (bit-identically) rather than duplicated."""
    if shard is None:
        return "fresh", {}
    if shard.get("done"):
        return "done", {}
    if ckpt is None:
        return "fresh", {"reason": "no checkpoint (died before the first resumable rung)"}
    if ckpt.get("config_key") != key:
        return "refuse", {"reason": "checkpoint config differs from this run's settings",
                          "checkpoint_config": ckpt.get("config_key"), "run_config": key}
    if [int(r) for r in ckpt.get("rungs", [])] != [int(r) for r in rungs]:
        return "refuse", {"reason": f"checkpoint ladder {ckpt.get('rungs')} != run ladder {list(rungs)}"}
    k = int(ckpt["rung_index"])
    n_json = len(shard.get("rungs", []))
    if n_json < k + 1:
        return "refuse", {"reason": f"shard JSON has {n_json} rungs but the checkpoint is "
                                    f"rung index {k} (JSON must be written before the core)"}
    return "resume", {"start_index": k + 1, "core": (ckpt["ferm_arr"], ckpt["bos_arr"]),
                      "n_rungs_keep": k + 1, "path": ckpt["path"]}
