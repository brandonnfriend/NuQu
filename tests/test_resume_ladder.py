"""Resume-from-checkpoint for the warm-grow ladder (classical infrastructure C1, 2026-10-08).

The contract: a ladder killed after rung k and resumed from its saved core is BIT-IDENTICAL
to one that never stopped. Checked at two levels on a tiny L=2 n_b=1 system:

  * `growing_ladder(resume=...)`: resuming from the core of rung index 2, or from the last
    select-stage rung, reproduces every later rung's E_var / core arrays exactly, and the
    rung dicts carry `rung_index` / `resumable` (False on select rungs but the last);
  * `misc.run_frame_shard --resume`: a shard process that dies right after saving rung k
    (test hook NUQU_TEST_DIE_AFTER_RUNG_INDEX) and is re-run with the same command line
    ends with the same rungs as an uninterrupted shard, records the restart, keeps its
    original manifest, and leaves one final core checkpoint; a finished shard re-run with
    --resume exits 0 untouched; a checkpoint from different settings is refused (exit 2).
  * `checkpoint.resume_plan` truncates a rung recorded after the last checkpoint.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("NUQU_NUM_WORKERS", "1")

from classical.trimci import build_from_eft  # noqa: E402
from classical.trimci import checkpoint as ck  # noqa: E402
from classical.trimci.run_cpp import growing_ladder  # noqa: E402

_H = build_from_eft(2, 3, 1, transform="bare")
_A = 3
_RUNGS = [100, 200, 400, 800]


def _ladder(**kw):
    cores = []
    out = growing_ladder(_H, _A, _RUNGS, phase0_runs=2, seed=1, verbose=False,
                         select_core=200, phase0_workers=1,
                         on_rung=lambda rung, res: cores.append((res.ferm_arr.copy(),
                                                                 res.bos_arr.copy())), **kw)
    return out, cores


def test_resume_is_bit_identical():
    full, cores = _ladder()
    assert [r["phase"] for r in full] == ["0-select", "0-select", "grow", "grow"]
    assert [r["rung_index"] for r in full] == [0, 1, 2, 3]
    assert [r["resumable"] for r in full] == [False, True, True, True]
    for k in (1, 2):                      # from the last select rung, and from a grow rung
        part, pcores = _ladder(resume={"core": cores[k], "start_index": k + 1})
        assert [r["core"] for r in part] == [r["core"] for r in full[k + 1:]]
        for a, b in zip(part, full[k + 1:]):
            assert a["E_var"] == b["E_var"] and a["dE_pt2"] == b["dE_pt2"], (k, a, b)
            assert a["rung_index"] == b["rung_index"] and a["phase"] == "grow"
        for (f1, b1), (f2, b2) in zip(pcores, cores[k + 1:]):
            assert np.array_equal(f1, f2) and np.array_equal(b1, b2)
    assert _ladder(resume={"core": cores[-1], "start_index": 4})[0] == []


def test_resume_plan_truncates_and_refuses():
    assert ck.resume_plan(None, None, _RUNGS, "k")[0] == "fresh"
    assert ck.resume_plan({"done": True, "rungs": []}, None, _RUNGS, "k")[0] == "done"
    assert ck.resume_plan({"done": False, "rungs": [1]}, None, _RUNGS, "k")[0] == "fresh"
    c = {"config_key": "k", "rungs": _RUNGS, "rung_index": 1, "ferm_arr": 0, "bos_arr": 0,
         "path": "p"}
    sh = {"done": False, "rungs": [{}, {}, {}]}          # rung 2 recorded, core 1 saved
    act, info = ck.resume_plan(sh, c, _RUNGS, "k")
    assert act == "resume" and info["start_index"] == 2 and info["n_rungs_keep"] == 2
    assert ck.resume_plan(sh, dict(c, config_key="other"), _RUNGS, "k")[0] == "refuse"
    assert ck.resume_plan(sh, dict(c, rungs=[100, 200]), _RUNGS, "k")[0] == "refuse"
    assert ck.resume_plan({"done": False, "rungs": [{}]}, c, _RUNGS, "k")[0] == "refuse"
    assert ck.config_key({"L": 2}, {"a": 1, "phase0_workers": 8, "x": {"phase0_workers": 1}}) \
        == ck.config_key({"L": 2}, {"a": 1, "phase0_workers": 2, "x": {}})


def _shard(out, extra_env=None, extra_args=()):
    cmd = [sys.executable, "-m", "misc.run_frame_shard", "--L", "2", "--n_b", "1", "--A", "3",
           "--frame", "bare", "--seed", "1", "--ladder-mode", "independent", "--warm-grow",
           "--ladder-start", "100", "--n-rungs", "4", "--max-core", "800",
           "--phase0-runs", "2", "--phase0-select-core", "200", "--phase0-init", "stratified",
           "--phase0-workers", "1", "--frame-runs", "1", "--orbopt-cycles", "1",
           "--phase0-core", "100", "--resume", "--out", out, *extra_args]
    env = dict(os.environ, NUQU_NUM_WORKERS="1", MPLBACKEND="Agg")
    env.pop("NUQU_TEST_DIE_AFTER_RUNG_INDEX", None)
    env.update(extra_env or {})
    return subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)


def test_shard_resume_end_to_end():
    tmp = tempfile.mkdtemp(prefix="resume_")
    try:
        full = os.path.join(tmp, "bare_full.json")
        p = _shard(full)
        assert p.returncode == 0, p.stderr[-2000:]
        F = json.load(open(full))
        assert F["done"] and [r["core"] for r in F["rungs"]] == _RUNGS
        assert all("mem" in r and r["mem"]["peak_rss_mb"] > 0 for r in F["rungs"])
        latest, prev = ck.checkpoint_paths(full)
        assert os.path.exists(latest) and not os.path.exists(prev)
        assert F["checkpoint"]["rung_index"] == 3 and F["checkpoint"]["core"] == 800

        # killed after rung index 2 (core 400) -> resumed with the SAME command line
        part = os.path.join(tmp, "bare_part.json")
        p = _shard(part, {"NUQU_TEST_DIE_AFTER_RUNG_INDEX": "2"})
        assert p.returncode == 137, (p.returncode, p.stderr[-800:])
        P = json.load(open(part))
        assert not P["done"] and len(P["rungs"]) == 3 and P["checkpoint"]["rung_index"] == 2
        p = _shard(part)
        assert p.returncode == 0 and "RESUME from rung index 2" in p.stdout, p.stderr[-2000:]
        R = json.load(open(part))
        assert R["done"] and len(R["rungs"]) == 4
        for a, b in zip(R["rungs"], F["rungs"]):
            for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext", "phase", "rung_index"):
                assert a[k] == b[k], (k, a[k], b[k])
        assert R["manifest"] == P["manifest"], "resume must keep the original manifest"
        assert len(R["resumed"]) == 1 and R["resumed"][0]["from_rung_index"] == 2
        assert R["wall_s"] >= P["wall_s"]
        # the final cores are the same, bit for bit
        with np.load(ck.checkpoint_paths(full)[0]) as zf, np.load(ck.checkpoint_paths(part)[0]) as zp:
            assert np.array_equal(zf["ferm_arr"], zp["ferm_arr"])
            assert np.array_equal(zf["bos_arr"], zp["bos_arr"])
            assert np.array_equal(zf["coeffs"], zp["coeffs"])

        # a finished shard is left alone
        before = open(part).read()
        p = _shard(part)
        assert p.returncode == 0 and "already done" in p.stdout and open(part).read() == before

        # a checkpoint from different settings is refused
        mis = os.path.join(tmp, "bare_mis.json")
        p = _shard(mis, {"NUQU_TEST_DIE_AFTER_RUNG_INDEX": "2"})
        assert p.returncode == 137
        p = _shard(mis, extra_args=("--pt2-max-core", "400"))
        assert p.returncode == 2 and "REFUSING" in p.stderr, (p.returncode, p.stderr[-500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_resume_is_bit_identical()
    test_resume_plan_truncates_and_refuses()
    test_shard_resume_end_to_end()
    print("test_resume_ladder: PASS  (resume from a saved core == uninterrupted ladder, bit for "
          "bit; shard die-after-rung-2 + --resume == full shard; done untouched; mismatch refused)")


if __name__ == "__main__":
    main()
