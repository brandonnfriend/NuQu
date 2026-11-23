"""EXTEND a finished warm-grow ladder to a deeper one from its saved top core (2026-11-22).

The contract: a ladder run to core K with PT2 capped below K, then EXTENDED to 2K with a
higher PT2 cap, is BIT-IDENTICAL to a fresh run of the deeper ladder -- the saved top rung is
re-recorded under the new PT2 cap (PT2 depends only on the core's determinants) and growth
continues with seed + i. Checked on a tiny L=2 n_b=1 system at two levels:

  * `growing_ladder(resume={..., "rerecord": ...})` reproduces the fresh ladder's records
    (E_var, dE_PT2, n_ext, core arrays) from the re-recorded rung onwards;
  * `misc.run_frame_shard --resume --extend`: a shard finished at max-core 400 / PT2 cap 200,
    re-run at max-core 800 / PT2 cap 800, ends with exactly the rungs of a fresh 800 shard,
    records the extension, keeps its manifest; a re-run is "already done"; an extension that
    changes anything beyond depth / PT2 cap is refused (exit 2); --extend without --resume
    exits 2;
  * `checkpoint.extend_plan` refuses non-prefix ladders and a checkpoint below the top.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Reproducibility contract = SINGLE-THREADED math: no official TrimCI Davidson (the cluster
# path, requirements-hpc.txt) and one OpenMP / BLAS thread. Multithreaded OpenBLAS / OpenMP
# reductions differ in the last bits run to run, which can tip a trim decision on these
# tiny systems (2026-11-22). Must be set BEFORE numpy is imported.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUQU_NUM_WORKERS"):
    os.environ[_k] = "1"
os.environ["NUQU_NO_OFFICIAL_DAVIDSON"] = "1"

import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from classical.trimci import build_from_eft  # noqa: E402
from classical.trimci import checkpoint as ck  # noqa: E402
from classical.trimci.run_cpp import growing_ladder  # noqa: E402

_H = build_from_eft(2, 3, 1, transform="bare")
_A = 3
_FULL = [100, 200, 400, 800]


def _ladder(rungs, pt2_cap, **kw):
    seen = []
    out = growing_ladder(_H, _A, rungs, phase0_runs=2, seed=1, verbose=False,
                         select_core=200, phase0_workers=1, pt2_max_core=pt2_cap,
                         on_rung=lambda rung, res: seen.append(
                             (res.ferm_arr.copy(), res.bos_arr.copy(),
                              np.asarray(res.coeffs).copy(), float(res.energy))), **kw)
    return out, seen


def test_extend_is_bit_identical():
    full, fseen = _ladder(_FULL, 800)
    short, sseen = _ladder(_FULL[:3], 200)
    assert short[2]["dE_pt2"] is None                    # top rung above the old PT2 cap
    f, b, c, e = sseen[2]
    ext, eseen = _ladder(_FULL, 800, resume={"core": (f, b), "start_index": 3,
                                             "rerecord": {"coeffs": c, "energy": e}})
    assert [r["core"] for r in ext] == [400, 800]
    assert ext[0]["rerecorded_from_checkpoint"] and ext[0]["rung_index"] == 2
    for a, r in zip(ext, full[2:]):
        for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext", "rung_index", "phase"):
            assert a[k] == r[k], (k, a[k], r[k])
    for (f1, b1, c1, _), (f2, b2, c2, _) in zip(eseen, fseen[2:]):
        assert np.array_equal(f1, f2) and np.array_equal(b1, b2) and np.array_equal(c1, c2)


def test_extend_plan_refusals():
    c = {"config_key": ck.config_key({"L": 2}, {"max_core": 400, "pt2_max_core": 200, "x": 1}),
         "rungs": _FULL[:3], "rung_index": 2, "ferm_arr": 0, "bos_arr": 0, "coeffs": 0,
         "energy": 0.0, "path": "p"}
    sh = {"done": True, "rungs": [{}, {}, {}]}
    key = ck.config_key({"L": 2}, {"max_core": 800, "pt2_max_core": 800, "x": 1})
    assert ck.extend_plan(sh, c, _FULL, key)[0] == "extend"
    bad = ck.config_key({"L": 2}, {"max_core": 800, "pt2_max_core": 800, "x": 2})
    assert ck.extend_plan(sh, c, _FULL, bad)[0] == "refuse"
    assert ck.extend_plan(sh, c, [100, 300, 400, 800], key)[0] == "refuse"
    assert ck.extend_plan(sh, c, _FULL[:3], key)[0] == "refuse"
    assert ck.extend_plan(sh, dict(c, rung_index=1), _FULL, key)[0] == "refuse"


def _shard(out, max_core, pt2_cap, extra=(), extend=True):
    cmd = [sys.executable, "-m", "misc.run_frame_shard", "--L", "2", "--n_b", "1", "--A", "3",
           "--frame", "bare", "--seed", "1", "--ladder-mode", "independent", "--warm-grow",
           "--ladder-start", "100", "--n-rungs", "4", "--max-core", str(max_core),
           "--pt2-max-core", str(pt2_cap),
           "--phase0-runs", "2", "--phase0-select-core", "200", "--phase0-init", "stratified",
           "--phase0-workers", "1", "--frame-runs", "1", "--orbopt-cycles", "1",
           "--phase0-core", "100", "--resume", *(("--extend",) if extend else ()),
           "--out", out, *extra]
    env = dict(os.environ, MPLBACKEND="Agg")         # inherits the pinned threads above
    env.pop("NUQU_TEST_DIE_AFTER_RUNG_INDEX", None)
    return subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)


def test_shard_extend_end_to_end():
    tmp = tempfile.mkdtemp(prefix="extend_")
    try:
        fresh = os.path.join(tmp, "bare_fresh.json")
        p = _shard(fresh, 800, 800, extend=False)
        assert p.returncode == 0, p.stderr[-2000:]
        F = json.load(open(fresh))
        assert [r["core"] for r in F["rungs"]] == _FULL

        ext = os.path.join(tmp, "bare_ext.json")
        p = _shard(ext, 400, 200, extend=False)
        assert p.returncode == 0, p.stderr[-2000:]
        S = json.load(open(ext))
        assert S["done"] and [r["core"] for r in S["rungs"]] == _FULL[:3]
        assert S["rungs"][2]["dE_pt2"] is None

        p = _shard(ext, 800, 800)
        assert p.returncode == 0 and "EXTEND" in p.stdout, (p.returncode, p.stderr[-2000:])
        E = json.load(open(ext))
        assert E["done"] and [r["core"] for r in E["rungs"]] == _FULL
        for a, b in zip(E["rungs"], F["rungs"]):
            for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext", "phase", "rung_index"):
                assert a[k] == b[k], (k, a["core"], a[k], b[k])
        assert E["manifest"] == S["manifest"], "extension must keep the original manifest"
        assert len(E["extended"]) == 1 and E["extended"][0]["from_ladder"] == _FULL[:3]
        assert E["extended"][0]["to_ladder"] == _FULL and E["extended"][0]["rerecord_index"] == 2
        assert E["rungs"][2].get("rerecorded_from_checkpoint") is True
        with np.load(ck.checkpoint_paths(fresh)[0]) as zf, np.load(ck.checkpoint_paths(ext)[0]) as ze:
            for k in ("ferm_arr", "bos_arr", "coeffs"):
                assert np.array_equal(zf[k], ze[k]), k

        before = open(ext).read()
        p = _shard(ext, 800, 800)
        assert p.returncode == 0 and "already done" in p.stdout and open(ext).read() == before

        # an extension that changes the search (not just depth / PT2 cap) is refused
        mis = os.path.join(tmp, "bare_mis.json")
        assert _shard(mis, 400, 200, extend=False).returncode == 0
        p = _shard(mis, 800, 800, extra=("--phase0-runs", "3"))
        assert p.returncode == 2 and "REFUSING" in p.stderr, (p.returncode, p.stderr[-500:])

        # --extend needs --resume
        cmd = [sys.executable, "-m", "misc.run_frame_shard", "--L", "2", "--seed", "1",
               "--extend", "--out", os.path.join(tmp, "x.json")]
        p = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
        assert p.returncode == 2 and "needs --resume" in p.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_extend_is_bit_identical()
    test_extend_plan_refusals()
    test_shard_extend_end_to_end()
    print("test_extend_ladder: PASS  (extend from the saved top core == fresh deeper ladder, bit "
          "for bit, at ladder and shard level; extension recorded; re-run done; refusals)")


if __name__ == "__main__":
    main()
