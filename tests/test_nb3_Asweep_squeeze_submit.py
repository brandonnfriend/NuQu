"""Guard the squeeze A-sweep + A=0/1 reference grid (`hpc/detsvsL/submit_nb3_Asweep_squeeze.sh`).

Runs the submit script with a stubbed `condor_submit` and checks:
  * every row has the 13 columns the `queue` line names;
  * default `all` = squeeze (gaussian) A=0..10 x L=2..4 x 3 seeds (99) + bare refs A=0,1 x
    L=2..4 x 3 seeds (18);
  * Phase-0 starts = 16 L capped at 64 in the squeeze arm (32/48/64), 32 for the bare refs
    (= cluster 293963); seed stride, select core, stratified starts, workers = P0W present;
  * per-L MAXCORE / PT2CAP / MEM / CPUS / MAXRUNGSEC match 293963's final, working sizing;
  * the frame is threaded into the runner arguments and the log names;
  * `squeeze`, `refs` and env trims select the right subsets; `test` submits one shard.
"""
import os
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_nb3_Asweep_squeeze.sh")
_VARS = ["FRAME", "NB", "L", "A", "SEED", "MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC",
         "PRIO", "P0W", "P0RUNS"]
_SIZING = {"2": ("1024000", "1024000", "32G", "4", "14400", "4"),
           "3": ("1024000", "512000", "128G", "8", "21600", "8"),
           "4": ("512000", "256000", "384G", "8", "21600", "8"),
           "5": ("128000", "65536", "384G", "8", "21600", "8")}


def _run(mode, env_extra=None):
    tmp = tempfile.mkdtemp(prefix="asq_")
    try:
        shutil.copy(_SCRIPT, tmp)
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "condor_submit")
        open(stub, "w").write("#!/bin/sh\necho fake-submit $*\n")
        os.chmod(stub, 0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("AS", "LS", "SEEDS", "REF_LS")}
        env.update(env_extra or {})
        env["PATH"] = binp + os.pathsep + env["PATH"]
        p = subprocess.run(["sh", os.path.basename(_SCRIPT), mode], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, f"{mode}: {p.stdout}\n{p.stderr}"
        cdir = os.path.join(tmp, [d for d in os.listdir(tmp) if d.startswith("campaign_")][0])
        grids = {f[:-4]: [ln.split() for ln in open(os.path.join(cdir, f)) if ln.strip()]
                 for f in os.listdir(cdir) if f.endswith(".txt")}
        subs = {f[:-4]: open(os.path.join(cdir, f)).read() for f in os.listdir(cdir)
                if f.endswith(".sub")}
        return grids, subs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_full_grid():
    grids, subs = _run("all")
    rows = grids["grid"]
    c = {v: i for i, v in enumerate(_VARS)}
    assert {len(r) for r in rows} == {len(_VARS)}
    sq = [r for r in rows if r[c["FRAME"]] == "gaussian"]
    rf = [r for r in rows if r[c["FRAME"]] == "bare"]
    assert len(sq) == 99 and len(rf) == 18 and len(rows) == 117
    assert {(r[c["L"]], r[c["A"]], r[c["SEED"]]) for r in sq} == {
        (str(L), str(A), str(s)) for L in (2, 3, 4) for A in range(11) for s in range(3)}
    assert {r[c["A"]] for r in rf} == {"0", "1"}
    for r in sq:
        assert r[c["P0RUNS"]] == str(min(16 * int(r[c["L"]]), 64))
    assert {r[c["P0RUNS"]] for r in rf} == {"32"}
    for r in rows:
        got = tuple(r[c[k]] for k in ("MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC", "P0W"))
        assert got == _SIZING[r[c["L"]]], (r, got)
        assert r[c["NB"]] == "3"
    sub = subs["Asweep"]
    q = [ln for ln in sub.splitlines() if ln.startswith("queue ")][0]
    assert q.split()[1] == ",".join(_VARS)
    for s in ("NUQU_PHASE0_RUNS=$(P0RUNS)", "NUQU_PHASE0_SEED_STRIDE=1000",
              "NUQU_PHASE0_SELECT_CORE=16000", "NUQU_PHASE0_INIT=stratified",
              "NUQU_PHASE0_WORKERS=$(P0W)", " $(FRAME) $(A) none ", "_$(FRAME)_L$(L)"):
        assert s in sub, s
    assert "NUQU_NOVEL" not in sub and "qis4" not in sub


def test_modes_and_trims():
    g, _ = _run("squeeze")
    assert len(g["grid"]) == 99 and {r[0] for r in g["grid"]} == {"gaussian"}
    g, _ = _run("refs")
    assert len(g["grid"]) == 18 and {r[0] for r in g["grid"]} == {"bare"}
    g, _ = _run("all", {"LS": "2 3 4 5", "SEEDS": "0"})
    sq5 = [r for r in g["grid"] if r[0] == "gaussian" and r[2] == "5"]
    assert len(sq5) == 11 and {r[-1] for r in sq5} == {"64"}
    g, s = _run("test")
    assert len(g["smoke"]) == 1 and g["smoke"][0][0] == "gaussian" and g["smoke"][0][-1] == "64"


def main():
    test_full_grid()
    test_modes_and_trims()
    print("test_nb3_Asweep_squeeze_submit: PASS  (99 squeeze + 18 bare refs; starts 16L<=64 vs 32; "
          "293963 sizing; frame threaded; modes, trims, 1-shard smoke)")


if __name__ == "__main__":
    main()
