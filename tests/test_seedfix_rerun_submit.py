"""Guard the seed-sharing close-out re-runs (`hpc/detsvsL/submit_seedfix_rerun.sh`).

Stubbed condor_submit; checks that every arm reproduces its legacy campaign's grid and settings
with ONLY the search levers changed:
  * baseline = 292477's filling-1.0 cells, seeds {1,2} x L=2..5, 11 rungs with PT2 to the
    per-L cap, 32 starts; SEEDS="0 1 2" adds seed 0; NB2=1 adds the 292478 n_b=2 arm;
  * gate = nb_energy_gate (L=2: n_b{2,3,4} x A{0,1,32}, 8 rungs, no PT2, 64 starts, n_b=4 at
    128k) + nb_energy_gate_L3 (n_b{2,3} x A{0,1,27}, 9 rungs), seeds 1-4;
  * volscaling = the primary @256k (9 rungs) and deep @512k (10 rungs) cells, seeds {1,2};
  * every .sub carries the stride/select/stratified levers, the auto-OOM policy lines, the
    qis1-3 pin and the 15 queue columns; `test` is one gate shard.
"""
import os
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_seedfix_rerun.sh")
_VARS = "ARM,NB,L,A,FILL,SEED,MAXCORE,NRUNGS,PT2CAP,P0RUNS,P0W,MEM,CPUS,MAXRUNGSEC,PRIO"
_LEVERS = "NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified"
_POLICY = ["request_memory          = ifThenElse($(MEMGROW_MB) < $(MEMCAP_MB), $(MEMGROW_MB), $(MEMCAP_MB))",
           "periodic_release        = (HoldReasonCode == 34) && (NumHolds < 4)",
           "on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))"]
C = {k: i for i, k in enumerate(_VARS.split(","))}


def _run(mode, env_extra=None):
    tmp = tempfile.mkdtemp(prefix="seedfix_")
    try:
        shutil.copy(_SCRIPT, tmp)
        binp = os.path.join(tmp, "bin"); os.makedirs(binp)
        with open(os.path.join(binp, "condor_submit"), "w") as f:
            f.write("#!/bin/sh\necho fake-submit $*\n")
        os.chmod(os.path.join(binp, "condor_submit"), 0o755)
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"], **(env_extra or {}))
        for k in ("SEEDS", "GATE_SEEDS", "VOL_SEEDS", "NB2"):
            if not env_extra or k not in env_extra:
                env.pop(k, None)
        p = subprocess.run(["sh", os.path.basename(_SCRIPT), mode], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        cdir = os.path.join(tmp, [d for d in os.listdir(tmp) if d.startswith("campaign_")][0])
        grids, subs = {}, {}
        for fn in os.listdir(cdir):
            if fn.endswith(".txt"):
                grids[fn[:-4]] = [ln.split() for ln in open(os.path.join(cdir, fn)) if ln.strip()]
            elif fn.endswith(".sub"):
                subs[fn[:-4]] = open(os.path.join(cdir, fn)).read()
        return grids, subs, p.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _check_sub(sub):
    assert f"queue {_VARS} from " in sub and _LEVERS in sub
    assert "NUQU_PHASE0_RUNS=$(P0RUNS) NUQU_PHASE0_WORKERS=$(P0W)" in sub
    assert "NUQU_N_RUNGS=$(NRUNGS) NUQU_PT2_MAX_CORE=$(PT2CAP)" in sub
    for line in _POLICY:
        assert line in sub, line
    assert "qis3.hep.wisc.edu" in sub and "qis4" not in sub
    assert "request_disk            = 2560M" in sub      # qis1/qis3 have ~3 GB execute disk (HPC_WORKFLOW 6b)
    assert "$(L) $(SEED) " in sub and "-$(ARM)-nb$(NB) bare $(A) $(FILL) $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)" in sub


def test_baseline_arm_matches_292477():
    grids, subs, out = _run("baseline")
    rows = grids["baseline"]; _check_sub(subs["baseline"])
    assert len(rows) == 8 and all(len(r) == 15 for r in rows) and "jobs=8" in out
    assert {(r[C["L"]], r[C["SEED"]]) for r in rows} == {(L, s) for L in "2345" for s in "12"}
    assert all(r[C["NB"]] == "3" and r[C["FILL"]] == "1.0" and r[C["A"]] == "1" and r[C["NRUNGS"]] == "11"
               and r[C["P0RUNS"]] == "32" for r in rows)
    caps = {r[C["L"]]: (r[C["MAXCORE"]], r[C["PT2CAP"]]) for r in rows}       # 292477's per-L caps
    assert caps == {"2": ("1024000", "1024000"), "3": ("1024000", "512000"),
                    "4": ("512000", "256000"), "5": ("128000", "65536")}
    assert {r[C["L"]]: r[C["MEM"]] for r in rows} == {"2": "64", "3": "192", "4": "384", "5": "384"}
    grids, _, _ = _run("baseline", {"SEEDS": "0 1 2", "NB2": "1"})
    rows = grids["baseline"]
    assert len(rows) == 12 + 3
    assert [(r[C["NB"]], r[C["L"]], r[C["SEED"]]) for r in rows if r[C["NB"]] == "2"] == \
        [("2", "2", "0"), ("2", "3", "0"), ("2", "4", "0")]


def test_gate_arm_matches_energy_gate_grids():
    grids, subs, _ = _run("gate")
    rows = grids["gate"]; _check_sub(subs["gate"])
    assert len(rows) == 36 + 24
    l2 = [r for r in rows if r[C["L"]] == "2"]; l3 = [r for r in rows if r[C["L"]] == "3"]
    assert {(r[C["NB"]], r[C["A"]]) for r in l2} == {(nb, a) for nb in "234" for a in ("0", "1", "32")}
    assert {(r[C["NB"]], r[C["A"]]) for r in l3} == {(nb, a) for nb in "23" for a in ("0", "1", "27")}
    assert {r[C["SEED"]] for r in rows} == {"1", "2", "3", "4"}
    assert all(r[C["PT2CAP"]] == "1" and r[C["P0RUNS"]] == "64" and r[C["FILL"]] == "none" for r in rows)
    assert all(r[C["NRUNGS"]] == "8" for r in l2) and all(r[C["NRUNGS"]] == "9" for r in l3)
    assert {r[C["MAXCORE"]] for r in l2 if r[C["NB"]] == "4"} == {"131072"}
    assert {r[C["MAXCORE"]] for r in l2 if r[C["NB"]] != "4"} == {"262144"}
    grids, _, _ = _run("gate", {"GATE_SEEDS": "1"})
    assert len(grids["gate"]) == 9 + 6


def test_volscaling_arm_matches_primary_and_deep():
    grids, subs, _ = _run("volscaling")
    rows = grids["volscaling"]; _check_sub(subs["volscaling"])
    prim = [r for r in rows if r[C["ARM"]] == "volscaling"]; deep = [r for r in rows if r[C["ARM"]] == "voldeep"]
    assert len(prim) == 12 and len(deep) == 8
    assert {(r[C["NB"]], r[C["L"]], r[C["A"]]) for r in prim} == \
        {("3", "4", "0"), ("3", "4", "1"), ("4", "3", "0"), ("4", "3", "1"), ("4", "4", "0"), ("4", "4", "1")}
    assert all(r[C["MAXCORE"]] == "262144" and r[C["NRUNGS"]] == "9" for r in prim)
    assert {(r[C["NB"]], r[C["L"]]) for r in deep} == {("3", "3"), ("3", "4"), ("4", "3"), ("4", "4")}
    assert all(r[C["MAXCORE"]] == "524288" and r[C["NRUNGS"]] == "10" and r[C["A"]] == "1" for r in deep)
    assert {r[C["SEED"]] for r in rows} == {"1", "2"}


def test_all_and_test_modes():
    grids, subs, out = _run("all")
    assert set(grids) == {"baseline", "gate", "volscaling"} and set(subs) == set(grids)
    assert sum(len(v) for v in grids.values()) == 88
    assert all(int(r[C["PRIO"]]) < 10 for v in grids.values() for r in v)   # below the squeeze campaign
    grids, subs, out = _run("test")
    assert list(grids) == ["smoke"] and len(grids["smoke"]) == 1
    r = grids["smoke"][0]
    assert (r[C["ARM"]], r[C["NB"]], r[C["L"]], r[C["A"]], r[C["SEED"]]) == ("gate", "3", "2", "1", "1")
    _check_sub(subs["smoke"])
