"""Guard `hpc/detsvsL/submit_squeeze_L4_deep.sh` (2026-11-22), the squeezed L=4 extension.

Runs the script with a stubbed `condor_submit` against a fake 294246 campaign directory and
checks: 33 rows (A=0..10 x seeds 0..2) at L=4; max-core 1,024,000 and PT2 cap 512,000; 24 cpus
and a 256 GB auto-growing request; NUQU_EXTEND=1 and 294246's search settings verbatim (64
starts, stride 1000, select 16000, stratified); the runner writes into 294246's shard directory;
seed 0 and A=0 first; the 512k JSONs are backed up once; `test` submits one shard (A=10 s0);
a missing checkpoint is refused (exit 2).
"""
import os
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_squeeze_L4_deep.sh")
_SRC = "20261008-184336-nb3AsweepSq"
_VARS = ["NB", "L", "A", "SEED", "MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC", "PRIO",
         "P0W", "P0RUNS"]


def _setup(skip=()):
    tmp = tempfile.mkdtemp(prefix="sqL4_")
    shutil.copy(_SCRIPT, tmp)
    sh = os.path.join(tmp, f"campaign_{_SRC}-nb3", "shards")
    os.makedirs(sh)
    for A in range(11):
        for s in range(3):
            stem = f"gaussian_L4d3_A{A}_s{s}"
            open(os.path.join(sh, stem + ".json"), "w").write("{}")
            if (A, s) not in skip:
                open(os.path.join(sh, stem + ".core.npz"), "w").write("x")
    os.makedirs(os.path.join(tmp, "bin"))
    stub = os.path.join(tmp, "bin", "condor_submit")
    open(stub, "w").write("#!/bin/sh\necho fake-submit $*\n")
    os.chmod(stub, 0o755)
    return tmp, sh


def _run(tmp, mode):
    env = {k: v for k, v in os.environ.items() if k not in ("AS", "SEEDS", "MEM", "CPUS")}
    env["PATH"] = os.path.join(tmp, "bin") + os.pathsep + env["PATH"]
    return subprocess.run(["sh", "submit_squeeze_L4_deep.sh", mode], cwd=tmp, env=env,
                          capture_output=True, text=True)


def _read(tmp):
    cdir = [d for d in os.listdir(tmp) if d.startswith("campaign_") and d.endswith("sqL4deep")][0]
    grid = [ln.split() for ln in open(os.path.join(tmp, cdir, "grid.txt")) if ln.strip()]
    sub = [open(os.path.join(tmp, cdir, f)).read() for f in os.listdir(os.path.join(tmp, cdir))
           if f.endswith(".sub")][0]
    return grid, sub


def test_full_grid():
    tmp, sh = _setup()
    try:
        p = _run(tmp, "all")
        assert p.returncode == 0, p.stderr
        grid, sub = _read(tmp)
        c = {v: i for i, v in enumerate(_VARS)}
        assert len(grid) == 33 and {len(r) for r in grid} == {len(_VARS)}
        assert {(r[c["A"]], r[c["SEED"]]) for r in grid} == {(str(A), str(s)) for A in range(11)
                                                             for s in range(3)}
        for r in grid:
            assert (r[c["L"]], r[c["MAXCORE"]], r[c["PT2CAP"]], r[c["MEM"]], r[c["CPUS"]],
                    r[c["P0RUNS"]]) == ("4", "1024000", "512000", "256", "24", "64")
        prio = {(r[c["A"]], r[c["SEED"]]): int(r[c["PRIO"]]) for r in grid}
        assert max(prio, key=prio.get) == ("0", "0")
        assert min(prio[(a, "0")] for a in map(str, range(11))) > max(prio[(a, "1")] for a in map(str, range(11)))
        q = [ln for ln in sub.splitlines() if ln.startswith("queue ")][0]
        assert q.split()[1] == ",".join(_VARS)
        for s in ("NUQU_EXTEND=1", "NUQU_PHASE0_RUNS=$(P0RUNS)", "NUQU_PHASE0_SEED_STRIDE=1000",
                  "NUQU_PHASE0_SELECT_CORE=16000", "NUQU_PHASE0_INIT=stratified",
                  "NUQU_PT2_MAX_CORE=$(PT2CAP)", "NUQU_N_RUNGS=11", "NUQU_DEEP_SOLVE=1",
                  f"$(SEED) {_SRC}-nb$(NB) gaussian $(A) none $(MAXCORE) independent 4 3 1000",
                  "periodic_release", "request_cpus            = $(CPUS)", "request_disk            = 300M"):
            assert s in sub, s
        bk = os.path.join(sh, "pre_extend_512k")
        assert len([f for f in os.listdir(bk) if f.endswith(".json")]) == 33
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_smoke_and_refusal():
    tmp, _ = _setup()
    try:
        p = _run(tmp, "test")
        assert p.returncode == 0, p.stderr
        grid, _ = _read(tmp)
        assert len(grid) == 1 and grid[0][2:4] == ["10", "0"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp, _ = _setup(skip={(5, 1)})
    try:
        p = _run(tmp, "all")
        assert p.returncode == 2 and "A5_s1" in p.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_full_grid()
    test_smoke_and_refusal()
    print("test_squeeze_L4_deep_submit: PASS  (33 extension rows, 1024k / PT2 512k, 24 cpus, 300M disk, 256 GB "
          "auto-grow, NUQU_EXTEND=1 + 294246 settings, backup, 1-shard smoke, missing-checkpoint refusal)")


if __name__ == "__main__":
    main()
