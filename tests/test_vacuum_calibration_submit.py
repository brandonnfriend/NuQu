"""Guard `hpc/detsvsL/submit_vacuum_calibration.sh` (2026-11-24): A=0 calibration runs at L=4/5.

Runs the script with a stubbed `condor_submit` against a fake 294246 campaign directory and
checks the three arms: bareL4 EXTENDS seeds 0,1 to 2,048,000 / PT2 1,024,000 in 294246's
directory with its bare-ref settings (32 starts) and a 960 GB cap, backing the 512k JSONs up;
bareL5 is fresh at 293963's sizing (128k / 65,536, 32 starts, 4 workers, 256 GB); sqL5 is fresh
squeezed to 1,024,000 / 512,000; the ladder reaches MAXCORE in NRUNGS doublings; a missing
checkpoint is refused (exit 2); a single arm submits only itself.
"""
import os
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_vacuum_calibration.sh")
_SRC = "20261008-184336-nb3AsweepSq"
_VARS = ["CAMP", "FRAME", "NB", "L", "A", "SEED", "MAXCORE", "PT2CAP", "NRUNGS", "MEM", "CPUS",
         "MAXRUNGSEC", "PRIO", "P0W", "P0RUNS", "EXT"]


def _setup(ckpt_seeds=(0, 1, 2)):
    tmp = tempfile.mkdtemp(prefix="vaccal_")
    shutil.copy(_SCRIPT, tmp)
    sh = os.path.join(tmp, f"campaign_{_SRC}-nb3", "shards")
    os.makedirs(sh)
    for s in range(3):
        open(os.path.join(sh, f"bare_L4d3_A0_s{s}.json"), "w").write("{}")
        if s in ckpt_seeds:
            open(os.path.join(sh, f"bare_L4d3_A0_s{s}.core.npz"), "w").write("x")
    os.makedirs(os.path.join(tmp, "bin"))
    stub = os.path.join(tmp, "bin", "condor_submit")
    open(stub, "w").write("#!/bin/sh\necho fake-submit $*\n")
    os.chmod(stub, 0o755)
    return tmp, sh


def _run(tmp, mode):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SEEDS_", "MAXHOLDS"))}
    env["PATH"] = os.path.join(tmp, "bin") + os.pathsep + env["PATH"]
    return subprocess.run(["sh", "submit_vacuum_calibration.sh", mode], cwd=tmp, env=env,
                          capture_output=True, text=True)


def _arms(tmp):
    cdir = [d for d in os.listdir(tmp) if d.startswith("campaign_") and d.endswith("vaccal")][0]
    out = {}
    for f in os.listdir(os.path.join(tmp, cdir)):
        if f.endswith(".txt"):
            arm = f[:-4]
            rows = [dict(zip(_VARS, ln.split())) for ln in open(os.path.join(tmp, cdir, f)) if ln.strip()]
            out[arm] = (rows, open(os.path.join(tmp, cdir, arm + ".sub")).read())
    return cdir, out


def test_all_arms():
    tmp, sh = _setup()
    try:
        p = _run(tmp, "all")
        assert p.returncode == 0, p.stderr
        cdir, arms = _arms(tmp)
        assert set(arms) == {"bareL4", "bareL5", "sqL5"}
        for arm, (rows, sub) in arms.items():
            assert all(len(r) == len(_VARS) for r in rows)
            for r in rows:   # the ladder 1000 * 2^k (k < NRUNGS, capped at MAXCORE) ends AT MAXCORE
                assert int(r["MAXCORE"]) in {1000 * 2 ** k for k in range(int(r["NRUNGS"]))}
                assert r["A"] == "0" and r["NB"] == "3"
            assert f"queue {','.join(_VARS)} from" in sub
            for s in ("NUQU_EXTEND=$(EXT)", "NUQU_PHASE0_RUNS=$(P0RUNS)", "NUQU_N_RUNGS=$(NRUNGS)",
                      "NUQU_PT2_MAX_CORE=$(PT2CAP)", "NUQU_PHASE0_SEED_STRIDE=1000",
                      "NUQU_PHASE0_SELECT_CORE=16000", "NUQU_PHASE0_INIT=stratified",
                      "$(L) $(SEED) $(CAMP) $(FRAME) $(A) none $(MAXCORE) independent 4 3 1000",
                      "periodic_release", "request_disk            = 300M"):
                assert s in sub, (arm, s)

        rows, sub = arms["bareL4"]
        assert [r["SEED"] for r in rows] == ["0", "1"]
        for r in rows:
            assert (r["CAMP"], r["FRAME"], r["L"], r["MAXCORE"], r["PT2CAP"], r["P0RUNS"], r["EXT"]) == \
                (f"{_SRC}-nb3", "bare", "4", "2048000", "1024000", "32", "1")
        assert f"MEMCAP_MB               = {960 * 1024}" in sub
        bk = os.path.join(sh, "pre_extend_512k")
        assert sorted(os.listdir(bk)) == ["bare_L4d3_A0_s0.json", "bare_L4d3_A0_s1.json"]

        rows, _ = arms["bareL5"]
        assert [r["SEED"] for r in rows] == ["0", "1", "2"]
        for r in rows:
            assert (r["CAMP"], r["FRAME"], r["L"], r["MAXCORE"], r["PT2CAP"], r["P0RUNS"], r["P0W"],
                    r["MEM"], r["EXT"]) == (cdir[len("campaign_"):] + "-nb3", "bare", "5", "128000",
                                            "65536", "32", "4", "256", "0")

        rows, _ = arms["sqL5"]
        assert [r["SEED"] for r in rows] == ["0", "1"]
        for r in rows:
            assert (r["FRAME"], r["L"], r["MAXCORE"], r["PT2CAP"], r["EXT"]) == \
                ("gaussian", "5", "1024000", "512000", "0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_single_arm_and_refusal():
    tmp, _ = _setup()
    try:
        p = _run(tmp, "bareL5")
        assert p.returncode == 0, p.stderr
        _, arms = _arms(tmp)
        assert set(arms) == {"bareL5"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp, _ = _setup(ckpt_seeds=(0,))
    try:
        p = _run(tmp, "bareL4")
        assert p.returncode == 2 and "bare_L4d3_A0_s1" in p.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_all_arms()
    test_single_arm_and_refusal()
    print("test_vacuum_calibration_submit: PASS")


if __name__ == "__main__":
    main()
