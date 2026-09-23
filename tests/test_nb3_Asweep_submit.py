"""Guard the fixed-A n_b=3 classical sweep grid (`hpc/detsvsL/submit_nb3_Asweep.sh`).

Runs the submit script with a stubbed `condor_submit` and checks the emitted grid/.sub:
  * every row has exactly the 10 columns the `queue` line names;
  * the grid is A=2..10 x L=2..5 x seeds{0,1,2} minus the reused L=2 A=8 cell (105 shards);
  * per-L solver settings (MAXCORE, PT2CAP, CPUS, MAXRUNGSEC) match the 292477 baseline, so
    the new points and the reused L=2 A=8 point come from one pipeline;
  * explicit A (`filling none`), n_b=3, the qis1-3 pin and the small disk request survive;
  * the env overrides trim the grid, and `test` submits exactly one shard.
"""
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_nb3_Asweep.sh")
_VARS = ["NB", "L", "A", "SEED", "MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC", "PRIO"]
# 292477 (submit_nb3_baseline.sh) per-L: MAXCORE PT2CAP CPUS MAXRUNGSEC
_BASELINE = {"2": ("1024000", "1024000", "16", "14400"), "3": ("1024000", "512000", "16", "21600"),
             "4": ("512000", "256000", "24", "21600"), "5": ("128000", "65536", "24", "21600")}


def _run(mode, env_extra=None):
    tmp = tempfile.mkdtemp(prefix="asweep_")
    try:
        shutil.copy(_SCRIPT, tmp)
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "condor_submit")
        with open(stub, "w") as f:
            f.write("#!/bin/sh\necho fake-submit $*\n")
        os.chmod(stub, 0o755)
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"], **(env_extra or {}))
        for k in ("AS", "LS", "SEEDS"):
            if not env_extra or k not in env_extra:
                env.pop(k, None)
        p = subprocess.run(["sh", os.path.basename(_SCRIPT), mode], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, f"submit failed ({mode}):\n{p.stdout}\n{p.stderr}"
        cdir = os.path.join(tmp, [d for d in os.listdir(tmp) if d.startswith("campaign_")][0])
        grids, subs = {}, {}
        for fn in os.listdir(cdir):
            path = os.path.join(cdir, fn)
            if fn.endswith(".txt"):
                grids[fn[:-4]] = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
            elif fn.endswith(".sub"):
                subs[fn[:-4]] = open(path).read()
        return grids, subs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _check_sub(sub, fails, name):
    q = [ln for ln in sub.splitlines() if ln.startswith("queue ")]
    if len(q) != 1 or q[0].split()[1] != ",".join(_VARS):
        fails.append(f"{name}: queue line {q} does not name {_VARS}")
    if "qis1.hep.wisc.edu" not in sub or "qis4" in sub:
        fails.append(f"{name}: qis1-3 pin missing/incorrect")
    if "request_disk            = 2560M" not in sub:
        fails.append(f"{name}: request_disk not the 2560M that fits qis1/qis3")
    if " bare $(A) none " not in sub:
        fails.append(f"{name}: not an explicit-A bare run (filling must be 'none')")
    if "-nb$(NB)" not in sub or "NUQU_N_B=$(NB)" not in sub:
        fails.append(f"{name}: n_b not threaded into campaign dir / env")


def test_asweep_grid():
    fails = []
    grids, subs = _run("all")
    rows = grids.get("grid", [])
    _check_sub(subs.get("Asweep", ""), fails, "Asweep")
    c = {v: i for i, v in enumerate(_VARS)}
    for i, r in enumerate(rows):
        if len(r) != len(_VARS):
            fails.append(f"row {i}: {len(r)} columns: {r}")
    rows = [r for r in rows if len(r) == len(_VARS)]
    cells = {(r[c["L"]], r[c["A"]], r[c["SEED"]]) for r in rows}
    expect = {(str(L), str(A), str(s)) for L in range(2, 6) for A in range(2, 11)
              for s in range(3)} - {("2", "8", str(s)) for s in range(3)}
    if cells != expect:
        fails.append(f"grid cells differ: missing {sorted(expect - cells)[:5]}, "
                     f"extra {sorted(cells - expect)[:5]}")
    if len(rows) != 105:
        fails.append(f"{len(rows)} shards, expected 105")
    if {r[c["NB"]] for r in rows} != {"3"}:
        fails.append("not n_b=3")
    for r in rows:
        got = (r[c["MAXCORE"]], r[c["PT2CAP"]], r[c["CPUS"]], r[c["MAXRUNGSEC"]])
        if got != _BASELINE[r[c["L"]]]:
            fails.append(f"L={r[c['L']]} solver settings {got} != 292477 {_BASELINE[r[c['L']]]}")
            break
        if not r[c["MEM"]].endswith("G"):
            fails.append(f"MEM {r[c['MEM']]!r} is not a Condor size")
            break
    prio = {r[c["L"]]: int(r[c["PRIO"]]) for r in rows}
    if not prio["2"] > prio["3"] > prio["4"] > prio["5"]:
        fails.append(f"JobPrio not L-ascending: {prio}")

    trimmed, _ = _run("all", {"AS": "2 10", "LS": "2 3", "SEEDS": "0"})
    if len(trimmed.get("grid", [])) != 4:
        fails.append(f"env trim gave {len(trimmed.get('grid', []))} shards, expected 4")

    smoke, ssub = _run("test")
    _check_sub(ssub.get("smoke", ""), fails, "smoke")
    if len(smoke.get("smoke", [])) != 1:
        fails.append("test mode should submit exactly one shard")

    assert not fails, "A-sweep grid problems:\n  - " + "\n  - ".join(fails)


def main():
    try:
        test_asweep_grid()
    except AssertionError as e:
        print("test_nb3_Asweep_submit: FAILED\n", e)
        sys.exit(1)
    print("test_nb3_Asweep_submit: PASS  (105 shards A=2..10 x L=2..5 x 3 seeds minus L2A8, "
          "292477 solver settings, explicit A, qis1-3, 2560M disk, env trim, 1-shard smoke)")


if __name__ == "__main__":
    main()
