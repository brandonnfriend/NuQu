"""Guard the fixed-A n_b=3 classical sweep grid (`hpc/detsvsL/submit_nb3_Asweep.sh`).

Runs the submit script with a stubbed `condor_submit` and checks the emitted grid/.sub:
  * every row has exactly the 10 columns the `queue` line names;
  * the grid is A=2..10 x L=2..5 x seeds{0,1,2} = 108 shards (L=2 relaunched for every A,
    2026-09-25: the old-search L=2 data, incl. 292477's A=8, is superseded);
  * per-L solver settings (MAXCORE, PT2CAP, MAXRUNGSEC) match the 292477 baseline, so
    the grid stays comparable with the baseline; CPUS is 4 at L=2 and 8 at L>=3 (293962:
    deep rungs scale 1.3-1.4x from 4 to 16 threads);
  * the Phase-0 seed stride is set (independent seeds), and JobPrio is seed-major then L;
  * L=4/L=5 ask 192G and L=5 runs 4 select workers (293963 OOM holds at 144G/128G);
  * the chosen search levers are ON (select core 16000 + stratified starts, workers = P0W)
    and the "novel" lever is OFF;
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
_VARS = ["NB", "L", "A", "SEED", "MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC", "PRIO", "P0W"]
# 292477 (submit_nb3_baseline.sh) per-L: MAXCORE PT2CAP MAXRUNGSEC
_BASELINE = {"2": ("1024000", "1024000", "14400"), "3": ("1024000", "512000", "21600"),
             "4": ("512000", "256000", "21600"), "5": ("128000", "65536", "21600")}


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
    for lever in ("NUQU_PHASE0_SELECT_CORE=16000", "NUQU_PHASE0_INIT=stratified",
                  "NUQU_PHASE0_WORKERS=$(P0W)"):
        if lever not in sub:
            fails.append(f"{name}: search lever {lever} missing")
    if "NUQU_NOVEL" in sub:
        fails.append(f"{name}: the novel-config lever should stay OFF")
    if "NUQU_PHASE0_SEED_STRIDE=1000" not in sub:
        fails.append(f"{name}: Phase-0 seed stride missing (seeds would share inits)")
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
              for s in range(3)}
    if cells != expect:
        fails.append(f"grid cells differ: missing {sorted(expect - cells)[:5]}, "
                     f"extra {sorted(cells - expect)[:5]}")
    if len(rows) != 108:
        fails.append(f"{len(rows)} shards, expected 108")
    if {r[c["NB"]] for r in rows} != {"3"}:
        fails.append("not n_b=3")
    for r in rows:
        got = (r[c["MAXCORE"]], r[c["PT2CAP"]], r[c["MAXRUNGSEC"]])
        if r[c["CPUS"]] != ("4" if r[c["L"]] == "2" else "8"):
            fails.append(f"L={r[c['L']]} CPUS {r[c['CPUS']]} (want 4 at L=2, 8 at L>=3)")
            break
        if got != _BASELINE[r[c["L"]]]:
            fails.append(f"L={r[c['L']]} solver settings {got} != 292477 {_BASELINE[r[c['L']]]}")
            break
        if r[c["P0W"]] != ("4" if r[c["L"]] in ("2", "5") else "8"):
            fails.append(f"L={r[c['L']]} select workers {r[c['P0W']]} (want 4 at L=2/5, 8 at L=3/4)")
            break
        if r[c["L"]] in ("4", "5") and r[c["MEM"]] != "192G":
            fails.append(f"L={r[c['L']]} MEM {r[c['MEM']]} (293963 OOM: L=4/5 need 192G)")
            break
        if not r[c["MEM"]].endswith("G"):
            fails.append(f"MEM {r[c['MEM']]!r} is not a Condor size")
            break
    prio = {(r[c["SEED"]], r[c["L"]]): int(r[c["PRIO"]]) for r in rows}
    order = [prio[(s, L)] for s in "012" for L in "2345"]
    if order != sorted(order, reverse=True) or len(set(order)) != len(order):
        fails.append(f"JobPrio not seed-major then L-ascending: {prio}")

    trimmed, _ = _run("all", {"AS": "2 10", "LS": "2 3", "SEEDS": "0"})
    if len(trimmed.get("grid", [])) != 4:
        fails.append(f"env trim gave {len(trimmed.get('grid', []))} shards, expected 4")

    smoke, ssub = _run("test")
    _check_sub(ssub.get("smoke", ""), fails, "smoke")
    if len(smoke.get("smoke", [])) != 1:
        fails.append("test mode should submit exactly one shard")

    assert not fails, "A-sweep grid problems:\n  - " + "\n  - ".join(fails)


def test_phase0_seed_base():
    """Shard seeds get disjoint Phase-0 init blocks; seed 0 is unchanged; stride 1 = legacy."""
    sys.path.insert(0, _ROOT)
    from misc.run_frame_shard import phase0_seed_base
    runs = 32
    blocks = [set(range(phase0_seed_base(s, 1000, runs), phase0_seed_base(s, 1000, runs) + runs))
              for s in range(3)]
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])
    assert phase0_seed_base(0, 1000, runs) == 0 == phase0_seed_base(0, 1, runs)
    assert phase0_seed_base(2, 1, runs) == 2            # legacy overlapping blocks
    try:
        phase0_seed_base(1, 16, runs)
    except ValueError:
        pass
    else:
        raise AssertionError("stride < phase0_runs accepted")


def main():
    try:
        test_asweep_grid()
    except AssertionError as e:
        print("test_nb3_Asweep_submit: FAILED\n", e)
        sys.exit(1)
    test_phase0_seed_base()
    print("test_nb3_Asweep_submit: PASS  (108 shards A=2..10 x L=2..5 x 3 seeds, 292477 solver "
          "settings, 4/8 cpus, levers select+stratified, seed stride, seed-major prio, qis1-3, "
          "2560M disk, env trim, 1-shard smoke, disjoint Phase-0 seed blocks)")


if __name__ == "__main__":
    main()
