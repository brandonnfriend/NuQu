"""Guard the n_b=3 classical-baseline campaign grid (`hpc/detsvsL/submit_nb3_baseline.sh`).

A Condor `queue <vars> from <file>` silently mis-binds when a grid row has the wrong
number of columns -- the submit either fails to parse or, worse, binds MEM to a core
count. That is expensive to discover on the cluster, so this test runs the submit
script with a stubbed `condor_submit` and checks the emitted grids and `.sub` files:

  * every row has exactly the 8 columns the `queue` line names;
  * the headline arm is n_b=3, L=2..5, seeds {0,1,2};
  * the paired comparison arm is n_b=2 at identical per-L sizing (audit P0-1 wants the
    n_b=2 -> 3 delta measured at matched settings);
  * PT2 reaches at least one doubling below the ladder top (PT2CAP >= MAXCORE/2) -- audit
    P0-2 needs post-collapse PT2 points for a defensible extrapolation, and 290832 capped PT2
    pre-collapse. The bound is one doubling rather than "every rung" because PT2 costs
    ~650-700 B per external determinant at depth and PT2-everywhere held two L=3 shards;
  * the qis1-3 allocation pin and the campaign-per-n_b output split survive edits.
"""
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "hpc", "detsvsL", "submit_nb3_baseline.sh")
_QUEUE_VARS = ["NB", "L", "SEED", "MAXCORE", "PT2CAP", "MEM", "CPUS", "MAXRUNGSEC"]


def _run_arm(mode):
    """Run the submit script with a fake condor_submit; return {name: parsed rows}, subs."""
    tmp = tempfile.mkdtemp(prefix="nb3sub_")
    try:
        shutil.copy(_SCRIPT, tmp)
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "condor_submit")
        with open(stub, "w") as f:
            f.write("#!/bin/sh\necho fake-submit $*\n")
        os.chmod(stub, 0o755)
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"])
        p = subprocess.run(["sh", os.path.basename(_SCRIPT), mode], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, f"submit script failed ({mode}):\n{p.stdout}\n{p.stderr}"
        camp = [d for d in os.listdir(tmp) if d.startswith("campaign_")]
        assert len(camp) == 1, f"expected one campaign dir, got {camp}"
        cdir = os.path.join(tmp, camp[0])
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


def _check(rows, sub, name, fails):
    col = {v: i for i, v in enumerate(_QUEUE_VARS)}
    qline = [ln for ln in sub.splitlines() if ln.startswith("queue ")]
    if len(qline) != 1 or qline[0].split()[1] != ",".join(_QUEUE_VARS):
        fails.append(f"{name}: queue line does not name {_QUEUE_VARS}: {qline}")
    for i, r in enumerate(rows):
        if len(r) != len(_QUEUE_VARS):
            fails.append(f"{name} row {i}: {len(r)} columns, expected {len(_QUEUE_VARS)}: {r}")
            continue
        # PT2 must reach at least one doubling below the ladder top, so the post-collapse
        # basin keeps >=3 PT2 rungs for the T2 extrapolation. Requiring PT2CAP == MAXCORE
        # (the original rule) held two L=3 shards on memory: the EN-PT2 map is ~650-700 B per
        # external determinant at depth, so PT2 at L=3's 1M rung needs ~225 GB.
        cap, top = int(r[col["PT2CAP"]]), int(r[col["MAXCORE"]])
        if cap < top // 2:
            fails.append(f"{name} row {i}: PT2CAP={cap} is more than one doubling below "
                         f"MAXCORE={top} -- too few post-collapse PT2 rungs for T2")
        if cap > top:
            fails.append(f"{name} row {i}: PT2CAP={cap} exceeds MAXCORE={top}")
        if not r[col["MEM"]].endswith("G"):
            fails.append(f"{name} row {i}: MEM {r[col['MEM']]!r} is not a Condor size")
    if "qis1.hep.wisc.edu" not in sub or "qis4" in sub:
        fails.append(f"{name}: qis1-3 allocation pin missing/incorrect")
    if "-nb$(NB)" not in sub:
        fails.append(f"{name}: campaign dir is not split per n_b -> shard filenames collide")


def test_nb3_baseline_grids():
    fails = []
    grids, subs = _run_arm("all")
    if set(grids) != {"grid_nb3", "grid_nb2"}:
        fails.append(f"'all' should emit both arms, got {sorted(grids)}")
    col = {v: i for i, v in enumerate(_QUEUE_VARS)}

    nb3 = grids.get("grid_nb3", [])
    _check(nb3, subs.get("nb3", ""), "nb3", fails)
    if {r[col["NB"]] for r in nb3} != {"3"}:
        fails.append("headline arm is not n_b=3")
    if sorted({r[col["L"]] for r in nb3}) != ["2", "3", "4", "5"]:
        fails.append(f"headline L set is {sorted({r[col['L']] for r in nb3})}, expected 2..5")
    if sorted({r[col["SEED"]] for r in nb3}) != ["0", "1", "2"]:
        fails.append("headline arm needs >1 seed for the error-bar spread")
    if len(nb3) != 12:
        fails.append(f"headline arm has {len(nb3)} shards, expected 12 (4 L x 3 seeds)")

    nb2 = grids.get("grid_nb2", [])
    _check(nb2, subs.get("nb2", ""), "nb2", fails)
    if {r[col["NB"]] for r in nb2} != {"2"}:
        fails.append("comparison arm is not n_b=2")
    # matched sizing: same (MAXCORE, PT2CAP, MEM, CPUS, MAXRUNGSEC) per L as the nb3 arm
    size3 = {r[col["L"]]: tuple(r[3:]) for r in nb3}
    for r in nb2:
        if size3.get(r[col["L"]]) != tuple(r[3:]):
            fails.append(f"n_b=2 L={r[col['L']]} sizing {tuple(r[3:])} != n_b=3 "
                         f"{size3.get(r[col['L']])} -- the delta would not be paired")

    smoke, ssub = _run_arm("test")
    _check(smoke.get("smoke", []), ssub.get("smoke", ""), "smoke", fails)
    if len(smoke.get("smoke", [])) != 1:
        fails.append("smoke mode should submit exactly one shard")

    assert not fails, "submit-grid problems:\n  - " + "\n  - ".join(fails)


def main():
    try:
        test_nb3_baseline_grids()
    except AssertionError as e:
        print("test_nb3_baseline_submit: FAILED\n", e)
        sys.exit(1)
    print("test_nb3_baseline_submit: PASS  (nb3 12 shards L=2..5 x seeds{0,1,2}, "
          "nb2 paired arm, PT2 on every rung, qis1-3 pin, per-n_b campaign dirs)")


if __name__ == "__main__":
    main()
