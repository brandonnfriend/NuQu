"""DMRG calibration of the TrimCI extrapolation (classical_next_studies A1, 2026-10-08).

Pure guards, no block2, no cluster:
  * `dmrg_extrapolate` recovers a planted intercept from E = E0 + a*dw, quotes the
    extrapolation distance as delta, and degrades to "bound only" with one rung;
  * `compare` on planted DMRG + TrimCI shards reports gap ~ 0 when both agree and flags a
    TrimCI intercept that sits below what DMRG supports (TRIMCI_OVERSHOOTS), plus the
    DMRG-below-bound sanity flag; the table/figure writer runs end to end;
  * `submit_dmrg_calib.sh` (stubbed condor_submit): `test` is one short shard, `all` is the
    A grid at n_b=3 with the auto-OOM policy lines, the qis1-3 pin and a 48G disk request;
  * `run_dmrg_calib.sh` carries run_frame_shard.sh's provisioning functions VERBATIM, derives
    N_f = 2**n_b, and hands the cpu count to block2 (--n-threads);
  * `run_dmrg_shard.py` defaults N_f to 2**n_b and records the contact convention.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from misc.compare_dmrg_trimci import compare, dmrg_extrapolate, figure, table
from tests.test_asweep_frames import _shard as _trimci_shard

_SUBMIT = os.path.join(_ROOT, "hpc", "dmrg", "submit_dmrg_calib.sh")
_RUNNER = os.path.join(_ROOT, "hpc", "dmrg", "run_dmrg_calib.sh")
_FRAME_RUNNER = os.path.join(_ROOT, "hpc", "detsvsL", "run_frame_shard.sh")
_POLICY = ["NHOLDS                  = ifThenElse(isUndefined(NumHolds), 0, NumHolds)",
           "request_memory          = ifThenElse($(MEMGROW_MB) < $(MEMCAP_MB), $(MEMGROW_MB), $(MEMCAP_MB))",
           "periodic_release        = (HoldReasonCode == 34) && (NumHolds < 4)",
           "on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))"]


def _dmrg_rungs(E0, slope=2.0e5, dws=(1.1e-6, 3.2e-7, 1.2e-7), chis=(100, 200, 400)):
    return [{"chi": c, "E": E0 + slope * dw, "discarded_weight": dw, "S_max_bond": 1.7, "wall_s": 1.0}
            for c, dw in zip(chis, dws)]


def _dmrg_shard(path, A, E0, **kw):
    json.dump({"kind": "dmrg_shard", "L": 2, "dim": 3, "A": A, "N_f": 8, "n_b": 3, "sites": 8,
               "bond_dims": [100, 200, 400], "n_sweeps_per": 6, "results": _dmrg_rungs(E0, **kw),
               "done": True, "contact_convention": "wick"}, open(path, "w"))


def test_extrapolate_recovers_planted_intercept():
    x = dmrg_extrapolate(_dmrg_rungs(2306.5))
    assert x["chi_max"] == 400 and x["n_pts"] == 3
    assert abs(x["E_inf"] - 2306.5) < 1e-6
    assert abs(x["delta"] - 2.0e5 * 1.2e-7) < 1e-6           # the extrapolation distance
    assert abs(x["slope"] - 2.0e5) / 2.0e5 < 1e-6
    # n_fit=2 uses the deepest two only; rungs without dw are skipped
    r = _dmrg_rungs(10.0) + [{"chi": 800, "E": 9.0, "discarded_weight": None}]
    assert dmrg_extrapolate(r, n_fit=2)["chi_max"] == 400
    one = dmrg_extrapolate(_dmrg_rungs(5.0)[:1])
    assert one["E_inf"] is None and one["delta"] is None and one["E_chi_max"] == 5.0 + 2.0e5 * 1.1e-6
    assert dmrg_extrapolate([])["n_pts"] == 0


def test_compare_on_planted_shards():
    tmp = tempfile.mkdtemp(prefix="dmrgcal_")
    try:
        td, dd = os.path.join(tmp, "trimci"), os.path.join(tmp, "dmrg")
        os.makedirs(td); os.makedirs(dd)
        E = {2: 2306.5, 4: 2211.0, 6: 2085.0}
        for A, e in E.items():
            for seed in (0, 1):
                _trimci_shard(os.path.join(td, f"bare_L2d3_A{A}_s{seed}.json"), 2, A, seed, "bare", e)
        _dmrg_shard(os.path.join(dd, "dmrg_L2d3_A2_Nf8.json"), 2, E[2])          # agrees
        _dmrg_shard(os.path.join(dd, "dmrg_L2d3_A4_Nf8.json"), 4, E[4] + 5.0)    # DMRG 5 MeV above
        _dmrg_shard(os.path.join(dd, "dmrg_L2d3_A6_Nf8.json"), 6, E[6], chis=(50,), dws=(1e-5,))
        json.dump({"L": 2, "dim": 3, "A": 2, "N_f": 4, "results": _dmrg_rungs(1.0)},
                  open(os.path.join(dd, "dmrg_L2d3_A2_Nf4.json"), "w"))          # other cutoff: ignored
        rows = compare([dd], [td], L=2, dim=3, n_b=3)
        assert [r["A"] for r in rows] == [2, 4, 6]
        r2, r4, r6 = rows
        assert r2["trimci_ok"] and abs(r2["gap"]) < 0.05 and abs(r2["gap_sigma"]) < 2.5
        # the planted TrimCI ladder sits ~260 MeV above E_fci at its deepest rung, DMRG far below it
        assert r2["flags"] == ["DMRG_BELOW_TRIMCI_BOUND"]
        assert r4["flags"] == ["TRIMCI_OVERSHOOTS"]            # DMRG above the bound here, so no bound flag
        assert r4["gap"] < -4.9 and r4["gap_sigma"] < -FLAG
        assert r6["dmrg"]["E_inf"] is None and r6["gap"] is None and r6["flags"] == ["DMRG_BELOW_TRIMCI_BOUND"]
        md = table(rows, os.path.join(tmp, "t.md"), 2, 3)
        assert "| 4 |" in md and "TRIMCI_OVERSHOOTS" in md and "bound only" in md
        for r in rows:
            r["_pts"] = [(p["discarded_weight"], p["E"]) for p in _dmrg_rungs(E[r["A"]])]
        figure(rows, os.path.join(tmp, "f.png"))
        assert os.path.getsize(os.path.join(tmp, "f.png")) > 1000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


FLAG = 2.0


def _run_submit(mode, env_extra=None):
    tmp = tempfile.mkdtemp(prefix="dmrgsub_")
    try:
        shutil.copy(_SUBMIT, tmp)
        binp = os.path.join(tmp, "bin"); os.makedirs(binp)
        with open(os.path.join(binp, "condor_submit"), "w") as f:
            f.write("#!/bin/sh\necho fake-submit $*\n")
        os.chmod(os.path.join(binp, "condor_submit"), 0o755)
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"], **(env_extra or {}))
        for k in ("AS", "L", "CHIS", "MEM", "CPUS"):
            if not env_extra or k not in env_extra:
                env.pop(k, None)
        p = subprocess.run(["sh", os.path.basename(_SUBMIT), mode], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        cdir = os.path.join(tmp, [d for d in os.listdir(tmp) if d.startswith("campaign_")][0])
        rows = [ln.split() for ln in open(os.path.join(cdir, "shards.txt")).read().splitlines() if ln.strip()]
        return rows, open(os.path.join(cdir, "campaign.sub")).read(), p.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_submit_grid_and_policy():
    rows, sub, out = _run_submit("all")
    assert [r[2] for r in rows] == ["2", "4", "6", "8", "10"]
    assert all(r[:2] == ["2", "3"] and r[3] == "3" and r[4] == "100+200+400+800"
               and r[5] == "48" and r[6] == "16" for r in rows)
    q = [ln for ln in sub.splitlines() if ln.startswith("queue ")]
    assert q == ["queue L,DIM,A,NB,BDIMS,MEM,CPUS from " + q[0].split()[-1]]
    for line in _POLICY:
        assert line in sub, line
    assert "MEMCAP_MB               = 786432" in sub and "request_disk            = 48G" in sub
    assert "qis3.hep.wisc.edu" in sub and "qis4" not in sub
    assert "Executable              = ./run_dmrg_calib.sh" in sub and "14400" in sub
    assert "jobs=5" in out
    rows, sub, _ = _run_submit("test")
    assert len(rows) == 1 and rows[0][2] == "2" and rows[0][4] == "50+100" and " 1800" in sub
    rows, _, _ = _run_submit("all", {"AS": "2 6", "L": "3", "CHIS": "100,200"})
    assert [r[:3] for r in rows] == [["3", "3", "2"], ["3", "3", "6"]] and rows[0][4] == "100+200"


def _fn(src, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", src, re.S | re.M)
    assert m, name
    return m.group(0)


def test_runner_shares_provisioning_and_passes_cutoff():
    run, frame = open(_RUNNER).read(), open(_FRAME_RUNNER).read()
    for name in ("retry", "provision_uv", "provision_python"):
        assert _fn(run, name) == _fn(frame, name), f"{name} drifted from run_frame_shard.sh"
    assert "N_F=$(( 1 << NB ))" in run and 'PROVISION_FAIL=3' in run
    assert '--n_b "$NB" --n-threads "$cpus"' in run
    assert 'exit "$PROVISION_FAIL"' in run.split("provision_wheels()")[0]      # repo check too
    assert '_A${A}_Nf${N_F}.json' in run                                        # same naming as 293938
    assert subprocess.run(["sh", "-n", _RUNNER]).returncode == 0


def test_shard_runner_defaults_N_f_to_2_pow_n_b(monkeypatch, tmp_path):
    import hpc.dmrg.run_dmrg_shard as mod

    class _H:
        meta = {"contact_convention": "wick"}

    seen = {}

    def fake_run_dmrg(L, dim, A, N_f, n_b, bond_dims, n_sweeps_per, on_chi, max_chi_seconds, n_threads):
        seen.update(N_f=N_f, n_b=n_b, n_threads=n_threads, bond_dims=bond_dims)
        on_chi({"chi": bond_dims[0], "E": 1.0, "S_max_bond": None, "discarded_weight": 1e-6})
        return [], _H()

    monkeypatch.setattr(mod, "run_dmrg", fake_run_dmrg)
    out = tmp_path / "x.json"
    monkeypatch.setattr(sys, "argv", ["prog", "--L", "2", "--A", "4", "--n_b", "3", "--n-threads", "16",
                                      "--bond-dims", "50,100", "--out", str(out)])
    mod.main()
    j = json.load(open(out))
    assert seen == {"N_f": 8, "n_b": 3, "n_threads": 16, "bond_dims": (50, 100)}
    assert j["N_f"] == 8 and j["done"] and j["contact_convention"] == "wick" and len(j["results"]) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
