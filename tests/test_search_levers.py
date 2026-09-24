"""The warm-grow SEARCH LEVERS (2026-09-24) and their OFF-by-default guarantee.

Levers (classical/trimci/graph_arrays.py + run_cpp.growing_ladder):
  * phase0 select core  -- grow every Phase-0 init to a larger core, keep the lowest there;
  * stratified starts   -- Phase-0 inits cycle evenly through nucleon-arrangement types;
  * novel ferm frac/keep -- reserve pool / core slots for new nucleon configurations.

Checks:
  * every lever OFF reproduces the pre-lever solve exactly (growing_ladder == the
    ensemble + warm-grow loop it always ran), so validated results are unchanged;
  * `site_partitions` / `nucleon_arrangement_starts` give A nucleons, <= 4 per site,
    and cover every arrangement type evenly;
  * `expand_arrays` with the lever keeps the default pool as a subset and only adds
    candidates whose nucleon configuration is new; `global_trim_arrays` reserves the
    requested slots and keeps the core size;
  * the select stage returns the run that is lowest at its LAST rung.
"""
import os
import sys
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("NUQU_NUM_WORKERS", "1")

from classical.trimci import build_from_eft  # noqa: E402
from classical.trimci import graph_arrays as ga  # noqa: E402
from classical.trimci.run_cpp import growing_ladder  # noqa: E402

_H = build_from_eft(2, 3, 1, transform="bare")       # L=2 3D, n_b=1: small and fast
_A = 3


def _occ_sites(H, occ):
    sites = ga.mode_sites(H)
    return Counter(int(sites[m]) for m in range(H.n_ferm_modes) if (occ >> m) & 1)


def test_levers_off_is_old_solve():
    rungs = [150, 300]
    new = growing_ladder(_H, _A, rungs, phase0_runs=3, seed=1, verbose=False)
    # the pre-lever loop, verbatim
    r0 = ga.ground_state_ensemble_arrays(_H, n_elec=_A, n_runs=3, n_dets=150, seed=1)
    r1 = ga.ground_state_arrays(_H, n_elec=_A, n_dets=300, initial_core=(r0.ferm_arr, r0.bos_arr),
                                seed=2)
    assert [r["phase"] for r in new] == ["0-ensemble", "grow"]
    assert np.array_equal(new[-1]["core"], r1.n_dets)
    from classical.trimci.pt2 import pt2_from_result
    assert new[-1]["E_var"] == float(pt2_from_result(_H, r1)["E_var"])


def test_partitions_and_stratified_starts():
    assert ga.site_partitions(5, 4, 8) == [(4, 1), (3, 2), (3, 1, 1), (2, 2, 1),
                                           (2, 1, 1, 1), (1, 1, 1, 1, 1)]
    assert ga.site_partitions(3, 4, 2) == [(3,), (2, 1)]      # at most 2 occupied sites
    sites = ga.mode_sites(_H)
    assert sites.max() + 1 == 8 and all((sites == j).sum() == 4 for j in range(8))
    n_types = len(ga.site_partitions(_A, 4, 8))
    starts = ga.nucleon_arrangement_starts(_H, _A, 4 * n_types, np.random.default_rng(0))
    types = Counter()
    for occ in starts:
        assert bin(occ).count("1") == _A
        c = _occ_sites(_H, occ)
        assert max(c.values()) <= 4
        types[tuple(sorted(c.values(), reverse=True))] += 1
    assert len(types) == n_types and set(types.values()) == {4}, types


def test_novel_expand_and_reserved_trim():
    r = ga.ground_state_arrays(_H, n_elec=_A, n_dets=100, seed=0)
    cf, cb, c = r.ferm_arr, r.bos_arr, r.coeffs
    pf0, pb0 = ga.expand_arrays(_H, cf, cb, c, 3)
    pf1, pb1 = ga.expand_arrays(_H, cf, cb, c, 3, novel_ferm_frac=0.5)
    N = cf.shape[0]
    assert pf1.shape[0] > pf0.shape[0]
    rows0 = np.hstack([pf0, pb0.astype(np.uint64)])
    rows1 = np.hstack([pf1, pb1.astype(np.uint64)])
    assert ga._rows_isin(rows0, rows1).all(), "lever dropped part of the default pool"
    added = rows1[~ga._rows_isin(rows1, rows0)]
    assert added.shape[0] <= int(np.ceil(0.5 * N))
    assert not ga._rows_isin(added[:, :pf1.shape[1]], cf).any(), "added a known nucleon config"
    keep = 80
    f, b, co, E = ga.global_trim_arrays(_H, pf1, pb1, keep, novel_keep_frac=0.25,
                                        prev_core_ferm=cf)
    assert f.shape[0] == keep
    n_new = int((~ga._rows_isin(f, cf)).sum())
    f0, *_ = ga.global_trim_arrays(_H, pf1, pb1, keep)
    assert n_new >= min(int((~ga._rows_isin(f0, cf)).sum()), int(np.ceil(0.25 * keep)))


def test_select_keeps_lowest_at_last_rung():
    trail, summary = ga.select_phase0_arrays(_H, _A, [100, 200], 3, seed=5, n_workers=1)
    assert len(trail) == 2 and len(summary) == 3
    assert trail[-1].energy == min(e[-1] for _, e in summary)


def test_lever_submit_grid():
    """submit_search_levers.sh: 6 arms x A{2,4,5,6,9} x 3 seeds, each arm's lever env."""
    import shutil
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="levers_")
    try:
        shutil.copy(os.path.join(_ROOT, "hpc", "detsvsL", "submit_search_levers.sh"), tmp)
        os.makedirs(os.path.join(tmp, "bin"))
        stub = os.path.join(tmp, "bin", "condor_submit")
        open(stub, "w").write("#!/bin/sh\necho fake $*\n")
        os.chmod(stub, 0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("AS", "SEEDS", "ARMS")}
        env["PATH"] = os.path.join(tmp, "bin") + os.pathsep + env["PATH"]
        p = subprocess.run(["sh", "submit_search_levers.sh", "all"], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        cdir = [d for d in os.listdir(tmp) if d.startswith("campaign_")][0]
        rows = [ln.split() for ln in open(os.path.join(tmp, cdir, "grid.txt")) if ln.strip()]
        sub = open(os.path.join(tmp, cdir, "levers.sub")).read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert len(rows) == 90 and {len(r) for r in rows} == {8}
    arms = {r[0]: tuple(r[3:7]) for r in rows}
    assert arms == {"base": ("0", "random", "0", "0"), "select": ("16000", "random", "0", "0"),
                    "strat": ("0", "stratified", "0", "0"),
                    "stratsel": ("16000", "stratified", "0", "0"),
                    "novel": ("0", "random", "0.5", "0.05"),
                    "all": ("16000", "stratified", "0.5", "0.05")}
    assert {r[1] for r in rows} == {"2", "4", "5", "6", "9"}
    assert "NUQU_PHASE0_SEED_STRIDE=1000" in sub and "NUQU_DEEP_SOLVE" not in sub
    assert "-$(ARM) bare $(A) none" in sub, "arms must write to separate campaign dirs"


def main():
    test_levers_off_is_old_solve()
    test_partitions_and_stratified_starts()
    test_novel_expand_and_reserved_trim()
    test_select_keeps_lowest_at_last_rung()
    test_lever_submit_grid()
    print("test_search_levers: PASS  (levers OFF == old solve; stratified starts cover every "
          "arrangement type evenly; novel pool/core reservation; select = lowest at last rung)")


if __name__ == "__main__":
    main()
