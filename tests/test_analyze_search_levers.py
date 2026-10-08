"""`misc/analyze_search_levers.py` — the lever hit rate, the choice-point penalty, and the
seed-independence verdict, on synthetic shards whose right answers are known by construction.

The report drives text in the manuscript ("N of 15 runs on the best basin", "the seeds were
not independent searches"), so the three judgements it makes are pinned here:

  * basin hit rate counts a run iff it is within BASIN_TOL_PS of the best any arm reached;
  * the choice-point penalty is measured on the LAST select rung, against the init the
    FIRST rung would have picked;
  * a legacy-numbered campaign whose seeds produced one ladder is called out as having no
    power, while disjoint-stride seeds that converge are not.
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc import analyze_search_levers as asl  # noqa: E402

SITES = 8       # L=2, dim=3


def _shard(path, A, seed, ladder, trail=None, stride=None, runs=32):
    """A minimal frame-shard JSON: `ladder` is E_var per rung, cores 1k doubling."""
    rungs = []
    for i, e in enumerate(ladder):
        r = {"core": 1000 * 2 ** i, "E_var": e, "E_pt2": e - 1.0, "wall_s": 10.0,
             "phase": "0-select" if trail and i == 0 else "grow"}
        if i == 0 and trail is not None:
            r["phase0_select"] = trail
        rungs.append(r)
    doc = {"L": 2, "dim": 3, "A": A, "seed": seed, "n_b": 3, "frame": "bare",
           "phase0_runs": runs, "rungs": rungs, "wall_s": 100.0}
    if stride is not None:
        doc["phase0_seed_stride"] = stride
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh)


def test_basin_hit_rate_and_spread(tmp_path):
    root = str(tmp_path / "levers")
    # Two arms, one A, three seeds. `good` is on the basin every time; `bad` only once.
    for s, e in enumerate([2000.0, 2000.0 + 0.4 * SITES, 2000.0 + 0.2 * SITES]):
        _shard(f"{root}/stratsel/bare_A4_s{s}.json", 4, s, [2100.0, e], stride=1000)
    for s, e in enumerate([2000.0, 2000.0 + 8.0 * SITES, 2000.0 + 9.0 * SITES]):
        _shard(f"{root}/base/bare_A4_s{s}.json", 4, s, [2100.0, e], stride=1000)

    grid = asl.lever_grid(root)
    rows = {r["arm"]: r for r in grid["rows"]}
    assert grid["best_ps"][4] == 2000.0 / SITES
    assert rows["stratsel"]["on_basin"] == 3          # 0.0, 0.4, 0.2 MeV/site — all within 0.5
    assert rows["base"]["on_basin"] == 1              # only the exact-best seed
    assert rows["base"]["spread_max_ps"] == 9.0       # worst minus best, per site


def test_choice_point_penalty_is_measured_at_the_last_rung(tmp_path):
    root = str(tmp_path / "levers")
    # init 0 wins at 1k and ends 16 MeV/site HIGH; init 1 loses at 1k and ends lowest.
    trail = [[0, [100.0, 2000.0 + 16.0 * SITES]], [1, [200.0, 2000.0]]]
    _shard(f"{root}/select/bare_A5_s0.json", 5, 0, [2400.0, 2000.0], trail=trail, stride=1000)
    grid = asl.lever_grid(root)
    cp = asl.choice_point(grid["arms"])
    assert cp["n"] == 1
    assert cp["same_winner"] == 0
    assert abs(cp["penalty_max_ps"] - 16.0) < 1e-9
    assert cp["n_penalty_over_tol"] == 1
    # perfectly inverted ranking of two items
    assert abs(cp["spearman_min"] + 1.0) < 1e-9


def test_choice_point_ignores_shards_without_a_select_trail(tmp_path):
    root = str(tmp_path / "levers")
    _shard(f"{root}/base/bare_A5_s0.json", 5, 0, [2400.0, 2000.0], stride=1000)
    assert asl.choice_point(asl.lever_grid(root)["arms"]) is None


def test_seed_audit_flags_legacy_shared_starts(tmp_path):
    legacy = str(tmp_path / "legacy")
    for s in (0, 1, 2):                                # no stride key -> legacy numbering
        _shard(f"{legacy}/bare_A8_s{s}.json", 8, s, [2400.0, 2000.0])
    disjoint = str(tmp_path / "disjoint")
    for s, e in ((0, 2000.0), (1, 2000.0), (2, 2010.0)):
        _shard(f"{disjoint}/bare_A8_s{s}.json", 8, s, [2400.0, e], stride=1000)

    rows = {r["campaign"]: r for r in asl.seed_audit([legacy, disjoint])}
    assert rows["legacy"]["legacy_numbering"] is True
    assert rows["legacy"]["n_distinct_ladders"] == 1        # one search, reported three times
    assert rows["legacy"]["spread_ps"] == 0.0
    assert rows["disjoint"]["legacy_numbering"] is False
    assert rows["disjoint"]["n_distinct_ladders"] == 2      # two converged, one did not


def test_seed_audit_groups_lever_arms_separately(tmp_path):
    """Sibling arm directories are different configurations, not repeat seeds of one."""
    root = str(tmp_path / "levers_999")
    for arm in ("base", "stratsel"):
        for s in (0, 1, 2):
            _shard(f"{root}/{arm}/bare_A4_s{s}.json", 4, s, [2400.0, 2000.0 + s], stride=1000)
    rows = asl.seed_audit([root])
    assert {r["campaign"] for r in rows} == {"levers_999/base", "levers_999/stratsel"}
    assert all(r["n_seeds"] == 3 for r in rows)
