"""Loader + figure support for squeezed shards and binding energies (infrastructure C5),
and the per-rung memory records (C3). 2026-10-08.

Synthetic bare + gaussian shards with PLANTED E_inf, in one campaign directory (as the
squeeze campaign writes them), check that:
  * `load(frame=...)` picks `<frame>_*.json` only, defaults to bare, and refuses to pool
    two frames into one (n_b, L) group;
  * `make_Asweep_figures.collect` works per frame and `binding_rows` returns
    BE = A E(1) - (A-1) E(0) - E(A) from the planted energies with the propagated sigma;
  * the figure script runs end to end with --frame gaussian --compare-frame bare --As 0..,
    writing the frame-suffixed figures, tables and the binding table;
  * `analyze_shard_memory` reads the per-rung `mem` records and names the peak stage.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc.aggregate_classical_energies import load  # noqa: E402
from misc.analyze_shard_memory import shard_peak, summarize, suggest  # noqa: E402
from misc.make_Asweep_figures import binding_rows, collect  # noqa: E402

DPS = (-40.0, -26.0, -17.0, -11.0, -7.0)


def _shard(path, L, A, seed, frame, E_fci, c=0.45, mem=False):
    sites = L ** 3
    rungs = [{"core": 1000, "E_var": E_fci + 260.0, "dE_pt2": -95.0, "wall_s": 1.0,
              "phase": "0-ensemble", "n_ext": 1000},
             {"core": 2000, "E_var": E_fci + 240.0, "dE_pt2": -80.0, "wall_s": 1.0,
              "phase": "grow", "n_ext": 2000}]
    for i, d in enumerate(DPS):
        rungs.append({"core": 4000 * 2 ** i, "E_var": E_fci + c * d - d, "dE_pt2": d,
                      "wall_s": 1.0, "phase": "grow", "n_ext": 4000 * 2 ** i})
    if mem:
        for k, r in enumerate(rungs):
            top = 1000.0 * (k + 1)
            r["mem"] = {"peak_rss_mb": top, "children_peak_rss_mb": 300.0 if k == 0 else 0.0,
                        "solver_peak_rss_mb": top, "max_pool": 4 * r["core"], "rounds": 2,
                        "trace": [[0, r["core"] // 2, 2 * r["core"], r["core"], top - 100, top - 50],
                                  [1, r["core"], 4 * r["core"], 2 * r["core"], top, top]]}
    json.dump({"kind": "frame_shard", "L": L, "dim": 3, "A": A, "filling": None, "frame": frame,
               "seed": seed, "n_b": 3, "N_f": 8, "sites": sites, "n_terms": 1777,
               "contact_convention": "wick", "rungs": rungs, "done": True, "wall_s": 10.0},
              open(path, "w"))


def _campaign(tmp):
    d = os.path.join(tmp, "2026-10-09", "Asweep_squeeze_nb3_1")
    os.makedirs(d)
    # planted TOTAL energies: E(A, L) = sites*300 + 100*A - 7*A^2 (so BE grows with A)
    E = lambda A, L: L ** 3 * 300.0 + 100.0 * A - 7.0 * A * A
    for L in (2, 3):
        for A in (0, 1, 2, 4):
            for seed in (0, 1):
                _shard(os.path.join(d, f"gaussian_L{L}d3_A{A}_s{seed}.json"), L, A, seed,
                       "gaussian", E(A, L) - 50.0 * L ** 3, mem=(seed == 0))
                if A in (0, 1):
                    _shard(os.path.join(d, f"bare_L{L}d3_A{A}_s{seed}.json"), L, A, seed,
                           "bare", E(A, L))
        _shard(os.path.join(d, f"bare_L{L}d3_A2_s0.json"), L, 2, 0, "bare", E(2, L))
    return d, E


def test_load_frame_filter_and_guard():
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        d, _ = _campaign(tmp)
        g, m = load([d], convention="wick", A=2)                      # default = bare
        assert set(g) == {(3, 2), (3, 3)} and all(v["frame"] == "bare" for v in m.values())
        assert {len(v) for v in g.values()} == {1}
        g, m = load([d], convention="wick", A=2, frame="gaussian")
        assert all(v["frame"] == "gaussian" for v in m.values()) and {len(v) for v in g.values()} == {2}
        g, m = load([d], convention="wick", A=4, frame="bare")
        assert not g, "no bare A=4 shards exist"
        try:
            load([d], convention="wick", A=2, frame="*")
        except ValueError as e:
            assert "frame" in str(e)
        else:
            raise AssertionError("two frames in one (n_b, L) group must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_binding_rows_from_planted_energies():
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        d, E = _campaign(tmp)
        rec = collect([d], frame="gaussian", As=[0, 1, 2, 4])
        assert set(rec) == {(A, L) for A in (0, 1, 2, 4) for L in (2, 3)}
        assert all(v["ok"] and v["frame"] == "gaussian" for v in rec.values())
        rows = binding_rows(rec)
        assert [(r["A"], r["L"]) for r in rows] == [(2, 2), (2, 3), (4, 2), (4, 3)]
        for r in rows:
            A, L = r["A"], r["L"]
            # the planted E has the -50/site frame shift in every sector: it cancels in BE,
            # as the pion vacuum does. With E(A) = 300 sites + 100 A - 7 A^2,
            # BE = A E1 - (A-1) E0 - EA = 7 A (A - 1).
            assert abs(r["BE"] - 7.0 * A * (A - 1)) < 0.5, (r, 7.0 * A * (A - 1))
            assert r["source"] == "E_inf" and r["BE_sigma"] is not None and r["BE_sigma"] >= 0
            assert abs(r["BE_per_A"] - r["BE"] / A) < 1e-9
        bare = collect([d], frame="bare", As=[0, 1, 2, 4])
        assert set(bare) == {(A, L) for A in (0, 1, 2) for L in (2, 3)}
        assert len(binding_rows(bare)) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_figure_script_end_to_end():
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        d, _ = _campaign(tmp)
        out = os.path.join(tmp, "out")
        p = subprocess.run([sys.executable, "-m", "misc.make_Asweep_figures", "--data", d,
                            "--out", out, "--frame", "gaussian", "--compare-frame", "bare",
                            "--As", "0", "1", "2", "4"],
                           cwd=_ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MPLBACKEND="Agg"))
        assert p.returncode == 0, p.stderr[-2000:]
        files = set(os.listdir(out))
        for f in ("classical_baseline_multiA_gaussian.png", "classical_baseline_perA_gaussian.png",
                  "classical_baseline_multiA_gaussian_table.md", "classical_baseline_multiA_gaussian.json",
                  "classical_binding_gaussian_table.md", "classical_binding_gaussian.png",
                  "classical_baseline_multiA_gaussian_vs_bare.png",
                  "classical_binding_table.md", "classical_binding.png"):
            assert f in files, (f, sorted(files))
        assert "classical_baseline_multiA.png" not in files, "bare outputs must not be clobbered"
        md = open(os.path.join(out, "classical_binding_gaussian_table.md")).read()
        assert "| 4 | 3 |" in md and "E_inf" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_shard_memory_records():
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        d, _ = _campaign(tmp)
        j = json.load(open(os.path.join(d, "gaussian_L2d3_A2_s0.json")))
        pk = shard_peak(j)
        assert pk[0] == 7000.0 and pk[1] == 64000 and pk[2] == "expand", pk
        assert shard_peak(json.load(open(os.path.join(d, "gaussian_L2d3_A2_s1.json")))) is None
        rows = summarize([d], frame="gaussian")
        assert len(rows) == 16 and sum(r["peak_mb"] is not None for r in rows) == 8
        sug = {(s["frame"], s["L"]): s for s in suggest(rows, headroom=1.3, floor_gb=16)}
        assert sug[("gaussian", 2)]["n"] == 4 and sug[("gaussian", 2)]["suggest_gb"] == 25  # ceil(1.3*7000/1024+16)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_load_frame_filter_and_guard()
    test_binding_rows_from_planted_energies()
    test_figure_script_end_to_end()
    test_shard_memory_records()
    print("test_asweep_frames: PASS  (frame filter + mix guard; BE from planted energies; figure "
          "script with --frame/--compare-frame/--As; memory records -> peak stage + suggestion)")


if __name__ == "__main__":
    main()
