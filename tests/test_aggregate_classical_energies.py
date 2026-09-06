"""End-to-end guard for `misc.aggregate_classical_energies` (audit 2026-09-05, P0-2 / P0-5).

The aggregator turns raw shard JSONs into the number the paper quotes, so the paths that
matter most are the ones no CURRENT dataset exercises: several seeds per L, two cutoffs
side by side, and the refusal to emit a value it cannot stand behind. Synthetic shards
with a PLANTED E_infinity let all of that be checked exactly.

Checks:
  * multi-seed pooling produces one row per (n_b, L) with an E_inf AND a sigma, and the
    planted E_infinity is recovered;
  * the seed spread is reported as a search diagnostic and kept OUT of sigma;
  * a short ladder is reported as a bound with NO invented central value;
  * the paired n_b=2 -> 3 cutoff-shift table appears when both cutoffs are present;
  * pre-vertex-fix directories are REFUSED by default (retired data must never reach a
    figure) and only an explicit opt-in flag lifts it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _shard(path, L, n_b, seed, E_fci, dps, sites=None, c=0.45):
    """A shard whose post-collapse rungs satisfy E_var + dE_PT2 = E_fci + c*dE_PT2, with a
    two-rung exploration basin planted in front (as the real warm-grow ladders show)."""
    sites = sites or L ** 3
    rungs = [{"core": 1000, "E_var": E_fci + 260.0, "dE_pt2": -95.0, "wall_s": 1.0,
              "phase": "0-ensemble", "n_ext": 1000},
             {"core": 2000, "E_var": E_fci + 240.0, "dE_pt2": -80.0, "wall_s": 1.0,
              "phase": "grow", "n_ext": 2000}]
    for i, d in enumerate(dps):
        rungs.append({"core": 4000 * 2 ** i, "E_var": E_fci + c * d - d, "dE_pt2": d,
                      "wall_s": 1.0, "phase": "grow", "n_ext": 4000 * 2 ** i})
    json.dump({"kind": "frame_shard", "L": L, "dim": 3, "A": sites, "filling": 1.0,
               "frame": "bare", "seed": seed, "n_b": n_b, "N_f": 2 ** n_b, "sites": sites,
               "n_terms": 1777, "rungs": rungs, "done": True, "wall_s": 10.0},
              open(path, "w"))


def _run(data_dirs, out_dir, extra=()):
    cmd = [sys.executable, "-m", "misc.aggregate_classical_energies",
           "--data", *data_dirs, "--out-dir", out_dir, *extra]
    p = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    return p


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="agg_")
    try:
        live = os.path.join(tmp, "2026-09-06", "bare_baseline_nb3_999999")
        os.makedirs(live)
        DPS = (-40.0, -26.0, -17.0, -11.0, -7.0)
        planted = {(3, 2): 1795.0, (3, 3): 8100.0, (2, 2): 1830.0}
        for (n_b, L), E in planted.items():
            for seed, shift in ((0, 0.0), (1, 4.0), (2, -4.0)):
                _shard(os.path.join(live, f"bare_L{L}d3_nb{n_b}_f1.0_s{seed}.json"),
                       L, n_b, seed, E + shift, DPS)
        # a deliberately short ladder: one post-collapse rung -> bound only
        _shard(os.path.join(live, "bare_L5d3_nb3_f1.0_s0.json"), 5, 3, 0, 46000.0, (-30.0,))

        out = os.path.join(tmp, "out")
        p = _run([live], out)
        if p.returncode != 0:
            fails.append(f"aggregator failed:\n{p.stdout}\n{p.stderr}")
        else:
            recs = json.load(open(os.path.join(out, "classical_energy_aggregate.json")))
            by = {(r["n_b"], r["L"]): r for r in recs}
            if set(by) != {(3, 2), (3, 3), (2, 2), (3, 5)}:
                fails.append(f"grouped keys {sorted(by)} != expected 4 (n_b, L) groups")
            for k, E in planted.items():
                r = by.get(k)
                if r is None:
                    continue
                if not r["ok"]:
                    fails.append(f"{k}: no extrapolation ({r['reason']})")
                    continue
                if r["n_seeds"] != 3:
                    fails.append(f"{k}: pooled {r['n_seeds']} seeds, expected 3")
                if abs(r["E_inf"] - E) > 5.0:
                    fails.append(f"{k}: E_inf {r['E_inf']:.2f} != planted {E} (tol 5 MeV)")
                if r["sigma"] is None:
                    fails.append(f"{k}: reported a value with no error bar")
                rob = r.get("seed_robustness") or {}
                if rob.get("n_seeds") != 3:
                    fails.append(f"{k}: robustness record missing/wrong ({rob})")
                if rob.get("spread") is None or rob["spread"] < 1.0:
                    fails.append(f"{k}: seed spread {rob.get('spread')} not reported")
                if r["sigma"] is not None and rob.get("spread") is not None \
                        and abs(r["sigma"] - rob["spread"]) < 1e-9:
                    fails.append(f"{k}: sigma equals the seed spread -- it leaked into sigma")
            short = by.get((3, 5))
            if short and short["ok"]:
                fails.append("a 1-rung post-collapse ladder produced a central value")
            if short and short["E_var_bound"] is None:
                fails.append("bound-only row lost its variational bound")

            md = open(os.path.join(out, "classical_energy_aggregate.md")).read()
            if "Cutoff shift $n_b$: 2 → 3" not in md:
                fails.append("paired n_b=2->3 shift table missing though both cutoffs present")
            if "Error budget" not in md or "SHCI ½-dist" not in md:
                fails.append("error-budget table missing or not on the SHCI convention")
            if "Search robustness" not in md:
                fails.append("search-robustness table missing -- the seed spread must be "
                             "reported somewhere once it is out of sigma")
            # the seed spread must not be presented as an uncertainty
            if "σ seed" in md:
                fails.append("'σ seed' column still present -- seed spread is a search "
                             "diagnostic, not an error-bar term")
            for f in ("classical_energy_aggregate.png", "classical_energy_aggregate.pdf"):
                if not os.path.exists(os.path.join(out, f)):
                    fails.append(f"{f} not written")

        # --- retired-data refusal --------------------------------------------------
        retired = os.path.join(tmp, "2026-08-13", "groupA_290813")
        os.makedirs(retired)
        _shard(os.path.join(retired, "bare_L2d3_f1.0_s0.json"), 2, 3, 0, 1500.0, DPS)
        out2 = os.path.join(tmp, "out2")
        p2 = _run([retired, live], out2)
        if p2.returncode != 0:
            fails.append(f"aggregator failed on the mixed run:\n{p2.stderr}")
        elif "REFUSED" not in p2.stdout:
            fails.append("pre-vertex-fix directory was NOT refused")
        else:
            recs2 = json.load(open(os.path.join(out2, "classical_energy_aggregate.json")))
            # the retired shard shadows live (n_b=3, L=2, seed 0) with E_fci=1500; if the
            # refusal leaked, that group's extrapolation would move off its planted 1795.
            r2 = next((r for r in recs2 if (r["n_b"], r["L"]) == (3, 2)), None)
            if r2 is None or not r2["ok"] or abs(r2["E_inf"] - planted[(3, 2)]) > 5.0:
                fails.append(f"retired shard leaked: (n_b=3,L=2) E_inf = "
                             f"{None if r2 is None else r2.get('E_inf')}, "
                             f"expected ~{planted[(3, 2)]}")
        p3 = _run([retired], os.path.join(tmp, "out3"), extra=("--allow-pre-vertex-fix",))
        if p3.returncode != 0:
            fails.append("--allow-pre-vertex-fix should still run (explicit, non-release opt-in)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print("test_aggregate_classical_energies: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print("test_aggregate_classical_energies: PASS  (3-seed pooling recovers the planted "
          "E_inf with a sigma; short ladder stays bound-only; cutoff-shift + error-budget "
          "tables emitted; pre-vertex-fix data refused)")


def test_aggregate_classical_energies():
    main()


if __name__ == "__main__":
    main()
