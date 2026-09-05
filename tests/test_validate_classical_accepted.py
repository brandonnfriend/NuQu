"""The classical release gate (`misc.validate_classical_accepted`) -- audit 2026-09-05 P0-5.

This is the check that is supposed to make the audit's central defect unrepeatable: an
n_b=2 baseline standing in for the selected n_b=3 model, retired pre-vertex-fix data
reaching a figure, or a result carried without provenance or with the wrong label. A gate
is only worth having if its refusals actually fire, so each one is exercised against a
synthetic shard tree built to trip exactly that check -- plus a clean tree that must pass
and produce a complete manifest.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc.validate_classical_accepted import RejectedError, validate  # noqa: E402

VERTEX_FIX = "9404fac4edf20646cb9862045159667a43e095a8"


def _head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode().strip()


def _shard(path, L, n_b, seed, E_fci=1795.0, dps=(-40.0, -26.0, -17.0, -11.0, -7.0),
           commit=None, c=0.45, rising=False, sites=None, filling=1.0, frame="bare",
           dirty=False):
    sites = sites or L ** 3
    rungs = [{"core": 1000, "E_var": E_fci + 260.0, "dE_pt2": -95.0, "wall_s": 1.0},
             {"core": 2000, "E_var": E_fci + 240.0, "dE_pt2": -80.0, "wall_s": 1.0}]
    for i, d in enumerate(dps):
        E = E_fci + c * d - d
        if rising:
            E = E_fci + 500.0 + 30.0 * i           # a non-variational ladder
        rungs.append({"core": 4000 * 2 ** i, "E_var": E, "dE_pt2": d, "wall_s": 1.0})
    j = {"kind": "frame_shard", "L": L, "dim": 3, "A": sites, "filling": filling,
         "frame": frame, "seed": seed, "n_b": n_b, "N_f": 2 ** n_b, "sites": sites,
         "n_terms": 1777, "rungs": rungs, "done": True, "wall_s": 10.0}
    if commit:
        j["manifest"] = {"git_commit": commit, "git_dirty": dirty,
                         "git_tracked_diff_hash": None, "git_branch": "remediation/vertex-fix",
                         "hostname": "qis1.hep.wisc.edu", "timestamp_utc": "2026-09-06T00:00:00+00:00",
                         "extra": {"solver": {"seed": seed, "pt2_max_core": 1024000}}}
    json.dump(j, open(path, "w"))


def _tree(tmp, name, shards):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    for kw in shards:
        _shard(os.path.join(d, f"bare_L{kw['L']}d3_nb{kw['n_b']}_f1.0_s{kw['seed']}.json"), **kw)
    return d


def _expect_reject(fn, needle, fails, what):
    try:
        fn()
    except RejectedError as e:
        if needle.lower() not in str(e).lower():
            fails.append(f"{what}: rejected for the wrong reason -> {e}")
        return
    except Exception as e:                          # pragma: no cover
        fails.append(f"{what}: raised {type(e).__name__} instead of RejectedError: {e}")
        return
    fails.append(f"{what}: NOT rejected")


def main():
    fails = []
    head = _head()
    tmp = tempfile.mkdtemp(prefix="valcls_")
    try:
        good = [dict(L=2, n_b=3, seed=s, E_fci=1795.0 + 4 * (s - 1), commit=head) for s in (0, 1, 2)]

        # --- the clean case must pass and produce a full manifest -------------------
        d = _tree(tmp, "clean", good)
        try:
            recs, info, prov = validate([d], expect_n_b=3)
            if len(info) != 3:
                fails.append(f"clean: hashed {len(info)} shards, expected 3")
            if not recs or recs[0]["label"] != "extrapolated":
                fails.append(f"clean: label {recs and recs[0]['label']!r}, expected 'extrapolated'")
            if recs and recs[0]["sigma"] is None:
                fails.append("clean: 'extrapolated' with no sigma slipped through")
            if prov["generating_commit"] != head:
                fails.append("clean: generating commit not taken from the shard manifest")
            if "embedded" not in prov["provenance_source"]:
                fails.append(f"clean: provenance source {prov['provenance_source']!r}")
            if any(s["sha256"] is None or len(s["sha256"]) != 64 for s in info):
                fails.append("clean: missing/short SHA-256")
            if not recs[0]["core_ladders"]:
                fails.append("clean: core ladders not recorded")
        except RejectedError as e:
            fails.append(f"clean tree was rejected: {e}")

        # --- the P0-1 defect: a baseline at the wrong cutoff ------------------------
        _expect_reject(lambda: validate([d], expect_n_b=2), "n_b=3, expected 2",
                       fails, "wrong cutoff")

        # --- retired pre-vertex-fix inputs, by path token and by commit ancestry ----
        ret = _tree(tmp, "2026-08-14", [dict(L=2, n_b=3, seed=0, commit=head)])
        _expect_reject(lambda: validate([ret], expect_n_b=3), "retired", fails,
                       "retired directory token")
        # a real ancestor of the vertex fix -> must be refused on ancestry, not on path
        parent = subprocess.check_output(["git", "rev-parse", f"{VERTEX_FIX}^"],
                                         cwd=_ROOT).decode().strip()
        old = _tree(tmp, "prefix_commit", [dict(L=2, n_b=3, seed=0, commit=parent)])
        _expect_reject(lambda: validate([old], expect_n_b=3), "predates the vertex fix",
                       fails, "pre-vertex-fix commit")
        unknown = _tree(tmp, "unknown_commit", [dict(L=2, n_b=3, seed=0, commit="0" * 40)])
        _expect_reject(lambda: validate([unknown], expect_n_b=3), "not in this clone",
                       fails, "unknown commit")

        # --- missing provenance, and the explicit operator-asserted escape hatch ----
        nom = _tree(tmp, "nomanifest", [dict(L=2, n_b=3, seed=s) for s in (0, 1, 2)])
        _expect_reject(lambda: validate([nom], expect_n_b=3), "no provenance manifest",
                       fails, "unmanifested shard")
        try:
            _, _, prov2 = validate([nom], expect_n_b=3, allow_unmanifested=True,
                                   assert_commit=head)
            if "OPERATOR-ASSERTED" not in prov2["provenance_source"]:
                fails.append("operator-asserted provenance was not marked as such")
        except RejectedError as e:
            fails.append(f"operator-asserted escape hatch rejected: {e}")
        _expect_reject(lambda: validate([nom], expect_n_b=3, allow_unmanifested=True,
                                        assert_commit=parent),
                       "not a descendant", fails, "asserted pre-fix commit")

        # --- corrupt / non-variational ladders --------------------------------------
        bad = _tree(tmp, "rising", [dict(L=2, n_b=3, seed=0, commit=head, rising=True)])
        _expect_reject(lambda: validate([bad], expect_n_b=3), "rises", fails,
                       "non-variational ladder")
        short = _tree(tmp, "short", [dict(L=2, n_b=3, seed=0, commit=head, dps=(-40.0,))])
        _expect_reject(lambda: validate([short], expect_n_b=3, min_rungs=6), "rungs <",
                       fails, "too-few rungs")

        # --- disagreeing configuration across shards --------------------------------
        mix = _tree(tmp, "mixedcfg", [dict(L=2, n_b=3, seed=0, commit=head),
                                      dict(L=2, n_b=3, seed=1, commit=head, filling=0.5)])
        _expect_reject(lambda: validate([mix], expect_n_b=3), "disagree", fails,
                       "mixed configuration")

        # --- the PT2-depth quality gate ---------------------------------------------
        _expect_reject(lambda: validate([d], expect_n_b=3, min_pt2_post=99),
                       "post-collapse PT2", fails, "PT2-depth gate")

        # --- CLI exits 2 on refusal (rejected) vs 0 on pass --------------------------
        r = subprocess.run([sys.executable, "-m", "misc.validate_classical_accepted",
                            "--data", d, "--expect-n-b", "2", "--label", "x",
                            "--out", os.path.join(tmp, "m.json")],
                           cwd=_ROOT, capture_output=True, text=True)
        if r.returncode != 2:
            fails.append(f"CLI refusal exited {r.returncode}, expected 2")
        r2 = subprocess.run([sys.executable, "-m", "misc.validate_classical_accepted",
                             "--data", d, "--expect-n-b", "3", "--label", "clean",
                             "--out", os.path.join(tmp, "ok.json")],
                            cwd=_ROOT, capture_output=True, text=True)
        if r2.returncode != 0:
            fails.append(f"CLI clean run exited {r2.returncode}:\n{r2.stderr}")
        else:
            man = json.load(open(os.path.join(tmp, "ok.json")))
            for k in ("dataset", "cutoff_n_b", "provenance", "dependencies",
                      "physical_and_solver_config", "seed_and_core_ladders",
                      "aggregation_rules", "results", "analysis_scripts", "outputs", "files"):
                if k not in man:
                    fails.append(f"manifest missing required section {k!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print("test_validate_classical_accepted: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print("test_validate_classical_accepted: PASS  (clean tree passes with a complete "
          "manifest; wrong cutoff, retired path, pre-fix commit, unknown commit, missing "
          "provenance, non-variational and short ladders, mixed config and the PT2-depth "
          "gate all refuse; CLI exits 2 on refusal)")


def test_validate_classical_accepted():
    main()


if __name__ == "__main__":
    main()
