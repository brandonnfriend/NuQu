"""The classical release gate (`misc.validate_classical_accepted`) -- audit 2026-09-05 P0-5.

This is the check that is supposed to make the audit's central defect unrepeatable: an
n_b=2 baseline standing in for the selected n_b=3 model, retired pre-vertex-fix data
reaching a figure, or a result carried without provenance or with the wrong label. A gate
is only worth having if its refusals actually fire, so each one is exercised against a
synthetic shard tree built to trip exactly that check -- plus a clean tree that must pass
and produce a complete manifest.

Includes MIXED provenance -- shards spanning more than one commit, which really happened
when Condor restarted four shards of cluster 292477 after the server checkout had moved,
and again when the deep nested arm (293917) was combined with the original sweep (292485).
Both shapes of mix -- two embedded commits, and embedded beside unmanifested -- must refuse
without `--allow-mixed-commits` and must be RECORDED as acknowledged with it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc.validate_classical_accepted import (RejectedError, validate,  # noqa: E402
                                              aggregation_rules as _agg_rules)

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

        # --- NESTED cutoff shards: a different observable, different failure modes ----
        def _nested(path, L, A, seed, commit, max_occ=7, delta=1e-3, bnd=5, seeded=100):
            rungs = []
            for i, c in enumerate((1000, 2000, 4000)):
                at_end = (i == 2)
                rungs.append({"core": c, "E_lo": 1000.0 - i,
                              "delta_shared": delta if at_end else 0.0,
                              "delta_shared_per_site": (delta if at_end else 0.0) / (L ** 3),
                              "delta_nested": 0.0, "delta_nested_per_site": 0.0,
                              "lo_max_occ": max_occ if at_end else 4,
                              "lo_n_boundary_dets": (bnd if at_end else 0),
                              "lo_boundary_weight": 1e-5 if at_end else 0.0,
                              "n_hi_only_seeded": seeded, "n_hi_only_kept": 0,
                              "hi_only_weight": 0.0, "wall_s": 1.0})
            json.dump({"kind": "nb_nested_shard", "L": L, "dim": 3, "A": A, "sites": L ** 3,
                       "seed": seed, "n_b_lo": 3, "n_b_hi": 4, "N_f_lo": 8, "N_f_hi": 16,
                       "rungs": rungs, "done": True, "wall_s": 5.0,
                       "manifest": {"git_commit": commit, "git_dirty": False,
                                    "hostname": "qis1", "timestamp_utc": "2026-09-09T00:00:00+00:00"}},
                      open(path, "w"))

        nd = os.path.join(tmp, "nested"); os.makedirs(nd)
        for sd in (0, 1):
            _nested(os.path.join(nd, f"nested_L2d3_A1_s{sd}.json"), 2, 1, sd, head)
        # a point whose ladder never reached the boundary -> must be labelled no_measurement
        _nested(os.path.join(nd, "nested_L4d3_A1_s0.json"), 4, 1, 0, head,
                max_occ=6, delta=0.0, bnd=0)
        try:
            recs, info, _ = validate([nd], expect_n_b=3, n_b_hi=4, kind="nb_nested_shard")
            labs = {(r["L"], r["A"]): r["label"] for r in recs}
            if labs.get((2, 1)) != "measured" or labs.get((4, 1)) != "no_measurement":
                fails.append(f"nested labels wrong: {labs} -- a ladder that never reached the "
                             "boundary must be no_measurement, never a zero shift")
            if len(info) != 3:
                fails.append(f"nested hashed {len(info)} shards, expected 3")
        except RejectedError as e:
            fails.append(f"clean nested tree rejected: {e}")
        _expect_reject(lambda: validate([nd], expect_n_b=2, n_b_hi=3, kind="nb_nested_shard"),
                       "cutoffs", fails, "nested wrong cutoff pair")
        # negative delta_shared is physically impossible on an identical determinant set
        badn = os.path.join(tmp, "nested_bad"); os.makedirs(badn)
        _nested(os.path.join(badn, "nested_L2d3_A1_s0.json"), 2, 1, 0, head, delta=-5.0)
        _expect_reject(lambda: validate([badn], expect_n_b=3, n_b_hi=4, kind="nb_nested_shard"),
                       "< 0", fails, "negative delta_shared")
        # a nonzero shift with no boundary population breaks the mis-scoring mechanism
        badm = os.path.join(tmp, "nested_mech"); os.makedirs(badm)
        _nested(os.path.join(badm, "nested_L2d3_A1_s0.json"), 2, 1, 0, head, bnd=0)
        _expect_reject(lambda: validate([badm], expect_n_b=3, n_b_hi=4, kind="nb_nested_shard"),
                       "no boundary population", fails, "shift without boundary population")
        # a shard that never showed the search the high-cutoff states
        badz = os.path.join(tmp, "nested_blind"); os.makedirs(badz)
        _nested(os.path.join(badz, "nested_L2d3_A1_s0.json"), 2, 1, 0, head, seeded=0)
        _expect_reject(lambda: validate([badz], expect_n_b=3, n_b_hi=4, kind="nb_nested_shard"),
                       "never looked", fails, "no high-only states seeded")

        # --- BACKEVAL (frame) shards: the P0-1 stall must stay unrepeatable -----------
        def _backeval(path, L, A, n_b, seed, commit, n_dets_ladder=(250, 1000, 4000, 16000, 64000),
                      dropped=None, ktl_break=False):
            res = []
            for nd in n_dets_ladder:
                r = {"core": nd, "n_dets": nd, "E_frame": 2000.0 - nd / 1e4,
                     "E_orig": 2000.0 - nd / 1e4, "reached_target": True, "growth_ok": True,
                     "converged": True, "solve_s": 1.0, "residual": 1.0, "eps_leak": 0.0,
                     "kato_temple_lower": (2500.0 if ktl_break else 1000.0)}
                if dropped is not None:
                    r["back_dropped_weight"] = dropped
                res.append(r)
            json.dump({"done": True, "results": res,
                       "metadata": {"L": L, "dim": 3, "A": A, "n_b": n_b, "N_f": 2 ** n_b,
                                    "frame": "bare", "seed": seed,
                                    "manifest": {"git_commit": commit, "git_dirty": False,
                                                 "hostname": "qis2",
                                                 "timestamp_utc": "2026-09-09T00:00:00+00:00"}}},
                      open(path, "w"))

        bd = os.path.join(tmp, "backeval"); os.makedirs(bd)
        _backeval(os.path.join(bd, "backeval_bare_L2d3nb3_A1_s0.json"), 2, 1, 3, 0, head)
        _backeval(os.path.join(bd, "backeval_bare_L2d3nb4_A1_s0.json"), 2, 1, 4, 0, head)
        try:
            recs, info, _ = validate([bd], expect_n_b=3, allow_n_b=[4], kind="backeval_shard")
            if len(info) != 2:
                fails.append(f"backeval hashed {len(info)} shards, expected 2")
        except RejectedError as e:
            fails.append(f"clean backeval tree rejected: {e}")
        # the n_b=4 cross-check arm must be DECLARED, never discovered silently
        _expect_reject(lambda: validate([bd], expect_n_b=3, kind="backeval_shard"),
                       "DECLARED set", fails, "undeclared cross-check cutoff")
        # THE key gate: the superseded bundle's ladders all stalled at 3,482 determinants
        st = os.path.join(tmp, "stalled"); os.makedirs(st)
        _backeval(os.path.join(st, "backeval_bare_L2d3nb3_A1_s0.json"), 2, 1, 3, 0, head,
                  n_dets_ladder=(250, 1000, 3482))
        _expect_reject(lambda: validate([st], expect_n_b=3, kind="backeval_shard"),
                       "P0-1 stall", fails, "P0-1-stalled ladder")
        # a cap-dominated map-back makes E_orig a truncated remnant
        cap = os.path.join(tmp, "capped"); os.makedirs(cap)
        _backeval(os.path.join(cap, "backeval_bare_L2d3nb3_A1_s0.json"), 2, 1, 3, 0, head,
                  dropped=0.40)
        _expect_reject(lambda: validate([cap], expect_n_b=3, kind="backeval_shard"),
                       "dropped", fails, "cap-dominated map-back")
        # E_orig must sit inside its own Kato-Temple bracket
        kt = os.path.join(tmp, "ktbreak"); os.makedirs(kt)
        _backeval(os.path.join(kt, "backeval_bare_L2d3nb3_A1_s0.json"), 2, 1, 3, 0, head,
                  ktl_break=True)
        _expect_reject(lambda: validate([kt], expect_n_b=3, kind="backeval_shard"),
                       "Kato-Temple", fails, "E_orig below its Kato-Temple bound")

        # --- the manifest's STATED POLICY must match what the code actually does -------
        # The emitted aggregation_rules once said the central value was the mean over seeds with
        # the spread folded into sigma. Both had ceased to be true (best-bound seed supplies the
        # value; the spread is a separate robustness diagnostic), and nothing caught the drift
        # because the text was never compared to behaviour. This does that comparison.
        from classical.trimci.extrapolation import combine_seeds as _cs
        dis = _tree(tmp, "disagree", [dict(L=2, n_b=3, seed=0, E_fci=1795.0, commit=head),
                                      dict(L=2, n_b=3, seed=1, E_fci=1799.0, commit=head),
                                      dict(L=2, n_b=3, seed=2, E_fci=1815.0, commit=head)])
        recs4, _, _ = validate([dis], expect_n_b=3)
        r4 = recs4[0]
        per = {}
        for f in sorted(os.listdir(dis)):
            if f.endswith(".json") and f.startswith("bare_"):
                j = json.load(open(os.path.join(dis, f)))
                per[j["seed"]] = j["rungs"]
        pooled = _cs(per, sites=r4["sites"])
        if not pooled.get("ok"):
            fails.append(f"policy fixture did not extrapolate: {pooled.get('reason')}")
        else:
            best = pooled["best_seed"]
            # POLICY 1: the central value is the best-bound seed's, not a mean over seeds
            if abs(r4["E_inf"] - pooled["per_seed"][best]["E_inf"]) > 1e-9:
                fails.append("manifest E_inf is not the best-bound seed's -- stated seed_rule "
                             "does not match combine_seeds")
            # POLICY 2: sigma is that seed's OWN uncertainty; the spread is not folded in
            if abs(r4["sigma"] - pooled["per_seed"][best]["sigma"]) > 1e-9:
                fails.append("manifest sigma differs from the best seed's own -- the seed spread "
                             "appears to be folded in, contradicting the stated uncertainty rule")
            if r4.get("sigma_seed") is None:
                fails.append("seed spread not reported alongside as a diagnostic")
        # POLICY TEXT: the strings must say those two things, so they cannot go stale silently
        rules = _agg_rules()
        if "best-bound seed" not in rules["seed_rule"] or "never an error-bar term" not in rules["seed_rule"]:
            fails.append(f"seed_rule text no longer states the best-bound/diagnostic policy: {rules['seed_rule']}")
        if "SEED SPREAD IS NOT IN SIGMA" not in rules["uncertainty"]:
            fails.append(f"uncertainty text no longer excludes the seed spread: {rules['uncertainty']}")
        if "mean" in rules["seed_rule"].lower():
            fails.append("seed_rule still describes a mean over seeds")

        # --- MIXED provenance: shards spanning more than one commit -------------------
        # Really happened: Condor restarted 4 shards of cluster 292477 after the server
        # checkout had moved, so they carry a later commit than the 8 launched earlier. The
        # first version of this gate reported ONE commit for that campaign, hiding the mix.
        mixdir = _tree(tmp, "mixed", [dict(L=2, n_b=3, seed=0, commit=head),
                                      dict(L=2, n_b=3, seed=1)])
        _expect_reject(lambda: validate([mixdir], expect_n_b=3, allow_unmanifested=True,
                                        assert_commit=head),
                       "mixed provenance", fails, "mixed manifested/unmanifested")
        try:
            _, _, pm = validate([mixdir], expect_n_b=3, allow_unmanifested=True,
                                assert_commit=head, allow_mixed_commits=True)
            if not pm.get("mixed_commits_acknowledged"):
                fails.append("acknowledged mix not flagged in the manifest")
            if pm.get("asserted_commit") != head or head not in pm.get("embedded_commits", []):
                fails.append(f"mixed record must carry BOTH commits: {pm}")
            if "MIXED" not in pm.get("provenance_source", ""):
                fails.append(f"provenance_source does not say MIXED: {pm}")
        except RejectedError as e:
            fails.append(f"--allow-mixed-commits still refused: {e}")
        parent2 = subprocess.check_output(["git", "rev-parse", "HEAD~1"],
                                          cwd=_ROOT).decode().strip()
        twodir = _tree(tmp, "twocommits", [dict(L=2, n_b=3, seed=0, commit=head),
                                           dict(L=2, n_b=3, seed=1, commit=parent2)])
        _expect_reject(lambda: validate([twodir], expect_n_b=3), "mixed provenance",
                       fails, "two embedded commits")
        # ... and the SAME dataset accepted under the acknowledgement must SAY it is mixed.
        # It did not: `mixed_commits_acknowledged` was computed from the embedded+unmanifested
        # case alone, so the deep nested manifest (2049447 + 93de7f5, both embedded, no
        # unmanifested shard) shipped with `false` while carrying two commits -- a manifest
        # contradicting itself, found by the 2026-09-14 checklist (item 3).
        try:
            _, _, p2c = validate([twodir], expect_n_b=3, allow_mixed_commits=True)
            if not p2c.get("mixed_commits_acknowledged"):
                fails.append("two EMBEDDED commits accepted under --allow-mixed-commits but "
                             "mixed_commits_acknowledged is false -- the manifest would deny "
                             "its own mixed provenance")
            if sorted(p2c.get("embedded_commits", [])) != sorted([head, parent2]):
                fails.append(f"both embedded commits must be retained: {p2c.get('embedded_commits')}")
            if "MIXED" not in p2c.get("provenance_source", ""):
                fails.append(f"provenance_source does not say MIXED for a two-commit dataset: "
                             f"{p2c.get('provenance_source')}")
            if p2c.get("generating_commit") is not None:
                fails.append("a two-commit dataset must not name a single generating_commit -- "
                             f"got {p2c.get('generating_commit')}")
        except RejectedError as e:
            fails.append(f"two embedded commits still refused under --allow-mixed-commits: {e}")
        # the flag must stay FALSE for a clean single-commit dataset, acknowledgement or not
        try:
            _, _, p1c = validate([d], expect_n_b=3, allow_mixed_commits=True)
            if p1c.get("mixed_commits_acknowledged"):
                fails.append("single-commit dataset flagged as mixed just because "
                             "--allow-mixed-commits was passed")
        except RejectedError as e:
            fails.append(f"clean tree rejected under --allow-mixed-commits: {e}")
        # and the acknowledgement must survive into the WRITTEN manifest, not just the call
        rmix = subprocess.run([sys.executable, "-m", "misc.validate_classical_accepted",
                               "--data", twodir, "--expect-n-b", "3", "--label", "twocommits",
                               "--allow-mixed-commits",
                               "--out", os.path.join(tmp, "mixed.json")],
                              cwd=_ROOT, capture_output=True, text=True)
        if rmix.returncode != 0:
            fails.append(f"CLI --allow-mixed-commits exited {rmix.returncode}:\n{rmix.stderr}")
        else:
            mman = json.load(open(os.path.join(tmp, "mixed.json")))["provenance"]
            if not mman.get("mixed_commits_acknowledged"):
                fails.append("written manifest records mixed_commits_acknowledged: false for a "
                             "two-commit dataset")
            if len(mman.get("embedded_commits", [])) != 2:
                fails.append(f"written manifest lost an embedded commit: {mman}")

        # --- an ASSERTED commit must itself be verified ------------------------------
        # It was not: the gate only ever checked the EMBEDDED commit, so a mixed dataset would
        # accept any --assert-commit value, including one that does not exist. A fabricated SHA
        # sat in two live manifests until this check was added.
        _expect_reject(lambda: validate([mixdir], expect_n_b=3, allow_unmanifested=True,
                                        allow_mixed_commits=True, assert_commit="0" * 40),
                       "not a descendant", fails, "unverifiable asserted commit")
        # several CANDIDATE commits are allowed for legacy data, but every one is checked
        _p2 = subprocess.check_output(["git", "rev-parse", "HEAD~1"],
                                      cwd=_ROOT).decode().strip()
        try:
            _, _, pc = validate([nom], expect_n_b=3, allow_unmanifested=True,
                                assert_commit=[head, _p2])
            if not pc.get("asserted_commit_candidates"):
                fails.append("candidate commit set not recorded")
        except RejectedError as e:
            fails.append(f"multi-candidate assertion rejected: {e}")
        _expect_reject(lambda: validate([nom], expect_n_b=3, allow_unmanifested=True,
                                        assert_commit=[head, "0" * 40]),
                       "not a descendant", fails, "one bad candidate among several")

        # --- the manifest file map must not collide -----------------------------------
        # It was keyed by BASENAME, so a dataset assembled from two directories with
        # same-named shards silently hashed fewer files than it validated (12 of 18).
        d2 = _tree(tmp, "samenames", [dict(L=2, n_b=3, seed=0, commit=head)])
        _, info2, _ = validate([d, d2], expect_n_b=3)
        from misc.validate_classical_accepted import build_manifest as _bm
        man2 = _bm("x", [d, d2], [], info2, {"generating_commit": head}, 3)
        if len(man2["files"]) != len(info2):
            fails.append(f"manifest hashed {len(man2['files'])} of {len(info2)} shards "
                         "-- file-map key collision")

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
          "gate all refuse; multi-commit provenance refuses without the acknowledgement and, "
          "with it, is recorded as mixed in both the call and the written manifest; CLI exits "
          "2 on refusal)")


def test_validate_classical_accepted():
    main()


if __name__ == "__main__":
    main()
