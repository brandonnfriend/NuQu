"""Release check + accepted-data manifest for CLASSICAL results (audit 2026-09-05, P0-5).

The n_b=3 quantum anchor has had a checksummed accepted-data manifest and a validator
since re-audit P0-6 (`misc/validate_nb3_anchor.py`). The classical headline, cutoff,
volume-scaling and frame materials had no equivalent, so an accepted classical number
could not be traced to the code and configuration that produced it. This is that
equivalent, and it is a GATE, not a report: every check raises, so a defective or
mislabelled shard cannot reach a figure.

What it records, per the audit's list:
  * the exact input shard list with SHA-256 hashes;
  * the generating commit (from each shard's own manifest where present, else
    operator-asserted and MARKED as such) and a dependency/environment snapshot;
  * the complete physical and solver configuration, cross-checked for agreement
    across shards;
  * the seed and core ladders actually run;
  * the aggregation, exclusion and quality-gate rules applied;
  * the analysis scripts and their output files, hashed.

What it refuses:
  * PRE-VERTEX-FIX inputs. The nucleon spin-isospin vertex bug was fixed in 9404fac
    (2026-08-18); everything earlier is retired and inadmissible. Checked by asking git
    whether the shard's commit is a descendant of the fix -- not by trusting a path.
  * The WRONG CUTOFF. `--expect-n-b` must match every shard (and N_f == 2**n_b), which
    is the P0-1 defect -- an n_b=2 baseline standing in for the n_b=3 model -- made
    mechanically impossible to repeat.
  * MISSING PROVENANCE, unless `--allow-unmanifested` is passed, which downgrades the
    record to operator-asserted and says so in the manifest.
  * A MISLABELLED RESULT. The classical analogue of the quantum compiled/projected
    label is bound-only vs extrapolated: a record claiming an extrapolated E_infinity
    must carry a sigma, and one without a defensible extrapolation must not carry a
    central value.
  * Non-variational or corrupt ladders (rising E_var, NaN/Inf, empty rungs).

    python -m misc.validate_classical_accepted \
        --data data/classical/<date>/bare_baseline_nb3_<cluster> \
        --expect-n-b 3 --label bare_baseline_nb3 \
        --analysis misc/aggregate_classical_energies.py misc/make_classical_baseline_figure.py \
        --outputs results/02_classical_baseline/classical_energy_aggregate.json
"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from classical.trimci.extrapolation import combine_seeds  # noqa: E402

VERTEX_FIX_COMMIT = "9404fac4edf20646cb9862045159667a43e095a8"
VERTEX_FIX_DATE = "2026-08-18"
# Directories from the retired pre-fix campaigns. Belt and braces alongside the git
# ancestry check, since legacy shards carry no manifest to check ancestry against.
RETIRED_DIR_TOKENS = ("2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17",
                      "groupA_290813", "groupB_290814", "290388")


class RejectedError(AssertionError):
    """A release-gate refusal. Distinct from a bug so callers can tell them apart."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dep_versions():
    import platform
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "matplotlib"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    return v


def _is_post_vertex_fix(commit, repo=_ROOT):
    """True/False/None (None = git could not answer, e.g. commit absent from this clone)."""
    if not commit:
        return None
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", VERTEX_FIX_COMMIT, commit],
                           cwd=repo, capture_output=True, timeout=15)
    except Exception:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None                                  # unknown commit / not a repo


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def validate(data_dirs, expect_n_b, expect_dim=3, expect_frame="bare",
             min_rungs=4, min_pt2_post=0, allow_unmanifested=False,
             assert_commit=None, repo=_ROOT):
    """Gate the shards and return (records, shard_info, provenance). Raises RejectedError."""
    files = []
    for d in data_dirs:
        hit = [t for t in RETIRED_DIR_TOKENS if t in d]
        if hit:
            raise RejectedError(
                f"REJECTED {d}: matches retired pre-vertex-fix campaign token {hit[0]!r}. "
                f"All pre-{VERTEX_FIX_DATE} data is inadmissible (vertex bug, fixed "
                f"{VERTEX_FIX_COMMIT[:7]}) and must never feed an accepted result.")
        found = sorted(glob.glob(f"{d}/bare_*.json"))
        if not found:
            raise RejectedError(f"REJECTED {d}: no bare_*.json shards found")
        files += found

    groups, shard_info, commits, unmanifested = defaultdict(dict), [], set(), []
    config = {}
    for f in files:
        tag = os.path.basename(f)
        j = json.load(open(f))
        man = j.get("manifest") or {}
        commit = man.get("git_commit")

        # --- provenance + vertex-fix gate ------------------------------------------
        if commit:
            if man.get("git_dirty") and not man.get("git_tracked_diff_hash"):
                raise RejectedError(f"REJECTED {tag}: generated from a dirty tree with no "
                                    f"recorded diff hash -- the source cannot be pinned")
            post = _is_post_vertex_fix(commit, repo=repo)
            if post is False:
                raise RejectedError(
                    f"REJECTED {tag}: generating commit {commit[:10]} PREDATES the vertex fix "
                    f"{VERTEX_FIX_COMMIT[:7]} ({VERTEX_FIX_DATE}) -- retired data.")
            if post is None:
                raise RejectedError(
                    f"REJECTED {tag}: generating commit {commit[:10]} is not in this clone, so "
                    f"its position relative to the vertex fix cannot be established. Fetch the "
                    f"commit or re-run.")
            commits.add(commit)
        else:
            if not allow_unmanifested:
                raise RejectedError(
                    f"REJECTED {tag}: no provenance manifest. Shards written before "
                    f"misc/run_frame_shard.py started emitting one need "
                    f"--allow-unmanifested --assert-commit <sha>, which records the "
                    f"provenance as OPERATOR-ASSERTED rather than embedded.")
            unmanifested.append(tag)

        # --- cutoff gate (the P0-1 defect, made unrepeatable) -----------------------
        n_b, N_f = j.get("n_b"), j.get("N_f")
        if n_b != expect_n_b:
            raise RejectedError(f"REJECTED {tag}: n_b={n_b}, expected {expect_n_b}. A baseline at "
                                f"a cutoff other than the selected one is not an accepted result.")
        if N_f != 2 ** n_b:
            raise RejectedError(f"REJECTED {tag}: N_f={N_f} != 2**n_b={2 ** n_b}")

        # --- configuration agreement ------------------------------------------------
        if j.get("dim") != expect_dim:
            raise RejectedError(f"REJECTED {tag}: dim={j.get('dim')}, expected {expect_dim}")
        if j.get("frame") != expect_frame:
            raise RejectedError(f"REJECTED {tag}: frame={j.get('frame')!r}, expected {expect_frame!r}")
        cfg = (j.get("filling"), j.get("ladder_mode"), j.get("boson_init_mean"))
        config.setdefault(cfg, []).append(tag)

        # --- ladder integrity -------------------------------------------------------
        rungs = sorted(j.get("rungs", []), key=lambda r: r["core"])
        if not rungs:
            raise RejectedError(f"REJECTED {tag}: no rungs")
        for r in rungs:
            for k in ("core", "E_var"):
                if not _finite(r.get(k)):
                    raise RejectedError(f"REJECTED {tag}: rung {r.get('core')} has non-finite {k}")
            if r.get("dE_pt2") is not None and not _finite(r["dE_pt2"]):
                raise RejectedError(f"REJECTED {tag}: rung {r['core']} has non-finite dE_pt2")
        if len(rungs) < min_rungs:
            raise RejectedError(f"REJECTED {tag}: {len(rungs)} rungs < required {min_rungs}")
        if rungs[-1]["E_var"] > rungs[0]["E_var"]:
            raise RejectedError(f"REJECTED {tag}: E_var RISES across the ladder "
                                f"({rungs[0]['E_var']:.3f} -> {rungs[-1]['E_var']:.3f}) -- "
                                f"a variational ladder must not increase")
        if j.get("A") != (j["sites"] if j.get("filling") == 1.0 else j.get("A")):
            raise RejectedError(f"REJECTED {tag}: A={j.get('A')} inconsistent with filling "
                                f"{j.get('filling')} over {j['sites']} sites")

        groups[(n_b, int(j["L"]))][int(j["seed"])] = rungs
        shard_info.append({
            "file": os.path.relpath(f, _ROOT) if f.startswith(_ROOT) else f,
            "sha256": _sha256(f), "L": j["L"], "n_b": n_b, "N_f": N_f, "A": j["A"],
            "sites": j["sites"], "seed": j["seed"], "done": bool(j.get("done")),
            "n_rungs": len(rungs), "cores": [r["core"] for r in rungs],
            "n_pt2_rungs": sum(1 for r in rungs if r.get("dE_pt2") is not None),
            "wall_s": j.get("wall_s"), "git_commit": commit,
            "host": man.get("hostname"), "timestamp_utc": man.get("timestamp_utc"),
            "solver": ((man.get("extra") or {}).get("solver") if man else None),
        })

    if len(config) > 1:
        raise RejectedError(f"REJECTED: shards disagree on (filling, ladder_mode, boson_init_mean): "
                            f"{ {str(k): v for k, v in config.items()} }")

    # --- derived records + the bound-only / extrapolated LABEL gate ------------------
    records = []
    for (n_b, L), per_seed in sorted(groups.items()):
        sites = next(s["sites"] for s in shard_info if (s["n_b"], s["L"]) == (n_b, L))
        pooled = combine_seeds(per_seed, sites=sites)
        label = "extrapolated" if pooled["ok"] else "bound_only"
        if label == "extrapolated" and pooled.get("sigma") is None:
            raise RejectedError(f"REJECTED (n_b={n_b}, L={L}): labelled 'extrapolated' but carries "
                                f"no uncertainty. A central value without an error bar is not a "
                                f"reportable result (audit P0-2).")
        if label == "bound_only" and pooled.get("E_inf") is not None:
            raise RejectedError(f"REJECTED (n_b={n_b}, L={L}): labelled 'bound_only' but carries a "
                                f"central value {pooled['E_inf']}")
        if min_pt2_post:
            worst = min(v.get("n_pt2_post", 0) for v in pooled["per_seed"].values())
            if worst < min_pt2_post:
                raise RejectedError(
                    f"REJECTED (n_b={n_b}, L={L}): only {worst} post-collapse PT2 rungs "
                    f"(need {min_pt2_post}) -- run with PT2 on every rung")
        records.append({
            "n_b": n_b, "L": L, "sites": sites, "label": label,
            "seeds": pooled["seeds"], "n_seeds": pooled["n_seeds"],
            "E_var_bound": pooled["E_var_bound"], "E_var_bound_per_site": pooled["E_var_bound_ps"],
            "E_inf": pooled.get("E_inf"), "E_inf_per_site": pooled.get("E_inf_ps"),
            "sigma": pooled.get("sigma"), "sigma_per_site": pooled.get("sigma_ps"),
            "sigma_seed": pooled.get("sigma_seed"), "reason": pooled.get("reason"),
            "core_ladders": {str(s): [r["core"] for r in rr] for s, rr in per_seed.items()},
        })

    if len(commits) > 1:
        raise RejectedError(f"REJECTED: shards span multiple generating commits "
                            f"{sorted(c[:10] for c in commits)} -- an accepted result must come "
                            f"from one.")
    provenance = {
        "generating_commit": (sorted(commits)[0] if commits else assert_commit),
        "provenance_source": ("embedded shard manifest" if commits and not unmanifested else
                              "OPERATOR-ASSERTED (shards predate per-shard manifests)"),
        "unmanifested_shards": unmanifested,
        "vertex_fix_commit": VERTEX_FIX_COMMIT, "vertex_fix_date": VERTEX_FIX_DATE,
    }
    if provenance["generating_commit"] is None:
        raise RejectedError("REJECTED: no generating commit -- pass --assert-commit for "
                            "unmanifested shards.")
    if unmanifested:
        post = _is_post_vertex_fix(provenance["generating_commit"], repo=repo)
        if post is not True:
            raise RejectedError(
                f"REJECTED: asserted commit {provenance['generating_commit'][:10]} is not a "
                f"descendant of the vertex fix {VERTEX_FIX_COMMIT[:7]} (or is unknown here).")
    return records, shard_info, provenance


def build_manifest(label, data_dirs, records, shard_info, provenance, expect_n_b,
                   analysis=(), outputs=(), gates=None):
    def hashed(paths):
        out = {}
        for p in paths:
            full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
            out[p] = _sha256(full) if os.path.exists(full) else None
        return out

    return {
        "dataset": label,
        "kind": "classical selected-CI (TrimCI) accepted data",
        "cutoff_n_b": expect_n_b,
        "input_dirs": list(data_dirs),
        "provenance": provenance,
        "dependencies": _dep_versions(),
        "physical_and_solver_config": {
            "note": "per-shard configuration is embedded in each shard's manifest.extra "
                    "(physical + solver); the cross-shard invariants checked here are dim, "
                    "frame, filling, ladder mode, boson init and the cutoff.",
            "shards": shard_info,
        },
        "seed_and_core_ladders": {f"n_b={r['n_b']},L={r['L']}": r["core_ladders"]
                                  for r in records},
        "aggregation_rules": gates or {},
        "results": records,
        "analysis_scripts": hashed(analysis),
        "outputs": hashed(outputs),
        "files": {os.path.basename(s["file"]): s["sha256"] for s in shard_info},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--expect-n-b", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--frame", default="bare")
    ap.add_argument("--min-rungs", type=int, default=4)
    ap.add_argument("--min-pt2-post", type=int, default=0,
                    help="require at least this many POST-collapse PT2 rungs per seed")
    ap.add_argument("--analysis", nargs="*", default=[])
    ap.add_argument("--outputs", nargs="*", default=[])
    ap.add_argument("--allow-unmanifested", action="store_true")
    ap.add_argument("--assert-commit", default=None)
    ap.add_argument("--out", default=None, help="manifest path (default: <first data dir>/accepted_data_manifest.json)")
    args = ap.parse_args()

    try:
        records, shard_info, prov = validate(
            args.data, args.expect_n_b, expect_dim=args.dim, expect_frame=args.frame,
            min_rungs=args.min_rungs, min_pt2_post=args.min_pt2_post,
            allow_unmanifested=args.allow_unmanifested, assert_commit=args.assert_commit)
    except RejectedError as e:
        # A gate refusal is an expected outcome, not a crash -- print it plainly and exit 2
        # so a release script can distinguish "rejected" (2) from "broken" (1).
        print(f"[validate classical] FAIL\n  {e}", file=sys.stderr)
        sys.exit(2)

    gates = {
        "basin_split": "ladder split at the largest single-doubling E_var drop; only "
                       "POST-collapse rungs are fitted (pre-collapse PT2 extrapolation is "
                       "unreliable -- +19 MeV/site at L=2)",
        "extrapolators": "SHCI/PT2 intercept preferred when PT2 exists on >=3 post-collapse "
                         "rungs; power law E_inf + a*N^-b otherwise, requiring >=4 rungs so the "
                         "3-parameter fit keeps a degree of freedom",
        "uncertainty": "quadrature of fit, method (PT2 vs power-law disagreement), stability "
                       "(leave-one-out refit) and seed spread; covers the N->inf limit of this "
                       "ladder ONLY -- not n_b, lattice/finite volume, EFT truncation, or search "
                       "basin risk",
        "exclusions": "an extrapolation with no estimable uncertainty, or one landing above the "
                      "deepest Ritz value, is refused -- the row becomes bound_only",
        "seed_rule": "tightest (lowest) E_var across seeds is the quoted bound; E_inf is the mean "
                     "over seeds that extrapolated, with the spread folded into sigma",
        "min_rungs": args.min_rungs, "min_pt2_post": args.min_pt2_post,
    }
    man = build_manifest(args.label, args.data, records, shard_info, prov, args.expect_n_b,
                         analysis=args.analysis, outputs=args.outputs, gates=gates)
    out = args.out or os.path.join(args.data[0], "accepted_data_manifest.json")
    json.dump(man, open(out, "w"), indent=2)
    lab = ", ".join(f"n_b{r['n_b']}/L{r['L']}:{r['label']}" for r in records)
    print(f"[validate classical] PASS — {len(shard_info)} shards, n_b={args.expect_n_b}, "
          f"commit {prov['generating_commit'][:10]} ({prov['provenance_source']}); {lab}")
    print(f"[manifest] wrote {out}")


if __name__ == "__main__":
    main()
