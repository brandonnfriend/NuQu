"""Accepted-data validation + manifest for the n_b=3 quantum anchor (re-audit P0-6).

Runs the SAME acceptance invariants used for the round-3 n_b=2 anchor (pruning-within-budget,
walk-T fit ≥3 samples with residual <1, coherent total-T arithmetic, finite/positive fields, single
clean generating commit, unique L) on the exact n_b=3 compile shards, and writes a checksummed
manifest (SHA-256 + code/env provenance). Raises on any failure so a defective shard can never feed
`make_nb3_headline`.

    python -m misc.validate_nb3_anchor --data data/quantum/nb3_anchor
"""
import argparse
import glob
import hashlib
import json
import math
import os

from src_PI.estimation.qpe_cost import walk_queries, WALK_QUERY_CONSTANT_HEISENBERG as PI


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dep_versions():
    import platform
    import sys
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "matplotlib"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    return v


def validate(data_dir):
    files = [f for f in sorted(glob.glob(f"{data_dir}/*fock_pauli_nb3*.json"))
             if "rep2" not in os.path.basename(f)]
    assert files, f"no n_b=3 shards in {data_dir}"
    recs, commits = [], set()
    for f in files:
        j = json.load(open(f))
        assert j.get("done"), f"{os.path.basename(f)}: not done"
        r = j["results"][0]
        b = r.get("QPE_Budget") or {}
        fit = b.get("walk_T_fit") or {}
        man = (j.get("metadata") or {}).get("manifest") or {}
        tag = f"L={r['L']} n_b={r['n_b']} ({os.path.basename(f)})"
        commits.add(man.get("git_commit"))
        assert man.get("git_dirty") is False, f"{tag}: generated from a dirty tree"
        assert b.get("delta_E") == 1.0, f"{tag}: delta_E != 1 MeV ({b.get('delta_E')})"
        assert r["n_b"] == 3, f"{tag}: not n_b=3"
        assert b.get("prune_within_budget") is True, f"{tag}: pruning not within budget"
        assert (fit.get("n_samples") or 0) >= 3, f"{tag}: walk-T fit <3 samples"
        assert fit.get("resid") is not None and fit["resid"] < 1.0, f"{tag}: fit resid not <1 ({fit.get('resid')})"
        eps = b.get("eps_qpe")
        for k, v in dict(lam=r["Physical_Lambda"], q=r["Logical_Qubits"], walkT=r["Walk_T_Count"],
                         nwalk=r["QPE_Walk_Queries"], qpeT=r["QPE_Total_T_Count"], eps=eps).items():
            assert v is not None and math.isfinite(v) and v > 0, f"{tag}: {k} missing/nonfinite ({v})"
        # raw stored arithmetic coherent
        prod = r["QPE_Walk_Queries"] * r["Walk_T_Count"]
        assert abs(prod - r["QPE_Total_T_Count"]) <= 1e-9 * r["QPE_Total_T_Count"], \
            f"{tag}: raw total-T arithmetic incoherent"
        # adopted-π recompute self-consistent (walk_queries(lam,eps,π)*walkT)
        pi_T = walk_queries(r["Physical_Lambda"], eps, PI) * r["Walk_T_Count"]
        assert pi_T > 0 and math.isfinite(pi_T), f"{tag}: π T recompute failed"
        # PADDING-LAW consistency (guards against a stale/mislabeled walk_T — the re-audit L=7 lesson):
        # pyLIQTR pads PREPARE to a power of 2, so the rotation count must be derived from THIS shard's
        # own term count. rot == 2·2^⌈log2(terms)⌉. (L=6,7 sharing a bin → identical rot is LEGITIMATE;
        # this catches only a rot that doesn't match its own terms — the true corruption signature.)
        terms, rot = r.get("Pauli_Term_Count"), r.get("Rotation_Count")
        assert terms and rot, f"{tag}: missing Pauli_Term_Count/Rotation_Count"
        P = 2 ** math.ceil(math.log2(terms))
        assert rot == 2 * P, f"{tag}: rotation count {rot} != 2·2^⌈log2({terms})⌉={2*P} (walk_T not from this shard)"
        assert abs(fit["b"] / P - 6.0206) < 0.05, f"{tag}: b/P={fit['b']/P:.4f} off the synthesis constant 6.02"
        recs.append(dict(L=r["L"], file=f, bP=fit["b"] / P))
    Ls = [r["L"] for r in recs]
    assert len(Ls) == len(set(Ls)), f"duplicate L in n_b=3 anchor: {Ls}"
    assert len(commits) == 1 and None not in commits, f"multiple/absent commits: {commits}"
    # cross-shard: the per-rotation synthesis slope b/P is a fixed constant — all shards must agree
    bps = [x["bP"] for x in recs]
    assert max(bps) - min(bps) < 0.02, f"b/P inconsistent across shards: {bps}"
    print(f"[validate nb3] PASS — {len(recs)} shards L={sorted(Ls)}, commit {list(commits)[0][:10]}, "
          f"all pruning-clean, fit resid<1, arithmetic coherent, padding-law rot==2·2^⌈log2 terms⌉, "
          f"b/P const {sum(bps)/len(bps):.3f}")
    return sorted(Ls), list(commits)[0], files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quantum/nb3_anchor")
    args = ap.parse_args()
    Ls, commit, files = validate(args.data)
    manifest = {
        "dataset": "nb3_anchor (fock_pauli, n_b=3)", "cutoff_n_b": 3,
        "L_compiled": Ls, "generating_commit": commit,
        "note": "n_b=3 quantum anchor — validated with the round-3 acceptance invariants + the "
                "padding-law rotation-count check (rot==2·2^ceil(log2 terms)). L=1..%d compiled; L>%d "
                "in the headline are padding-model projections (not in this manifest)."
                % (max(Ls), max(Ls)),
        "dependencies": _dep_versions(),
        "files": {os.path.basename(f): _sha256(f) for f in files},
    }
    out = f"{args.data}/accepted_data_manifest_nb3.json"
    json.dump(manifest, open(out, "w"), indent=2)
    print(f"[manifest] wrote {out}  ({len(files)} shards hashed)")


if __name__ == "__main__":
    main()
