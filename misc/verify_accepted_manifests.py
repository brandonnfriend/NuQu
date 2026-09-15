"""Independent SHA-256 re-verification of EVERY accepted-data manifest in the tree.

The release gate (`misc/validate_classical_accepted.py`, `misc/validate_nb3_anchor.py`) hashes
a dataset at the moment it is accepted. That is the wrong moment to check twice: files get
regenerated afterwards. The 2026-09-14 checklist found exactly that -- the 33-shard nested
manifest was still certifying a script and a PDF that the deep arm had overwritten -- so this
script re-hashes every listed artifact against the recorded digest, now, from the outside.

It also cross-checks the provenance bookkeeping each manifest emits about itself:
a record carrying more than one embedded commit, or embedded shards beside unmanifested ones,
must say `mixed_commits_acknowledged: true`. A manifest that denies its own mixed provenance is
reported as INCONSISTENT even when every hash matches.

Two manifest schemas are in the tree and both are handled: the classical one keys `files` by
REPO-RELATIVE path, the quantum ones key by BASENAME (a dict for the n_b=3 anchor, a list of
records for the historical n_b=2 headline figure). Basenames are ambiguous -- the same shard
names exist under several campaign directories with different contents -- so a basename manifest
is resolved by finding the directory in which EVERY entry matches, and the search refuses to
guess if zero or more than one such directory exists.

Files named `SUPERSEDED_*` are skipped by design: superseding a manifest is how a record is
retired, and a retired record is allowed to have stale hashes.

    python -m misc.verify_accepted_manifests            # exit 0 = every manifest clean
"""
import argparse
import glob
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = ("files", "analysis_scripts", "outputs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def entries(man):
    """[(section, name, recorded_sha)] across both manifest schemas."""
    out = []
    for sect in SECTIONS:
        v = man.get(sect)
        if isinstance(v, dict):
            out += [(sect, k, s) for k, s in v.items()]
        elif isinstance(v, list):
            out += [(sect, e["file"], e.get("sha256")) for e in v if isinstance(e, dict)]
    return out


def _resolve_dir(named, manifest_path, root=_ROOT):
    """For a BASENAME manifest: the unique directory whose files MATCH the recorded digests.

    `named` is [(basename, recorded_sha)]. Searched over the manifest's own directory first,
    then every directory under `data/` holding the first listed name. Presence alone is not
    enough to resolve: the historical quantum headline names 17 shards that exist under BOTH
    `vertexfix_r2_290818` and `vertexfix_r3_290826` -- same names, different runs, different
    contents. Resolution is therefore by digest: a directory matching every entry wins, and if
    two do, that is a real ambiguity and the function refuses (None) rather than guess. When no
    directory matches everything -- a STALE manifest, the case a release check exists for -- the
    best-matching directory is returned so the offending entries can be named individually
    instead of the whole record collapsing to UNRESOLVED.

    Returns (dir, candidates), candidates being every directory holding any listed name.
    """
    names = [n for n, _ in named]
    own = os.path.dirname(manifest_path)
    seeds = [own] + sorted({os.path.dirname(p) for p in
                            glob.glob(os.path.join(root, "data", "**", names[0]), recursive=True)})
    present = [d for d in dict.fromkeys(seeds)
               if any(os.path.isfile(os.path.join(d, n)) for n in names)]

    def _score(d):
        return sum(os.path.isfile(os.path.join(d, n)) and sha256(os.path.join(d, n)) == h
                   for n, h in named)
    scores = {d: _score(d) for d in present}
    exact = [d for d in present if scores[d] == len(named)]
    if len(exact) == 1:
        return exact[0], present
    if exact:                                    # two directories both match every digest
        return None, present                     # -- a real ambiguity, do not guess
    # No directory matches everything. That is the interesting case for a release check: a
    # manifest has gone stale. Resolve to the BEST-matching directory anyway so the offending
    # entries are named as HASH/MISSING instead of the whole manifest collapsing to UNRESOLVED.
    best = [d for d in present if scores[d] == max(scores.values())] if present else []
    return (best[0] if len(best) == 1 else None), present


def verify(manifest_path, root=_ROOT):
    man = json.load(open(manifest_path))
    ents = entries(man)
    repo_rel = [(s, n, h) for s, n, h in ents if os.path.sep in n or "/" in n]
    basenames = [(s, n, h) for s, n, h in ents if (s, n, h) not in repo_rel]
    base_dir, cands = (None, [])
    if basenames:
        base_dir, cands = _resolve_dir([(n, h) for _, n, h in basenames], manifest_path, root)

    rows = []
    for sect, name, recorded in ents:
        if (sect, name, recorded) in repo_rel:
            path = name if os.path.isabs(name) else os.path.join(root, name)
        elif base_dir is None:
            rows.append((sect, name, "UNRESOLVED", None))
            continue
        else:
            path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            rows.append((sect, name, "MISSING", None))
        else:
            got = sha256(path)
            rows.append((sect, name, "OK" if got == recorded else "HASH", got))

    prov = man.get("provenance") or {}
    embedded = prov.get("embedded_commits") or []
    unmanifested = prov.get("unmanifested_shards") or []
    mixed = len(embedded) > 1 or bool(embedded and unmanifested)
    prov_ok = (not mixed) or bool(prov.get("mixed_commits_acknowledged"))
    return {
        "manifest": os.path.relpath(manifest_path, root),
        "rows": rows,
        "n": len(rows),
        "bad": [r for r in rows if r[2] != "OK"],
        "mixed": mixed,
        "acknowledged": prov.get("mixed_commits_acknowledged"),
        "provenance_ok": prov_ok,
        "embedded_commits": embedded,
        "basename_dir": (os.path.relpath(base_dir, root) if base_dir else None),
        "basename_candidates": [os.path.relpath(c, root) for c in cands],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_ROOT)
    ap.add_argument("--glob", default="data/**/accepted_data_manifest*.json")
    args = ap.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(args.root, args.glob), recursive=True)
                   if not os.path.basename(p).startswith("SUPERSEDED"))
    if not paths:
        print("no accepted manifests found", file=sys.stderr)
        return 1
    total = failed = 0
    for p in paths:
        r = verify(p, args.root)
        total += r["n"]
        ok = not r["bad"] and r["provenance_ok"]
        failed += 0 if ok else 1
        prov = ("MIXED, acknowledged" if r["mixed"] and r["acknowledged"] else
                "MIXED, NOT ACKNOWLEDGED" if r["mixed"] else "single-commit")
        print(f"{'PASS' if ok else 'FAIL'}  {r['manifest']}")
        print(f"        {r['n'] - len(r['bad'])}/{r['n']} hashes verified · {prov}"
              + (f" · basenames resolved in {r['basename_dir']}" if r["basename_dir"] else "")
              + ("" if r["provenance_ok"] else "  <-- manifest denies its own mixed provenance"))
        if r["basename_candidates"] and r["basename_dir"] is None:
            print(f"        AMBIGUOUS basename directory: {r['basename_candidates']}")
        for sect, name, why, got in r["bad"]:
            print(f"        {why:10} {sect}/{name}" + (f"  now {got[:16]}…" if got else ""))
    print(f"\n{total} artifacts across {len(paths)} accepted manifests; "
          f"{failed} manifest(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
