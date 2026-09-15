"""The two release-verification tools added for the 2026-09-14 remediation checklist.

`misc/verify_accepted_manifests.py` is the OUTSIDE check: it re-hashes every artifact an
accepted manifest certifies, now, rather than at the moment of acceptance. That distinction is
the whole point -- the 33-shard nested manifest passed its gate and then went stale when the
deep arm overwrote a script and a figure it had certified, and nothing noticed for five days.

`misc/diff_import_closure.py` is what makes a `--allow-mixed-commits` acknowledgement a check
instead of an assertion: it walks the shard entry point's first-party import graph and says
which files differ between the two commits, and whether the difference is executable at all.

Both are exercised against synthetic trees, including the two traps that bit the real data: a
docstring-only change that must NOT be called a code change, and same-named shards in two
directories where only one set matches the recorded digests.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc.verify_accepted_manifests import verify  # noqa: E402
from misc.diff_import_closure import import_closure, classify  # noqa: E402


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _manifest(path, files, prov=None, sect="files", key=os.path.basename):
    """`key` picks the manifest schema: basename (the quantum one) or a repo-relative path."""
    json.dump({"dataset": "synthetic", "provenance": prov or {},
               sect: {key(f): _sha(f) for f in files}}, open(path, "w"))


def _check_verifier(tmp, fails):
    d = os.path.join(tmp, "data", "ds")
    os.makedirs(d)
    shards = []
    for i in range(3):
        p = os.path.join(d, f"shard_{i}.json")
        open(p, "w").write(f'{{"i": {i}}}')
        shards.append(p)
    man = os.path.join(d, "accepted_data_manifest.json")

    # clean, single-commit -> every hash verifies, provenance consistent
    _manifest(man, shards, prov={"embedded_commits": ["a" * 40], "unmanifested_shards": []})
    r = verify(man, root=tmp)
    if r["bad"] or not r["provenance_ok"]:
        fails.append(f"clean manifest did not verify: {r['bad']}")
    if r["basename_dir"] is None:
        fails.append("basename manifest not resolved to its own directory")

    # the OTHER schema: keys are repo-relative paths, resolved against --root, no search
    rel = os.path.join(d, "accepted_data_manifest_rel.json")
    _manifest(rel, shards, key=lambda f: os.path.relpath(f, tmp))
    rr = verify(rel, root=tmp)
    if rr["bad"] or rr["basename_dir"] is not None:
        fails.append(f"repo-relative manifest mishandled: bad={rr['bad']} "
                     f"dir={rr['basename_dir']}")
    os.remove(rel)

    # THE defect the checklist found: an artifact regenerated after acceptance
    open(shards[1], "w").write('{"i": 1, "regenerated": true}')
    r = verify(man, root=tmp)
    if [b[2] for b in r["bad"]] != ["HASH"]:
        fails.append(f"a rewritten artifact was not reported as a hash mismatch: {r['bad']}")
    open(shards[1], "w").write('{"i": 1}')

    # a deleted artifact is MISSING, not silently skipped
    os.rename(shards[2], shards[2] + ".bak")
    r = verify(man, root=tmp)
    if [b[2] for b in r["bad"]] != ["MISSING"]:
        fails.append(f"a deleted artifact was not reported MISSING: {r['bad']}")
    os.rename(shards[2] + ".bak", shards[2])

    # two embedded commits with mixed_commits_acknowledged false = the item-3 contradiction
    _manifest(man, shards, prov={"embedded_commits": ["a" * 40, "b" * 40],
                                 "unmanifested_shards": [],
                                 "mixed_commits_acknowledged": False})
    r = verify(man, root=tmp)
    if r["bad"]:
        fails.append("provenance check should not disturb hashing")
    if r["provenance_ok"] or not r["mixed"]:
        fails.append("a manifest denying its own two-commit provenance was accepted")
    _manifest(man, shards, prov={"embedded_commits": ["a" * 40, "b" * 40],
                                 "unmanifested_shards": [],
                                 "mixed_commits_acknowledged": True})
    if not verify(man, root=tmp)["provenance_ok"]:
        fails.append("an acknowledged mix was still reported inconsistent")

    # BASENAME AMBIGUITY: the same shard names under a second, superseded campaign directory.
    # Presence alone cannot resolve it; the digests can, and must.
    d2 = os.path.join(tmp, "data", "ds_round2")
    os.makedirs(d2)
    for i in range(3):
        open(os.path.join(d2, f"shard_{i}.json"), "w").write(f'{{"i": {i}, "round": 2}}')
    r = verify(man, root=tmp)
    if r["basename_dir"] != os.path.relpath(d, tmp):
        fails.append(f"basename resolution picked {r['basename_dir']}, not the digest-matching "
                     f"directory {os.path.relpath(d, tmp)}")
    if len(r["basename_candidates"]) < 2:
        fails.append("the ambiguous second directory was not even seen as a candidate")
    if r["bad"]:
        fails.append(f"digest-resolved manifest reported failures: {r['bad']}")

    # a SUPERSEDED record is allowed to be stale, and the CLI must skip it
    sup = os.path.join(d, "SUPERSEDED_accepted_data_manifest.json")
    shutil.copy(man, sup)
    open(shards[0], "w").write('{"i": 0, "regenerated": true}')
    out = subprocess.run([sys.executable, "-m", "misc.verify_accepted_manifests",
                          "--root", tmp, "--glob", "**/accepted_data_manifest*.json"],
                         cwd=_ROOT, capture_output=True, text=True)
    if "SUPERSEDED_accepted_data_manifest.json" in out.stdout:
        fails.append("the CLI verified a SUPERSEDED manifest instead of skipping it")
    if out.returncode == 0:
        fails.append("the CLI exited 0 with a stale artifact in a live manifest")
    open(shards[0], "w").write('{"i": 0}')
    out = subprocess.run([sys.executable, "-m", "misc.verify_accepted_manifests",
                          "--root", tmp, "--glob", "**/accepted_data_manifest*.json"],
                         cwd=_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        fails.append(f"the CLI exited {out.returncode} on a clean tree:\n{out.stdout}")


def _check_closure(tmp, fails):
    # RELATIVE imports must be followed: `classical/trimci/__init__.py` reaches extrapolation.py
    # only through `from .extrapolation import ...`, and a walker that skipped that reported a
    # 5-file closure instead of 40 -- and would have declared the real dataset unaffected
    # without ever looking at the file that actually changed.
    closure = import_closure("misc/run_nb_nested_shard.py")
    for expect in ("misc/run_nb_nested_shard.py", "classical/trimci/__init__.py",
                   "classical/trimci/extrapolation.py", "src_PI/utils/manifest.py",
                   "src_PI/hamiltonians/core/MixedHamiltonian.py"):
        if expect not in closure:
            fails.append(f"import closure missed {expect} ({len(closure)} files found)")
    if "tests/test_nb_nested.py" in closure:
        fails.append("import closure wandered into the tests")

    # classify(): a docstring-only edit is COSMETIC, a body edit is REAL and names the symbol.
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")

    def git(*a):
        return subprocess.check_output(["git"] + list(a), cwd=repo, env=env).decode().strip()

    git("init", "-q")
    src = os.path.join(repo, "m.py")
    open(src, "w").write('"""doc A."""\n\n\ndef f(x):\n    return x + 1\n\n\ndef g(x):\n    return x\n')
    git("add", "-A"); git("commit", "-qm", "a")
    ca = git("rev-parse", "HEAD")
    open(src, "w").write('"""doc B, rewritten."""\n\n\ndef f(x):\n    # a comment\n    return x + 1\n\n\ndef g(x):\n    return x\n')
    git("add", "-A"); git("commit", "-qm", "b")
    cb = git("rev-parse", "HEAD")
    kind, syms = classify("m.py", ca, cb, root=repo)
    if kind != "cosmetic":
        fails.append(f"a docstring/comment-only change was classified {kind!r}, not cosmetic")
    open(src, "w").write('"""doc B, rewritten."""\n\n\ndef f(x):\n    return x + 2\n\n\ndef g(x):\n    return x\n')
    git("add", "-A"); git("commit", "-qm", "c")
    cc = git("rev-parse", "HEAD")
    kind, syms = classify("m.py", ca, cc, root=repo)
    if kind != "real" or syms != ["f"]:
        fails.append(f"a changed function body gave {kind!r} {syms}, expected real ['f']")
    if classify("m.py", ca, ca, root=repo)[0] != "identical":
        fails.append("a file compared against itself was not 'identical'")


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="relman_")
    try:
        _check_verifier(tmp, fails)
        _check_closure(tmp, fails)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("test_release_manifest_tools: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print("test_release_manifest_tools: PASS  (re-hashing catches a regenerated and a deleted "
          "artifact; a manifest denying its own mixed provenance is refused; basename "
          "ambiguity is resolved by digest, not by presence; SUPERSEDED records are skipped; "
          "the import closure follows relative imports and separates cosmetic from executable "
          "change)")


def test_release_manifest_tools():
    main()


if __name__ == "__main__":
    main()
