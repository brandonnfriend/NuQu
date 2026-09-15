"""Is a mixed-commit dataset scientifically reproducible? Diff the ENTRY POINT'S IMPORT CLOSURE.

A long campaign can span commits: Condor restarts a shard after the server checkout has
moved, so late shards re-read the code and carry a later commit than the launch. The
release gate (`misc/validate_classical_accepted.py`) refuses that unless the operator
passes `--allow-mixed-commits`, and its help says to pass it only "AFTER diffing the shard
entry point and everything it imports". This is that diff, done mechanically instead of by
eye, so the acknowledgement is a CHECK rather than an assertion.

What it does:
  1. walks the first-party import graph from an entry-point script (absolute and relative
     imports, packages and modules), giving every repo file that the shard can load;
  2. asks git which of those files differ between the two commits;
  3. classifies each difference as COMMENT/DOCSTRING-ONLY (no executable change) or REAL,
     by comparing the `ast.dump` of the two parses with docstrings stripped.

A dataset whose closure differs only in comments/docstrings is byte-equivalent in behaviour.
A REAL difference is not automatically disqualifying -- a changed function the shard never
calls is still inert -- but it must then be named and argued in the justification, not waved
past. The report prints the changed symbols so that argument can be made against a list.

    python -m misc.diff_import_closure --entry misc/run_nb_nested_shard.py \
        --commits 2049447 93de7f5
"""
import argparse
import ast
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_PARTY = ("src_PI", "classical", "misc", "hpc", "tests")


def _candidates(mod):
    p = mod.replace(".", "/")
    return [c for c in (p + ".py", p + "/__init__.py")
            if os.path.exists(os.path.join(_ROOT, c))]


def import_closure(entry, root=_ROOT):
    """Every first-party repo file reachable by import from `entry` (transitively).

    Relative imports are resolved against the importing file's package, which matters here:
    `classical/trimci/__init__.py` pulls in `extrapolation` with `from .extrapolation import
    ...`, and a walker that skipped relative imports would report a far smaller -- and wrong
    -- closure.
    """
    files, stack = set(), [entry]
    while stack:
        f = stack.pop()
        if f in files:
            continue
        files.add(f)
        try:
            tree = ast.parse(open(os.path.join(root, f)).read())
        except (OSError, SyntaxError):
            continue
        pkg = os.path.dirname(f).replace("/", ".")
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level:                       # relative: climb `level-1` packages
                    parts = pkg.split(".")
                    base = ".".join(parts[:len(parts) - (n.level - 1)]) if n.level > 1 else pkg
                    m = f"{base}.{n.module}" if n.module else base
                else:
                    m = n.module or ""
                if not m:
                    continue
                mods.add(m)
                mods.update(f"{m}.{a.name}" for a in n.names)   # `from pkg import module`
        for m in mods:
            if m.split(".")[0] in FIRST_PARTY:
                stack.extend(c for c in _candidates(m) if c not in files)
    return sorted(files)


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _blob(commit, path, root=_ROOT):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root,
                       capture_output=True)
    return r.stdout.decode() if r.returncode == 0 else None


def _defs(src):
    """Top-level and one-level-nested def/class names, for reporting WHAT changed."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = ast.dump(_strip_docstrings(n))
    return out


def classify(path, commit_a, commit_b, root=_ROOT):
    """'identical' | 'cosmetic' (comments/docstrings only) | 'real', plus changed symbols."""
    sa, sb = _blob(commit_a, path, root), _blob(commit_b, path, root)
    if sa is None or sb is None:
        return ("added/removed", [])
    if sa == sb:
        return ("identical", [])
    try:
        da = ast.dump(_strip_docstrings(ast.parse(sa)))
        db = ast.dump(_strip_docstrings(ast.parse(sb)))
    except SyntaxError:
        return ("real", [])
    if da == db:
        return ("cosmetic", [])
    fa, fb = _defs(sa), _defs(sb)
    changed = sorted(set(fa) ^ set(fb)) + sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
    return ("real", changed)


def report(entry, commit_a, commit_b, root=_ROOT):
    closure = import_closure(entry, root)
    diffed = set(subprocess.check_output(
        ["git", "diff", "--name-only", commit_a, commit_b], cwd=root).decode().split())
    rows = [(f,) + classify(f, commit_a, commit_b, root)
            for f in closure if f in diffed]
    return closure, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True, help="entry-point script, repo-relative")
    ap.add_argument("--commits", nargs=2, required=True, metavar=("A", "B"))
    args = ap.parse_args()
    a, b = args.commits
    closure, rows = report(args.entry, a, b)
    print(f"[import closure] {args.entry} -> {len(closure)} first-party files")
    print(f"[commits] {a} .. {b}")
    if not rows:
        print("  NO file on the import closure differs between the two commits.")
    real = [r for r in rows if r[1] == "real"]
    for f, kind, syms in rows:
        print(f"  {kind.upper():14} {f}" + (f"   [{', '.join(syms)}]" if syms else ""))
    print(f"\n{len(rows)} of {len(closure)} closure files differ; "
          f"{len(real)} carr{'ies' if len(real) == 1 else 'y'} an EXECUTABLE change.")
    if real:
        print("Each executable change must be argued inert on the shard's call path, "
              "by name, in the dataset's mixed-commit justification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
