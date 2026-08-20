"""Reproducibility manifest for a resource-estimation run (task 34, I3).

Pins *what code and parameters produced a data file* so a sweep JSON (laptop or
HPC) is independently reproducible: the git commit, whether the tree was dirty,
the host, a UTC timestamp, and any caller-supplied extras (config, run args).
Task 04 calls for exactly this — "a manifest pinning the git commit hash so the
figures stay reproducible."

Kept dependency-free (stdlib only) so it runs unchanged inside a self-provisioning
HPC sandbox.
"""

import os
import socket
import subprocess
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _git(args):
    try:
        out = subprocess.check_output(
            ['git', *args], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL, timeout=10)
        return out.decode().strip()
    except Exception:
        return None


def git_commit():
    return _git(['rev-parse', 'HEAD'])


def git_is_dirty():
    """True iff TRACKED source files are modified. Untracked files (campaign output,
    the nested `data/` repo, caches) do NOT count — otherwise every HPC run that writes
    output beside the checkout falsely reports dirty (codex data-audit provenance point).
    `--untracked-files=no` restricts the check to tracked modifications = real source drift."""
    status = _git(['status', '--porcelain', '--untracked-files=no'])
    if status is None:
        return None
    return bool(status.strip())


def git_tracked_diff_hash():
    """Short hash of the tracked-file diff when the source is dirty (None if clean/unavailable).
    Lets a dirty run stay auditable: the exact source delta is pinned even without a commit."""
    diff = _git(['diff', 'HEAD'])
    if not diff:
        return None
    import hashlib
    return hashlib.sha1(diff.encode('utf-8', 'replace')).hexdigest()[:12]


def build_manifest(extra=None):
    """Return a manifest dict: git commit + dirty flag + branch, hostname, UTC
    timestamp, and any `extra` (e.g. the Config dict, the shard's run args)."""
    manifest = {
        'git_commit': git_commit(),
        'git_dirty': git_is_dirty(),                 # tracked-source modifications only
        'git_tracked_diff_hash': git_tracked_diff_hash(),
        'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'hostname': socket.gethostname(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest['extra'] = extra
    return manifest
