#!/bin/sh
# ONE-TIME tool provisioning for the frame-shard jobs (2026-10-08, infrastructure C4).
#
# Populates $TOOLS (default /nfs_scratch/bfriend3/NuQu/tools) with everything a job used to
# download at start-up, so a network blip no longer costs a shard (294238):
#   uv        a pinned uv binary (x86_64 linux-gnu)
#   uvpy/     a uv-managed CPython 3.10 (python-build-standalone, relocatable)
#   wheels/   a wheelhouse for requirements-hpc.txt (binary wheels for this platform)
#   MANIFEST  what was installed, when, from which requirements file
# run_frame_shard.sh COPIES these into the job sandbox (never writes to $TOOLS), and falls
# back to the downloads if a piece is missing. Each piece is staged and renamed atomically,
# so re-running this while jobs are starting is safe. The submit node and the qis execute
# nodes are the same platform (AlmaLinux 9, x86_64, glibc 2.34), so what is built here runs
# there. Nothing is written to $HOME: the uv cache lives in a temp dir that is removed.
#
#   ssh hep-submit
#   cd /nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL && sh provision_tools.sh
#   NUQU_UV_VERSION=0.12.23 sh provision_tools.sh /nfs_scratch/bfriend3/NuQu/tools
#
# Re-run after bumping requirements-hpc.txt or NUQU_UV_VERSION (keep it equal to UV_PIN in
# run_frame_shard.sh). Verify with: ls -la $TOOLS; cat $TOOLS/MANIFEST
set -eu
TOOLS="${1:-/nfs_scratch/bfriend3/NuQu/tools}"
UV_VERSION="${NUQU_UV_VERSION:-0.12.23}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REQ="$HERE/requirements-hpc.txt"
ARCH="x86_64-unknown-linux-gnu"
[ -r "$REQ" ] || { echo "ERROR: $REQ not found" >&2; exit 1; }
mkdir -p "$TOOLS"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/nuqu-provision.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
export UV_CACHE_DIR="$TMP/cache" UV_NO_PROGRESS=1 UV_PYTHON_DOWNLOADS=automatic
unset UV_PYTHON_INSTALL_DIR

swap_in() {   # swap_in <staged> <final>: atomic replace of a file or directory
  rm -rf "$2.old"
  [ -e "$2" ] && mv "$2" "$2.old"
  mv "$1" "$2"
  rm -rf "$2.old"
}

echo "[provision] uv $UV_VERSION -> $TOOLS/uv"
curl -LsSf --retry 3 --connect-timeout 20 -o "$TMP/uv.tgz" \
  "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$ARCH.tar.gz"
tar -xzf "$TMP/uv.tgz" -C "$TMP"
install -m 755 "$TMP/uv-$ARCH/uv" "$TOOLS/uv.new"
swap_in "$TOOLS/uv.new" "$TOOLS/uv"
UV="$TOOLS/uv"
"$UV" --version

echo "[provision] CPython 3.10 -> $TOOLS/uvpy"
rm -rf "$TOOLS/uvpy.new"
UV_PYTHON_INSTALL_DIR="$TOOLS/uvpy.new" "$UV" python install 3.10
swap_in "$TOOLS/uvpy.new" "$TOOLS/uvpy"
ls "$TOOLS/uvpy" | grep '^cpython-3\.10' >/dev/null || { echo "ERROR: no cpython-3.10 in $TOOLS/uvpy" >&2; exit 1; }

echo "[provision] wheelhouse for $REQ -> $TOOLS/wheels"
UV_PYTHON_INSTALL_DIR="$TOOLS/uvpy" UV_PYTHON_DOWNLOADS=never "$UV" venv --python 3.10 "$TMP/venv" >/dev/null
VIRTUAL_ENV="$TMP/venv" "$UV" pip install -q pip
rm -rf "$TOOLS/wheels.new"
"$TMP/venv/bin/python" -m pip download -q -r "$REQ" -d "$TOOLS/wheels.new" --only-binary=:all:
# prove the wheelhouse is complete OFFLINE, exactly as a job will use it
UV_PYTHON_INSTALL_DIR="$TOOLS/uvpy" UV_PYTHON_DOWNLOADS=never "$UV" venv --python 3.10 "$TMP/venv2" >/dev/null
VIRTUAL_ENV="$TMP/venv2" "$UV" pip install -q --no-index --find-links "$TOOLS/wheels.new" -r "$REQ"
"$TMP/venv2/bin/python" -c 'import numpy, scipy, openfermion, pybind11; print("[provision] offline install OK:", numpy.__version__, scipy.__version__)'
swap_in "$TOOLS/wheels.new" "$TOOLS/wheels"

{
  echo "provisioned_utc: $(date -u +%FT%TZ)"
  echo "host: $(hostname)"
  echo "uv: $UV_VERSION ($("$UV" --version))"
  echo "python: $(ls "$TOOLS/uvpy" | grep '^cpython-3\.10' | head -1)"
  echo "requirements_sha256: $(sha256sum "$REQ" | cut -c1-16)  ($REQ)"
  echo "wheels: $(ls "$TOOLS/wheels" | wc -l) files"
} > "$TOOLS/MANIFEST"
cat "$TOOLS/MANIFEST"
echo "[provision] done: $TOOLS"
