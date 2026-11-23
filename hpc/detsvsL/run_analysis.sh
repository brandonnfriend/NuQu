#!/bin/sh
# Run one Python ANALYSIS module on a qis node (2026-11-22): the same pinned-tool provisioning
# as run_frame_shard.sh (uv, CPython 3.10, wheelhouse from $NUQU_TOOLS, 3-attempt retries,
# provisioning failure -> exit 3 so the submit file re-queues), but no C++ build and no solve.
#     run_analysis.sh <python module> [args...]      e.g. misc.ab_collapse_rule --data-root ...
# Threads: the module parallelises itself (e.g. --workers); BLAS/OpenMP are pinned to 1 so the
# workers do not oversubscribe the slot.
set -u
PROVISION_FAIL=3
MOD="$1"; shift
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_frame_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit "$PROVISION_FAIL"; }
detect_cpus() {
  for v in "${NUQU_CPUS:-}" "${_CONDOR_REQUEST_CPUS:-}" "${OMP_THREAD_LIMIT:-}" "${PYTHON_CPU_COUNT:-}"; do
    case "$v" in ''|*[!0-9]*) ;; *) [ "$v" -gt 0 ] && { echo "$v"; return; } ;; esac
  done
  if [ -n "${_CONDOR_MACHINE_AD:-}" ] && [ -r "$_CONDOR_MACHINE_AD" ]; then
    c=$(awk '$1 == "Cpus" && $2 == "=" {print $3; exit}' "$_CONDOR_MACHINE_AD")
    case "$c" in ''|*[!0-9]*) ;; *) echo "$c"; return ;; esac
  fi
  echo 2
}
cpus="$(detect_cpus)"
export NUQU_CPUS_RESOLVED="$cpus" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"
echo "[analysis] host=$(hostname) module=$MOD cpus=$cpus"
TOOLS="${NUQU_TOOLS:-/nfs_scratch/bfriend3/NuQu/tools}"   # see provision_tools.sh
UV_PIN="${NUQU_UV_VERSION:-0.12.23}"
REQ="$REPO/hpc/detsvsL/requirements-hpc.txt"
# retry N cmd...: up to N attempts, sleeping 20 s, 40 s, ... between them (a blip, not an outage)
retry() {
  n="$1"; shift; i=1
  while :; do
    "$@" && return 0
    [ "$i" -ge "$n" ] && return 1
    echo "[analysis] attempt $i/$n failed: $*  (retrying)" >&2
    sleep $(( i * 20 )); i=$(( i + 1 ))
  done
}
# uv: the pinned binary from $TOOLS, copied into the sandbox; else the versioned installer.
provision_uv() {
  mkdir -p "$UV_INSTALL_DIR"
  if [ -x "$TOOLS/uv" ] && cp "$TOOLS/uv" "$UV_INSTALL_DIR/uv" 2>/dev/null; then
    UV_SRC="tools"
  else
    # download THEN run (a `curl | sh` pipeline reports sh's status, so a failed curl
    # looked like success and only the later `uv not on PATH` check caught it: 294238)
    retry 3 sh -c "curl -LsSf --connect-timeout 20 --max-time 300 -o '$SANDBOX/uv-install.sh' https://astral.sh/uv/$UV_PIN/install.sh && sh '$SANDBOX/uv-install.sh' >/dev/null 2>&1" \
      || { echo "ERROR: uv install failed" >&2; return 1; }
    UV_SRC="download"
  fi
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; return 1; }
}
# CPython 3.10: the managed install from $TOOLS/uvpy copied into the sandbox (relocatable
# python-build-standalone; copying avoids any concurrent write to NFS); else uv downloads it.
provision_python() {
  mkdir -p "$UV_PYTHON_INSTALL_DIR"
  if [ -d "$TOOLS/uvpy" ] && ls "$TOOLS/uvpy" 2>/dev/null | grep -q '^cpython-3\.10' \
     && cp -r "$TOOLS/uvpy/." "$UV_PYTHON_INSTALL_DIR/" 2>/dev/null; then
    PY_SRC="tools"; export UV_PYTHON_DOWNLOADS=never
  else
    rm -rf "$UV_PYTHON_INSTALL_DIR"; mkdir -p "$UV_PYTHON_INSTALL_DIR"
    retry 3 uv python install 3.10 || { echo "ERROR: uv python install failed" >&2; return 1; }
    PY_SRC="download"
  fi
  uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; return 1; }
}
# deps: the wheelhouse in $TOOLS/wheels, installed from its compiled lock (no network, no
# resolving); else PyPI with retries.
provision_wheels() {
  if [ -r "$TOOLS/wheels/requirements.lock" ] && VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q --no-index \
       --find-links "$TOOLS/wheels" -r "$TOOLS/wheels/requirements.lock" >/dev/null 2>&1; then
    WHL_SRC="tools"
  else
    retry 3 env VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REQ" \
      || { echo "ERROR: pip install failed" >&2; return 1; }
    WHL_SRC="pypi"
  fi
}
provision_uv      || exit "$PROVISION_FAIL"
provision_python  || exit "$PROVISION_FAIL"
provision_wheels  || exit "$PROVISION_FAIL"
PY="$SANDBOX/venv/bin/python"
echo "[analysis] provisioned uv=$UV_SRC python=$PY_SRC wheels=$WHL_SRC ($("$PY" --version 2>&1))"
cd "$REPO" || exit "$PROVISION_FAIL"
"$PY" -m "$MOD" "$@"
