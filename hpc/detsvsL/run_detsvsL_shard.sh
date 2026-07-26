#!/bin/sh
# Condor executable for ONE (L, seed) shard of the parallel dets-vs-L campaign.
# args: $1=L  $2=seed  $3=campaign_id  $4=max_core (optional, default 128000)
#
# Self-provisions in the Condor sandbox, but shares uv's cache + managed CPython on
# /nfs_scratch (within the project) across shards, so only the first shard downloads
# (uv's cache is concurrency-safe by design). Writes its per-shard JSON straight to
# the shared campaign dir (nothing transfers back). Runs in the compacting per-mode
# squeeze frame (transform=gaussian). Combine with misc/combine_detsvsL.py afterward.
set -u
L="$1"; SEED="$2"; CAMPAIGN="$3"; MAXCORE="${4:-128000}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_detsvsL_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-2}"
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus" \
       BLIS_NUM_THREADS="$cpus" NUMEXPR_NUM_THREADS="$cpus" MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin"
# Per-shard cache + interpreter IN THE SANDBOX (full isolation). A shared /nfs_scratch
# dir corrupts under many-way concurrent cold `uv python install` (NFS locking is too
# weak): the shared interpreter's _sysconfigdata gets truncated and every pip install
# then fails. Each shard downloads its own (~30-60s); reliable > deduplicated.
export UV_CACHE_DIR="$SANDBOX/uvcache"
export UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[shard] host=$(hostname) L=$L seed=$SEED campaign=$CAMPAIGN cpus=$cpus max_core=$MAXCORE"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "ERROR: uv install failed" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }
uv python install 3.10 >/dev/null 2>&1
uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; exit 1; }
PY="$SANDBOX/venv/bin/python"
VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REPO/hpc/detsvsL/requirements-hpc.txt" \
    || { echo "ERROR: pip install failed" >&2; exit 1; }

# build the mixed_ci C++ hot path in the sandbox
cp "$REPO/classical/trimci/backend_fork/mixed_ci_pybind.cpp" \
   "$REPO/classical/trimci/backend_fork/mixed_ci.hpp" "$SANDBOX/"
PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
c++ -O3 -Wall -shared -std=c++17 -fPIC -I"$PYBIND_INC" -I"$PY_INC" \
    "$SANDBOX/mixed_ci_pybind.cpp" -o "$SANDBOX/mixed_ci${EXT}" \
    || { echo "ERROR: C++ build failed" >&2; exit 1; }

OUTDIR="$REPO/hpc/detsvsL/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$SANDBOX:$REPO"
# n-rungs 11 so ladder_start=250 x2^k reaches 256k (250..256000); max-core caps per L.
"$PY" -m misc.run_detsvsL_shard --L "$L" --seed "$SEED" --dim 3 --A 1 --n_b 2 \
    --transform gaussian --ladder-start 250 --n-rungs 11 --max-core "$MAXCORE" \
    --out "$OUTDIR/L${L}_s${SEED}.json"
status=$?
echo "[shard] done status=$status -> $OUTDIR/L${L}_s${SEED}.json"
exit "$status"
