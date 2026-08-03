#!/bin/sh
# Condor executable for ONE (L, series) quantum resource-estimation shard.
# args: $1=L  $2=series  $3=campaign_id  $4=A_values (csv, opt)  $5=mode (opt: "test")
#
# Self-provisions in the Condor sandbox (per-sandbox uv cache + managed CPython;
# a SHARED /nfs_scratch uv dir corrupts under concurrent cold installs — see
# HPC_WORKFLOW.md §4). Unlike the classical shard there is NO C++ build step:
# the quantum estimate is pure-Python/numpy symbolic counting. Writes its per-shard
# JSON straight to the shared campaign dir (nothing transfers back).
#
# `mode=test` runs the pyLIQTR/Julia/gmpy2 smoke test instead of a real shard —
# run this once on a qis node before launching a campaign.
set -u
L="$1"; SERIES="$2"; CAMPAIGN="$3"; AVALS="${4:-1,2,4}"; MODE="${5:-run}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_quantum_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

# Quantum estimate is symbolic (not BLAS-bound); keep threads modest+deterministic.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin"
# Per-sandbox cache + interpreter (full isolation; shared NFS uv dir corrupts under
# concurrent cold `uv python install`). Also give juliacall/juliapkg a sandbox-local
# depot so any Julia auto-provision stays inside the sandbox and is discarded on exit.
export UV_CACHE_DIR="$SANDBOX/uvcache"
export UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export JULIA_DEPOT_PATH="$SANDBOX/julia_depot"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[qshard] host=$(hostname) L=$L series=$SERIES campaign=$CAMPAIGN A=$AVALS mode=$MODE"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "ERROR: uv install failed" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }
uv python install 3.10 >/dev/null 2>&1
uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; exit 1; }
PY="$SANDBOX/venv/bin/python"
VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REPO/hpc/quantum/requirements-hpc-quantum.txt" \
    || { echo "ERROR: pip install failed" >&2; exit 1; }

export PYTHONPATH="$REPO"
cd "$REPO" || { echo "ERROR: cd repo failed" >&2; exit 1; }

if [ "$MODE" = "test" ]; then
    "$PY" -m misc.run_quantum_shard --test
    status=$?
    echo "[qshard] smoke-test status=$status"
    exit "$status"
fi

OUTDIR="$REPO/hpc/quantum/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
"$PY" -m misc.run_quantum_shard --L "$L" --series "$SERIES" --dim 3 \
    --A-values "$AVALS" --out "$OUTDIR/L${L}_${SERIES}.json"
status=$?
echo "[qshard] done status=$status -> $OUTDIR/L${L}_${SERIES}.json"
exit "$status"
