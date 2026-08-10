#!/bin/sh
# Condor executable for ONE Architecture A-vs-B resource shard (task 34).
# args: $1=L  $2=n_b_values ('+'-separated)  $3=squeeze_r  $4=campaign_id
#       $5=mode(opt:"test")  $6=encoder(opt:"pauli_lcu"|"sparse", default pauli_lcu)
# pauli_lcu BUILDS the walk circuit (size-capped ~L=3); sparse is the analytical
# Gilyen-Lemma-30 aggregation (fock_squeezed->fock_native_squeezed) that scales to L=10.
#
# Runs bare `fock` vs squeezed `fock_squeezed@r*` walks across an n_b sweep at fixed L
# (see misc/run_frame_AB_shard.py). Self-provisions per-sandbox (uv cache + managed
# CPython) exactly like run_quantum_shard.sh — NO C++ build (pure-Python symbolic
# counting), and NO classical.trimci import (squeeze_r is computed off-cluster and
# passed in, so the job needs only src_PI + pyLIQTR from requirements-hpc-quantum.txt).
# Writes its per-shard JSON straight to the shared campaign dir (nothing transfers back).
set -u
L="$1"; NB="$2"; R="$3"; CAMPAIGN="$4"; MODE="${5:-run}"; ENCODER="${6:-pauli_lcu}"
# Condor `queue ... from file` splits columns on commas -> the n_b list uses '+' as a
# comma-free separator; normalize back here (see HPC_WORKFLOW.md §10).
NB="$(printf '%s' "$NB" | tr '+' ',')"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_frame_AB_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

# Symbolic estimate (not BLAS-bound); pin threads deterministic.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin"
# Per-sandbox cache + interpreter (a shared NFS uv dir corrupts under concurrent cold
# installs). Sandbox-local Julia depot so any juliacall auto-provision stays contained.
export UV_CACHE_DIR="$SANDBOX/uvcache"
export UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export JULIA_DEPOT_PATH="$SANDBOX/julia_depot"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[AB] host=$(hostname) L=$L n_b=$NB r=$R campaign=$CAMPAIGN mode=$MODE"
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
    "$PY" -m misc.run_frame_AB_shard --test
    status=$?
    echo "[AB] smoke-test status=$status"
    exit "$status"
fi

OUTDIR="$REPO/hpc/quantum/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/L${L}_${ENCODER}_AB.json"
# shellcheck disable=SC2086
"$PY" -m misc.run_frame_AB_shard --L "$L" --dim 3 --n-b-values "$NB" \
    --squeeze-r "$R" --frames bare,squeeze --encoder "$ENCODER" --out "$OUT"
status=$?
echo "[AB] done status=$status -> $OUT"
exit "$status"
