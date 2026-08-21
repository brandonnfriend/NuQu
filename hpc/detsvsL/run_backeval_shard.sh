#!/bin/sh
# Condor executable for ONE LF back-evaluation benchmark shard (one L, frame, filling).
# args: $1=L $2=frame $3=campaign $4=filling(none|float) $5=cores(csv) $6=n_b(1) $7=dim(3)
#       $8=A(1, used when filling=none)  $9=num_runs(16)  $10=seed(0)
#
# Self-provisions per-sandbox (shared NFS uv dir corrupts under concurrency), builds the
# mixed_ci C++ hot path, then runs misc/run_backeval_benchmark.py: solve the framed H at
# geometric cores, map the solved state back with the EXACT composed unitary (squeeze∘LF),
# score E_orig=<psi|H_bare|psi>, and record support growth / wall / peak-mem / convergence.
# Writes its JSON straight to the shared campaign dir (incremental per core).
set -u
L="$1"; FRAME="$2"; CAMPAIGN="$3"; FILLING="$4"; CORES="${5:-250,1000,4000,16000}"
NB="${6:-1}"; DIM="${7:-3}"; A="${8:-1}"; NRUNS="${9:-16}"; SEED="${10:-0}"
CORES="$(printf '%s' "$CORES" | tr '+' ',')"      # '+' -> ',' (comma-free shards file)
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_backeval_benchmark.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-2}"
# Fork ensemble (the solver's num_runs restarts) parallelizes across cpus; all numeric libs
# pinned to 1 thread (fork-safe BLAS). Same pattern as the frame shard.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export NUQU_NUM_WORKERS="$cpus"
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[bkshard] host=$(hostname) L=$L dim=$DIM n_b=$NB frame=$FRAME filling=$FILLING A=$A cores=$CORES cpus=$cpus"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "ERROR: uv install failed" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }
uv python install 3.10 >/dev/null 2>&1
uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; exit 1; }
PY="$SANDBOX/venv/bin/python"
VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REPO/hpc/detsvsL/requirements-hpc.txt" \
    || { echo "ERROR: pip install failed" >&2; exit 1; }
cp "$REPO/classical/trimci/backend_fork/mixed_ci_pybind.cpp" \
   "$REPO/classical/trimci/backend_fork/mixed_ci.hpp" "$SANDBOX/"
PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
c++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp -I"$PYBIND_INC" -I"$PY_INC" \
    "$SANDBOX/mixed_ci_pybind.cpp" -o "$SANDBOX/mixed_ci${EXT}" \
    || { echo "ERROR: C++ build failed" >&2; exit 1; }

OUTDIR="$REPO/hpc/detsvsL/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$SANDBOX:$REPO"
FILL_ARG=""; FTAG="_A${A}"
[ "$FILLING" != "none" ] && { FILL_ARG="--filling $FILLING"; FTAG="_f${FILLING}"; }
OUT="$OUTDIR/backeval_${FRAME}_L${L}d${DIM}nb${NB}${FTAG}_s${SEED}.json"
# shellcheck disable=SC2086
"$PY" -m misc.run_backeval_benchmark --L "$L" --dim "$DIM" --n_b "$NB" --frame "$FRAME" \
    --cores "$CORES" --A "$A" $FILL_ARG --num-runs "$NRUNS" --seed "$SEED" --out "$OUT"
status=$?
echo "[bkshard] done status=$status -> $OUT"
exit "$status"
