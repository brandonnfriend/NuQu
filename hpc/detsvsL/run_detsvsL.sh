#!/bin/sh
# Condor executable — self-provisions its Python env on the compute node, then runs
# the classical dets-vs-L driver.
#
# Probe (cluster 289774, qis3) established that a qis execute node: reads
# /nfs_scratch, has outbound internet + g++11, but has NO uv, only python3.9, and no
# home mount. So we install uv + CPython 3.10 + the pinned deps and compile the
# mixed_ci C++ hot path INTO THE CONDOR SANDBOX (all discarded on exit). The repo is
# only READ from /nfs_scratch; only the small rundir (outputs) transfers back.
#
# arg $1 = mode (test|full).
set -u
MODE="${1:-full}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
RUNDIR="$(ls -d "$SANDBOX"/rundir* 2>/dev/null | head -1)"
[ -n "$RUNDIR" ] || { echo "ERROR: no rundir in sandbox" >&2; exit 1; }
[ -r "$REPO/misc/run_dets_vs_L.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-2}"
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus" \
       BLIS_NUM_THREADS="$cpus" NUMEXPR_NUM_THREADS="$cpus" MPLBACKEND=Agg
# keep ALL tool state in the writable sandbox (the real home isn't mounted on the node)
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy" UV_CACHE_DIR="$SANDBOX/uvcache"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[run] host=$(hostname) mode=$MODE cpus=$cpus"
echo "[run] installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "ERROR: uv install failed" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }
echo "[run] uv: $(uv --version)"
uv python install 3.10 >/dev/null 2>&1
uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; exit 1; }
PY="$SANDBOX/venv/bin/python"
echo "[run] python: $("$PY" --version)"
VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REPO/hpc/detsvsL/requirements-hpc.txt" \
    || { echo "ERROR: pip install failed" >&2; exit 1; }

echo "[run] building mixed_ci C++ hot path..."
cp "$REPO/classical/trimci/backend_fork/mixed_ci_pybind.cpp" \
   "$REPO/classical/trimci/backend_fork/mixed_ci.hpp" "$SANDBOX/"
PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
c++ -O3 -Wall -shared -std=c++17 -fPIC -I"$PYBIND_INC" -I"$PY_INC" \
    "$SANDBOX/mixed_ci_pybind.cpp" -o "$SANDBOX/mixed_ci${EXT}" \
    || { echo "ERROR: C++ build failed" >&2; exit 1; }
echo "[run] built mixed_ci${EXT}"

# --- run parameters ----------------------------------------------------------
if [ "$MODE" = "test" ]; then
  DIM=1; LVALUES="2"; A=2; NB=1; EPS="1.0"; MAXCORE=400; NRUNS=2; NRUNGS=3; LADDER0=100; RUNGCAP=120
else
  DIM=3; LVALUES="2 3"; A=1; NB=2; EPS="1.0 0.1"; MAXCORE=50000; NRUNS=4; NRUNGS=8; LADDER0=500; RUNGCAP=3600
fi
STAMP="$(date +%Y%m%d-%H%M%S)"; LABEL="detsvsL_hpc_${MODE}_${STAMP}"
# -----------------------------------------------------------------------------

export PYTHONPATH="$SANDBOX:$REPO"    # $SANDBOX has the freshly-built mixed_ci.so
cd "$RUNDIR"                          # driver writes data/classical/ here -> returned
echo "[run] dim=$DIM L=[$LVALUES] A=$A N_f=$((1<<NB)) max_core=$MAXCORE n_runs=$NRUNS cap=${RUNGCAP}s label=$LABEL"
"$PY" -m misc.run_dets_vs_L --hpc --dim "$DIM" --L $LVALUES --A "$A" --n_b "$NB" \
    --eps $EPS --max-core "$MAXCORE" --n_runs "$NRUNS" --n-rungs "$NRUNGS" \
    --ladder-start "$LADDER0" --max-rung-seconds "$RUNGCAP" --label "$LABEL" \
    > "${LABEL}.log" 2>&1
status=$?
echo "[run] driver exit=$status"; tail -25 "${LABEL}.log" 2>/dev/null
echo "[run] rundir outputs: $(ls data/classical/ 2>/dev/null | tr '\n' ' ')"
exit "$status"
