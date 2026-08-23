#!/bin/sh
# One n_b-convergence STUDY shard (empirical Fock-cutoff convergence on the CORRECTED H).
# args: $1=study (A|B|Bdense|Cdilute|Cdense|C|G) $2=campaign
# Self-provisions per-sandbox (uv + mixed_ci C++ build), then runs misc/run_nb_convergence.py
# for ONE study, writing its JSON to $REPO/hpc/nb_cutoff/campaign_$2/shards (rsync'd back).
set -u
STUDY="$1"; CAMPAIGN="$2"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_nb_convergence.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-8}"
# The nb sweep's heavy path is the C++ SpMV/H-build (threaded via OpenMP) + selected-CI ensemble.
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export NUQU_NUM_WORKERS="$cpus"
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[nb] host=$(hostname) study=$STUDY campaign=$CAMPAIGN cpus=$cpus"
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

OUTDIR="$REPO/hpc/nb_cutoff/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$SANDBOX:$REPO" NUQU_NB_OUT_DIR="$OUTDIR"
"$PY" -m misc.run_nb_convergence "$STUDY"
status=$?
echo "[nb] done status=$status study=$STUDY -> $OUTDIR"
exit "$status"
