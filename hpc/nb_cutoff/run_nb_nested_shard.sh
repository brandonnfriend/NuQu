#!/bin/sh
# Condor executable for ONE nested boson-cutoff shard (task 35, T3 / audit P0-3).
# args: $1=L $2=dim $3=A $4=seed $5=campaign $6=max_core $7=n_rungs $8=phase0_runs
#       $9=max_rung_seconds $10=also_independent(0|1)
#
# Same self-provisioning pattern as run_frame_shard.sh (per-sandbox uv dirs -- a shared NFS
# uv dir corrupts under concurrent cold installs), builds the mixed_ci C++ hot path, then runs
# misc/run_nb_nested_shard.py with INCREMENTAL per-rung saving.
set -u
L="$1"; DIM="$2"; A="$3"; SEED="$4"; CAMPAIGN="$5"; MAXCORE="${6:-262144}"
NRUNGS="${7:-11}"; P0RUNS="${8:-32}"; MAXRUNGSEC="${9:-14400}"; ALSOIND="${10:-1}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_nb_nested_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-2}"
# The phase-0 ensemble is the only parallel part (fork ensemble); the nested solves are single
# warm-started trajectories. Pin every numeric lib to 1 thread (fork-safe BLAS) and let the
# ensemble fan across the requested cpus.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export NUQU_NUM_WORKERS="$cpus"
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[nested] host=$(hostname) L=$L dim=$DIM A=$A seed=$SEED max_core=$MAXCORE cpus=$cpus"
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
export PYTHONPATH="$SANDBOX:$REPO"
OUT="$OUTDIR/nested_L${L}d${DIM}_A${A}_s${SEED}.json"
IND_ARG=""; [ "$ALSOIND" = "1" ] && IND_ARG="--also-independent"

"$PY" -m misc.run_nb_nested_shard --L "$L" --dim "$DIM" --A "$A" --seed "$SEED" \
    --ladder-start 1000 --n-rungs "$NRUNGS" --max-core "$MAXCORE" \
    --phase0-runs "$P0RUNS" --max-rung-seconds "$MAXRUNGSEC" $IND_ARG --out "$OUT"
status=$?
echo "[nested] done status=$status -> $OUT"
exit "$status"
