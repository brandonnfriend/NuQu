#!/bin/sh
# Condor executable for ONE frame shard (deep dilute run + frame x filling comparison).
# args: $1=L $2=seed $3=campaign $4=frame $5=A $6=filling(none|float) $7=max_core
#       $8=ladder_mode(grow|independent) $9=runs(64) $10=orbopt_cycles(10)
#       $11=phase0_core(2000) $12=max_rung_seconds(14400)
#
# Self-provisions per-sandbox (a shared NFS uv dir corrupts under concurrency), builds
# the mixed_ci C++ hot path, then runs misc/run_frame_shard.py which builds the frame
# (bare|gaussian|coo|gaussian+coo) and runs a deep ladder with INCREMENTAL per-rung
# saving (a shard that OOMs/times-out at a deep rung keeps everything it finished).
set -u
L="$1"; SEED="$2"; CAMPAIGN="$3"; FRAME="$4"; A="$5"; FILLING="$6"; MAXCORE="${7:-1024000}"
LADDER_MODE="${8:-grow}"; RUNS="${9:-64}"; ORBOPTCYCLES="${10:-10}"
PHASE0CORE="${11:-2000}"; MAXRUNGSEC="${12:-14400}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/misc/run_frame_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-2}"
# Parallelism = the TrimCI ENSEMBLE across fork'd worker processes (the independent
# random-init solves). Pin ALL numeric libs to 1 thread/process: no core over-
# subscription, and single-thread BLAS is fork-safe (no thread pool to corrupt). The
# earlier fork "stall/OOM" on 290074 was actually random_core hanging on C(108,A)
# enumeration (fixed in d8d43d2), NOT a fork deadlock; the fork ensemble is validated
# bit-identical + ~3-4x. Worker count is adaptive in Python (reads _CONDOR_REQUEST_CPUS).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export NUQU_NUM_WORKERS="$cpus"
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[shard] host=$(hostname) L=$L seed=$SEED frame=$FRAME A=$A filling=$FILLING max_core=$MAXCORE cpus=$cpus"
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
# -fopenmp: threads the H-build (Linux g++ on qis supports it natively).
c++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp -I"$PYBIND_INC" -I"$PY_INC" \
    "$SANDBOX/mixed_ci_pybind.cpp" -o "$SANDBOX/mixed_ci${EXT}" \
    || { echo "ERROR: C++ build failed" >&2; exit 1; }

OUTDIR="$REPO/hpc/detsvsL/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$SANDBOX:$REPO"
FILL_ARG=""
[ "$FILLING" != "none" ] && FILL_ARG="--filling $FILLING"
# distinct shard name per filling (or per A, for explicit-A grids)
if [ "$FILLING" != "none" ]; then FTAG="_f${FILLING}"; else FTAG="_A${A}"; fi
OUT="$OUTDIR/${FRAME}_L${L}${FTAG}_s${SEED}.json"
# Optional knobs via Condor `environment` (defaults reproduce the crossover behavior):
NB="${NUQU_N_B:-2}"                    # boson bits/mode -> N_f=2^NB (the N_f cutoff study)
NRUNGS="${NUQU_N_RUNGS:-12}"           # ladder depth in #rungs; MAXCORE is the real cap
LNRUNS="${NUQU_LADDER_NRUNS:-1}"       # ensemble runs per ladder rung (>1 = unbiased-init convergence)
BIM_ARG=""; [ -n "${NUQU_BOSON_INIT_MEAN:-}" ] && BIM_ARG="--boson-init-mean ${NUQU_BOSON_INIT_MEAN}"
# grow: Phase-0 ensemble + Phase-1 co-evolution + warm-start growth (deep/convergence runs).
# independent: fit the frame ONCE (cheap; NO Phase-1 co-evolution, which is the 60+ min
# cost) then grow a FROZEN frame -- for cheap frame COMPARISONS at equal footing.
if [ "$LADDER_MODE" = "independent" ]; then
  "$PY" -m misc.run_frame_shard --L "$L" --seed "$SEED" --dim 3 --n_b "$NB" --frame "$FRAME" \
      --A "$A" $FILL_ARG --ladder-mode independent --ladder-start 1000 --n-rungs "$NRUNGS" \
      --max-core "$MAXCORE" --frame-runs "$RUNS" --phase0-core "$PHASE0CORE" \
      --orbopt-cycles "$ORBOPTCYCLES" --max-rung-seconds "$MAXRUNGSEC" \
      --ladder-n-runs "$LNRUNS" $BIM_ARG --out "$OUT"
else
  "$PY" -m misc.run_frame_shard --L "$L" --seed "$SEED" --dim 3 --n_b "$NB" --frame "$FRAME" \
      --A "$A" $FILL_ARG --ladder-mode grow --ladder-start 1000 --max-core "$MAXCORE" \
      --phase0-runs "$RUNS" --orbopt-cycles "$ORBOPTCYCLES" \
      --max-rung-seconds "$MAXRUNGSEC" --out "$OUT"
fi
status=$?
echo "[shard] done status=$status -> $OUT"
exit "$status"
