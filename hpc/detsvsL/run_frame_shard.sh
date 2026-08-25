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
# TWO parallelism modes (comparison switch):
#  * default (fork ensemble): parallelism = independent random-init solves across
#    fork'd workers; every numeric lib pinned to 1 thread (fork-safe BLAS, no core
#    oversubscription). Right for the many-seed frame campaigns.
#  * NUQU_DEEP_SOLVE=1 (single deep solve): parallelism = the C++ SpMV threaded across
#    ALL cores via OpenMP (the Hermitian row-gather matvec), NO fork ensemble. This is
#    the 1M+ path -- one warm-grown solve whose eigsh matvec fills the node. mixed_ci is
#    already compiled with -fopenmp below, so the only difference is the thread env.
if [ -n "${NUQU_DEEP_SOLVE:-}" ]; then
  export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
  export NUQU_NUM_WORKERS=1
else
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
  export NUQU_NUM_WORKERS="$cpus"
fi
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
OUT="$OUTDIR/${FRAME}_L${L}d${NUQU_DIM:-3}${FTAG}_s${SEED}.json"   # d<dim> keeps 1D/3D distinct
# Optional knobs via Condor `environment` (defaults reproduce the crossover behavior):
NB="${NUQU_N_B:-2}"                    # boson bits/mode -> N_f=2^NB (the N_f cutoff study)
NRUNGS="${NUQU_N_RUNGS:-12}"           # ladder depth in #rungs; MAXCORE is the real cap
LNRUNS="${NUQU_LADDER_NRUNS:-1}"       # ensemble runs per ladder rung (>1 = unbiased-init convergence)
BIM_ARG=""; [ -n "${NUQU_BOSON_INIT_MEAN:-}" ] && BIM_ARG="--boson-init-mean ${NUQU_BOSON_INIT_MEAN}"
# PT2 external space ~223x core -> OOMs (~150GB at 1M) before the E_var solve does. Deep
# runs cap it low so the ladder reaches 1M+ on E_var (PT2 kept on the shallow rungs).
PT2CAP_ARG=""; [ -n "${NUQU_PT2_MAX_CORE:-}" ] && PT2CAP_ARG="--pt2-max-core ${NUQU_PT2_MAX_CORE}"
# WARM-GROW: after the one-time frame fit, grow the core rung-to-rung warm-started from
# the previous rung (monotone -> smooth convergence curve), not a fresh solve per rung.
WARMGROW_ARG=""; [ -n "${NUQU_WARM_GROW:-}" ] && WARMGROW_ARG="--warm-grow"
P0RUNS="${NUQU_PHASE0_RUNS:-64}"       # warm-grow Phase-0 basin-escape ensemble size
# LADDER_START = the phase-0 ENSEMBLE core (rungs[0]) in warm-grow mode. Bigger = the compact
# ground-state basin is more distinguishable at phase-0, so heavy restarts there can escape the
# delocalized basin at a SMALL core (the "basin collapse" search fix). Default 1000 (legacy).
LADDERSTART="${NUQU_LADDER_START:-1000}"
DIM="${NUQU_DIM:-3}"                   # lattice dim (Tier-1 exact-anchor uses 1D/2D chains)
# --exact-ref: guarded Lanczos true E_inf for the Tier-1 cost anchor (small ED systems).
EXACT_ARG=""; [ -n "${NUQU_EXACT_REF:-}" ] && EXACT_ARG="--exact-ref --exact-max-mem-gb ${NUQU_EXACT_MAX_MEM_GB:-24}"
# --back-eval (GAUSSIAN-ONLY): map each rung's framed |psi~> back through exp(G_sq) onto the
# ORIGINAL bare H -> VARIATIONAL E_orig (>= E_bare) alongside the frame-internal E_var. Makes
# the classical baseline a genuine upper bound. Squeeze's map-back is grow~1 (tractable at all
# L); guarded off for LF/COO frames inside the python. SUPPORT_CAP bounds the deep-rung map-back
# memory (weight-truncate the input state; dropped_weight logged to convergence-test the cap).
BACKEVAL_ARG=""; [ -n "${NUQU_BACK_EVAL:-}" ] && BACKEVAL_ARG="--back-eval"
BACKCAP_ARG=""; [ -n "${NUQU_BACK_SUPPORT_CAP:-}" ] && BACKCAP_ARG="--back-support-cap ${NUQU_BACK_SUPPORT_CAP}"
# COO-paper co-evolution knobs (grow mode only; 5a2112d wired these into run_frame_shard.py).
# Depth caps matter because grow-mode Phase-2 is a single warm-grown trajectory (NO deep-solve
# OpenMP), so an uncapped hpc ceiling (2^(22-L) = 1M at L=2) is intractable single-threaded.
PROFILE="${NUQU_PROFILE:-hpc}"                 # hpc | smoke ladder sizes
P1MODE="${NUQU_PHASE1_MODE:-coevolve}"         # coevolve (faithful) | doubling-fresh (legacy A/B)
SQOPT="${NUQU_SQUEEZE_OPT:-analytic}"          # analytic r* | numerical (the r* study)
P1MAX_ARG=""; [ -n "${NUQU_PHASE1_MAX_DETS:-}" ] && P1MAX_ARG="--phase1-max-dets ${NUQU_PHASE1_MAX_DETS}"
P2MAX_ARG=""; [ -n "${NUQU_PHASE2_MAX_DETS:-}" ] && P2MAX_ARG="--phase2-max-dets ${NUQU_PHASE2_MAX_DETS}"
# grow: Phase-0 ensemble + Phase-1 co-evolution + warm-start growth (deep/convergence runs).
# independent: fit the frame ONCE (cheap; NO Phase-1 co-evolution, which is the 60+ min
# cost) then grow a FROZEN frame -- for cheap frame COMPARISONS at equal footing.
if [ "$LADDER_MODE" = "independent" ]; then
  "$PY" -m misc.run_frame_shard --L "$L" --seed "$SEED" --dim "$DIM" --n_b "$NB" --frame "$FRAME" \
      --A "$A" $FILL_ARG --ladder-mode independent --ladder-start "$LADDERSTART" --n-rungs "$NRUNGS" \
      --max-core "$MAXCORE" --frame-runs "$RUNS" --phase0-core "$PHASE0CORE" \
      --orbopt-cycles "$ORBOPTCYCLES" --max-rung-seconds "$MAXRUNGSEC" --phase0-runs "$P0RUNS" \
      --ladder-n-runs "$LNRUNS" $BIM_ARG $PT2CAP_ARG $EXACT_ARG $WARMGROW_ARG \
      $BACKEVAL_ARG $BACKCAP_ARG --out "$OUT"
else
  # shellcheck disable=SC2086
  "$PY" -m misc.run_frame_shard --L "$L" --seed "$SEED" --dim "$DIM" --n_b "$NB" --frame "$FRAME" \
      --A "$A" $FILL_ARG --ladder-mode grow --ladder-start 1000 --max-core "$MAXCORE" \
      --phase0-runs "$RUNS" --orbopt-cycles "$ORBOPTCYCLES" \
      --profile "$PROFILE" --phase1-mode "$P1MODE" --squeeze-opt "$SQOPT" $P1MAX_ARG $P2MAX_ARG \
      $BACKEVAL_ARG $BACKCAP_ARG --max-rung-seconds "$MAXRUNGSEC" --out "$OUT"
fi
status=$?
echo "[shard] done status=$status -> $OUT"
exit "$status"
