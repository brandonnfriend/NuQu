#!/bin/sh
# Condor executable for the classical dets-vs-L HPC test run.
#
# Runs in the Condor execute sandbox. Reads code + venv from the shared
# /nfs_scratch checkout (qis nodes mount it), writes outputs BOTH into
# $REPO/data/classical/ (durable, rsync-able) and the returned rundir.
#
# Provisioning (venv + C++ backend) is done once by setup_env.sh — this script
# only activates and runs.
REPO="/nfs_scratch/bfriend3/NuQu/NuQu"

# --- capture the transferred (returned-ON_EXIT) rundir ----------------------
cd rundir* 2>/dev/null || { echo "ERROR: no rundir in sandbox" >&2; exit 1; }
SANDBOX="$(pwd)"

# --- thread env = requested cpus (numpy/scipy BLAS) + headless plotting ------
cpus="${_CONDOR_REQUEST_CPUS:-4}"
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus" \
       BLIS_NUM_THREADS="$cpus" NUMEXPR_NUM_THREADS="$cpus"
export MPLBACKEND=Agg

# --- locate the shared venv python (no activation needed) --------------------
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY missing — run hpc/detsvsL/setup_env.sh on the server first" >&2; exit 1; }
cd "$REPO" || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }

# =============================================================================
#  RUN PARAMETERS  (edit here to rescale)
#  First HPC TEST: L=2,3 dilute A=1, N_f=4, cores up to 50k, n_runs=4 (matches
#  the validated laptop config so this isolates the "deeper cores on HPC" variable
#  that Phase C is blocked on). The per-rung wall cap self-limits large L, so the
#  ladder climbs toward 50k and stops honestly where the budget runs out (the JSON
#  records exactly where). PRODUCTION science run bumps n_runs to >=16 (the Phase-D
#  seed-fragility fix) and adds L=4.
# =============================================================================
STAMP="$(date +%Y%m%d-%H%M%S)"
LABEL="detsvsL_hpc_${STAMP}"
DIM=3
L_VALUES="2 3"
A=1
N_B=2                 # N_f = 2^n_b = 4  (matches the laptop dilute reference runs)
EPS="1.0 0.1"         # per-site accuracy targets (MeV/site)
MAX_CORE=50000
N_RUNS=4
N_RUNGS=8
LADDER_START=500
MAX_RUNG_SECONDS=3600 # 1 h/rung wall cap -> bounds total wall time
# -----------------------------------------------------------------------------

LOG="data/classical/${LABEL}.log"
echo "[run] host=$(hostname) cpus=$cpus"
echo "[run] label=$LABEL"
echo "[run] dim=$DIM L=[$L_VALUES] A=$A N_f=$((1<<N_B)) max_core=$MAX_CORE n_runs=$N_RUNS rung_cap=${MAX_RUNG_SECONDS}s"

"$PY" -m misc.run_dets_vs_L --hpc \
    --dim "$DIM" --L $L_VALUES --A "$A" --n_b "$N_B" \
    --eps $EPS --max-core "$MAX_CORE" --n_runs "$N_RUNS" \
    --n-rungs "$N_RUNGS" --ladder-start "$LADDER_START" \
    --max-rung-seconds "$MAX_RUNG_SECONDS" --label "$LABEL" \
    > "$LOG" 2>&1
status=$?

cat "$LOG"    # mirror the driver's console output into the Condor stdout

# --- return compact artifacts inside the rundir (durable copies stay in data/) --
for ext in json png log; do
    f="data/classical/${LABEL}.${ext}"
    [ -f "$f" ] && cp "$f" "$SANDBOX/"
done

echo "[run] done (status=$status); artifacts: data/classical/${LABEL}.{json,png,log}"
exit "$status"
