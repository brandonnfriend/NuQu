#!/bin/sh
# HEAVY PHASE-0 RESTART — is the L>=3 "extensivity trap" real, or a search artifact?
#
# The bare deep-solve ladders (290832) show a "basin collapse": E_var sits in a delocalized basin
# and only drops onto the compact ground-state basin at a LARGE core (L=2: 64k->128k). The compact
# state is ~3 dominant dets, so it FITS in a small core — the failure is the greedy warm-grow SEARCH
# (expansion only proposes dets connected to the current, wrong support), not capacity. See
# results/02_classical_baseline/basin_collapse_and_search_note.md.
#
# Fix per user: stay in the BARE basis (no frame optimization during growth — pure expand+trim),
# invest the search budget in a HEAVY phase-0 ensemble at a LARGER phase-0 core (so the compact
# basin is distinguishable there), then GROW the winner warm-started (NOT re-randomizing per rung).
# Levers: NUQU_PHASE0_RUNS (restart count) + NUQU_LADDER_START (phase-0 ensemble core).
#
# GOAL: improve the classical HEADLINE (bare TrimCI convergence + PT2 + extrapolation — all trusted).
# If phase-0 finds the compact basin at small core, then (a) L=2 converges at a much smaller core
# (proving search, not extensivity), and (b) PT2 is computed IN the compact basin (trustworthy),
# unlocking a real PT2/E_inf extrapolation with error bars for L=3 too.
#
# Multiple seeds per L = independent grow-trajectories: extra basin-finding chances AND independent
# deep solves (the input the extrapolator wants). E_var stays a rigorous variational upper bound.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/detsvsL
#   sh submit_heavy_restart.sh test   # 1 L=2 shard — measure phase-0 cost + does rung0 find 226/site?
#   sh submit_heavy_restart.sh        # L=2,3,4 x seeds
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-heavyRestart"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"

# bare, warm-grow, HEAVY phase-0 (128 restarts) at a LARGE phase-0 core (32k), deep-solve, n_b=2.
# PT2 cap is per-L (grid col). phase0_runs comes from the env; the positional runs (4) is frame-runs
# (irrelevant for bare). phase0_core positional (1000) is the frame-opt core (irrelevant for bare).
ENV_COMMON='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=8 NUQU_PHASE0_RUNS=128 NUQU_LADDER_START=32000'
# args: L seed campaign frame A filling max_core ladder_mode frame_runs cycles phase0_core maxrungsec

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = 2 0 ${BASE} bare 1 1.0 524288 independent 4 3 1000 14400
environment             = "${ENV_COMMON} NUQU_PT2_MAX_CORE=131072"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 48G
request_disk            = 8G
Output                  = ${DIR}/logs/test_L2.out
Error                   = ${DIR}/logs/test_L2.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (heavy phase-0 L=2): watch rung0 (core=32000) E_var/site ->"
  echo "  ~226 = compact basin FOUND at small core (search was the issue); ~255 = need bigger phase-0."
  exit 0
fi

# grid cols: L SEED MAXCORE MEM PT2CAP
SH="$DIR/grid.txt"; : > "$SH"
printf '%s\n' \
  '2 0 524288 48G 131072' '2 1 524288 48G 131072' '2 2 524288 48G 131072' \
  '3 0 524288 96G 131072' '3 1 524288 96G 131072' '3 2 524288 96G 131072' \
  '4 0 262144 128G 65536' '4 1 262144 128G 65536' > "$SH"

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) \$(SEED) ${BASE} bare 1 1.0 \$(MAXCORE) independent 4 3 1000 14400
environment             = "${ENV_COMMON} NUQU_PT2_MAX_CORE=\$(PT2CAP)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 17
Output                  = ${DIR}/logs/hr_L\$(L)_s\$(SEED).out
Error                   = ${DIR}/logs/hr_L\$(L)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,SEED,MAXCORE,MEM,PT2CAP from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=$(wc -l < "$SH")  (heavy phase-0 restart: L=2,3 x3 seeds, L=4 x2 seeds)"
echo "pull -> data/classical/<date>/heavy_restart/ ; compare rung0 + convergence vs bare_baseline_290832"
