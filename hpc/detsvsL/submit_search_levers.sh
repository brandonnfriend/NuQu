#!/bin/sh
# SEARCH-LEVER EVALUATION at L=2 (2026-09-24) -- which warm-grow search levers find the
# lowest nucleon-arrangement basin, and how reliably across independent seeds.
#
# WHY. The fixed-A L=2 pass (293959) left A=2,5,6 without an extrapolation: their ladders
# step down (a new nucleon arrangement) at 256k-1024k. Locally, A=5 grown from 8 single
# inits lands in discrete basins {274.7, ~285, 291, ~294, 305} MeV/site at 16k, and 2/8
# reach 274.7 -- BELOW 293959's 1M-det bound (276.1). The energy at 1k barely predicts it.
#
# ARMS (each lever is OFF by default in the code; OFF == the validated solve, bit-identical):
#   base      no lever (seed 0 reproduces 293959 s0 to 128k -- a free reproducibility check)
#   select    NUQU_PHASE0_SELECT_CORE=16000: grow all 32 Phase-0 inits to 16k, keep lowest
#   strat     NUQU_PHASE0_INIT=stratified: Phase-0 inits cycle evenly over arrangement types
#   stratsel  strat + select
#   novel     NUQU_NOVEL_FERM_FRAC=0.5 NUQU_NOVEL_KEEP_FRAC=0.05: reserve pool/core slots for
#             new nucleon configurations every round (more hopping)
#   all       stratsel + novel
# GRID: arms x A in {2,4,5,6,9} (4 = a converged control) x seeds {0,1,2} = 90 shards, ladder
# 1k..128k (8 rungs, PT2 every rung). Seeds use the disjoint Phase-0 stride (1000).
# RESOURCES: 4 cpus / 24G (293959 L=2 peaked 22 GB at 1M; 128k is far below). NOT deep-solve:
# OMP=1 and fork workers = cpus, so the Phase-0 ensemble / select stage run 4-wide and no
# forked child inherits an OpenMP pool. ~1-2 h/shard.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit), after reconciling:
#     sh submit_search_levers.sh test   # 1 shard: A=5 s0, all levers (select+strat+novel)
#     sh submit_search_levers.sh all    # the 90-shard grid
# Retrieve: rsync -az hep:.../campaign_<BASE>-<arm>/shards/ data/classical/<date>/levers_<cluster>/<arm>/
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-levers"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
AS="${AS:-2 4 5 6 9}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-base select strat stratsel novel all}"

# args to run_frame_shard.sh: L seed campaign frame A filling max_core ladder_mode
#                             frame_runs orbopt_cycles phase0_core max_rung_seconds
ARGS='2 $(SEED) '"${BASE}"'-$(ARM) bare $(A) none 128000 independent 4 3 1000 7200'
ENVSTR='NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=3 NUQU_N_RUNGS=8 NUQU_PT2_MAX_CORE=128000 NUQU_PHASE0_RUNS=32 NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=$(SELCORE) NUQU_PHASE0_INIT=$(P0INIT) NUQU_NOVEL_FERM_FRAC=$(NFF) NUQU_NOVEL_KEEP_FRAC=$(NKF)'

arm_env() {   # arm -> "SELCORE P0INIT NFF NKF"
  case "$1" in
    base)     echo "0     random     0   0" ;;
    select)   echo "16000 random     0   0" ;;
    strat)    echo "0     stratified 0   0" ;;
    stratsel) echo "16000 stratified 0   0" ;;
    novel)    echo "0     random     0.5 0.05" ;;
    all)      echo "16000 stratified 0.5 0.05" ;;
    *) echo "ERROR unknown arm $1" >&2; exit 1 ;;
  esac
}

submit_grid() {   # $1=name  $2=grid file
  cat > "campaign_${BASE}/$1.sub" <<SUB
Executable              = ./run_frame_shard.sh
arguments               = ${ARGS}
environment             = "${ENVSTR}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 4
request_memory          = 24G
request_disk            = 2560M
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/\$(ARM)_A\$(A)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/\$(ARM)_A\$(A)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue ARM,A,SEED,SELCORE,P0INIT,NFF,NKF,PRIO from $2
SUB
  condor_submit "campaign_${BASE}/$1.sub"
  echo "$1  CAMPAIGN=${BASE}  jobs=$(wc -l < "$2")"
}

mkdir -p "campaign_${BASE}/logs"
G="campaign_${BASE}/grid.txt"; : > "$G"
if [ "$MODE" = "test" ]; then
  echo "all 5 0 $(arm_env all) 50" > "$G"
  submit_grid smoke "$G"
  exit 0
fi
if [ "$MODE" = "all" ]; then
  for S in $SEEDS; do
    for ARM in $ARMS; do
      for A in $AS; do echo "$ARM $A $S $(arm_env "$ARM") $(( 10 * (2 - S) + 10 ))" >> "$G"; done
    done
  done
  submit_grid levers "$G"
  exit 0
fi
echo "usage: sh submit_search_levers.sh {test|all}" >&2
exit 2
