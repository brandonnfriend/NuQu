#!/bin/sh
# 1M DEEP-SOLVE SMOKE. One expensive L=3 selected-CI solve grown to ~1M determinants
# in the NEW single-deep-solve mode: the C++ Hermitian row-gather SpMV threaded across
# ALL cores via OpenMP (NUQU_DEEP_SOLVE=1 -> OMP=cpus, no fork ensemble), a bounded
# eigsh maxiter (no runaway at large N), and the marshaling-free complex matvec.
# Validates that 1M is SAFE (no OOM, no eigsh stall) and that the matvec fills the node.
#
#   sh submit_deep_solve.sh [FRAME] [FILLING] [L] [MAXCORE] [CPUS] [MEM]
# default: gaussian frame, half-filling (A=54 at L=3, the most-connected / hardest), 1M.
set -eu
FRAME="${1:-gaussian}"; FILLING="${2:-0.5}"; L="${3:-3}"; MAXCORE="${4:-1024000}"
CPUS="${5:-48}"; MEM="${6:-192G}"
# MEMORY scales steeply with FILLING (connection count) and core. Measured: L=3
# half-filling (A=54) hit ~64G by 128k and OOM'd climbing to 256k. Guidance to reach 1M:
# filling<=0.5 (dilute, A<=14) -> ~192G is ample; filling 1.0 (A=27) -> ~256G; half-
# filling (A=54) -> ~512G. Default is dilute A=14 (the DMRG-comparison point) at 192G.
CAMPAIGN="deep-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = ${L} 0 ${CAMPAIGN} ${FRAME} 1 ${FILLING} ${MAXCORE} independent 4 3 1000 21600
environment             = "NUQU_DEEP_SOLVE=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=64000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 25
Output                  = ${DIR}/logs/deep_${FRAME}_f${FILLING}_L${L}.out
Error                   = ${DIR}/logs/deep_${FRAME}_f${FILLING}_L${L}.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  frame=${FRAME}  filling=${FILLING}  L=${L}  max_core=${MAXCORE}  cpus=${CPUS}  mem=${MEM}"
echo "watch: per-rung E_var/mean_occ save incrementally; deepest rung ~= ${MAXCORE} dets"
