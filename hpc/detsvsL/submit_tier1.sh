#!/bin/sh
# TIER-1 exact-anchor campaign. Small ED-feasible systems with --exact-ref (guarded
# Lanczos = the TRUE E_inf) + a TrimCI core ladder + per-rung support metrics. Gives the
# RIGOROUS cost-to-fixed-accuracy core*(dE) against the exact reference, and its SCALING
# with system size -- the defensible route to "L=3 would need 10^x dets / CPU-h". Where
# the sector is too big for ED the exact ref auto-skips (E_exact=null); that shard still
# contributes the TrimCI ladder + support (Tier-2). Small systems, so cheap + bounded.
#
#   sh submit_tier1.sh LABEL "L:dim:A ..." "FRAMES" [MAXCORE] [MEM] [CPUS] [NB]
# e.g. sh submit_tier1.sh t1 "2:1:1 2:1:2 3:1:1 3:1:2 4:1:1 1:3:1 1:3:2 2:3:1" "bare gaussian+lf" 64000 24G 8 2
set -eu
LABEL="$1"; POINTS="$2"; FRAMES="$3"; MAXCORE="${4:-64000}"; MEM="${5:-24G}"; CPUS="${6:-8}"; NB="${7:-2}"
CAMPAIGN="${LABEL}-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"

: > "$DIR/shards.txt"
for FR in $FRAMES; do
  for P in $POINTS; do
    L="${P%%:*}"; rest="${P#*:}"; DIM="${rest%%:*}"; A="${rest##*:}"
    echo "$FR $L $DIM $A" >> "$DIR/shards.txt"
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) 0 ${CAMPAIGN} \$(FR) \$(A) none ${MAXCORE} independent 4 3 1000 3600
environment             = "NUQU_N_B=${NB} NUQU_LADDER_NRUNS=1 NUQU_DIM=\$(DIM) NUQU_EXACT_REF=1 NUQU_EXACT_MAX_MEM_GB=20 NUQU_PT2_MAX_CORE=64000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/\$(FR)_L\$(L)d\$(DIM)_A\$(A).out
Error                   = ${DIR}/logs/\$(FR)_L\$(L)d\$(DIM)_A\$(A).err
Log                     = ${DIR}/logs/campaign.log
queue FR,L,DIM,A from ${DIR}/shards.txt
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  label=${LABEL}  jobs=${NJOBS}  points=[${POINTS}]  frames=[${FRAMES}]  N_f=2^${NB}  mem=${MEM}"
echo "combine: rsync the shards, then the Tier-1 cost analysis (core*(dE) vs E_exact + support)"
