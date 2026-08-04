#!/bin/sh
# Submitter for block2-DMRG isospectrality-reference shards (one (L,A) point per shard).
# block2 is ONE multithreaded process, so CPUS is real parallelism (not a fork ensemble).
#
#   sh submit_dmrg.sh LABEL "L:A L:A ..." BOND_DIMS NSWEEPS MEM CPUS [N_F]
#
# e.g. cheap L=2:  sh submit_dmrg.sh iso_L2 "2:4 2:8" 100,200,400,800,1200 8 48G 12
#      heavy L=3:  sh submit_dmrg.sh iso_L3 "3:14 3:27 3:54" 200,400,800 6 192G 24
#
# Run from $REPO/hpc/dmrg/ on `ssh hep-submit`. Combine/analyze locally with
# data/.../isospectrality_check.py (extrapolate E_inf, compare to gaussian+lf).
set -eu
LABEL="$1"; POINTS="$2"; BOND_DIMS="$3"; NSWEEPS="$4"; MEM="$5"; CPUS="$6"; N_F="${7:-4}"
CAMPAIGN="${LABEL}-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"

: > "$DIR/shards.txt"
for P in $POINTS; do
  L="${P%%:*}"; A="${P##*:}"
  echo "$L $A" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_dmrg_shard.sh
arguments               = \$(L) \$(A) ${CAMPAIGN} ${N_F} ${BOND_DIMS} ${NSWEEPS} ${CPUS}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_dmrg_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/dmrg_L\$(L)_A\$(A).out
Error                   = ${DIR}/logs/dmrg_L\$(L)_A\$(A).err
Log                     = ${DIR}/logs/campaign.log
queue L,A from ${DIR}/shards.txt
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  label=${LABEL}  jobs=${NJOBS}  points=[${POINTS}]  bond_dims=${BOND_DIMS}  cpus=${CPUS}  mem=${MEM}"
echo "pull: rsync -aq 'hep-submit:${DIR#campaign_}...' ; analyze with isospectrality_check.py"
