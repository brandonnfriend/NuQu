#!/bin/sh
# TASK 1 — deep dilute run: push to 1M states, L=2-5, A=1, gaussian frame, to CERTAINLY
# converge each L (kill the L-dependent extrapolation bias that muddied the 4-pt exponent).
# Incremental per-rung saving means big-L shards that can't reach 1M still keep their
# deepest converged rungs. qis nodes are ~1TB, so memory is generous.
#
# Run from $REPO/hpc/detsvsL/ on ssh hep-submit:  sh submit_deep_dilute.sh [n_seeds]
set -eu
NSEEDS="${1:-4}"   # grow mode does a 64-seed Phase-0 ensemble PER shard; few outer seeds
LLIST="${2:-2 3 4 5}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# per-L RAM (grows ~sites x core). All target 1024k; the 4h/rung cap + incremental save
# let big-L shards stop at the deepest rung they finish (that's plenty to converge).
: > "$DIR/shards.txt"
for L in $LLIST; do
  case "$L" in 2) MEM=64G ;; 3) MEM=192G ;; 4) MEM=384G ;; 5) MEM=640G ;; *) MEM=128G ;; esac
  s=0
  while [ "$s" -lt "$NSEEDS" ]; do echo "$L $s $MEM" >> "$DIR/shards.txt"; s=$((s + 1)); done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) \$(SEED) ${CAMPAIGN} gaussian 1 none 1024000
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = 4
request_memory          = \$(MEM)
request_disk            = 12G
queue L,SEED,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "combine: python -m misc.combine_detsvsL --shard-dir ${DIR}/shards --label detsvsL_deep_${CAMPAIGN}"
