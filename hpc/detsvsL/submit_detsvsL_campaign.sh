#!/bin/sh
# Generate + submit the PARALLEL (L, seed) dets-vs-L campaign.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git pull
#     cd hpc/detsvsL && sh submit_detsvsL_campaign.sh [n_seeds] "[L list]"
#   defaults: n_seeds=16, L="2 3 4"  -> 48 shards
#
# The n_runs ensemble is parallelised across jobs: 1 shard = 1 (L, seed), n_runs=1,
# full ladder, in the compacting squeeze frame. Wall clock = the single slowest shard
# (~L=4 one seed), not the serial sum. Combine afterward with misc/combine_detsvsL.py.
set -eu
NSEEDS="${1:-16}"
LLIST="${2:-2 3 4 5}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# per-shard list: "L seed mem max_core". Memory + ladder depth both grow with L:
#  - L=2 is converged by ~64k -> no deepening (128k).
#  - L>=3 go to 256k to pin the tighter eps; memory grows ~sites x core (the CSC/eigsh),
#    so L=4/256k wants ~128G.
#  - L=5 stays at 128k: it is the load-bearing 4th point and L=5/256k on 125 sites is too
#    memory-heavy to risk (the shard saves only at the end, so an OOM loses ALL its rungs).
# A held shard (memory spike) is recoverable and the combine tolerates a few missing seeds:
#    condor_qedit <cluster> RequestMemory <MB> && condor_release <cluster>
: > "$DIR/shards.txt"
for L in $LLIST; do
  case "$L" in
    2) MEM=8G;   MAXCORE=128000 ;;
    3) MEM=48G;  MAXCORE=256000 ;;
    4) MEM=128G; MAXCORE=256000 ;;
    5) MEM=128G; MAXCORE=128000 ;;
    *) MEM=24G;  MAXCORE=128000 ;;
  esac
  s=0
  while [ "$s" -lt "$NSEEDS" ]; do
    echo "$L $s $MEM $MAXCORE" >> "$DIR/shards.txt"
    s=$((s + 1))
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_detsvsL_shard.sh
arguments               = \$(L) \$(SEED) ${CAMPAIGN} \$(MAXCORE)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_detsvsL_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = 2
request_memory          = \$(MEM)
request_disk            = 8G
queue L,SEED,MEM,MAXCORE from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "combine when done: python -m misc.combine_detsvsL --shard-dir ${DIR}/shards --label detsvsL_hpc_${CAMPAIGN}"
