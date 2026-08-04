#!/bin/sh
# Quantum campaign ROUND 2 (task 34) — gap-fill + extend after 290367.
#
# Run from $REPO/hpc/quantum/ on the pinned submit node, after reconciling to the
# campaign branch (see submit_overnight.sh header). Fills the L-grid holes the
# first campaign left (L=5,7,9), pushes sparse to L=12, extends the Watson (Tier-0)
# baseline to L=6 to match sparse, and fills the ns curve (L=5,8). Same shard
# mechanics as submit_overnight.sh (A-values '+'-separated; run script converts).
set -eu
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

mem_for_L() {
  case "$1" in
    2|3|4) echo 8G ;;
    5|6)   echo 12G ;;
    7|8|9) echo 16G ;;
    *)     echo 24G ;;   # L>=10 (incl. the L=12 stretch)
  esac
}

# "L series avals frameocc"
PLAN="
5 sparse 4 -
7 sparse 4 -
9 sparse 4 -
12 sparse 4 -
5 sparse 4 0.045
7 sparse 4 0.045
9 sparse 4 0.045
10 sparse 4 0.045
6 watson 4 -
5 ns 4 -
8 ns 4 -
"

: > "$DIR/shards.txt"
echo "$PLAN" | while read -r L SERIES AVALS FOCC; do
  [ -z "${L:-}" ] && continue
  MEM="$(mem_for_L "$L")"
  # watson is the expensive amplitude/PauliLCU baseline (n_b ~26 at L=6) -> more RAM.
  [ "$SERIES" = "watson" ] && MEM=24G
  echo "$L $SERIES $AVALS $FOCC $MEM" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) \$(SERIES) ${CAMPAIGN} \$(AVALS) run \$(FOCC)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_\$(SERIES)_\$(FOCC).out
Error                   = ${DIR}/logs/L\$(L)_\$(SERIES)_\$(FOCC).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 10G
periodic_remove         = (JobStatus == 2) && (time() - JobCurrentStartDate > 21600)
queue L,SERIES,AVALS,FOCC,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "combine when done: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
