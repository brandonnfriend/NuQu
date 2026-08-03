#!/bin/sh
# Generate + submit the PARALLEL (L, series) quantum resource-estimation campaign.
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q && git reset --hard origin/main
#     cd hpc/quantum
#     sh submit_quantum_sweep.sh test                 # <-- FIRST: 1 pyLIQTR smoke job
#     sh submit_quantum_sweep.sh "2 3 4 6 8" "sparse" # then the real campaign
#   defaults: L="2 3 4 6 8", series="sparse"  (one shard per (L, series))
#
# 1 shard = 1 (L, series) with an A-sweep. The quantum estimate is symbolic
# (pure Python/numpy, no C++ build, ~seconds/point after the JW-cache/analytic
# fixes), so shards are light. Combine afterward with misc/combine_quantum_shards.py.
set -eu
MODE_OR_L="${1:-2 3 4 6 8}"
SERIES_LIST="${2:-sparse}"
AVALS="${3:-1,2,4,8}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# --- smoke-test mode: a single pyLIQTR/Julia/gmpy2 shakedown job on a qis node ---
if [ "$MODE_OR_L" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = 2 sparse ${CAMPAIGN} 1 test
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = 4G
request_disk            = 6G
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (1 job)  logs=${DIR}/logs/smoketest.{out,err}"
  echo "check: grep '\[qshard:test\] OK' ${DIR}/logs/smoketest.out  (and no Julia/GMP errors)"
  exit 0
fi

# --- real campaign: one shard per (L, series) ---
LLIST="$MODE_OR_L"
: > "$DIR/shards.txt"
for L in $LLIST; do
  # Quantum estimate is symbolic; RAM is modest and grows with the operator size.
  case "$L" in
    2|3|4) MEM=4G ;;
    6)     MEM=8G ;;
    8)     MEM=16G ;;
    *)     MEM=32G ;;     # L>=10
  esac
  for S in $SERIES_LIST; do
    echo "$L $S $MEM" >> "$DIR/shards.txt"
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) \$(SERIES) ${CAMPAIGN} ${AVALS}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_\$(SERIES).out
Error                   = ${DIR}/logs/L\$(L)_\$(SERIES).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 6G
queue L,SERIES,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  A_values=${AVALS}  shard_dir=${DIR}/shards"
echo "combine when done: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
