#!/bin/sh
# The NuQu quantum resource-estimation OVERNIGHT campaign (task 34).
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit), AFTER a
# clean `sh submit_quantum_sweep.sh test`:
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout quantum-frame-qpe-campaign \
#       && git reset --hard origin/quantum-frame-qpe-campaign
#     cd hpc/quantum && sh submit_overnight.sh
#
# One shard = one (L, series, A-values [, frame_occupation]) with per-A incremental
# save. Shards run in parallel on qis1-3; wall = the slowest shard. Sizing is from
# the laptop calibration (2026-08-03): sparse cheap+A-flat (L=10 ~20-30min); ns
# ~L=6 (17min); watson the Tier-0 baseline, expensive (n_b 19-25, ~L=4-5). A only
# moves the resource estimate for the n_b-growing series (watson, sparse_heuristic);
# sparse/ns are A-flat under tong, so one representative A=4 per L.
#
# Deliverables:
#   * L-scaling per series (headline: how far each encoding scales).
#   * Tier 0 (watson) vs Tier 1 (sparse/ns) reduction (~1e5x in QPE-T, matched (L,A)).
#   * A-scaling of the n_b-growing series (watson, sparse_heuristic).
#   * Tier-2 occupation-reduced sparse (frame_occupation=0.045 = the verified bare
#     vacuum -> n_b=3): the aggressive realistic register.
set -eu
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# memory by L (quantum estimate is symbolic + lean; generous enough to avoid OOM
# churn on an unattended run, trivial on the ~1TB qis nodes).
mem_for_L() {
  case "$1" in
    2|3|4) echo 8G ;;
    5|6)   echo 12G ;;
    *)     echo 16G ;;   # L>=8
  esac
}

# The plan: "L series avals frameocc" (frameocc "-" = the series' own cutoff). One
# row per (L, series, frameocc) -> unique output file. For the n_b-growing series
# (watson, sparse_heuristic) the A-grid rows at low L already cover A=4, so there's
# no separate A=4-only row there (it would collide on the filename).
# NOTE: A-values use '+' (not ',') as the separator -- Condor's `queue ... from`
# splits columns on commas, so a comma-list in a column breaks the parse. The run
# script normalizes '+' back to ',' for --A-values.
PLAN="
2 sparse 4 -
3 sparse 4 -
4 sparse 4 -
6 sparse 4 -
8 sparse 4 -
10 sparse 4 -
2 ns 4 -
3 ns 4 -
4 ns 4 -
6 ns 4 -
2 watson 1+2+4+8+16+32+64 -
3 watson 1+2+4+8+16 -
4 watson 4 -
5 watson 4 -
2 sparse_heuristic 1+2+4+8+16+32+64+100 -
3 sparse_heuristic 1+2+4+8+16+32+64 -
4 sparse_heuristic 1+2+4+8+16+32 -
6 sparse_heuristic 4 -
8 sparse_heuristic 4 -
2 sparse 4 0.045
3 sparse 4 0.045
4 sparse 4 0.045
6 sparse 4 0.045
8 sparse 4 0.045
"

: > "$DIR/shards.txt"
echo "$PLAN" | while read -r L SERIES AVALS FOCC; do
  [ -z "${L:-}" ] && continue
  echo "$L $SERIES $AVALS $FOCC $(mem_for_L "$L")" >> "$DIR/shards.txt"
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
# Safety cap: remove any shard still running after 6h (incremental save keeps its
# finished points). Guards a watson/deep-L point that runs longer than calibrated.
periodic_remove         = (JobStatus == 2) && (time() - JobCurrentStartDate > 21600)
queue L,SERIES,AVALS,FOCC,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "combine when done: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
