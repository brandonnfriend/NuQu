#!/bin/sh
# Quantum campaign — D_WALK / REACTION-LIMITED sweep (task 30/34). Full L=1..10 at
# every L, sparse-family series, to harvest the walk-step Toffoli-depth band
# (Reaction_Depth: D_walk {serial,qroam,log} + adaptive N_walk*D_walk) that feeds the
# reaction-limited runtime figure. Each shard now persists `Sparse_Breakdown` (L_eff)
# and computes `Reaction_Depth` in-record (src_PI/estimation/hardware/walk_depth.py).
#
# Run from $REPO/hpc/quantum/ on the pinned submit node AFTER reconciling to the
# campaign branch and running the on-node smoke test once (see submit_overnight.sh
# header + `sh submit_quantum_sweep.sh test`). The smoke test (L=2 A=1 sparse) now
# also exercises the D_walk atom build on-node.
#
# Series (all give D_walk at every L; n_b differs -> distinct atom depths):
#   * sparse            tong cutoff (n_b~5)      -- realistic-from-vacuum headline.
#   * sparse +focc0.045 occupation-reduced n_b   -- the fully-optimized (Tier-2) curve.
#   * sparse_heuristic  heuristic cutoff (high n_b, grows with A) -- high-n_b comparison.
# Feasibility (from 290367/290370 + publication timings): sparse & sparse+occ are
# A-FLAT (JW cached across A -> a full A-sweep is ~one JW, ~15 min even at L=10) -> do
# all L. sparse_heuristic n_b grows with A (cache miss per A) -> full grid to L=8, a
# reduced grid at L=9,10 to stay under the wall. Incremental per-A save + a 6h
# periodic_remove keep partial sweeps if a large point runs long.
set -eu
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# Watson-spanning A-grid (nuclei-dense at low A, out to A=100). '+' not ',' — Condor
# `queue ... from` splits columns on commas (run script converts '+' back to ',').
AF="1+2+3+4+6+8+10+16+24+32+48+64+100"
AF_SMALL="1+4+16+64+100"

mem_for_L() {
  case "$1" in
    1|2|3|4) echo 8G ;;
    5|6)     echo 12G ;;
    7|8|9)   echo 16G ;;
    *)       echo 24G ;;
  esac
}

# Build the plan: "L series avals frameocc".
PLANF="$DIR/plan.txt"; : > "$PLANF"
add() { echo "$1 $2 $3 $4" >> "$PLANF"; }
for L in 1 2 3 4 5 6 7 8 9 10; do add "$L" sparse "$AF" -; done
for L in 1 2 3 4 5 6 7 8 9 10; do add "$L" sparse "$AF" 0.045; done
for L in 1 2 3 4 5 6 7 8;      do add "$L" sparse_heuristic "$AF" -; done
add 9  sparse_heuristic "$AF_SMALL" -
add 10 sparse_heuristic "$AF_SMALL" -

: > "$DIR/shards.txt"
while read -r L SERIES AVALS FOCC; do
  [ -z "${L:-}" ] && continue
  MEM="$(mem_for_L "$L")"
  echo "$L $SERIES $AVALS $FOCC $MEM" >> "$DIR/shards.txt"
done < "$PLANF"
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
