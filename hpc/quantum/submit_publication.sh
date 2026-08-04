#!/bin/sh
# Quantum campaign — PUBLICATION A-sweeps (task 34). Full sweeps over nucleon
# number A at every tractable (L, series), for the resource-vs-A figures that
# overlay the Watson curves (esp. at L=10). Run from $REPO/hpc/quantum/ on the
# pinned submit node after reconciling to the campaign branch (see
# submit_overnight.sh header).
#
# Feasibility (from the 290367/290370 timings + a local A-sweep cache test):
#  - sparse (tong) & sparse+occ are A-FLAT: the fermion-JW is cached across A, so
#    A>1 is ~1 s -> a full A-sweep even at L=10 is ~one JW (~15 min). Do all L.
#  - sparse_heuristic n_b grows with A (cache miss per A) -> cheap to L=8; L=10
#    gets a reduced A-grid to stay under the wall.
#  - ns (amp/PauliLCU) is A-flat but pricier per point -> A-sweep only to L=4.
#  - watson (Lemma-5, the Tier-0 A-GROWING baseline, the direct Watson overlay):
#    full A-sweep to L=4; L=5 a reduced grid. L>=6 full sweeps are infeasible
#    (watson L=6 single point was 73 min); its L=10 baseline comes from the
#    analytic Lemma-5 formula in the task-11 comparison, not from a run here.
# Incremental per-A save + a 6h periodic_remove cap keep partial sweeps if a
# high-A/large-L point runs long.
set -eu
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# Watson-spanning A-grid (nuclei-dense at low A, out to A=100). '+' not ',' —
# Condor `queue ... from` splits columns on commas (run script converts back).
AF="1+2+3+4+6+8+10+16+24+32+48+64+100"

mem_for_L() {
  case "$1" in
    2|3|4) echo 8G ;;
    5|6)   echo 12G ;;
    7|8|9) echo 16G ;;
    *)     echo 24G ;;
  esac
}

# Build the plan: "L series avals frameocc".
PLANF="$DIR/plan.txt"; : > "$PLANF"
add() { echo "$1 $2 $3 $4" >> "$PLANF"; }
for L in 2 3 4 6 8 10; do add "$L" sparse "$AF" -; done
for L in 2 3 4 6 8 10; do add "$L" sparse "$AF" 0.045; done
for L in 2 3 4 6 8;    do add "$L" sparse_heuristic "$AF" -; done
add 10 sparse_heuristic "1+4+16+64+100" -
for L in 2 3 4;        do add "$L" ns "$AF" -; done
for L in 2 3 4;        do add "$L" watson "$AF" -; done
add 5 watson "1+16+100" -

: > "$DIR/shards.txt"
while read -r L SERIES AVALS FOCC; do
  [ -z "${L:-}" ] && continue
  MEM="$(mem_for_L "$L")"
  [ "$SERIES" = "watson" ] && MEM=24G   # amplitude PauliLCU at high n_b wants RAM
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
