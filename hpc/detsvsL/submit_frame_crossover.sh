#!/bin/sh
# FRAME CROSSOVER campaign — which frame wins vs nucleon filling, cheap & fast.
#
# Design (2026-07-29): the cost driver is FRAME OPTIMIZATION (LF/COO), not ladder depth.
# The full 3-phase grow spends 60+ min in Phase-1 co-evolution per shard -> too slow. For a
# COMPARISON we don't need co-evolution: fit each frame ONCE (independent mode: optimize_frame
# at a small core with a few ensemble runs), FREEZE it, then grow the frozen frame (cheap:
# a 16k ladder rung is ~75s). So: L=2, fit at core 1000 with 3 runs, grow to 64k, 24G RAM.
# Bare/gaussian shards are minutes; LF/COO ones ~10-15 min. Whole campaign ~1h on our qis share.
#
# Why this grid: 2026-07-27 data (at 256k) showed the best frame is FILLING-DEPENDENT --
# squeeze at eighth-filling, squeeze+LF wins by ~170 MeV at quarter-filling, COO ~a no-op --
# but half-filling was unresolved (LF/COO timed out). This maps the squeeze->LF crossover and
# resolves half-filling. Every rung (1k..64k) is saved so the ORDERING can be checked for
# depth-stability (the shallow-core read was misleading before).
#
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit`:
#   sh submit_frame_crossover.sh ["fillings"] [n_seeds] ["frames"] [max_core]
set -eu
FILLINGS="${1:-0.5 0.75 1.0 1.5 2.0}"     # A = filling*8; 1.0=quarter, 2.0=half-filling
NSEEDS="${2:-2}"
FRAMES="${3:-bare gaussian lf gaussian+lf gaussian+coo}"
MAXCORE="${4:-64000}"
L=2
MEM=24G
LADDER_MODE=independent
FRAME_RUNS=3          # ensemble runs inside the one-shot frame fit
ORBOPT_CYCLES=3       # COO/LF optimization cycles
PHASE0_CORE=1000      # core at which the frame is fit (small = cheap)
MAXRUNGSEC=1800       # 30 min/rung cap
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

: > "$DIR/shards.txt"
for FILL in $FILLINGS; do
  for FR in $FRAMES; do
    s=0
    while [ "$s" -lt "$NSEEDS" ]; do echo "$FR $FILL $s" >> "$DIR/shards.txt"; s=$((s + 1)); done
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = ${L} \$(SEED) ${CAMPAIGN} \$(FR) 1 \$(FILL) ${MAXCORE} ${LADDER_MODE} ${FRAME_RUNS} ${ORBOPT_CYCLES} ${PHASE0_CORE} ${MAXRUNGSEC}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/\$(FR)_f\$(FILL)_s\$(SEED).out
Error                   = ${DIR}/logs/\$(FR)_f\$(FILL)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = 4
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 30
queue FR,FILL,SEED from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  L=${L}  mode=${LADDER_MODE}  max_core=${MAXCORE}  fillings=[${FILLINGS}]"
echo "combine: python -m misc.combine_detsvsL --shard-dir ${DIR}/shards --by-frame --label crossover_${CAMPAIGN}"
