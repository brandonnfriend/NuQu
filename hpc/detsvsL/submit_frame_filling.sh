#!/bin/sh
# TASK 2 — frame x filling comparison: at higher nucleon filling (A = filling*sites, so
# the fermion sector is dense and the interaction is strong), how do the FERMIONIC frame
# (COO orbital optimization, TrimCI-style) and its combination with the boson squeeze
# compare to squeeze-only and bare? This was blocked at laptop sizes; it's the interesting
# HPC regime. Grid: {bare, gaussian, coo, gaussian+coo} x L x seeds at a fixed filling.
#
# Run from $REPO/hpc/detsvsL/ on ssh hep-submit:
#   sh submit_frame_filling.sh [filling] [n_seeds] "[L list]"     (default 1.0, 8, "2 3")
set -eu
FILLING="${1:-1.0}"
NSEEDS="${2:-8}"
LLIST="${3:-2 3}"
# full frame set: bare, boson-squeeze, fermionic-COO, squeeze+COO, Lang-Firsov polaron
# (targets the fermion-boson coupling H_AV), and squeeze+LF. Pass $4 to run a subset,
# e.g. "lf gaussian+lf" to add the LF frames to an existing comparison.
FRAMES="${4:-bare gaussian coo gaussian+coo lf gaussian+lf}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# grid "frame L seed mem". A = filling*sites -> dense fermion sector, so RAM grows fast
# with L. max_core fixed at 256k: this is a frame COMPARISON at equal footing, not a
# convergence push.
: > "$DIR/shards.txt"
for FR in $FRAMES; do
  for L in $LLIST; do
    case "$L" in 2) MEM=64G ;; 3) MEM=384G ;; 4) MEM=640G ;; *) MEM=128G ;; esac
    s=0
    while [ "$s" -lt "$NSEEDS" ]; do echo "$FR $L $s $MEM" >> "$DIR/shards.txt"; s=$((s + 1)); done
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) \$(SEED) ${CAMPAIGN} \$(FR) 1 ${FILLING} 256000
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/\$(FR)_L\$(L)_s\$(SEED).out
Error                   = ${DIR}/logs/\$(FR)_L\$(L)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = 4
request_memory          = \$(MEM)
request_disk            = 12G
queue FR,L,SEED,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  filling=${FILLING}  shard_dir=${DIR}/shards"
echo "combine: python -m misc.combine_detsvsL --shard-dir ${DIR}/shards --by-frame --label frame_fill${FILLING}_${CAMPAIGN}"
