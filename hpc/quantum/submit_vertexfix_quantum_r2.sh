#!/bin/sh
# Vertex-fix QUANTUM round 2 — clean precision-aware Fock/PauliLCU rerun.
# Addresses the 2026-08-20 data audit: total-T-optimal error budget (issue 1), pruned
# one-norm recorded vs budget (issue 2), tracked-only git provenance, relabeled fields.
# Fock/PauliLCU ANCHOR ONLY — amplitude watson/ns, the frame-occupation seam, and sparse
# are deferred/omitted per the audit.
#
# Every shard runs `--optimize-budget` (the publication accuracy contract): split ΔE=1 MeV
# between QPE resolution and block-encoding synthesis, pick the allocation that MINIMIZES
# total QPE T, and flag any point whose pruned coefficient one-norm exceeds its budget slice
# (the high-n_b points the audit flagged). NOTE: --optimize-budget does 2 pyLIQTR samples,
# so ~2× the estimate wall vs round 1 — memory is unchanged (operator materialized once).
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/quantum
#     sh submit_vertexfix_quantum_r2.sh test    # 1 fock_pauli L=2 n_b=2 --optimize-budget job
#     sh submit_vertexfix_quantum_r2.sh         # the full clean grid
#
# Combine ONLY done=true shards: python -m misc.combine_quantum_shards --shard-dir <dir>.
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = 2 fock_pauli ${CAMPAIGN} 1 run - - 2 opt -
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
${QIS}
request_cpus            = 1
request_memory          = 8G
request_disk            = 15G
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (fock_pauli L=2 n_b=2 --optimize-budget)"
  echo "check: grep 'f\*=' ${DIR}/logs/smoketest.out  (budget optimizer ran, done status=0)"
  exit 0
fi

# --- clean grid: columns = L NB REP MEM  (series=fock_pauli, A=1, --optimize-budget) ---
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s %s\n' "$1" "$2" "$3" "$4" >> "$SH"; }

# Anchor: fixed n_b=2, direct compiled L=1..7, + L=8,9,10 DIRECT retry on big-mem.
row 1  2 - 4G
row 2  2 - 4G
row 3  2 - 6G
row 4  2 - 8G
row 5  2 - 12G
row 6  2 - 20G
row 7  2 - 32G
row 8  2 - 128G
row 9  2 - 224G
row 10 2 - 320G

# Cost-vs-cutoff response curve (n_b=2 already in the anchor). High n_b will flag pruning.
row 2  1 - 4G
row 2  3 - 8G
row 2  4 - 12G
row 3  1 - 8G
row 3  3 - 16G
row 3  4 - 32G

# Determinism / reproducibility repeat of one representative point.
row 3  2 rep2 6G

NJOBS=$(wc -l < "$SH")
cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) fock_pauli ${CAMPAIGN} 1 run - - \$(NB) opt \$(REP)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB)\$(REP).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB)\$(REP).err
Log                     = ${DIR}/logs/campaign.log
${QIS}
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 15G
queue L,NB,REP,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "anchor n_b=2 L=1..10 (L8-10 big-mem retry) + cutoff L=2,3 n_b={1,3,4} + L=3 n_b=2 rep"
echo "combine (done=true only): python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
