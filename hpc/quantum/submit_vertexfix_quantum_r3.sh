#!/bin/sh
# Quantum round-3 — the PRUNING-FIXED regeneration (valid L=10 + valid high-n_b).
# Round-2 (290818) L=10 and the high-n_b cutoff points were invalid: the Λ-coupled prune floor
# discarded real small-coefficient physics at large Λ. Fixed (commit c037265): machine-noise
# prune floor (1e-12·max|c|) → the Fock estimate is of the EXACT target Hamiltonian at every
# L/n_b (clean L≤9 numbers unchanged byte-for-byte; L=10 + high-n_b now valid). Also records
# Pauli_Term_Count + Rotation_Count and validates the walk-T fit on 3 (incl. interior) points.
#
# Same grid as r2 (anchor n_b=2 L=1..10 + cutoff sweep L=2,3 n_b={1,3,4} + determinism rep), all
# --optimize-budget. Deep-L/high-n_b memory is bumped: keeping the full term list + the 3-sample
# fit raises the transient peak (round-2 held L7-9 on OOM).
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/quantum
#     sh submit_vertexfix_quantum_r3.sh test   # 1 fock_pauli L=2 n_b=4 (the smallest FIXED point)
#     sh submit_vertexfix_quantum_r3.sh        # full grid
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = 2 fock_pauli ${CAMPAIGN} 1 run - - 4 opt -
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 1
request_memory          = 32G
request_disk            = 15G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (fock_pauli L=2 n_b=4 --optimize-budget; expect pruned=0, within=True)"
  exit 0
fi

# --- grid: columns = L NB REP MEM ---
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s %s\n' "$1" "$2" "$3" "$4" >> "$SH"; }
# Anchor n_b=2, L=1..10 (deep-L memory bumped for the full term list + 3-sample fit spike).
row 1  2 - 8G
row 2  2 - 8G
row 3  2 - 8G
row 4  2 - 8G
row 5  2 - 16G
row 6  2 - 24G
row 7  2 - 128G
row 8  2 - 256G
row 9  2 - 384G
row 10 2 - 448G
# Cutoff sweep (now valid — full term list kept): L=2,3 n_b={1,3,4}.
row 2  1 - 8G
row 2  3 - 16G
row 2  4 - 48G
row 3  1 - 12G
row 3  3 - 64G
row 3  4 - 128G
# Determinism repeat.
row 3  2 rep2 8G
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) fock_pauli ${CAMPAIGN} 1 run - - \$(NB) opt \$(REP)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 15G
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB)\$(REP).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB)\$(REP).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,REP,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "pruning-fixed: expect ALL points pruned=0/within-budget (incl. L=10 + high-n_b)"
echo "combine (done=true only): python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
