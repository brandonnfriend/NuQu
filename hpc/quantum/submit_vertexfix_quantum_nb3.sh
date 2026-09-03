#!/bin/sh
# n_b=3 ANCHOR RECOMPUTE — the energy gate showed n_b=2 is under-converged (E_0 off 4-7 MeV, binding
# energy ~91 MeV); n_b=3 is the converged cutoff. So recompute the compiled PauliLCU anchor at n_b=3.
#
# MEMORY WALL: n_b=3 (N_f=8) costs ~5x the n_b=2 memory at the same L (L=3: 12G->64G), and n_b=2
# already hit 448G at L=10 — so a full L=1..10 n_b=3 compile (~2TB at L=10) is INFEASIBLE. This runs
# the exact n_b=3 compile where it fits (L=1..7); L=8..10 are then filled by lambda-scaling the n_b=2
# curve by the EXACT, L-independent ratio lambda(n_b=3)/lambda(n_b=2)=3.80 (verified at L=2,3, and
# this run reconfirms it up to L=7). fock_pauli is A-independent at fixed n_b (one estimate per L).
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/quantum
#     sh submit_vertexfix_quantum_nb3.sh test   # 1 fock_pauli L=2 n_b=3
#     sh submit_vertexfix_quantum_nb3.sh        # L=1..7 at n_b=3
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nb3"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = 2 fock_pauli ${CAMPAIGN} 1 run - - 3 opt -
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 1
request_memory          = 16G
request_disk            = 15G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (fock_pauli L=2 n_b=3; expect pruned=0, within-budget)"
  exit 0
fi

# grid: L NB MEM  (n_b=3 for L=1..7; memory ~5x the n_b=2 anchor, L=7 best-effort)
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s\n' "$1" 3 "$2" >> "$SH"; }
row 1  8G
row 2  16G
row 3  64G
row 4  96G
row 5  192G
row 6  384G
row 7  768G
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) fock_pauli ${CAMPAIGN} 1 run - - \$(NB) opt -
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 15G
JobPrio                 = 15
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (n_b=3 anchor L=1..7; L=8..10 lambda-scaled x3.80)"
echo "combine: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"