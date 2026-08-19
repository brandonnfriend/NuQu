#!/bin/sh
# Vertex-fix regeneration — round 1 QUANTUM campaign (REMEDIATION_PLAN N2/N4).
# Regenerates the quantum resource anchor on the CORRECTED Hamiltonian and pushes
# the compiled PauliLCU walk as deep in L as the cluster allows.
#
# Run from $REPO/hpc/quantum/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix \
#         && git reset --hard origin/remediation/vertex-fix
#     cd hpc/quantum
#     sh submit_vertexfix_quantum.sh test     # <-- FIRST: 1 real fock_pauli L=2 anchor job
#     sh submit_vertexfix_quantum.sh          # then the full Q1-Q4 grid
#
# 1 shard = 1 (L, series, n_b[, frame_occ]) point. The compiled PauliLCU anchor
# (series=fock_pauli) is A-INDEPENDENT at a fixed n_b (the block encoding encodes
# the OPERATOR, not the A-nucleon state), so each anchor shard is a single estimate.
# Combine afterward with misc/combine_quantum_shards.py.
#
# Job groups (see REMEDIATION_PLAN N4 + the session plan):
#   Q1  compiled PauliLCU deep-L anchor      fock_pauli, n_b=2, L=1..10
#   Q2  n_b convergence / resource-vs-cutoff fock_pauli, L in {2,3,4,5}, n_b in {1,3,4}
#   Q3  frame->QPE seam demo                 fock_pauli, --frame-occupation 0.045 -> n_b=3
#   Q4  amplitude baselines (A/B table)      watson (Lemma-5) + ns (Nyquist-Shannon)
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# --- smoke test: ONE real fock_pauli L=2 n_b=2 anchor estimate on a qis node. ---
# Validates the pyLIQTR/Julia/gmpy2 deps AND the actual compiled anchor code path
# (better than the generic sparse import check); also banks the L=2 anchor point.
if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = 2 fock_pauli ${CAMPAIGN} 1 run - - 2
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
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (1 job: fock_pauli L=2 n_b=2 anchor)"
  echo "check: grep '\[qshard\] done status=0' ${DIR}/logs/smoketest.out  (no Julia/GMP errors)"
  echo "then:  sh submit_vertexfix_quantum.sh   # full Q1-Q4 grid"
  exit 0
fi

# --- full Q1-Q4 grid: columns = L SERIES AVALS NB FRAMEOCC MEM ---------------- #
# AVALS uses '+' (comma-free) — Condor's `queue ... from file` splits on commas.
# NB / FRAMEOCC = '-' means "unset" (the run script drops the flag). MEM per shard.
SH="$DIR/shards.txt"
: > "$SH"
row() { printf '%s %s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$SH"; }

# Q1 — compiled PauliLCU deep-L anchor (fock_pauli, n_b=2). One shard per L.
# Memory grows ~ sites (~13 GB projected at L=10); generous on the deep points for
# a clean first pass — trim next round from the real MemoryUsage peaks.
row 1  fock_pauli 1 2 - 4G
row 2  fock_pauli 1 2 - 4G
row 3  fock_pauli 1 2 - 6G
row 4  fock_pauli 1 2 - 8G
row 5  fock_pauli 1 2 - 12G
row 6  fock_pauli 1 2 - 20G
row 7  fock_pauli 1 2 - 32G
row 8  fock_pauli 1 2 - 48G
row 9  fock_pauli 1 2 - 96G
row 10 fock_pauli 1 2 - 128G

# Q2 — n_b convergence / resource-vs-cutoff curve (fock_pauli). n_b=2 already in Q1.
row 2  fock_pauli 1 1 - 4G
row 2  fock_pauli 1 3 - 6G
row 2  fock_pauli 1 4 - 8G
row 3  fock_pauli 1 1 - 6G
row 3  fock_pauli 1 3 - 10G
row 3  fock_pauli 1 4 - 16G
# frame-justified n_b=3 pushed a bit deeper (the physical-cutoff anchor line).
row 4  fock_pauli 1 3 - 12G
row 5  fock_pauli 1 3 - 20G

# Q3 — frame->QPE seam demo (fock_pauli, measured <n> -> n_b via the 5-sigma rule).
# 0.045 is the pre-fix near-vacuum <n> placeholder; repin from corrected classical.
# NB left '-' so the frame seam sets n_b (expect n_b=3 -> consistency-checks Q2).
row 2  fock_pauli 1 - 0.045 6G
row 3  fock_pauli 1 - 0.045 10G

# Q4 — amplitude baselines (regenerate retired Watson/NS numbers; A/B comparison).
# watson (Lemma-5, n_b~19-25, Lambda~1e10) IS A-dependent (n_b grows with E_bound*A).
row 2  watson 1+2+4+8 - - 8G
row 3  watson 1+2+4+8 - - 24G
row 4  watson 1+2+4   - - 48G
# ns (Nyquist-Shannon, tong register; tong is A-flat but keep a few A for the record).
row 2  ns 1+2+4 - - 4G
row 3  ns 1+2+4 - - 6G
row 4  ns 1+2+4 - - 10G
row 5  ns 1+2+4 - - 20G
row 6  ns 1+2+4 - - 40G

NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) \$(SERIES) ${CAMPAIGN} \$(AVALS) run \$(FRAMEOCC) - \$(NB)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_\$(SERIES)_nb\$(NB)_f\$(FRAMEOCC).out
Error                   = ${DIR}/logs/L\$(L)_\$(SERIES)_nb\$(NB)_f\$(FRAMEOCC).err
Log                     = ${DIR}/logs/campaign.log
${QIS}
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 15G
queue L,SERIES,AVALS,NB,FRAMEOCC,MEM from ${SH}
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "groups: Q1 deep-L anchor(10) + Q2 n_b sweep(8) + Q3 frame(2) + Q4 amplitude(8)"
echo "combine when done: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
