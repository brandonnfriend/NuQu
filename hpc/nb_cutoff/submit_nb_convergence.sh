#!/bin/sh
# Publication-grade n_b (Fock-cutoff) convergence study on the CORRECTED Hamiltonian
# (codex 03_cutoff: report convergence EMPIRICALLY). Fans out one study per shard:
#
#   A       ED-exact anchor      L=2 d=1 A=1  (exact Lanczos to N_f=16 — validates TrimCI+PT2)
#   G       ED-exact occ vs A    L=2 d=1      (exact <N>(A))
#   B       dilute headline      L=2 d=3 A=1
#   Bdense  DENSE (nuclear)      L=2 d=3 A=8  (filling 1.0 — max pion source)
#   Cdilute dilute larger volume L=3 d=3 A=1
#   Cdense  DENSE larger volume  L=3 d=3 A=27 (filling 1.0 — heaviest)
#
# Each shard emits energy-convergence (ΔE vs n_b -> the n_b for 1 MeV), the |c|²-weighted
# boundary population (leaked weight vs cutoff), and observable (<N>/mode) convergence.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_convergence.sh test   # 1 tiny study-A shard (ED anchor; validates the env path)
#   sh submit_nb_convergence.sh        # full 6-study grid
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = A ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 16G
request_disk            = 10G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (study A: ED anchor L2d1; expect n_b=2 converges <1 MeV, exact-matched)"
  exit 0
fi

# --- grid: STUDY MEM ---
SH="$DIR/shards.txt"; : > "$SH"
printf 'A 16G\nG 16G\nB 64G\nBdense 160G\nCdilute 160G\nCdense 224G\n' > "$SH"
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = \$(STUDY) ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 32
request_memory          = \$(MEM)
request_disk            = 10G
JobPrio                 = 20
Output                  = ${DIR}/logs/\$(STUDY).out
Error                   = ${DIR}/logs/\$(STUDY).err
Log                     = ${DIR}/logs/campaign.log
queue STUDY,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "pull: rsync hep:${REPO:-/nfs_scratch/bfriend3/NuQu/NuQu}/hpc/nb_cutoff/${DIR}/shards -> data/classical/nb_convergence/"
echo "then: python -m misc.run_nb_convergence plot  (combined convergence figures)"
