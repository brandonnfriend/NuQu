#!/bin/sh
# CHEAP, schedulable n_b (Fock-cutoff) revalidation — post-vertex-fix.
#
# Purpose: EARN the boson-cutoff claim on the CORRECTED Hamiltonian — show n_b=2 (N_f=4) or n_b=3
# (N_f=8) is enough AND rule out n_b=1 (N_f=2) — via E_var(N_f) convergence at a FIXED core + the
# |c|²-weighted occupation tail. The old submit_nb_convergence.sh dense shards OOM'd at 160-224G
# (N_f=16 boson dim) and never scheduled; this caps N_f=(2,4,8) and turns PT2 off on the heavy
# shards, so every shard is ≤64G and schedules on the qis nodes.
#
#   A              ED-exact anchor   L=2 d=1 A=1     (exact Lanczos — validates TrimCI energy conv.)
#   Bdilute_cheap  dilute curve      L=2 d=3 A=1     N_f=(2,4,8,16), PT2 on   (clean single-nucleon)
#   Bdense_cheap   PHYSICAL regime   L=2 d=3 A=8     N_f=(2,4,8),   PT2 off   (filling 1.0 — the one)
#   Cdilute_cheap  larger volume     L=3 d=3 A=1     N_f=(2,4,8),   PT2 off   (n_b=2 holds at L=3)
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_cheap.sh test   # 1 Bdense_cheap shard — de-risks the DENSE memory (the OOM risk)
#   sh submit_nb_cheap.sh        # A + Bdilute_cheap + Bdense_cheap + Cdilute_cheap
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nbCheap"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = Bdense_cheap ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 48G
request_disk            = 10G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (Bdense_cheap L2d3A8): expect it FITS in 48G (PT2 off) and shows"
  echo "  E_var drops N_f=2→4 then plateaus N_f=4→8 (n_b=1 ruled out, n_b=2 enough). Watch mem + ΔE."
  exit 0
fi

# grid cols: STUDY MEM  (all ≤64G → schedulable)
SH="$DIR/shards.txt"; : > "$SH"
printf 'A 16G\nBdilute_cheap 48G\nBdense_cheap 48G\nCdilute_cheap 64G\n' > "$SH"
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = \$(STUDY) ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
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
echo "pull: rsync hep-submit:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/nb_cutoff/${DIR}/shards -> data/classical/nb_convergence/"
echo "then: python -m misc.run_nb_convergence plot"
