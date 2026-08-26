#!/bin/sh
# PUSH-FURTHER n_b cutoff: the two adds that are now cheap since PT2 (not A, not volume) was the
# old 160-224G cost.
#
#   Cdense_cheap   L=3 d=3 A=27 (filling 1.0) — the larger-volume DENSE confirmation (was the 224G
#                  monster; PT2-off makes it cost the same as the L=3 dilute shard — the H is
#                  A-independent). Completes the volume×density grid at L≤3.
#   Ddilute_cheap  L=4 d=3 A=1               — the volume-trend extension (64 sites). RISK shard:
#                  H-build ∝ n_terms ∝ L³, so small core + PT2 off. Smoke this first.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_push.sh test   # 1 Ddilute_cheap (L=4) shard — de-risks the L³ H-build
#   sh submit_nb_push.sh        # Cdense_cheap + Ddilute_cheap
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nbPush"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = Ddilute_cheap ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 48G
request_disk            = 10G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (Ddilute_cheap L=4): watch mem + wall — if it fits/finishes fast,"
  echo "  L=4 is in reach; if it OOMs or drags, we cap the volume trend at L=3 and drop it."
  exit 0
fi

# grid cols: STUDY MEM
SH="$DIR/shards.txt"; : > "$SH"
printf 'Cdense_cheap 16G\nDdilute_cheap 48G\n' > "$SH"
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = \$(STUDY) ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 10G
JobPrio                 = 20
Output                  = ${DIR}/logs/\$(STUDY).out
Error                   = ${DIR}/logs/\$(STUDY).err
Log                     = ${DIR}/logs/campaign.log
queue STUDY,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (L=3 dense A=27 + L=4 dilute A=1)"
echo "pull -> data/classical/nb_convergence/"
