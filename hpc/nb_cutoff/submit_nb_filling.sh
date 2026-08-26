#!/bin/sh
# HIGHER-FILLING boson-cutoff stress test — does n_b=2 survive maximum pion source?
#
# The main cheap grid (submit_nb_cheap.sh) did L=2 d=3 A=8 (filling 1.0). Higher A packs more
# nucleons onto the same 8 sites (max A=32 = fully filled, 4 spin-isospin states/site), maximizing
# the PION SOURCE — the harshest boson-cutoff test. The fermion side stays cheap (A=32 is a single
# determinant; the fixed-core cost is A-flat), so this is nearly free. If the n_b=2 occupation tail
# stays small up to A=32, the cutoff claim is robust at ANY density, not just filling 1.0.
#
#   Bdense_cheap_A16  L=2 d=3 A=16  (filling 2.0 — biggest fermion sector)
#   Bdense_cheap_A24  L=2 d=3 A=24  (filling 3.0)
#   Bdense_cheap_A32  L=2 d=3 A=32  (filling 4.0 — MAX source, trivial fermion sector)
# (A=8 / filling 1.0 already ran in submit_nb_cheap.sh.)
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_filling.sh test   # 1 A=32 shard (max source; de-risks the densest path)
#   sh submit_nb_filling.sh        # A=16, 24, 32
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nbFilling"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = Bdense_cheap_A32 ${CAMPAIGN}
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
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (Bdense_cheap_A32 — MAX pion source): watch the n_b=2 leaked weight."
  exit 0
fi

# grid cols: STUDY MEM  (A=16 is the biggest fermion sector -> a touch more headroom)
SH="$DIR/shards.txt"; : > "$SH"
printf 'Bdense_cheap_A16 24G\nBdense_cheap_A24 16G\nBdense_cheap_A32 12G\n' > "$SH"
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
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (filling sweep A=16,24,32 at L=2 d=3)"
echo "pull -> data/classical/nb_convergence/ ; the A=8..32 series shows if n_b=2 holds at max density"
