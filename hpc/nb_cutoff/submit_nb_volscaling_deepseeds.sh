#!/bin/sh
# VOLUME-SCALING deep-core SEED FILL (re-audit 2026-09-04): the 512k deep probe had only ONE seed at
# L=3 and L=4, so a paired same-seed Δ34 could not be reported with a seed range there. The audit asks
# for ≥3 comparable paired seeds at each reported deep point. This runs seeds 1 and 2 (seed 0 already
# exists) for L∈{3,4} × n_b∈{3,4} at A=1, to a 512k core — which ALSO yields the 256k rung, so it
# lands 3 paired seeds at BOTH 256k and 512k for L=3 and L=4 (n_b=4 s2 previously stalled at 128k).
#
# With 3 seeds the analysis (misc/make_nb_volscaling, now paired same-seed + quality-checked, median +
# seed range) can report the DEEPEST cores properly instead of falling back to L=4's shallow 128k (its
# only current 3-seed depth, which sits on a +0.18 MeV core-incompleteness excursion). The solve is
# seed-insensitive at these sizes, so this firms the seed statistics; it is not expected to move the
# central values, only to make the seed range legitimate and push the reported core to 512k.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/nb_cutoff
#     sh submit_nb_volscaling_deepseeds.sh test   # 1 cell (L=3 n_b=3 A=1 s1 @512k) — fast sanity
#     sh submit_nb_volscaling_deepseeds.sh        # 8 shards: L={3,4} x n_b={3,4} x seeds{1,2} @512k
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbVolSeeds"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="../detsvsL/run_frame_shard.sh"
ENV10="NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_RUNGS=10 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 3 1 ${BASE}-deep-nb3 bare 1 none 524288 independent 4 3 1000 21600
environment             = "${ENV10} NUQU_N_B=3"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 96G
request_disk            = 12G
Output                  = ${DIR}/logs/test.out
Error                   = ${DIR}/logs/test.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (L=3 n_b=3 A=1 s1 @512k)."
  exit 0
fi

# grid: L NB A SEED MEM  (seeds 1,2; L=3 light, L=4 heavy; n_b=4/N_f=16 heaviest)
SH="$DIR/grid.txt"; : > "$SH"
for S in 1 2; do
  printf '3 3 1 %s 96G\n'  "$S" >> "$SH"
  printf '3 4 1 %s 96G\n'  "$S" >> "$SH"
  printf '4 3 1 %s 192G\n' "$S" >> "$SH"
  printf '4 4 1 %s 256G\n' "$S" >> "$SH"
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = \$(L) \$(SEED) ${BASE}-deep-nb\$(NB) bare \$(A) none 524288 independent 4 3 1000 21600
environment             = "${ENV10} NUQU_N_B=\$(NB)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 12G
JobPrio                 = 17
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,A,SEED,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=${NJOBS}  (deep-seed fill: L={3,4} x n_b={3,4} x seeds{1,2} @512k)"
echo "pull -> data/classical/nb_volscaling_deep/nb{3,4}/ ; then python -m misc.make_nb_volscaling"
