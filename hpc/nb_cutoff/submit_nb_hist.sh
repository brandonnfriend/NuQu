#!/bin/sh
# DEEP-reference occupation histogram — validate n_b=2 against a reference two levels deeper.
#
# The adequacy grid used N_f up to 8 (n_b=3) as its deepest solve, so "leak above N_f=4" was measured
# relative to n_b=3. To VALIDATE that n_b=2 is adequate (not undercounted by a too-shallow reference),
# solve to N_f=16 (n_b=4) with PT2 off and store the full per-level population p(n): we can then SEE
# the tail die off well before the reference cutoff and quantify the n_b=2 cut (keep n≤3) as <1%.
#
# Key configs (PT2 off -> cheap; N_f=16 is only ~2× the N_f=8 smoke = 977MB/57s):
#   hist_L2A8   L=2 A=8   (physical filling 1.0 — the anchor regime)
#   hist_L2A32  L=2 A=32  (MAX density — the worst-case tail, 0.87% at the n_b=3 reference)
#   hist_L3A27  L=3 A=27  (dense larger volume)
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_hist.sh test   # 1 hist_L2A8 shard — confirms N_f=16 PT2-off fits/is fast
#   sh submit_nb_hist.sh        # L2A8 + L2A32 + L3A27
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nbHist"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = hist_L2A8 ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 24G
request_disk            = 10G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE (hist_L2A8, N_f up to 16): watch mem + that p(n) dies off by n~4."
  exit 0
fi

# grid cols: STUDY MEM
SH="$DIR/shards.txt"; : > "$SH"
printf 'hist_L2A8 24G\nhist_L2A32 24G\nhist_L3A27 32G\n' > "$SH"
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
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (deep n_b=4 histograms: L2A8, L2A32, L3A27)"
echo "pull -> data/classical/nb_convergence/ ; then python -m misc.make_nb_histogram"
