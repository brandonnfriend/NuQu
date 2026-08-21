#!/bin/sh
# LF back-evaluation BENCHMARK campaign (codex audit 05_classical_frames, steps 4-5).
# Answers: is original-H back-evaluation tractable at production-shaped selected-CI cores,
# and does the frame lower the PHYSICAL (original-H) energy vs bare at matched core?
# One shard = one (L, frame, filling), solving geometric cores and back-evaluating each.
# Production geometry: dim=3, n_b=1 (N_f=2). Small + quick — even L=2 is a key data point.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_backeval.sh test     # 1 tiny gaussian+lf L=2 shard (C++ build + back-eval path)
#     sh submit_backeval.sh          # full grid: L={2,3} x frame x filling
#
# Combine: rsync campaign_<CID>/shards then compare E_orig(frame) vs E_var(bare) at matched
# core/wall, and read support_out / backeval_s / backeval_peak_mb scaling (the tractability Q).
set -eu
MODE="${1:-run}"
CAMPAIGN="bkeval-$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = 2 gaussian+lf ${CAMPAIGN} 1.0 250+1000 1 3 1 8 0
environment             = "NUQU_NUM_WORKERS=8"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 16G
request_disk            = 8G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (gaussian+lf L=2 dim=3, cores 250,1000)"
  echo "check: grep '\[bkshard\] done status=0' ${DIR}/logs/smoketest.out"
  exit 0
fi

# --- full grid: columns = L FRAME FILLING CORES MEM  (dim=3, n_b=1, num_runs=16) ---
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" >> "$SH"; }
FRAMES="bare gaussian lf gaussian+lf"
for fr in $FRAMES; do
  # L=2 dim=3 (8 sites): fillings 0.5 and 1.0, full geometric ladder.
  for fill in 0.5 1.0; do row 2 "$fr" "$fill" "250+1000+4000+16000" 24G; done
  # L=3 dim=3 (27 sites): filling 1.0, shorter ladder (heavier solve + map-back).
  row 3 "$fr" 1.0 "250+1000+4000" 48G
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) 1 3 1 16 0
environment             = "NUQU_NUM_WORKERS=8"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 15
Output                  = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING).out
Error                   = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING).err
Log                     = ${DIR}/logs/campaign.log
queue L,FRAME,FILLING,CORES,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "grid: L={2,3} dim=3 x {bare,gaussian,lf,gaussian+lf} x filling; geometric cores"
echo "analyze: E_orig(frame) vs E_var(bare) at matched core + support/wall/peak-mem scaling"
