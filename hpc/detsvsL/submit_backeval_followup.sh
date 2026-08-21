#!/bin/sh
# LF back-evaluation FOLLOW-UP — the physical-energy frame comparison in the TRACTABLE regime.
# The first benchmark (290820) showed the map-back is tractable at DILUTE filling (support
# grow× ~1) but SUPERLINEAR at dense filling (gaussian+lf f=1.0 blew up at L=3). So this run
# does the clean comparison at dilute fillings (exact map-back), plus a DENSE set with the
# support-cap fallback (bounded fan-out; still variational; dropped_weight convergence-tested).
# Question answered: does any frame LOWER the physical original-H energy vs bare at matched core?
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_backeval_followup.sh test    # 1 tiny dilute L=2 gaussian+lf shard
#     sh submit_backeval_followup.sh         # full grid
set -eu
MODE="${1:-run}"
CAMPAIGN="bkeval2-$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
FRAMES="bare gaussian lf gaussian+lf"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = 2 gaussian+lf ${CAMPAIGN} 0.25 250+1000 1 3 1 8 0
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
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (dilute L=2 gaussian+lf f=0.25)"
  exit 0
fi

# --- grid: columns = L FRAME FILLING CORES CAP MEM  (dim=3, n_b=1, num_runs=16) ---
# CAP='-' = exact map-back (dilute, tractable). CAP=<int> = bounded fan-out (dense fallback).
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$SH"; }
for fr in $FRAMES; do
  # L=2 dim=3: dilute fillings (exact) — the clean physical-energy comparison.
  for fill in 0.1 0.25 0.5; do row 2 "$fr" "$fill" "250+1000+4000+16000" - 24G; done
  # L=2 dim=3: dense filling (1.0) with the support-cap fallback (tractable + cap-convergence).
  row 2 "$fr" 1.0 "250+1000+4000+16000" 50000 32G
  # L=3 dim=3: dilute (exact) — where dense blew up but dilute is tractable.
  for fill in 0.25 0.5; do row 3 "$fr" "$fill" "250+1000+4000" - 48G; done
done
# L=4 dim=3: the most dilute point for the volume trend (a few frames).
for fr in bare gaussian gaussian+lf; do row 4 "$fr" 0.1 "250+1000+4000" - 64G; done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) 1 3 1 16 0
environment             = "NUQU_NUM_WORKERS=8 NUQU_SUPPORT_CAP=\$(CAP)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 15
Output                  = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_cap\$(CAP).out
Error                   = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_cap\$(CAP).err
Log                     = ${DIR}/logs/campaign.log
queue L,FRAME,FILLING,CORES,CAP,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "dilute (exact) L=2,3,4 + dense(f=1.0) support-capped at L=2; frames {bare,gaussian,lf,gaussian+lf}"
echo "analyze: E_orig(frame) vs E_var(bare) at matched core; cap-convergence via dropped_weight"
