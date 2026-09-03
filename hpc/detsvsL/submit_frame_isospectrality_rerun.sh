#!/bin/sh
# RE-RUN of the 26 OOM'd shards from the first frame-isospectrality campaign (cluster 291037).
# Root cause: squeeze-containing frames were routed into back-evaluation, whose Taylor map-back
# fans the squeeze over EVERY boson mode up to N_f_ref (24 modes at L2d3, 81 at L3d3) -> combinatorial
# OOM; plus a few L3 solves near the 48G limit. FIX (committed): squeeze/COO are operator-identity
# (E_frame already variational -> no map-back); LF frames back-evaluate ONLY the LF displacement and
# score against the SQUEEZED operator-identity reference (no all-modes fan-out). This re-run adds
# headroom memory (L2 32G, L3 96G) and a support-cap on the dense-filling LF frames.
#
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit` AFTER the fix is pushed + reconciled:
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/detsvsL
#   sh submit_frame_isospectrality_rerun.sh test   # 1 shard: the previously-exploding gaussian+lf L2d3
#   sh submit_frame_isospectrality_rerun.sh        # all 26
set -eu
MODE="${1:-run}"
CAMPAIGN="frameiso-rerun-$(date +%Y%m%d-%H%M%S)-$$"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# the 26 not-done shards from cluster 291037:  "FRAME L DIM AF NB"
SET="
gaussian 2 3 A1 2
gaussian 2 3 A1 3
gaussian 2 3 A1 4
gaussian 2 3 f0.5 2
gaussian 2 3 f1.0 2
gaussian+lf 2 3 A1 2
gaussian+lf 2 3 A1 3
gaussian+lf 2 3 A1 4
gaussian+lf 2 3 f0.5 2
gaussian+lf 2 3 f1.0 2
gaussian 3 3 A1 2
gaussian 3 3 A1 3
gaussian 3 3 A1 4
gaussian 3 3 f0.5 2
gaussian 3 3 f1.0 2
gaussian+coo 3 3 A1 2
gaussian+coo 3 3 A1 3
gaussian+coo 3 3 A1 4
gaussian+coo 3 3 f1.0 2
gaussian+lf 3 3 A1 2
gaussian+lf 3 3 A1 3
gaussian+lf 3 3 A1 4
gaussian+lf 3 3 f0.5 2
gaussian+lf 3 3 f1.0 2
lf 3 3 f0.5 2
lf 3 3 f1.0 2
"

# columns: L DIM FRAME A FILLING CORES NB BKNF CAP MEM
SH="$DIR/shards.txt"; : > "$SH"
printf '%s\n' "$SET" | while read -r fr L dim af nb; do
  [ -z "${fr:-}" ] && continue
  if [ "$af" = "A1" ]; then fill=none; else fill=$(printf '%s' "$af" | sed 's/^f//'); fi
  if [ "$L" = "2" ]; then cores="250+1000+4000+16000"; mem=32G; else cores="250+1000+4000"; mem=96G; fi
  cap="-"                                     # support-cap only for DENSE-filling LF map-backs
  case "$fr" in *lf*) [ "$af" != "A1" ] && cap=200000 ;; esac
  printf '%s %s %s 1 %s %s %s 3 %s %s\n' "$L" "$dim" "$fr" "$fill" "$cores" "$nb" "$cap" "$mem" >> "$SH"
done

if [ "$MODE" = "test" ]; then
  # the previously-exploding case, now fixed: gaussian+lf L2d3 nb2 A1, 2 cores
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = 2 gaussian+lf ${CAMPAIGN} none 250+1000 2 3 1 16 0
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=3 NUQU_EXACT_REF=-"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 32G
request_disk            = 8G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  RERUN SMOKE (gaussian+lf L2d3 nb2 — was the OOM case)"
  echo "check: grep '\[bench\]' ${DIR}/logs/smoketest.out  (want bounded support, conv=True)"
  exit 0
fi

NJOBS=$(wc -l < "$SH")
cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) \$(NB) \$(DIM) \$(A) 16 0
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=\$(BKNF) NUQU_SUPPORT_CAP=\$(CAP) NUQU_EXACT_REF=-"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_A\$(A)f\$(FILLING)_nb\$(NB).out
Error                   = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_A\$(A)f\$(FILLING)_nb\$(NB).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,FRAME,A,FILLING,CORES,NB,BKNF,CAP,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "re-runs: gaussian(10, now operator-identity) + gaussian+lf(10, LF-only map) + gaussian+coo(4, more mem) + lf-filling(2, capped)"
