#!/bin/sh
# Targeted re-run of the 2 densest LF shards that OOM'd even at 96G / cap 200k:
# gaussian+lf and lf at L3 d3 f1.0 (27 nucleons -> the LF map-back of a dense state fans out
# hugest). Tighter support-cap (50k, ~4x less map-back memory) + incremental per-core save so
# even if the deep 4000-core rung OOMs we keep 250/1000. NOTE: a tighter cap DROPS more weight
# (dropped_weight logged) -> a looser (but still variational) E_orig; low scientific value since
# LF is already shown marginal/non-stacking at L2 + dilute + L3 f0.5. Cap-convergence-test via
# dropped_weight. All d3 => ED-free.
#
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit` after reconciling to the fix commit:
#   sh submit_frame_iso_denself_rerun.sh
set -eu
CAMPAIGN="frameiso-denself-$(date +%Y%m%d-%H%M%S)-$$"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# columns: L DIM FRAME A FILLING CORES NB BKNF CAP MEM
SH="$DIR/shards.txt"; : > "$SH"
printf '3 3 %s 1 1.0 250+1000+4000 2 3 50000 96G\n' "gaussian+lf" >> "$SH"
printf '3 3 %s 1 1.0 250+1000+4000 2 3 50000 96G\n' "lf" >> "$SH"
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
Output                  = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_f\$(FILLING)_nb\$(NB).out
Error                   = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_f\$(FILLING)_nb\$(NB).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,FRAME,A,FILLING,CORES,NB,BKNF,CAP,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (gaussian+lf & lf, L3 d3 f1.0, cap 50k)"
echo "watch dropped_weight — a large drop means the cap bit hard (E_orig still variational, just looser)"
