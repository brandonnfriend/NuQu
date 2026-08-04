#!/bin/sh
# Flexible submitter for the "special" classical studies (A-grid crossover-point,
# LF isospectrality, N_f cutoff, unbiased-init). All fit-once (independent) shards;
# the qis self-provisioning + fork-ensemble path is unchanged (see HPC_WORKFLOW.md).
#
#   sh submit_special.sh LABEL KIND "VALUES" "FRAMES" NSEEDS MAXCORE L MEM [NB] [BIM] [LNRUNS]
#
# KIND   = fill  -> VALUES are fillings (A = round(fill*sites))
#          A     -> VALUES are explicit nucleon counts A (filling reported as none)
# NB     = boson bits/mode -> N_f = 2^NB (default 2 => N_f=4; the cutoff study)
# BIM    = boson init mean: 'none' = UNIFORM/unbiased (no vacuum anchor); else keep solver 0.5
# LNRUNS = ensemble runs PER LADDER RUNG (default 1; use >1 with an unbiased init to converge)
set -eu
LABEL="$1"; KIND="$2"; VALUES="$3"; FRAMES="$4"; NSEEDS="$5"; MAXCORE="$6"; L="$7"; MEM="$8"
NB="${9:-2}"; BIM="${10:-}"; LNRUNS="${11:-1}"
# LABEL + pid go INTO the campaign id so two studies submitted in the same second get
# DISTINCT dirs (the earlier nf_nb1/nf_nb2 collision was a timestamp-only dir clobbering
# shard files). The id is passed verbatim to run_frame_shard.sh, which derives the
# execute-side OUTDIR from it, so submit-side DIR and execute-side OUTDIR always match.
CAMPAIGN="${LABEL}-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"

: > "$DIR/shards.txt"
for FR in $FRAMES; do
  for V in $VALUES; do
    s=0
    while [ "$s" -lt "$NSEEDS" ]; do
      if [ "$KIND" = "A" ]; then echo "$FR $V none $s" >> "$DIR/shards.txt"
      else echo "$FR 1 $V $s" >> "$DIR/shards.txt"; fi   # A=1 placeholder; filling=$V drives A
      s=$((s + 1))
    done
  done
done
NJOBS=$(wc -l < "$DIR/shards.txt")

ENV="NUQU_N_B=${NB} NUQU_LADDER_NRUNS=${LNRUNS}"
[ -n "$BIM" ] && ENV="${ENV} NUQU_BOSON_INIT_MEAN=${BIM}"

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = ${L} \$(SEED) ${CAMPAIGN} \$(FR) \$(A) \$(FILL) ${MAXCORE} independent 4 3 1000 3600
environment             = "${ENV}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
# 8 cpus: the LF frame-fit scan is now a FLAT fork over 17 scales x n_runs seeds (~51
# tasks; frame.optimize_displacement), so it fans across all requested cpus instead of
# being capped at the ensemble's n_runs. The deep single-run ladder rungs don't use the
# extra cpus (peak RAM is unchanged -> MEM stays L-sized), so 8 is a schedulable balance.
request_cpus            = 8
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/\$(FR)_A\$(A)_f\$(FILL)_s\$(SEED).out
Error                   = ${DIR}/logs/\$(FR)_A\$(A)_f\$(FILL)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue FR,A,FILL,SEED from ${DIR}/shards.txt
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  label=${LABEL}  jobs=${NJOBS}  L=${L}  max_core=${MAXCORE}  env=[${ENV}]"
echo "combine: python -m misc.combine_detsvsL --shard-dir ${DIR}/shards --by-frame --label ${CAMPAIGN}"
