#!/bin/sh
# CORRECTED convergence run (v2) — after the P0-1 max_rounds fix (commit adds _rounds_for so the
# solver actually grows to the requested core instead of stalling at ~3482 dets). Codex plan A:
# bare vs Gaussian only, n_b=3, the four (L,filling) cases, GENUINE deep core ladders, seeds
# {0,1,2} as independent inits, + one n_b=4 case (L2 f1.0) to check the advantage isn't special to
# a particular finite Fock subspace. Every rung records ACTUAL n_dets + reached_target/stop_reason
# + wall time (solve_s) so the curve is plotted vs actual n_dets, never requested core. bare/Gaussian
# are operator-identity => no back-eval (BKNF/CAP off). All d3 = ED-free.
#
# Overnight-friendly: incremental per-core save; cancel stragglers with condor_rm any time.
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit` after reconciling to the fix commit.
set -eu
CAMPAIGN="frameiso-conv2-$(date +%Y%m%d-%H%M%S)-$$"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
CORES="250+1000+4000+16000+64000"

# columns: L DIM FRAME A FILLING CORES NB MEM SEED
SH="$DIR/shards.txt"; : > "$SH"
for fr in bare gaussian; do
  for L in 2 3; do
    if [ "$L" = "2" ]; then mem=48G; else mem=96G; fi
    for af in A1 f1.0; do
      if [ "$af" = "A1" ]; then fill=none; else fill=1.0; fi
      for s in 0 1 2; do
        printf '%s 3 %s 1 %s %s 3 %s %s\n' "$L" "$fr" "$fill" "$CORES" "$mem" "$s" >> "$SH"
      done
    done
  done
  # n_b=4 representative (Codex A.5): L2 f1.0, seed 0
  printf '2 3 %s 1 1.0 %s 4 48G 0\n' "$fr" "$CORES" >> "$SH"
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) \$(NB) \$(DIM) \$(A) 8 \$(SEED)
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=- NUQU_SUPPORT_CAP=- NUQU_EXACT_REF=-"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 10
Output                  = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_nb\$(NB)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_nb\$(NB)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,FRAME,A,FILLING,CORES,NB,MEM,SEED from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (bare/gaussian x L{2,3} x {dilute,f1.0} x seed{0,1,2} nb3; +nb4 L2f1.0)"
echo "analyze: E vs ACTUAL n_dets (+ wall time); dets-to-cross-bare-best; check reached_target/stop_reason per rung"
