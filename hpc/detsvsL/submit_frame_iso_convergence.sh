#!/bin/sh
# CONVERGENCE + MULTI-SEED run — hardens the frame-isospectrality headline for the audit.
# Point: bare and squeeze are the SAME H (isospectral at fixed n_b), so the Track-B "win" is a
# COMPACTION / convergence-RATE effect (squeeze reaches a given accuracy with far fewer dets), not
# a lower E-infinity. This run traces E(core) deep for bare vs squeeze (+LF curves) so the curves
# can be shown converging, and adds seeds {0,1,2} on bare/squeeze for the scatter bars that §7.2
# of docs/frame_isospectrality_results.md flags as missing. n_b=3 (converged). All d3 = ED-free.
#
# Overnight-friendly: deep cores at L3 f1.0 may not finish — incremental per-core save keeps every
# rung, so cancel stragglers with `condor_rm <cluster>` any time and the shallow rungs survive.
#
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit` after reconciling to the pushed commit.
set -eu
CAMPAIGN="frameiso-conv-$(date +%Y%m%d-%H%M%S)-$$"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# columns: L DIM FRAME A FILLING CORES NB BKNF CAP MEM SEED
SH="$DIR/shards.txt"; : > "$SH"
for fr in bare gaussian gaussian+lf lf; do
  case "$fr" in bare|gaussian) seeds="0 1 2" ;; *) seeds="0" ;; esac      # bars only where cheap/central
  for L in 2 3; do
    if [ "$L" = "2" ]; then cores="250+1000+4000+16000+64000+256000"; mem=48G
    else                    cores="250+1000+4000+16000+64000";        mem=128G; fi
    for af in A1 f1.0; do
      if [ "$af" = "A1" ]; then fill=none; else fill=1.0; fi
      cap="-"; case "$fr" in *lf*) [ "$af" != "A1" ] && cap=50000 ;; esac   # cap dense-LF map-back
      for s in $seeds; do
        printf '%s 3 %s 1 %s %s 3 3 %s %s %s\n' "$L" "$fr" "$fill" "$cores" "$cap" "$mem" "$s" >> "$SH"
      done
    done
  done
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) \$(NB) \$(DIM) \$(A) 16 \$(SEED)
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=\$(BKNF) NUQU_SUPPORT_CAP=\$(CAP) NUQU_EXACT_REF=-"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 10
Output                  = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_\$(FRAME)_f\$(FILLING)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,FRAME,A,FILLING,CORES,NB,BKNF,CAP,MEM,SEED from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  (bare/squeeze x L{2,3} x {dilute,f1.0} x seed{0,1,2}; +LF curves seed0)"
echo "analyze: E(core) convergence curves bare vs squeeze (compaction/dets-to-accuracy) + seed scatter"
