#!/bin/sh
# DEEP CONVERGENCE SWEEP across L -- deeper-than-before E_var(core) curves for the energy
# extrapolation (§4 of the cost writeup), one campaign per L. Warm-grow (monotone), the
# production gaussian+lf frame, filling 1.0 (A = sites = L^3), N_b=2 -- same recipe as the
# L=2/L=3 anchors, just bigger L. The "half the final core each step up in L" schedule the
# anchors set:
#     L=2 -> 1M (done)   L=3 -> 512k (done)   L=4 -> 256k   L=5 -> 128k   L=6 -> 64k
#
# PT2 is gated LOWER as L grows: the EN-PT2 external space scales ~ (connections) x core and
# connections grow ~L^3, so a fixed cap that is safe at L=3 would OOM at L=6. Gating PT2 to a
# shallow rung keeps its footprint ~10-15 GB while E_var (the observable we extrapolate) is
# still recorded on EVERY rung. Memory is generous (qis nodes ~1 TB) to avoid OOM holds.
#
#   sh submit_deep_L_sweep.sh ["4 5 6"]      # default sweeps L=4,5,6 at 48c/192G
#
# Tight-packing overrides (to fit a partially-free qis node when the allocation is busy):
#   NUQU_CPUS=24 NUQU_MEM_OVERRIDE=128G sh submit_deep_L_sweep.sh "5 6"
# The C++ SpMV threads to request_cpus, so fewer cores = a slower (but scheduling-friendly)
# solve; per-rung incremental saving means a capped deep rung still keeps everything below it.
set -eu
FRAME=gaussian+lf; FILLING=1.0; NB=2; P0RUNS=32; MAXRUNGSEC=21600
CPUS="${NUQU_CPUS:-48}"
LS="${1:-4 5 6}"
for L in $LS; do
  case "$L" in
    2) MAXCORE=1024000; MEM=96G;  PT2CAP=64000 ;;   # done (anchor) -- here for completeness
    3) MAXCORE=512000;  MEM=160G; PT2CAP=64000 ;;   # done (anchor)
    4) MAXCORE=256000;  MEM=192G; PT2CAP=32000 ;;
    5) MAXCORE=128000;  MEM=192G; PT2CAP=16000 ;;
    6) MAXCORE=64000;   MEM=192G; PT2CAP=8000  ;;
    *) echo "ERROR: no schedule for L=$L" >&2; exit 1 ;;
  esac
  [ -n "${NUQU_MEM_OVERRIDE:-}" ] && MEM="$NUQU_MEM_OVERRIDE"   # tight-pack a busy node
  CAMPAIGN="deepL-$(date +%Y%m%d-%H%M%S)-L${L}-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = ${L} 0 ${CAMPAIGN} ${FRAME} 1 ${FILLING} ${MAXCORE} independent 4 3 1000 ${MAXRUNGSEC}
environment             = "NUQU_DEEP_SOLVE=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=${NB} NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=${PT2CAP} NUQU_WARM_GROW=1 NUQU_PHASE0_RUNS=${P0RUNS}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 25
Output                  = ${DIR}/logs/deep_L${L}.out
Error                   = ${DIR}/logs/deep_L${L}.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  L=${L}  frame=${FRAME}  filling=${FILLING} (A=$((L*L*L)))  max_core=${MAXCORE}  mem=${MEM}  pt2_cap=${PT2CAP}"
done
echo "watch: per-rung E_var saves incrementally; a rung that OOMs/times-out keeps all prior rungs."
