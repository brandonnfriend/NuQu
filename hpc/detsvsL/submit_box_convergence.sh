#!/bin/sh
# FIXED-A BOX-CONVERGENCE binding energies -- the "the model predicts physics" panel.
# For each box side L, solve four nucleon sectors: the pion vacuum (A=0), one nucleon
# (A=1), and the target nuclei A=2 (deuteron) and A=4 (alpha). Then
#     BE(A,L) = A*E(1,L) - (A-1)*E(0,L) - E(A,L)
# cancels the extensive +202.5*sites pion vacuum and gives the binding energy; watch it
# converge as the physical box L*a_L (a_L=2.2 fm) grows past the nuclear size.
#
# Recipe: gaussian frame (EXACTLY isospectral -> exact eigenvalues, unlike the truncated
# LF), warm-grow (monotone convergence curve per sector), n_b=2, dim=3, Watson params, NO
# LEC fit. Dilute (A<=4) => far cheaper than the filling=1.0 dets-vs-L runs; A=0 (full
# pion field) is the heaviest per L, so memory is sized for it.
#
#   sh submit_box_convergence.sh ["2 3 4 5 6"] ["0 1 2 4"]
set -eu
FRAME=gaussian; NB=2; P0RUNS=16; MAXRUNGSEC=10800; PT2CAP=64000
LS="${1:-2 3 4 5 6}"; AS="${2:-0 1 2 4}"
CAMPAIGN="bind-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
SUB="$DIR/campaign.sub"; : > "$SUB"
n=0
for L in $LS; do
  case "$L" in
    2) MAXCORE=64000;  MEM=48G;  CPUS=16 ;;
    3) MAXCORE=128000; MEM=96G;  CPUS=16 ;;
    4) MAXCORE=256000; MEM=128G; CPUS=24 ;;
    5) MAXCORE=256000; MEM=160G; CPUS=24 ;;
    6) MAXCORE=128000; MEM=192G; CPUS=24 ;;
    *) echo "ERROR: no schedule for L=$L" >&2; exit 1 ;;
  esac
  for A in $AS; do
    n=$((n + 1))
    cat >> "$SUB" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = ${L} 0 ${CAMPAIGN} ${FRAME} ${A} none ${MAXCORE} independent 4 3 1000 ${MAXRUNGSEC}
environment             = "NUQU_DEEP_SOLVE=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=${NB} NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=${PT2CAP} NUQU_WARM_GROW=1 NUQU_PHASE0_RUNS=${P0RUNS}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/bind_L${L}_A${A}.out
Error                   = ${DIR}/logs/bind_L${L}_A${A}.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  done
done
condor_submit "$SUB"
echo "CAMPAIGN=${CAMPAIGN}  ${n} shards  L={${LS}} x A={${AS}}  frame=${FRAME} n_b=${NB} dim=3"
echo "pull shards/<frame>_L<L>d3_A<A>_s0.json ; analyze with misc/run_binding_analysis.py"
