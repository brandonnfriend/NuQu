#!/bin/sh
# ENERGY GATE at L=3 — does the "n_b=3, not n_b=2" conclusion generalize in volume? The per-mode
# cutoff should be ~L-independent, so the n_b=2->3 shift ought to appear at L=3 too. RISK: L=3 hits
# the extensivity trap (E_0 not fully core-converged at reachable cores), so the absolute residual is
# large; the test is whether the n_b=2->3 shift still stands ABOVE that residual.
#
# n_b={2,3,4} x A={0,1,27} x seed{0..4} = 45 shards. A=0 vacuum + A=1 dilute + A=27 dense (filling 1.0
# at L=3). Deeper ladder (-> 256k) than L=2; PT2 off; per-run stored. run_frame_shard writes per-rung
# E_var + occ + <N>; BE(27)=27*E(1)-26*E(0)-E(27) derived post-hoc.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/nb_cutoff
#     sh submit_nb_energy_gate_L3.sh test   # 1 heaviest cell (n_b=4,A=27,seed0) — de-risk L=3 mem/wall
#     sh submit_nb_energy_gate_L3.sh        # n_b={2,3,4} x A={0,1,27} x seed{0..4} = 45 shards
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbEgateL3"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="../detsvsL/run_frame_shard.sh"
# warm-grow, PT2 off, deeper ladder (n_rungs=9 -> ~256k), heavy phase-0. args: L seed camp frame A filling maxcore mode runs cyc p0core maxsec
ENV_COMMON="NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_RUNGS=9 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 3 0 ${BASE}-nb4 bare 27 none 262144 independent 4 3 1000 21600
environment             = "${ENV_COMMON} NUQU_N_B=4"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 48G
request_disk            = 12G
Output                  = ${DIR}/logs/test.out
Error                   = ${DIR}/logs/test.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (L=3 n_b=4/N_f=16, A=27, seed0 @256k): de-risk mem/wall + core-convergence."
  exit 0
fi

# grid: NB A SEED MAXCORE MEM
SH="$DIR/grid.txt"; : > "$SH"
for NB in 2 3 4; do
  case "$NB" in 2) MEM=24G;; 3) MEM=32G;; 4) MEM=48G;; esac
  for A in 0 1 27; do
    for S in 0 1 2 3 4; do printf '%s %s %s 262144 %s\n' "$NB" "$A" "$S" "$MEM" >> "$SH"; done
  done
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 3 \$(SEED) ${BASE}-nb\$(NB) bare \$(A) none \$(MAXCORE) independent 4 3 1000 21600
environment             = "${ENV_COMMON} NUQU_N_B=\$(NB)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 12G
JobPrio                 = 16
Output                  = ${DIR}/logs/nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue NB,A,SEED,MAXCORE,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=${NJOBS}  (L=3 energy gate: n_b={2,3,4} x A={0,1,27} x seed{0..4})"
echo "pull -> data/classical/nb_energy_gate_L3/ ; extend make_nb_energy_gate for L=3 (residual caveat)"