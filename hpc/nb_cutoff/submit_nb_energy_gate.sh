#!/bin/sh
# ENERGY GATE for the boson cutoff (the 2026-09-02 audit's required experiment), L=2 where the core
# converges. The occupation-tail study is a diagnostic; THIS measures the quantity the cutoff claim
# actually needs: the ground-energy change E_0(N_f=4)->E_0(8)->E_0(16) at CORE-CONVERGED states, with
# per-seed uncertainty and observable (<N>/mode) convergence.
#
# One shard = one (n_b, A, seed) warm-grow CORE-LADDER (run_frame_shard stores per-rung E_var + occ +
# <N>), so the grid n_b={2,3,4} x A={1,32} x seed={0..4} = 30 shards gives E_0(N_f, core, seed).
# n_b=2/3/4 <-> N_f=4/8/16. A=1 dilute + A=32 dense (max source) = the representative dilute/worst
# points; L=3 dense is NOT included (extensivity/H-build wall — stays diagnostic). PT2 OFF (E_var is
# the variational energy; PT2 is only a labeled diagnostic and the N_f=16 memory hog). Per-run stored
# (not best-of-ensemble): the seed SPREAD is the uncertainty, and captures the basin-collapse variance.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_energy_gate.sh test   # 1 heaviest shard (n_b=4,A=32,seed0) — de-risk N_f=16 @256k
#   sh submit_nb_energy_gate.sh        # n_b={2,3,4} x A={0,1,32} x seed{0..4} = 45 shards
#
# Observables tracked per rung (run_frame_shard stores them): E_var (E_0), <N>/mode (pion occupation),
# occupation tail. Derived post-hoc: BINDING ENERGY BE(A)=A*E(1)-(A-1)*E(0)-E(A) (needs A=0,1,A), and
# lambda(n_b) (the block-encoding 1-norm that sets N_walk) reported separately as the cost sensitivity.
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbEgate"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="../detsvsL/run_frame_shard.sh"

# warm-grow deep-solve core-ladder; PT2 off (cap=1); heavy-ish phase0 for basin finding.
# per-L_bos memory: N_f=16 (n_b=4) is the heavy one. args: L seed camp frame A filling maxcore mode runs cyc p0core maxsec
ENV_FOR() { # $1=n_b -> echo the environment string
  echo "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$1 NUQU_N_RUNGS=8 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"
}

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 2 0 ${BASE}-nb4 bare 32 none 131072 independent 4 3 1000 14400
environment             = "$(ENV_FOR 4)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 64G
request_disk            = 10G
Output                  = ${DIR}/logs/test.out
Error                   = ${DIR}/logs/test.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (n_b=4/N_f=16, A=32, seed0 @256k): de-risk the heaviest cell's mem/wall."
  exit 0
fi

# build the grid: NB A SEED MAXCORE MEM.  A=0 (pion vacuum, no nucleons) is the reference for the
# BINDING ENERGY BE(A) = A*E(1) - (A-1)*E(0) - E(A), which subtracts the large pion-vacuum energy
# (~202.5/site) — so BE's convergence in n_b is the physically meaningful observable, not raw E_0.
# n_b={2,3,4} x A={0,1,32} x seed{0..4} = 45 shards (A=0 is a cheap boson-only solve).
SH="$DIR/grid.txt"; : > "$SH"
for NB in 2 3 4; do
  case "$NB" in 2) MC=262144; MEM=32G;; 3) MC=262144; MEM=48G;; 4) MC=131072; MEM=64G;; esac
  for A in 0 1 32; do
    for S in 0 1 2 3 4; do printf '%s %s %s %s %s\n' "$NB" "$A" "$S" "$MC" "$MEM" >> "$SH"; done
  done
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 2 \$(SEED) ${BASE}-nb\$(NB) bare \$(A) none \$(MAXCORE) independent 4 3 1000 14400
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=\$(NB) NUQU_N_RUNGS=8 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 10G
JobPrio                 = 18
Output                  = ${DIR}/logs/nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue NB,A,SEED,MAXCORE,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=${NJOBS}  (energy gate: n_b={2,3,4} x A={1,32} x seed{0..4}, L=2)"
echo "output: run_frame_shard writes bare_L2*.json per (seed) to detsvsL campaign dir; NUQU_N_B distinguishes n_b."
echo "then: pull + build E_0(N_f) convergence + <N> convergence + seed-spread figure"