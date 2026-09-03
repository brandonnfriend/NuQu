#!/bin/sh
# VOLUME-SCALING of the boson-cutoff shift (re-audit P0-4, route 1): does the n_b=3 cutoff hold to
# L=10 in the TOTAL energy? A per-mode cutoff being L-independent doesn't bound a total-energy error
# at 1000 sites (small local truncations can accumulate). So measure the PAIRED shift
# Δ34(L) = E(n_b=4) - E(n_b=3) at COMMON cores for L=2,3,4, fit Δ34 per site, and propagate to L=10
# with a band. The shift is a small difference that is stable across the ladder even when the absolute
# energy is not converged (verified at L=2: Δ34 ~ 0.001-0.009 MeV/component).
#
# Already have: L=2 (n_b=3,4, A=0,1) from the energy gate; L=3 n_b=3 from the L=3 gate. This fills the
# gaps: L=3 n_b=4, and L=4 n_b={3,4} — at A={0,1} (dilute+vacuum: best core-convergence + the BE
# reference; the L=2 gate showed Δ34 is ~A-independent). 3 seeds (determinism check; solve is
# seed-insensitive at these sizes). PT2 off, warm-grow, deep ladder (n_rungs=9 -> 256k).
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/nb_cutoff
#     sh submit_nb_volscaling.sh test   # 1 heaviest cell (L=4 n_b=4 A=1) — de-risk L=4/N_f=16 mem
#     sh submit_nb_volscaling.sh        # L=3 n_b=4 + L=4 n_b={3,4}, A={0,1}, 3 seeds = 18 shards
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbVolscale"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="../detsvsL/run_frame_shard.sh"
ENV_COMMON="NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_RUNGS=9 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 4 0 ${BASE}-nb4 bare 1 none 262144 independent 4 3 1000 21600
environment             = "${ENV_COMMON} NUQU_N_B=4"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 96G
request_disk            = 12G
Output                  = ${DIR}/logs/test.out
Error                   = ${DIR}/logs/test.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (L=4 n_b=4/N_f=16 A=1 @256k): de-risk the heaviest volume-scaling cell."
  exit 0
fi

# grid: L NB A SEED MEM  (fills the gaps; L=2 + L=3-n_b3 reused from the gates)
SH="$DIR/grid.txt"; : > "$SH"
add() { for S in 0 1 2; do printf '%s %s %s %s %s\n' "$1" "$2" "$3" "$S" "$4" >> "$SH"; done; }
add 3 4 0 48G; add 3 4 1 48G                         # L=3 n_b=4 (n_b=3 already have)
add 4 3 0 48G; add 4 3 1 48G                         # L=4 n_b=3
add 4 4 0 96G; add 4 4 1 96G                         # L=4 n_b=4
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = \$(L) \$(SEED) ${BASE}-nb\$(NB) bare \$(A) none 262144 independent 4 3 1000 21600
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
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,A,SEED,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=${NJOBS}  (volume scaling: L=3 n_b=4 + L=4 n_b={3,4}, A={0,1}, 3 seeds)"
echo "combine with L=2 gate + L=3-n_b3 gate; fit Delta34(L) per site -> propagate to L=10 with a band"