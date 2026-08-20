#!/bin/sh
# Vertex-fix regeneration — round-1 CLASSICAL campaign (REMEDIATION_PLAN N2/N3).
# Corrected Hamiltonian, N_b=2, dim=3, production frame gaussian+lf, filling 1.0 (A=L^3).
#
#   Group A  deep E_inf convergence      independent + deep-solve + warm-grow, L=2..6, 2 seeds
#            (the established 1M-core frozen-frame path; feeds energies/binding/uncertainty)
#   Group B  co-evolution grow-mode regen  grow + faithful Phase-1 co-evolution, L=2..4, 2 seeds
#            (validates + regenerates the 5a2112d COO-paper workflow on corrected H)
#
# Group A is the frozen-frame deep-solve (OpenMP SpMV across 48 cores -> reaches 1M). Group B is
# the NEW co-evolution (fork-ensemble Phase-0 + slow γ=1.1 warm-started Phase-1 frame refit); its
# Phase-2 is single-threaded so its depth is CAPPED (validation, not a 1M run — Group A does depth).
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_vertexfix_classical.sh test     # <-- FIRST: 1 tiny L=2 grow shard (C++ build + co-evolution path)
#     sh submit_vertexfix_classical.sh          # then Group A (10) + Group B (6) = 16 shards
#
# Per-rung incremental save: a shard that OOMs/times-out at a deep rung keeps every rung below it.
set -eu
MODE="${1:-run}"
BASE="vfc-$(date +%Y%m%d-%H%M%S)"
SEEDS="0 1"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# --- smoke test: ONE tiny L=2 co-evolution grow shard on a qis node ---------- #
# Validates the self-provision + C++ mixed_ci build AND the new grow-mode co-evolution
# code path (5a2112d) end-to-end before the campaign. 'smoke' profile = tiny ladder.
if [ "$MODE" = "test" ]; then
  DIR="campaign_${BASE}-test"; mkdir -p "$DIR/logs"
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = 2 0 ${BASE}-test gaussian+lf 1 1.0 999999 grow 8 3 500 3600
environment             = "NUQU_N_B=2 NUQU_PROFILE=smoke NUQU_PHASE1_MODE=coevolve NUQU_SQUEEZE_OPT=analytic"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 16G
request_disk            = 8G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}-test  SMOKE TEST (1 job: L=2 gaussian+lf grow/co-evolve, smoke ladder)"
  echo "check: grep '\[shard\] done status=0' ${DIR}/logs/smoketest.out  (no C++ build / import errors)"
  echo "then:  sh submit_vertexfix_classical.sh"
  exit 0
fi

# ======================= Group A: deep E_inf convergence ===================== #
# Frozen-frame deep-solve warm-grow (submit_deep_L_sweep recipe): OpenMP SpMV to deep cores.
# PT2 gated LOWER as L grows (EN-PT2 external space ~ connections·core; would OOM at high L).
CA="${BASE}-A"; DA="campaign_${CA}"; mkdir -p "$DA/logs"
GA="$DA/groupA_shards.txt"; : > "$GA"
for L in 2 3 4 5 6; do
  case "$L" in
    2) MAXCORE=1024000; MEM=96G;  PT2CAP=64000 ;;
    3) MAXCORE=512000;  MEM=160G; PT2CAP=64000 ;;
    4) MAXCORE=256000;  MEM=192G; PT2CAP=32000 ;;
    5) MAXCORE=128000;  MEM=192G; PT2CAP=16000 ;;
    6) MAXCORE=64000;   MEM=192G; PT2CAP=8000  ;;
  esac
  for S in $SEEDS; do echo "$L $S $MAXCORE $MEM $PT2CAP" >> "$GA"; done
done
cat > "$DA/groupA.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) \$(SEED) ${CA} gaussian+lf 1 1.0 \$(MAXCORE) independent 4 3 1000 21600
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=\$(PT2CAP) NUQU_PHASE0_RUNS=32"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 48
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 25
Output                  = ${DA}/logs/A_L\$(L)_s\$(SEED).out
Error                   = ${DA}/logs/A_L\$(L)_s\$(SEED).err
Log                     = ${DA}/logs/campaign.log
queue L,SEED,MAXCORE,MEM,PT2CAP from ${GA}
EOF
condor_submit "$DA/groupA.sub"
echo "GROUP A  CAMPAIGN=${CA}  jobs=$(wc -l < "$GA")  (deep E_inf: gaussian+lf L=2..6 x seeds{${SEEDS}})"

# ==================== Group B: co-evolution grow-mode regen ================== #
# Faithful COO-paper co-evolution. Phase-1/2 depth CAPPED (single-threaded Phase-2) — this is
# the workflow-validation + frame-trajectory regen, not a depth run. Fork ensemble on 32 cores.
CB="${BASE}-B"; DB="campaign_${CB}"; mkdir -p "$DB/logs"
GB="$DB/groupB_shards.txt"; : > "$GB"
for L in 2 3 4; do
  case "$L" in
    2) P1MAX=16000; P2MAX=64000; MEM=48G ;;
    3) P1MAX=8000;  P2MAX=32000; MEM=48G ;;
    4) P1MAX=8000;  P2MAX=16000; MEM=32G ;;
  esac
  for S in $SEEDS; do echo "$L $S $P1MAX $P2MAX $MEM" >> "$GB"; done
done
cat > "$DB/groupB.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) \$(SEED) ${CB} gaussian+lf 1 1.0 999999 grow 32 10 1000 21600
environment             = "NUQU_N_B=2 NUQU_PROFILE=hpc NUQU_PHASE1_MODE=coevolve NUQU_SQUEEZE_OPT=analytic NUQU_PHASE1_MAX_DETS=\$(P1MAX) NUQU_PHASE2_MAX_DETS=\$(P2MAX)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 32
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DB}/logs/B_L\$(L)_s\$(SEED).out
Error                   = ${DB}/logs/B_L\$(L)_s\$(SEED).err
Log                     = ${DB}/logs/campaign.log
queue L,SEED,P1MAX,P2MAX,MEM from ${GB}
EOF
condor_submit "$DB/groupB.sub"
echo "GROUP B  CAMPAIGN=${CB}  jobs=$(wc -l < "$GB")  (co-evolution regen: gaussian+lf L=2..4 x seeds{${SEEDS}})"
echo ""
echo "combine A: rsync the campaign_${CA}/shards then analyze the E_var(core) ladders + PT2 extrapolation"
echo "combine B: campaign_${CB}/shards — check the Phase-1 co-evolution trajectory + frame refit logs"
