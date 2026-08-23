#!/bin/sh
# Vertex-fix VARIATIONAL BASELINE campaign (REMEDIATION_PLAN N2/N3 — the replan of Group A).
#
# WHY this replaces Group A's gaussian+lf deep-E_inf run: Group A's energies were the
# frame-internal E_var of the gaussian+LF frame -- NON-variational (LF non-isospectrality) and
# un-back-evaluatable at dense filling (the LF map-back fans out superlinearly, L>=3). So its
# deep L5/L6 points (which also OOM'd at 256G) were neither variational nor fixable. This run
# instead produces a GENUINE VARIATIONAL upper bound at every L, via the gaussian-only path:
#
#   * bare arm      -> E_var(bare) IS variational (no frame). The comparison baseline.
#   * gaussian arm  -> solve the SQUEEZED H (compact), then map the framed |psi~> back through
#                      exp(G_sq) onto the ORIGINAL bare H -> E_orig >= E_bare (Ritz-valid).
#                      Squeeze's map-back is grow~1 (support doesn't fan out) so it stays
#                      TRACTABLE at every L -- unlike LF. E_orig(gaussian) vs E_var(bare) at
#                      matched core = the honest, VARIATIONAL frame value (headline 2).
#
# Validated locally: the wired back-eval reproduces the 290820 benchmark E_orig EXACTLY
# (L=2 d3 n_b=1 f=1.0: E_orig=1950.203, resid=75.09), confirming squeeze helps -72 MeV (var)
# at L=2. This run resolves whether that gain survives at L=3/4/5 once the core is deep (the
# benchmark's L=3 gaussian sat +78 above bare, but only at core=4000 -- deeply under-converged).
#
# Deep-solve warm-grow (OpenMP SpMV, 48 cores), n_b=2, filling 1.0, ONE seed (Group A's two
# seeds were byte-identical -- the deep solve is deterministic once the frame is fit).
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_vertexfix_baseline.sh test   # 1 tiny gaussian L=2 back-eval shard (validate the env path)
#     sh submit_vertexfix_baseline.sh        # full grid (bare + gaussian, L=2,3,4,5)
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-baseline"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  D="campaign_${BASE}-test"; mkdir -p "$D/logs"
  cat > "$D/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = 2 0 ${BASE}-test gaussian 1 1.0 8000 independent 8 3 1000 3600
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=4 NUQU_PT2_MAX_CORE=8000 NUQU_PHASE0_RUNS=16 NUQU_BACK_EVAL=1 NUQU_BACK_SUPPORT_CAP=300000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 32G
request_disk            = 8G
Output                  = ${D}/logs/smoketest.out
Error                   = ${D}/logs/smoketest.err
Log                     = ${D}/logs/campaign.log
queue
EOF
  condor_submit "$D/campaign.sub"
  echo "CAMPAIGN=${BASE}-test  SMOKE (gaussian L=2 back-eval; expect rungs with E_orig>=E_var and back_converged=true)"
  exit 0
fi

# ---- per-L ceilings (gaussian-only + bare are lighter than gaussian+lf; PT2 gated low deep) --
#      cols: L MAXCORE MEM PT2CAP.  L=5 is BEST-EFFORT: Group A's gaussian+lf OOM'd L5 at 256G,
#      but gaussian-only/bare are lighter (no LF displacement terms) -> real shot; if the deep
#      rung still OOMs, the incremental per-rung save keeps every shallower rung.
LS="2 3 4 5"
lrow() {
  case "$1" in
    2) echo "2 1024000 96G  64000" ;;
    3) echo "3 512000  160G 64000" ;;
    4) echo "4 256000  192G 32000" ;;
    5) echo "5 128000  256G 16000" ;;
  esac
}

# ============================ bare arm (the baseline) ======================== #
CBARE="${BASE}-bare"; DBARE="campaign_${CBARE}"; mkdir -p "$DBARE/logs"
GBARE="$DBARE/shards.txt"; : > "$GBARE"
for L in $LS; do lrow "$L" >> "$GBARE"; done
cat > "$DBARE/bare.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) 0 ${CBARE} bare 1 1.0 \$(MAXCORE) independent 4 3 1000 21600
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
Output                  = ${DBARE}/logs/bare_L\$(L).out
Error                   = ${DBARE}/logs/bare_L\$(L).err
Log                     = ${DBARE}/logs/campaign.log
queue L,MAXCORE,MEM,PT2CAP from ${GBARE}
EOF
condor_submit "$DBARE/bare.sub"
echo "BARE arm  CAMPAIGN=${CBARE}  jobs=$(wc -l < "$GBARE")  (variational baseline: bare L=2,3,4,5[L5 best-effort])"

# ==================== gaussian arm (back-evaluated E_orig) =================== #
CG="${BASE}-gauss"; DG="campaign_${CG}"; mkdir -p "$DG/logs"
GG="$DG/shards.txt"; : > "$GG"
# cols: L MAXCORE MEM PT2CAP  (support cap fixed at 300000 -- bounds the deep-rung map-back;
# the compact squeeze state saturates well below it at L=2, so dropped_weight~0 there).
for L in $LS; do lrow "$L" >> "$GG"; done
cat > "$DG/gauss.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) 0 ${CG} gaussian 1 1.0 \$(MAXCORE) independent 4 3 1000 21600
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=\$(PT2CAP) NUQU_PHASE0_RUNS=32 NUQU_BACK_EVAL=1 NUQU_BACK_SUPPORT_CAP=300000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 48
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 25
Output                  = ${DG}/logs/gauss_L\$(L).out
Error                   = ${DG}/logs/gauss_L\$(L).err
Log                     = ${DG}/logs/campaign.log
queue L,MAXCORE,MEM,PT2CAP from ${GG}
EOF
condor_submit "$DG/gauss.sub"
echo "GAUSS arm CAMPAIGN=${CG}  jobs=$(wc -l < "$GG")  (variational E_orig via squeeze map-back: gaussian L=2,3,4,5[L5 best-effort])"
echo ""
echo "combine: rsync campaign_${CBARE}/shards + campaign_${CG}/shards, then compare"
echo "  E_orig(gaussian) vs E_var(bare) at matched core -> the VARIATIONAL frame value per L."
