#!/bin/sh
# Vertex-fix VARIATIONAL BASELINE + dilute frame-value probe (REMEDIATION_PLAN N2/N3).
#
# TWO campaigns, selected by the first arg (test | bare | probe | all; default all):
#
#   bare  -> the CERTIFIABLE variational classical baseline. E_var(bare) is variational with NO
#            map-back (bare frame = identity), so it needs none of the back-eval machinery.
#            Deep-solve warm-grow, n_b=2, filling 1.0, L=2,3,4,5(best-effort), one seed.
#            (Replaces retired Group A gaussian+lf deep-E_inf: those were frame-internal,
#            un-back-evaluatable, non-converging past L=2, redundant-seed, and OOM'd L5/L6.)
#
#   probe -> DILUTE gaussian back-eval probe. The DENSE (filling 1.0) gaussian map-back is
#            INTRACTABLE at n_b=2: the smoke (cluster 290831) fanned 8k dets past a 300k cap and
#            STILL dropped 49% of the weight -> E_orig was a garbage 49%-truncated remnant. The
#            squeezed vacuum over the boson modes is a dense object at n_b=2 (N_f=4), exactly like
#            LF. BUT a DILUTE ground state is far more COMPACT, so its fanned-out image may fit the
#            cap with negligible dropped weight -> a clean, variational E_orig AT the physical n_b=2.
#            This sweep (A=1, f=0.25, f=0.5 x L=2,3) MAPS the tractability boundary: read
#            back_dropped_weight per shard. ~0 -> E_orig is the honest n_b=2 frame value; ~0.5 ->
#            dilution did not rescue it (a clean negative result). Pairs with the clean DENSE n_b=1
#            frame value (benchmark 290820: squeeze helps -72 MeV variationally at L=2).
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_vertexfix_baseline.sh test    # 1 tiny dilute gaussian L=2 back-eval shard (env check)
#     sh submit_vertexfix_baseline.sh probe   # the dilute gaussian back-eval sweep (bare already ran)
#     sh submit_vertexfix_baseline.sh bare    # the bare variational baseline only
#     sh submit_vertexfix_baseline.sh all     # both (default)
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-baseline"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

if [ "$MODE" = "test" ]; then
  D="campaign_${BASE}-test"; mkdir -p "$D/logs"
  cat > "$D/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = 2 0 ${BASE}-test gaussian 1 0.25 8000 independent 8 3 1000 3600
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=4 NUQU_PT2_MAX_CORE=8000 NUQU_PHASE0_RUNS=16 NUQU_BACK_EVAL=1 NUQU_BACK_SUPPORT_CAP=1000000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 48G
request_disk            = 8G
Output                  = ${D}/logs/smoketest.out
Error                   = ${D}/logs/smoketest.err
Log                     = ${D}/logs/campaign.log
queue
EOF
  condor_submit "$D/campaign.sub"
  echo "CAMPAIGN=${BASE}-test  SMOKE (dilute f=0.25 gaussian L=2 back-eval; check back_dropped_weight)"
  exit 0
fi

# ============================ bare arm (the baseline) ======================== #
# cols: L MAXCORE MEM PT2CAP.  L=5 best-effort (gaussian+lf OOM'd L5 at 256G; bare is lighter).
if [ "$MODE" = "bare" ] || [ "$MODE" = "all" ]; then
  CBARE="${BASE}-bare"; DBARE="campaign_${CBARE}"; mkdir -p "$DBARE/logs"
  GBARE="$DBARE/shards.txt"; : > "$GBARE"
  printf '2 1024000 96G  64000\n3 512000  160G 64000\n4 256000  192G 32000\n5 128000  256G 16000\n' > "$GBARE"
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
fi

# ==================== probe arm (dilute gaussian back-eval) ================== #
# cols: L FILL A MAXCORE  ('none' filling => explicit A=1). Shared: n_b=2, cap=1e6, PT2=16000,
# max_core kept modest (dilute converges shallow; deeper core = bigger input = more fan-out).
if [ "$MODE" = "probe" ] || [ "$MODE" = "all" ]; then
  CG="${BASE}-probe"; DG="campaign_${CG}"; mkdir -p "$DG/logs"
  GG="$DG/shards.txt"; : > "$GG"
  for L in 2 3; do
    printf '%s none 1 16000\n%s 0.25 1 16000\n%s 0.5 1 16000\n' "$L" "$L" "$L" >> "$GG"
  done
  cat > "$DG/probe.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) 0 ${CG} gaussian \$(A) \$(FILL) \$(MAXCORE) independent 4 3 1000 14400
environment             = "NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=6 NUQU_PT2_MAX_CORE=16000 NUQU_PHASE0_RUNS=32 NUQU_BACK_EVAL=1 NUQU_BACK_SUPPORT_CAP=1000000"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 64G
request_disk            = 8G
JobPrio                 = 22
Output                  = ${DG}/logs/probe_L\$(L)_f\$(FILL).out
Error                   = ${DG}/logs/probe_L\$(L)_f\$(FILL).err
Log                     = ${DG}/logs/campaign.log
queue L,FILL,A,MAXCORE from ${GG}
EOF
  condor_submit "$DG/probe.sub"
  echo "PROBE arm CAMPAIGN=${CG}  jobs=$(wc -l < "$GG")  (dilute gaussian back-eval: A=1,f=0.25,f=0.5 x L=2,3)"
  echo "  -> read back_dropped_weight per shard: ~0 = clean n_b=2 E_orig (frame value); ~0.5 = still fans out."
fi
