#!/bin/sh
# A=0 VACUUM CALIBRATION RUNS AT L=4 AND L=5 (2026-11-24), for gate 5 of nuqu-2026-10.
#
# WHY. Gate 5 (family calibration) reports a (frame, n_b, L) family's limits only if its A=0
# cell lands within sigma of the exact pion vacuum. After it was built: bare n_b=3 L=4 FAILS
# (512k ladder, +12.94 MeV/site = 2.4 sigma), bare L=5 has NO A=0 run, and the squeezed frame
# has none at L=5. The exact vacuum is now registered at L=5 too (352.328627 MeV/site,
# frame.boson_vacuum_energy). Three arms, one row per (arm, seed):
#
#   bareL4  EXTEND 294246's bare L=4 A=0 shards 512k -> 2,048,000 dets, PT2 cap 256k -> 1,024,000
#           (two doublings; the 512k rung is re-recorded with PT2). The 512k ladder was still
#           falling 4.4 MeV/site per doubling at its top and sits 23.6/site above the vacuum.
#           Settings must match 294246's bare refs (32 starts) or the shard refuses. Seeds 0,1:
#           the three 512k seeds agree to 0.001/site at every rung, so two give the matched-depth
#           spread at half the node-days. A deeper PASS here would not certify the 512k A>0 cells
#           (any failing entry fails the family); it would say the bare A-sweep needs this depth.
#   bareL5  FRESH bare L=5 A=0 at EXACTLY the bare L=5 A-sweep's settings (293963: 128k top, PT2
#           to 65,536, 32 starts, 4 select workers), so it calibrates that family as run. Seeds
#           0,1,2. Each finished shard checkpoints its 128k core, so it can be EXTENDED later.
#   sqL5    FRESH squeezed L=5 A=0 to 1,024,000 with PT2 to 512,000 (the depth squeezed L=4
#           needed). 16 select starts: at A=0 there are no nucleon arrangements to stratify over,
#           so more starts only re-sample the boson init. Seeds 0,1 (squeezed L=4 A=0 seeds agree
#           to 0.000/site).
#
# WHAT TO EXPECT (computed 2026-11-24, frame.analytic_bogoliubov): the bare Fock vacuum's overlap
# with the exact pion vacuum is 0.57 / 7.3e-2 / 8.4e-4 / 3.5e-7 at L=2/3/4/5 (<N> = 1.2 / 5.5 /
# 14.9 / 31.4 quanta). The bare determinant count needed grows exponentially in the volume, so the
# bare arms will most likely CONFIRM the miss (bareL4 tightens the bound and shows whether the
# limit walks toward 344.596; bareL5 most likely refuses or misses, giving the family 'failed'
# instead of 'missing'). The squeezed frame starts from the per-mode squeezed vacuum: sqL5 should
# land within sigma, extending the squeezed calibration from L=4 to L=5.
#
# MEMORY (from the 294246 / 293963 / 319703 per-rung records). Bare L=4: 512k grow peaked
# 155 GB with 120M PT2 externals at 256k (~700 B each) -> 1024k grow ~310 GB, PT2 at 512k ~330 GB,
# 2048k grow + PT2 at 1024k ~650 GB. Base 384 GB, doubles on a hold, cap 960 GB (a qis node is
# 1,007 GB), so the 2048k rung runs ~alone on a node. Bare L=5: 293963's sizing (256 GB; the 128k
# rung peaked 222 GB at A=125). Squeezed L=5: ~2x squeezed L=4's per det (375 vs 192 modes):
# 512k PT2 ~200 GB, 1024k grow ~450 GB -> base 384, cap 768. Every hold resumes from the last
# checkpointed rung.
#
# COST. bareL4 ~1.5-2 days (no select stage); bareL5 ~18-20 h (293963's L=5 wall); sqL5 ~2 days
# (select ~4-5 h + ladder). 7 jobs; the bare L=5 ones fit beside the running 319703 jobs, the
# deep ones wait for a mostly free node.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit), after the push:
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout <branch> && git reset --hard origin/<branch>
#     cd hpc/detsvsL
#     sh submit_vacuum_calibration.sh all             # all three arms
#     sh submit_vacuum_calibration.sh bareL5          # or one arm: bareL4 | bareL5 | sqL5
# Retrieve (bareL4 lands in 294246's directory, the others in this campaign's):
#     rsync -az --exclude '*.npz' --exclude 'pre_extend_512k' hep:<shards dir>/ data/classical/$(date +%F)/vacuum_calib_<cluster>/
set -eu
MODE="${1:-all}"
SRC_CAMPAIGN="${SRC_CAMPAIGN:-20261008-184336-nb3AsweepSq}"     # 294246: the bare L=4 A=0 checkpoints
BASE="$(date +%Y%m%d-%H%M%S)-vaccal"                             # fresh L=5 shards + all logs
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"
NB=3
SEEDS_L4="${SEEDS_L4:-0 1}"
SEEDS_BL5="${SEEDS_BL5:-0 1 2}"
SEEDS_SQ5="${SEEDS_SQ5:-0 1}"
MAXHOLDS="${MAXHOLDS:-4}"

SRC_SHARDS="campaign_${SRC_CAMPAIGN}-nb${NB}/shards"

# runner args: L seed campaign frame A filling max_core ladder_mode ...  (campaign per row)
ARGS='$(L) $(SEED) $(CAMP) $(FRAME) $(A) none $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=$(NRUNGS) NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=$(P0RUNS) NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified NUQU_PHASE0_WORKERS=$(P0W) NUQU_EXTEND=$(EXT)'
VARS="CAMP,FRAME,NB,L,A,SEED,MAXCORE,PT2CAP,NRUNGS,MEM,CPUS,MAXRUNGSEC,PRIO,P0W,P0RUNS,EXT"

row() { printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' "$@"; }

submit_grid() {   # $1=arm  $2=grid  $3=memory cap (GB)
  arm="$1"; grid="$2"; cap="$3"
  cat > "campaign_${BASE}/${arm}.sub" <<EOF
Executable              = ${RUNNER}
arguments               = ${ARGS}
environment             = "${ENVSTR}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = \$(CPUS)
MEMCAP_MB               = $(( cap * 1024 ))
NHOLDS                  = ifThenElse(isUndefined(NumHolds), 0, NumHolds)
MEMGROW_MB              = (\$(MEM) * 1024 * (1 + \$(NHOLDS)))
request_memory          = ifThenElse(\$(MEMGROW_MB) < \$(MEMCAP_MB), \$(MEMGROW_MB), \$(MEMCAP_MB))
periodic_release        = (HoldReasonCode == 34) && (NumHolds < ${MAXHOLDS})
on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))
request_disk            = 300M
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/${arm}_\$(FRAME)_L\$(L)_A\$(A)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/${arm}_\$(FRAME)_L\$(L)_A\$(A)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue ${VARS} from ${grid}
EOF
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm}  LOGS=campaign_${BASE}  jobs=$(wc -l < "$grid")"
}

arm_bareL4() {
  [ -d "$SRC_SHARDS" ] || { echo "ERROR: $SRC_SHARDS not found (run from hpc/detsvsL on hep-submit)" >&2; exit 2; }
  for S in $SEEDS_L4; do
    [ -e "$SRC_SHARDS/bare_L4d3_A0_s${S}.core.npz" ] || { echo "ERROR: no 512k checkpoint for bare_L4d3_A0_s${S}" >&2; exit 2; }
  done
  mkdir -p "$SRC_SHARDS/pre_extend_512k"
  for S in $SEEDS_L4; do
    f="$SRC_SHARDS/bare_L4d3_A0_s${S}.json"
    [ -e "$SRC_SHARDS/pre_extend_512k/$(basename "$f")" ] || cp -p "$f" "$SRC_SHARDS/pre_extend_512k/"
  done
  G="campaign_${BASE}/bareL4.txt"; : > "$G"
  for S in $SEEDS_L4; do
    # 294246's bare-ref settings verbatim (32 starts, select 16k, stratified) + the deeper ladder
    row "${SRC_CAMPAIGN}-nb${NB}" bare "$NB" 4 0 "$S" 2048000 1024000 12 384 24 172800 $(( 90 - S )) 8 32 1 >> "$G"
  done
  submit_grid bareL4 "$G" 960
}

arm_bareL5() {
  G="campaign_${BASE}/bareL5.txt"; : > "$G"
  for S in $SEEDS_BL5; do
    # 293963's bare L=5 sizing verbatim: 128k, PT2 65536, 32 starts, 4 select workers, 256 GB
    row "${BASE}-nb${NB}" bare "$NB" 5 0 "$S" 128000 65536 11 256 16 43200 $(( 80 - S )) 4 32 0 >> "$G"
  done
  submit_grid bareL5 "$G" 768
}

arm_sqL5() {
  G="campaign_${BASE}/sqL5.txt"; : > "$G"
  for S in $SEEDS_SQ5; do
    row "${BASE}-nb${NB}" gaussian "$NB" 5 0 "$S" 1024000 512000 11 384 24 86400 $(( 85 - S )) 8 16 0 >> "$G"
  done
  submit_grid sqL5 "$G" 768
}

mkdir -p "campaign_${BASE}/logs"
case "$MODE" in
  all)    arm_bareL4; arm_bareL5; arm_sqL5 ;;
  bareL4) arm_bareL4 ;;
  bareL5) arm_bareL5 ;;
  sqL5)   arm_sqL5 ;;
  *) echo "usage: sh submit_vacuum_calibration.sh {all|bareL4|bareL5|sqL5}" >&2; exit 2 ;;
esac
echo
echo "Fresh L=5 shards: hpc/detsvsL/campaign_${BASE}-nb${NB}/shards/   bare L=4: ${SRC_SHARDS}/"
echo "After they land: python -m misc.build_family_calibration --data-root data/classical  (gate-5 registry)"
exit 0
