#!/bin/sh
# DEEPER SQUEEZED L=4 LADDERS by EXTENSION (2026-11-22): 512k -> 1,024,000 dets, PT2 cap 256k -> 512k.
#
# WHY. In the squeeze A-sweep (cluster 294246) the L=4 ladders stop at 512k with PT2 to 256k, and
# 8 of 11 cells come out bound-only: a smooth 0.3 MeV/site step around 64k-128k leaves 2-3 PT2
# rungs to fit. One more doubling, with PT2 on the 512k rung, gives the fit its points and the
# squeezed L=4 family its first calibration against the exact pion vacuum (A=0: 344.596 MeV/site;
# the 512k bound is 345.03). This demonstrates the squeezed frame at the largest lattice we run.
#
# HOW: EXTEND, not re-run. Each 294246 shard checkpointed its finished 512k core beside its JSON
# (`<stem>.core.npz`, 33 of them on /nfs_scratch). `run_frame_shard.py --resume --extend`
# (NUQU_EXTEND=1) reloads that core, RE-RECORDS the 512k rung under the new PT2 cap (PT2 depends
# only on the core's determinants), and grows the 1,024k rung with seed + i -- bit-identical to a
# fresh run of the deeper ladder (tests/test_extend_ladder.py; caveat: bit-identity needs single-
# threaded math, multithreaded OpenBLAS/OpenMP sums can differ in the last bits). It skips the
# select stage and the first 9 rungs: ~10-14 h saved per shard. Anything other than ladder depth
# and PT2 cap must match the original run, or the shard refuses (exit 2) instead of mixing runs,
# so ENVSTR below is 294246's verbatim apart from NUQU_PT2_MAX_CORE / MAXCORE / NUQU_EXTEND.
#
# The extended shards are written IN PLACE in 294246's campaign directory (that is where their
# checkpoints are). The 512k JSONs are copied to shards/pre_extend_512k/ first, so the as-run
# 294246 record survives on the cluster as well as in data/classical/2026-10-08/.
#
# PARALLELISM. 24 cpus per job (319703 was launched at 32 and qedited to 24 before matching). With no select stage, the work is the 1,024k grow rung (OpenMP
# connection sweep + matvec, deep-solve: OMP threads = cpus) and PT2 at 512k (OpenMP). Scaling is
# sub-linear (293962: 4->16 threads 1.3-1.4x; the candidate-score reduction is serial), but memory,
# not cores, limits how many run at once: 3 x 256 GB per ~1 TB qis node, and qis1 had only 87
# free cores (3 x 24 fits, 3 x 32 does not). Condor DISK limits it too: see request_disk below.
#
# MEMORY. From 294246's per-rung records: the 512k rung peaked at 100-116 GB, PT2 at 256k had
# 50-65M external dets (~700 B each). So PT2 at 512k is ~85 GB above the solve and the 1,024k
# grow rung ~220-250 GB. Base 256 GB with the automatic memory policy (request x (1 + holds),
# cap MEMCAP_GB) and resume: a hold costs one rung, never the shard.
#
# COST. ~2-6 h for PT2 at 512k + ~5-10 h for the 1,024k rung at 32 cpus; 33 shards at 9 at a time
# -> roughly 1-1.5 days wall.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit), after the push:
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout <branch> && git reset --hard origin/<branch>
#     cd hpc/detsvsL
#     sh submit_squeeze_L4_deep.sh test   # extends ONE production shard (A=10 s0): exit 0, rungs to
#                                        # 1,024k, rung 9 re-recorded with PT2, "extended" in JSON
#     sh submit_squeeze_L4_deep.sh all    # the rest (a shard already extended exits 0 at once)
set -eu
MODE="${1:-all}"
SRC_CAMPAIGN="${SRC_CAMPAIGN:-20261008-184336-nb3AsweepSq}"     # 294246, whose L=4 shards we extend
BASE="$(date +%Y%m%d-%H%M%S)-sqL4deep"                           # this submission's logs only
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"
AS="${AS:-0 1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-0 1 2}"
NB=3
L=4
MAXCORE=1024000
PT2CAP=512000
MEM="${MEM:-256}"                 # GB, base of the auto-growing request
CPUS="${CPUS:-24}"
MAXRUNGSEC="${MAXRUNGSEC:-86400}" # the 1,024k rung is the last one; the budget only stops growth
P0RUNS=64                         # 294246's L=4 start count -- must match the checkpoint
P0W=8                             # ignored by the checkpoint key; no select stage on extension
MEMCAP_GB="${MEMCAP_GB:-768}"
MAXHOLDS="${MAXHOLDS:-4}"

SHARDS="campaign_${SRC_CAMPAIGN}-nb${NB}/shards"
[ -d "$SHARDS" ] || { echo "ERROR: $SHARDS not found (run from hpc/detsvsL on hep-submit)" >&2; exit 2; }

# 294246's runner arguments with the DEEPER max_core; the campaign name points at its shards
ARGS='$(L) $(SEED) '"${SRC_CAMPAIGN}"'-nb$(NB) gaussian $(A) none $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=$(P0RUNS) NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified NUQU_PHASE0_WORKERS=$(P0W) NUQU_EXTEND=1'
VARS="NB,L,A,SEED,MAXCORE,PT2CAP,MEM,CPUS,MAXRUNGSEC,PRIO,P0W,P0RUNS"

row() { printf '%s %s %s %s %s %s %s %s %s %s %s %s\n' "$@"; }

submit_grid() {   # $1=armname  $2=grid file
  arm="$1"; grid="$2"
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
MEMCAP_MB               = $(( MEMCAP_GB * 1024 ))
NHOLDS                  = ifThenElse(isUndefined(NumHolds), 0, NumHolds)
MEMGROW_MB              = (\$(MEM) * 1024 * (1 + \$(NHOLDS)))
request_memory          = ifThenElse(\$(MEMGROW_MB) < \$(MEMCAP_MB), \$(MEMGROW_MB), \$(MEMCAP_MB))
periodic_release        = (HoldReasonCode == 34) && (NumHolds < ${MAXHOLDS})
on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))
request_disk            = 300M
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/${arm}_L\$(L)_A\$(A)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/${arm}_L\$(L)_A\$(A)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue ${VARS} from ${grid}
EOF
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm}  LOGS=campaign_${BASE}  SHARDS=${SHARDS}  jobs=$(wc -l < "$grid")"
}

backup() {   # keep 294246's as-run 512k JSONs (small; the cores stay where they are)
  mkdir -p "$SHARDS/pre_extend_512k"
  for f in "$SHARDS"/gaussian_L4d3_*.json; do
    [ -e "$SHARDS/pre_extend_512k/$(basename "$f")" ] || cp -p "$f" "$SHARDS/pre_extend_512k/"
  done
  echo "backup: $(ls "$SHARDS/pre_extend_512k" | wc -l) as-run L=4 JSONs in $SHARDS/pre_extend_512k/"
}

missing=""
for A in $AS; do for S in $SEEDS; do
  [ -e "$SHARDS/gaussian_L4d3_A${A}_s${S}.core.npz" ] || missing="$missing A${A}_s${S}"
done; done
[ -z "$missing" ] || { echo "ERROR: no 512k checkpoint for:$missing" >&2; exit 2; }

mkdir -p "campaign_${BASE}/logs"
backup

G="campaign_${BASE}/grid.txt"; : > "$G"
if [ "$MODE" = "test" ]; then
  row "$NB" "$L" 10 0 "$MAXCORE" "$PT2CAP" "$MEM" "$CPUS" "$MAXRUNGSEC" 60 "$P0W" "$P0RUNS" > "$G"
  submit_grid smoke "$G"
  echo "SMOKE: check ExitCode 0; '[frameshard] EXTEND' in the .out; the JSON's rungs end at 1024000,"
  echo "  rung index 9 has dE_pt2 and rerecorded_from_checkpoint; 'extended' has one entry."
  exit 0
fi
[ "$MODE" = "all" ] || { echo "usage: sh submit_squeeze_L4_deep.sh {test|all}" >&2; exit 2; }
for S in $SEEDS; do for A in $AS; do
  # seed 0 first, and A=0 (the exact-vacuum calibration) first within a seed
  pr=$(( 50 + 100 * (2 - S) - A ))
  row "$NB" "$L" "$A" "$S" "$MAXCORE" "$PT2CAP" "$MEM" "$CPUS" "$MAXRUNGSEC" "$pr" "$P0W" "$P0RUNS" >> "$G"
done; done
submit_grid deep "$G"
echo "  -> squeeze L=4 A={${AS}} x seeds {${SEEDS}}: 512k -> ${MAXCORE}, PT2 cap ${PT2CAP}, ${CPUS} cpus, ${MEM} GB base"
echo
echo "Retrieve:  rsync -az --exclude '*.npz' --exclude 'pre_extend_512k' \\"
echo "             hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/${SHARDS}/ data/classical/\$(date +%F)/squeeze_L4_deep_<cluster>/"
exit 0
