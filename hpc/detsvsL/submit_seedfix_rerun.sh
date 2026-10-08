#!/bin/sh
# Submitter for the SEED-SHARING CLOSE-OUT re-runs (2026-10-08).
#
# WHY. Before 846e9e7, shard seed s drew its Phase-0 starts from seeds s, s+1, ..., s+n-1, so
# neighbouring seeds shared all but one or two starts and usually kept the same winner: a 3- or
# 5-seed "robustness check" was one search reported several times. Three legacy datasets the paper
# still cites carry that defect (results/02_classical_baseline/search_v1_seed_dependence_note.md):
#   bare_baseline_nb3_292477   filling 1.0, n_b=3, L=2..5          (claims C1/C2)  3 of 4 cells
#   nb_energy_gate(+_L3)       n_b={2,3,4} x A={0,1,full}, L=2/3   (n_b gate)      13 of 15 cells
#   nb_volscaling(+_deep)      L={3,4} x n_b={3,4} x A={0,1}       (volume scaling) 9 of 10 cells
# This is NOT a seed repair: the re-run uses the CURRENT production search (stride 1000 + select
# core 16000 + stratified starts, the levers 293963 validated), so central values may move at
# L>=3 (bounds came down 7.8-9.5 MeV/site at L=2 for A=4, 10). Seed 0 is identical under either
# seed numbering but NOT under the new levers, so SEEDS="0 1 2" re-runs the full set.
#
# Everything else matches each legacy campaign: ladder depth (N_RUNGS), PT2 cap, Phase-0 start
# count (32 for the baseline = the sweep; 64 for the gate/volscaling arms = their own runs),
# MAXCORE, wall caps. Shards land in campaign_<BASE>-<arm>-nb<NB>/shards/ with the legacy file
# names (bare_L<L>d3_f1.0_s<seed>.json / bare_L<L>d3_A<A>_s<seed>.json) and carry the
# phase0_seed_stride key the audit keys on.
#
#   sh submit_seedfix_rerun.sh test          # one gate shard (L=2 n_b=3 A=1 seed 1): minutes
#   sh submit_seedfix_rerun.sh baseline      # 292477 seeds 1,2 x L=2..5           (8 shards)
#   sh submit_seedfix_rerun.sh gate          # L=2 nb{2,3,4} + L=3 nb{2,3}, seeds 1-4 (60 shards)
#   sh submit_seedfix_rerun.sh volscaling    # primary @256k + deep @512k, seeds 1,2 (20 shards)
#   sh submit_seedfix_rerun.sh all           # the three arms
#   SEEDS="0 1 2" sh submit_seedfix_rerun.sh baseline      # include seed 0 under the new search
#   GATE_SEEDS="1 2" VOL_SEEDS="1" ...                      # trims
#   NB2=1 sh submit_seedfix_rerun.sh baseline               # also the paired n_b=2 arm (292478, seed 0, L=2..4)
#
# Run from $REPO/hpc/detsvsL/ on `ssh hep-submit` after the launch-approval loop (HPC_WORKFLOW).
# Same automatic memory/provisioning recovery and per-rung resume as the A-sweep submits.
# Priorities sit BELOW the squeeze campaign (10-45) so these ride along in spare slots.
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-seedfix"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"
SEEDS="${SEEDS:-1 2}"
GATE_SEEDS="${GATE_SEEDS:-1 2 3 4}"
VOL_SEEDS="${VOL_SEEDS:-1 2}"
NB2="${NB2:-0}"
MEMCAP_GB="${MEMCAP_GB:-768}"
MAXHOLDS="${MAXHOLDS:-4}"

# the validated search (293963): disjoint Phase-0 seed blocks, select-at-16k, stratified starts
LEVERS='NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified'
ARGS='$(L) $(SEED) '"${BASE}"'-$(ARM)-nb$(NB) bare $(A) $(FILL) $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=$(NRUNGS) NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=$(P0RUNS) NUQU_PHASE0_WORKERS=$(P0W) '"${LEVERS}"
VARS="ARM,NB,L,A,FILL,SEED,MAXCORE,NRUNGS,PT2CAP,P0RUNS,P0W,MEM,CPUS,MAXRUNGSEC,PRIO"

row() { printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' "$@"; }

submit_grid() {   # $1=arm  $2=grid file
  arm="$1"; grid="$2"
  cat > "campaign_${BASE}/${arm}.sub" <<SUB
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
request_disk            = 10G
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/\$(ARM)_nb\$(NB)_L\$(L)_A\$(A)\$(FILL)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/\$(ARM)_nb\$(NB)_L\$(L)_A\$(A)\$(FILL)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue ${VARS} from ${grid}
SUB
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm}  CAMPAIGN=${BASE}  jobs=$(wc -l < "$grid" | tr -d ' ')"
}

# ---- baseline (292477): filling 1.0, PT2 on every rung, 11 rungs from 1000 ----------------
# L -> "MAXCORE PT2CAP MEM(GB) CPUS MAXRUNGSEC P0W"  (292477's caps; the sweep's memory + workers)
baseline_sizing() {
  case "$1" in
    2) echo "1024000 1024000 64  8  14400 4" ;;
    3) echo "1024000 512000  192 16 21600 8" ;;
    4) echo "512000  256000  384 16 21600 8" ;;
    5) echo "128000  65536   384 16 21600 8" ;;
    *) echo "ERROR unknown L=$1" >&2; exit 1 ;;
  esac
}
baseline_rows() {   # $1=grid
  for L in 2 3 4 5; do
    set -- $(baseline_sizing "$L"); mc=$1; pc=$2; mem=$3; cpus=$4; mrs=$5; p0w=$6
    for S in $SEEDS; do row baseline 3 "$L" 1 1.0 "$S" "$mc" 11 "$pc" 32 "$p0w" "$mem" "$cpus" "$mrs" 8 >> "$G"; done
  done
  if [ "$NB2" = "1" ]; then
    for L in 2 3 4; do
      set -- $(baseline_sizing "$L"); mc=$1; pc=$2; mem=$3; cpus=$4; mrs=$5; p0w=$6
      row baseline 2 "$L" 1 1.0 0 "$mc" 11 "$pc" 32 "$p0w" "$mem" "$cpus" "$mrs" 7 >> "$G"
    done
  fi
}

# ---- gate (nb_energy_gate + _L3): no PT2, 64 starts, 8 rungs at L=2 / 9 at L=3 -----------
gate_rows() {
  for NB in 2 3 4; do
    case "$NB" in 4) mc=131072 ;; *) mc=262144 ;; esac
    for A in 0 1 32; do
      for S in $GATE_SEEDS; do row gate "$NB" 2 "$A" none "$S" "$mc" 8 1 64 8 16 16 14400 6 >> "$G"; done
    done
  done
  for NB in 2 3; do
    case "$NB" in 2) mem=32 ;; *) mem=48 ;; esac
    for A in 0 1 27; do
      for S in $GATE_SEEDS; do row gate "$NB" 3 "$A" none "$S" 262144 9 1 64 8 "$mem" 16 21600 6 >> "$G"; done
    done
  done
}

# ---- volscaling (primary @256k, 9 rungs) + deep (@512k, 10 rungs, A=1) -------------------
vol_rows() {
  for S in $VOL_SEEDS; do
    for A in 0 1; do
      row volscaling 3 4 "$A" none "$S" 262144 9 1 64 8 128 16 21600 5 >> "$G"
      row volscaling 4 3 "$A" none "$S" 262144 9 1 64 8 48  16 21600 5 >> "$G"
      row volscaling 4 4 "$A" none "$S" 262144 9 1 64 8 96  16 21600 5 >> "$G"
    done
    row voldeep 3 3 1 none "$S" 524288 10 1 64 8 96  16 21600 4 >> "$G"
    row voldeep 4 3 1 none "$S" 524288 10 1 64 8 96  16 21600 4 >> "$G"
    row voldeep 3 4 1 none "$S" 524288 10 1 64 8 192 16 21600 4 >> "$G"
    row voldeep 4 4 1 none "$S" 524288 10 1 64 8 256 16 21600 4 >> "$G"
  done
}

mkdir -p "campaign_${BASE}/logs"
case "$MODE" in
  test)
    G="campaign_${BASE}/smoke.txt"; : > "$G"
    row gate 3 2 1 none 1 262144 8 1 64 8 16 16 14400 9 >> "$G"
    submit_grid smoke "$G"
    echo "SMOKE: L=2 n_b=3 A=1 seed 1 under the new levers; legacy value 298.416 MeV/site at 128k"
    echo "  (shards: campaign_${BASE}-gate-nb3/shards/bare_L2d3_A1_s1.json; expect phase0_seed_stride=1000)" ;;
  baseline|gate|volscaling|all)
    if [ "$MODE" = baseline ] || [ "$MODE" = all ]; then
      G="campaign_${BASE}/baseline.txt"; : > "$G"; baseline_rows; submit_grid baseline "$G"; fi
    if [ "$MODE" = gate ] || [ "$MODE" = all ]; then
      G="campaign_${BASE}/gate.txt"; : > "$G"; gate_rows; submit_grid gate "$G"; fi
    if [ "$MODE" = volscaling ] || [ "$MODE" = all ]; then
      G="campaign_${BASE}/volscaling.txt"; : > "$G"; vol_rows; submit_grid volscaling "$G"; fi ;;
  *) echo "usage: sh submit_seedfix_rerun.sh {test|baseline|gate|volscaling|all}" >&2; exit 1 ;;
esac
echo
echo "pull (per arm, per n_b), e.g.:"
echo "  rsync -az --exclude '*.npz' hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_${BASE}-baseline-nb3/shards/ data/classical/\$(date +%F)/bare_baseline_nb3_<cluster>/"
echo "  rsync -az --exclude '*.npz' hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_${BASE}-gate-nb<NB>/shards/ data/classical/\$(date +%F)/nb_energy_gate_<cluster>/nb<NB>/"
echo "audit: python -m misc.analyze_search_levers --seed-audit <dirs> --out-dir <out>   (expect 0 collapsed cells)"
