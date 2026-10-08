#!/bin/sh
# FIXED-A n_b=3 SWEEP IN THE GAUSSIAN-SQUEEZE FRAME + A=0/1 REFERENCES (2026-10-08).
#
# WHY. The bare fixed-A sweep (293963, `submit_nb3_Asweep.sh`) finished with ladders that are
# pre-asymptotic at L>=3: the variational energies decay as N^-0.08 at L=3 and show no
# curvature at L=4, so COO's power law cannot resolve a limit, and its gap to the PT2-linear
# intercept is 7-10 MeV/site at L=3. The analytic per-mode squeeze compacted the ground state
# 64x (L=2) to 256x (L=3) in the frame study, so the same search in the squeezed frame should
# put L=3-4 in the regime the TrimCI/COO error-bar conventions assume. The A=0 (pion vacuum)
# and A=1 (dressed nucleon) sectors are the references BE(A) = A E(1) - (A-1) E(0) - E(A) needs;
# the error in E(1) is multiplied by A, so they are run at the sweep's own settings.
#
# ARMS (one campaign directory; the shard file prefix is the frame: gaussian_* / bare_*)
#   squeeze  gaussian frame (analytic per-mode squeeze, fixed r*, `optimize_frame`), A=0..10.
#            The squeezed truncated H is a different operator from the truncated bare H at
#            finite N_f; both are variational bounds on the untruncated H (operator identity,
#            no back-evaluation needed). Compare frames as a reported shift, not as one series.
#   refs     bare frame, A=0 and A=1, at 293963's exact settings (32 starts), so the bare
#            A=2..10 sweep gets its own binding-energy references.
#
# WHAT'S THE SAME as 293963: dim=3, n_b=3, warm-grow deep solve, ladder 1000 x 2^k to the per-L
# MAXCORE, PT2 to one doubling below the top, seeds {0,1,2} with disjoint Phase-0 blocks
# (stride 1000), select core 16000 + stratified starts, the per-L memory that finally worked.
# WHAT'S NEW: the number of Phase-0 starts grows with L in the squeeze arm,
#   P0RUNS = 16 L  ->  32 (L=2), 48 (L=3), 64 (L=4); L=5 capped at 64.
# The arrangement TYPES (site partitions of A, <= 4 per site) number 1-23 for A <= 10 and barely
# depend on L once sites >= A; what grows with L is the number of distinct PLACEMENTS (which
# sites, how far apart). 16 L keeps >= 2 starts per type at L=3 for every A <= 10 and ~3 at
# L=4, and costs linearly: the select stage is the only part that scales with the start count
# (293963 select: 0.4-1.2 h at L=3 and 1.3-1.7 h at L=4 with 32 starts / 8 workers).
# Each start's arrangement is now recorded per shard (`rungs[0].phase0_select_starts`).
# The bare refs keep 32 so they match 293963. NOTE: at L>=3 the squeeze arm therefore differs
# from the bare sweep in TWO ways (frame and start count); L=2 is the clean frame comparison.
#
# GRID (defaults): squeeze A=0..10 x L=2..4 x 3 seeds = 99, refs A=0,1 x L=2..4 x 3 = 18.
# Trim / extend with env: AS="0 1 2 4 6 8 10"  LS="2 3 4 5"  SEEDS="0"  REF_LS="2 3".
#
# COST (bare-frame wall-clocks as the guide; the squeezed H has ~0.65x the bare term count):
# L=2 ~3-5 h/4c, L=3 ~6-8 h/8c, L=4 ~16-22 h/8c (+ ~1.5 h for the larger select stage);
# L=5 (opt-in) ~25-30 h/8c at 384G. Memory, not cores, caps concurrency at L=4 (2 per qis node).
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit), after the push:
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_nb3_Asweep_squeeze.sh test     # 1 squeeze shard: L=4 A=10 s0 to 64k, 64 starts
#     sh submit_nb3_Asweep_squeeze.sh oomtest  # 1 undersized L=2 shard: exercises auto-OOM recovery + resume
#     sh submit_nb3_Asweep_squeeze.sh all      # squeeze + refs (117 shards)
#     sh submit_nb3_Asweep_squeeze.sh squeeze  # squeeze arm only
#     sh submit_nb3_Asweep_squeeze.sh refs     # bare A=0/1 references only
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-nb3AsweepSq"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"
AS="${AS:-0 1 2 3 4 5 6 7 8 9 10}"
LS="${LS:-2 3 4}"
REF_LS="${REF_LS:-2 3 4}"
SEEDS="${SEEDS:-0 1 2}"

# args to run_frame_shard.sh: L seed campaign frame A filling max_core ladder_mode
#                             frame_runs orbopt_cycles phase0_core max_rung_seconds
ARGS='$(L) $(SEED) '"${BASE}"'-nb$(NB) $(FRAME) $(A) none $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=$(P0RUNS) NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified NUQU_PHASE0_WORKERS=$(P0W)'
VARS="FRAME,NB,L,A,SEED,MAXCORE,PT2CAP,MEM,CPUS,MAXRUNGSEC,PRIO,P0W,P0RUNS"

# AUTOMATIC OUT-OF-MEMORY RECOVERY (2026-10-08, infrastructure C2; HTCondor 25.0 on hep-submit).
# 293963 took three manual rounds of condor_qedit RequestMemory + condor_release because the
# top-rung spike sits 100-250 GB above Condor's last 5-minute sample. Now a memory hold
# (HoldReasonCode 34, cgroup limit) re-queues itself with more memory and no human in the loop:
#   request_memory   = MEM GB x (1 + NumHolds), capped at MEMCAP_GB (qis nodes have ~1 TB);
#   periodic_release = the job releases itself after a memory hold, up to MAXHOLDS holds,
#                      then it stays held for a person to look at;
#   on_exit_remove   = a provisioning failure (run_frame_shard.sh exit 3: uv / python /
#                      wheels unreachable) is re-queued, up to 3 starts; everything else leaves.
# Every restart RESUMES from the last checkpointed rung (C1), so a hold costs one rung.
# NumHolds, not NumSystemHolds: in 293963's history the cgroup holds left NumSystemHolds at 0.
MEMCAP_GB="${MEMCAP_GB:-768}"
MAXHOLDS="${MAXHOLDS:-4}"

row() { printf '%s %s %s %s %s %s %s %s %s %s %s %s %s\n' "$@"; }
# L -> "MAXCORE PT2CAP MEM(GB) CPUS MAXRUNGSEC PRIO P0W"  (293963's final, working sizing; L=5 gets
# 8 select workers at 384G so the 64-start select stage stays ~7-10 h)
sizing_for_L() {
  case "$1" in
    2) echo "1024000 1024000 32  4 14400 40 4" ;;
    3) echo "1024000 512000  128 8 21600 30 8" ;;
    4) echo "512000  256000  384 8 21600 20 8" ;;
    5) echo "128000  65536   384 8 21600 10 8" ;;
    *) echo "ERROR unknown L=$1" >&2; exit 1 ;;
  esac
}
# Phase-0 starts: 16 L for the squeeze arm, capped at 64; 32 for the bare refs (= 293963)
p0runs_squeeze() { n=$(( 16 * $1 )); [ "$n" -gt 64 ] && n=64; echo "$n"; }

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
request_disk            = 2560M
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/${arm}_\$(FRAME)_L\$(L)_A\$(A)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/${arm}_\$(FRAME)_L\$(L)_A\$(A)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue ${VARS} from ${grid}
EOF
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm}  CAMPAIGN=${BASE}  jobs=$(wc -l < "$grid")"
}

add_rows() {   # $1=frame  $2=list of L  $3=list of A  $4=starts mode (squeeze|ref)  $5=grid
  fr="$1"; Ls="$2"; As="$3"; sm="$4"; g="$5"
  for LL in $Ls; do
    set -- $(sizing_for_L "$LL")     # MAXCORE PT2CAP MEM CPUS MAXRUNGSEC PRIO P0W
    if [ "$sm" = "squeeze" ]; then nr=$(p0runs_squeeze "$LL"); bump=0; else nr=32; bump=5; fi
    for A in $As; do
      # seed-major priority: seed 0 everywhere first; refs a notch above the sweep at equal L
      for S in $SEEDS; do
        row "$fr" 3 "$LL" "$A" "$S" "$1" "$2" "$3" "$4" "$5" $(( $6 + 100 * (2 - S) + bump )) "$7" "$nr" >> "$g"
      done
    done
  done
}

mkdir -p "campaign_${BASE}/logs"

if [ "$MODE" = "test" ]; then
  # one squeeze shard at the largest default start count: L=4 A=10 s0, 64 starts, to 64k with
  # PT2 every rung. Measures the squeezed select-stage cost and per-rung memory before the grid.
  G="campaign_${BASE}/smoke.txt"
  row gaussian 3 4 10 0 64000 64000 192 8 21600 50 8 64 > "$G"
  submit_grid smoke "$G"
  echo "SMOKE: expect ~4-6 h. Check: ExitCode 0; rung 0 '0-select' with 64 trails and 64"
  echo "  phase0_select_starts; frame 'gaussian'; select wall; MemoryUsage per rung."
  exit 0
fi

if [ "$MODE" = "oomtest" ]; then
  # the auto-recovery smoke: a cheap L=2 A=4 s0 shard to 16k, deliberately undersized at
  # OOM_MEM GB (default 1), so it should be held on memory, release itself with 2x, 3x, ...
  # and finish from its checkpoint. Check `condor_q -l <id> -af NumHolds RequestMemory` while
  # it runs and, when done, the shard JSON's "resumed" list and per-rung "mem".
  G="campaign_${BASE}/oomtest.txt"
  row gaussian 3 2 4 0 16000 16000 "${OOM_MEM:-1}" 4 7200 50 4 32 > "$G"
  submit_grid oomtest "$G"
  echo "OOMTEST: expect NumHolds >= 1, RequestMemory doubling, then ExitCode 0 within ~1 h."
  exit 0
fi

G="campaign_${BASE}/grid.txt"; : > "$G"
case "$MODE" in
  all)     add_rows gaussian "$LS" "$AS" squeeze "$G"; add_rows bare "$REF_LS" "0 1" ref "$G" ;;
  squeeze) add_rows gaussian "$LS" "$AS" squeeze "$G" ;;
  refs)    add_rows bare "$REF_LS" "0 1" ref "$G" ;;
  *) echo "usage: sh submit_nb3_Asweep_squeeze.sh {test|oomtest|all|squeeze|refs}" >&2; exit 2 ;;
esac
submit_grid Asweep "$G"
echo "  -> squeeze A={${AS}} x L={${LS}} and/or bare refs A={0 1} x L={${REF_LS}}, seeds {${SEEDS}}"
echo
echo "Retrieve:  rsync -az --exclude '*.npz' hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_${BASE}-nb3/shards/ \\"
echo "               data/classical/\$(date +%F)/Asweep_squeeze_nb3_<cluster>/"
exit 0
