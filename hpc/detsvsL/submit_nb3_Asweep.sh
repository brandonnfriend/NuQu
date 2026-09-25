#!/bin/sh
# CLASSICAL n_b=3 BASELINE vs NUCLEON NUMBER — A = 2..10 at L = 2..5 (2026-09-23).
#
# WHY. The headline classical baseline (292477, `bare_baseline_nb3_292477`) runs at filling 1.0,
# i.e. A = L^3 — a different nucleon number at every L. This campaign holds A FIXED across the
# volume ladder so the "classical estimates lose precision with volume" figure
# (docs/presentation/classical_baseline.png) can be drawn per A, for A = 2..10.
#
# WHAT'S THE SAME as 292477 (so the two datasets are one pipeline): bare TrimCI, dim=3, n_b=3
# (N_f=8), warm-grow deep solve, ladder_start=1000, 11 geometric rungs, NUQU_PHASE0_RUNS=32,
# PT2 on every rung up to one doubling below the ladder top, seeds {0,1,2}, the same per-L
# MAXCORE / PT2CAP / CPUS / MAXRUNGSEC.
#
# WHAT'S DIFFERENT:
#   1. Explicit A (`filling none`) -> shards are `bare_L<L>d3_A<A>_s<seed>.json`.
#   2. The Hamiltonian is the post-2026-09-14 WICK-ORDERED build (the code default), and each
#      shard now records `contact_convention: "wick"`. 292477 ran legacy and is untagged; the
#      loaders (`apply_wick_correction.shard_convention`) put both on one convention, so the
#      L=2 A=8 point below can be reused rather than re-run.
#   3. Memory trimmed: A <= 10 is far below 292477's A = L^3 at L >= 3, and PT2 dominates memory
#      (~650-700 B/ext). Measured locally at L=2, core 4000: n_ext(A=2) = 0.51x, n_ext(A=10) =
#      1.08x the A=8 value, so the external space is boson-dominated and grows sub-linearly in A.
#   4. request_disk 2560M (HPC_WORKFLOW s6b: 10G strands jobs off qis1/qis3).
#   5. JobPrio seed-major, then L: every seed-0 shard (cheap volumes first) is scheduled before
#      any seed 1, so a partial campaign still has one independent ladder at every (L, A).
#   6. (2026-09-24, from the L=2 pass 293959) INDEPENDENT SEEDS. The warm-grow Phase-0 ensemble
#      drew seeds s..s+31, so shards 0/1/2 shared 30 of 32 inits and kept the same best one:
#      every 293959 L=2 ladder is bit-identical across seeds (so are 292477's at L=3). Now
#      NUQU_PHASE0_SEED_STRIDE=1000 -> shard s uses inits 1000*s + k (disjoint). Seed 0 is
#      unchanged, so the 293959 / smoke seed-0 shards stay valid; their s1/s2 are superseded.
#   7. (2026-09-24) request_cpus 4, not 16/24. Measured 1.4-1.5 cores busy on average (smoke:
#      6714 CPU-s / 4698 s wall at 24c; L=2: 93 CPU-h used of 1014 allocated at 16c). Under
#      NUQU_DEEP_SOLVE only the SpMV is OpenMP-threaded (~3% of wall), so 4 threads cost
#      ~10% wall at most and free ~4x the slots.
#
# EXISTING DATA CHECKED (2026-09-23). The nb_nested shards (L=2 A=2,4,8; L=3,4 A=8) and the DMRG
# A=2 shards carry no PT2, so they cannot give the extrapolated E_inf +- sigma. 292477's L=2 A=8
# and 293959 (L=2, old search, seeds that were copies of each other) are superseded by this run.
#
#   8. (2026-09-25, from the lever grid 293961) SEARCH LEVERS ON: NUQU_PHASE0_SELECT_CORE=16000
#      (grow all 32 Phase-0 inits to 16k, keep the lowest there) + NUQU_PHASE0_INIT=stratified
#      (inits cycle evenly over nucleon-arrangement types). At L=2, A in {2,4,5,6,9} x 3
#      independent seeds, the old search put 5/15 runs on the best basin (seed spread 3-12
#      MeV/site); select 14/15, stratified+select 13/15; the "novel" (more hopping) lever did
#      nothing (5/15) at 2x cost and stays OFF. The levers found bounds at 128k below 293959's
#      1M ones (A=4: 276.9 vs 286.1). So L=2 is RELAUNCHED for every A, and A=8 is no longer
#      reused from 292477 (old search). Select workers = the job's cpus (NUQU_PHASE0_WORKERS),
#      each forced to 1 OpenMP thread (mixed_ci.set_num_threads) so deep-solve does not
#      oversubscribe.
#   9. (2026-09-25, 293962) CPUS resolved from the real allocation (detect_cpus). Deep rungs
#      scale weakly: 4->16 threads = 1.27x at L=2, 1.32-1.43x at L=3, so L>=3 asks 8 (memory,
#      not cores, caps concurrency there) and L=2 asks 4 (qis gives 4 minimum anyway).
#
# GRID: A in {2..10} x L in {2..5} x seed in {0,1,2} = 108 shards.
# Trim without editing (env overrides): AS="2 4 6 8 10"  LS="2 3 4"  SEEDS="0".
# (sigma is a per-seed quantity; the seed spread is a robustness check, so SEEDS="0" still
# yields E_inf +- sigma — at a third of the cost.)
#
# COST. Earlier wall-clocks (292477, 293959) ran at 2 threads; 4->8 threads buys ~1.2x. Per shard,
# incl. the select stage (32 inits to 16k across the job's cpus): L=2 ~3-5 h/4c, L=3 ~5-6 h/8c,
# L=4 ~12-14 h/8c, L=5 ~9-10 h/8c  ->  ~6.6k CPU-h allocated for 108 shards. Memory caps
# concurrency at L>=3 (~7 x 128-144G per qis node): ~2-3 days wall on qis1+qis2.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_nb3_Asweep.sh test   # 1 shard: L=4 A=10 s0 to 32k, PT2 every rung (n_ext check)
#     sh submit_nb3_Asweep.sh all    # the grid
#     LS=2 sh submit_nb3_Asweep.sh all              # L=2 relaunch alone (27 shards)
#     LS="3 4 5" sh submit_nb3_Asweep.sh all        # the L>=3 grid alone (81 shards)
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-nb3Asweep"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"
AS="${AS:-2 3 4 5 6 7 8 9 10}"
LS="${LS:-2 3 4 5}"
SEEDS="${SEEDS:-0 1 2}"

# args to run_frame_shard.sh: L seed campaign frame A filling max_core ladder_mode
#                             frame_runs orbopt_cycles phase0_core max_rung_seconds
ARGS='$(L) $(SEED) '"${BASE}"'-nb$(NB) bare $(A) none $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=32 NUQU_PHASE0_SEED_STRIDE=1000 NUQU_PHASE0_SELECT_CORE=16000 NUQU_PHASE0_INIT=stratified NUQU_PHASE0_WORKERS=$(CPUS)'

row() { printf '%s %s %s %s %s %s %s %s %s %s\n' "$@"; }
# L -> "MAXCORE PT2CAP MEM CPUS MAXRUNGSEC PRIO"   (MAXCORE/PT2CAP/MAXRUNGSEC = 292477)
#   292477 peaks at A=L^3: L=2 14-20 GB; L=4 PT2@256k 146 GB; L=3/L=5 ~118/106 GB + solve.
#   Measured at A=10: L=3 n_ext 15.9M@64k (0.71x A=27) -> PT2@512k ~125M ext ~105 GB;
#   L=4 19.2M@32k -> PT2@256k ~130M ~110 GB (292477 peaked 146 GB at A=64); L=5 is < A=125's
#   155M@64k by the same sub-linear A factor. ~830 B/ext incl. solve (292477 peak/n_ext).
#   An OOM is recoverable: condor_qedit RequestMemory + condor_release, the per-rung save keeps
#   every finished rung (HPC_WORKFLOW s6).
sizing_for_L() {
  case "$1" in
    2) echo "1024000 1024000 32G  4 14400 40" ;;
    3) echo "1024000 512000  128G 8 21600 30" ;;
    4) echo "512000  256000  144G 8 21600 20" ;;
    5) echo "128000  65536   128G 8 21600 10" ;;
    *) echo "ERROR unknown L=$1" >&2; exit 1 ;;
  esac
}

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
request_memory          = \$(MEM)
request_disk            = 2560M
JobPrio                 = \$(PRIO)
Output                  = campaign_${BASE}/logs/${arm}_L\$(L)_A\$(A)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/${arm}_L\$(L)_A\$(A)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue NB,L,A,SEED,MAXCORE,PT2CAP,MEM,CPUS,MAXRUNGSEC,PRIO from ${grid}
EOF
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm}  CAMPAIGN=${BASE}  jobs=$(wc -l < "$grid")"
}

mkdir -p "campaign_${BASE}/logs"

if [ "$MODE" = "test" ]; then
  # L=3 A=10 s0 to 64k with the levers, 8 cpus, deep-solve: the first run of the forked
  # select stage UNDER deep-solve (1-thread workers), and its cost at L=3.
  G="campaign_${BASE}/smoke.txt"
  row 3 3 10 0 64000 64000 48G 8 7200 50 > "$G"
  submit_grid smoke "$G"
  echo "SMOKE: expect ~1-2 h. Then check: ExitCode 0, rung 0 phase '0-select' with 32 trails,"
  echo "  manifest threads.NUQU_CPUS_RESOLVED == 8, condor RemoteUserCpu/wall > 1."
  exit 0
fi

if [ "$MODE" = "all" ]; then
  G="campaign_${BASE}/grid.txt"; : > "$G"
  for L in $LS; do
    set -- $(sizing_for_L "$L")
    for A in $AS; do
      # seed-major priority: seed 0 everywhere, then seed 1, then seed 2
      for S in $SEEDS; do row 3 "$L" "$A" "$S" "$1" "$2" "$3" "$4" "$5" $(( $6 + 100 * (2 - S) )) >> "$G"; done
    done
  done
  submit_grid Asweep "$G"
  echo "  -> bare TrimCI, dim=3, n_b=3, A={${AS}} x L={${LS}} x seed={${SEEDS}}, PT2 to MAXCORE/2"
  echo
  echo "Retrieve:  rsync -az hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_${BASE}-nb3/shards/ \\"
  echo "               data/classical/\$(date +%F)/bare_Asweep_nb3_<cluster>/"
  exit 0
fi

echo "usage: sh submit_nb3_Asweep.sh {test|all}" >&2
exit 2
