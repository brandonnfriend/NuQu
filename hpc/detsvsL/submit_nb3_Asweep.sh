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
# EXISTING DATA CHECKED (2026-09-23). The only compatible shards (bare, n_b=3, PT2 post-collapse,
# 3 seeds) in the A=2..10 window are L=2 A=8 = `bare_baseline_nb3_292477/bare_L2d3_f1.0_s*`,
# so that cell is SKIPPED. The nb_nested shards (L=2 A=2,4,8; L=3,4 A=8) and the DMRG A=2 shards
# carry no PT2, so they cannot give the extrapolated E_inf +- sigma and are not a substitute.
#
# GRID: A in {2..10} x L in {2..5} x seed in {0,1,2}, minus (L=2, A=8) = 105 shards.
# Trim without editing (env overrides): AS="2 4 6 8 10"  LS="2 3 4"  SEEDS="0".
# (sigma is a per-seed quantity; the seed spread is a robustness check, so SEEDS="0" still
# yields E_inf +- sigma — at a third of the cost.)
#
# COST. Measured L=2 (293959): 3.1-5.7 h/shard, 14-22 GB. From 292477 wall-clocks (per-rung
# walls at A=10 match A=64 in the smoke): L=3 ~6 h, L=4 ~15 h, L=5 ~10.5 h per shard. At 4c the
# L>=3 grid (81 shards) is ~3.4k CPU-h allocated (~0.5k used); wall is set by memory slots
# (qis2 ~6 x 128G concurrent) -> ~5-6 days for 3 seeds, ~2 days for seed 0 alone.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_nb3_Asweep.sh test   # 1 shard: L=4 A=10 s0 to 32k, PT2 every rung (n_ext check)
#     sh submit_nb3_Asweep.sh all    # the grid
#     LS="3 4 5" sh submit_nb3_Asweep.sh all        # the L>=3 grid (L=2 done in 293959)
#     LS=2 SEEDS="1 2" sh submit_nb3_Asweep.sh all  # L=2 independent seeds 1,2 (s0 reused)
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
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=32 NUQU_PHASE0_SEED_STRIDE=1000'

row() { printf '%s %s %s %s %s %s %s %s %s %s\n' "$@"; }
# L -> "MAXCORE PT2CAP MEM CPUS MAXRUNGSEC PRIO"   (MAXCORE/PT2CAP/MAXRUNGSEC = 292477)
#   292477 peaks at A=L^3: L=2 14-20 GB; L=4 PT2@256k 146 GB; L=3/L=5 ~118/106 GB + solve.
#   Measured at A<=10: L=2 14-22 GB; L=4 smoke n_ext = 0.51-0.59x A=64 -> PT2@256k ~75-80 GB.
#   An OOM is recoverable: condor_qedit RequestMemory + condor_release, the per-rung save keeps
#   every finished rung (HPC_WORKFLOW s6).
sizing_for_L() {
  case "$1" in
    2) echo "1024000 1024000 32G  4 14400 40" ;;
    3) echo "1024000 512000  128G 4 21600 30" ;;
    4) echo "512000  256000  128G 4 21600 20" ;;
    5) echo "128000  65536   128G 4 21600 10" ;;
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
  # Largest A at L=4, shallow: n_ext per rung vs 292477's A=64 (37.3M at 32k) re-checks the
  # L>=4 memory trim from real cluster numbers before the grid goes in.
  G="campaign_${BASE}/smoke.txt"
  row 3 4 10 0 32000 32000 64G 4 7200 50 > "$G"
  submit_grid smoke "$G"
  echo "SMOKE: expect ~1 h. Then check:"
  echo "  condor_history <cluster> -af ExitCode MemoryUsage RemoteWallClockTime"
  echo "  grep -o '\"n_ext\": [0-9]*' campaign_${BASE}-nb3/shards/bare_L4d3_A10_s0.json"
  exit 0
fi

if [ "$MODE" = "all" ]; then
  G="campaign_${BASE}/grid.txt"; : > "$G"
  for L in $LS; do
    set -- $(sizing_for_L "$L")
    for A in $AS; do
      # L=2 A=8 == 292477's L=2 filling-1.0 shards (same H, same settings) -> reuse them.
      if [ "$L" = 2 ] && [ "$A" = 8 ]; then continue; fi
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
