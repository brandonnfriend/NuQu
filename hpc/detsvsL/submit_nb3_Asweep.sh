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
#   5. JobPrio by L (L=2 highest) so the cheap volumes land first.
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
# COST (from 292477 wall-clocks, an upper bound at lower A): per shard L=2 ~2.5 h/16c,
# L=3 ~6 h/16c, L=4 ~15 h/24c, L=5 ~10.5 h/24c  =>  ~20k CPU-h ALLOCATED for the full grid,
# ~2-3 days wall on qis1-3 (~18 concurrent at L>=3). L=5 gives a bound only at A=L^3.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_nb3_Asweep.sh test   # 1 shard: L=4 A=10 s0 to 32k, PT2 every rung (n_ext check)
#     sh submit_nb3_Asweep.sh all    # the grid
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
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=32'

row() { printf '%s %s %s %s %s %s %s %s %s %s\n' "$@"; }
# L -> "MAXCORE PT2CAP MEM CPUS MAXRUNGSEC PRIO"   (MAXCORE/PT2CAP/CPUS/MAXRUNGSEC = 292477)
#   292477 peaks at A=L^3: L=2 14-20 GB; L=4 PT2@256k 146 GB; L=3/L=5 ~118/106 GB + solve.
#   An OOM is recoverable: condor_qedit RequestMemory + condor_release, the per-rung save keeps
#   every finished rung (HPC_WORKFLOW s6).
sizing_for_L() {
  case "$1" in
    2) echo "1024000 1024000 32G  16 14400 40" ;;
    3) echo "1024000 512000  128G 16 21600 30" ;;
    4) echo "512000  256000  160G 24 21600 20" ;;
    5) echo "128000  65536   160G 24 21600 10" ;;
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
  row 3 4 10 0 32000 32000 64G 24 7200 50 > "$G"
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
      for S in $SEEDS; do row 3 "$L" "$A" "$S" "$1" "$2" "$3" "$4" "$5" "$6" >> "$G"; done
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
