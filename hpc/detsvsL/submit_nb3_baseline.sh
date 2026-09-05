#!/bin/sh
# CLASSICAL HEADLINE BASELINE AT n_b=3 — closes audit P0-1 (2026-09-05).
#
# WHY. The published classical baseline (cluster 290832, `bare_baseline_290832`) is n_b=2 /
# N_f=4 — the cutoff the project's OWN L=2 energy gate REJECTS (E_0 off 4–7 MeV, BE off ~91 MeV).
# The quantum PauliLCU anchor and the frame study are both n_b=3. So the end-to-end comparison
# is currently two different finite-dimensional Hamiltonians. This campaign regenerates the
# PRIMARY bare-TrimCI baseline at n_b=3 so classical and quantum describe one model.
#
# WHAT'S DIFFERENT from 290832 (deliberately, only two things):
#   1. n_b=3 (N_f=8) instead of n_b=2 (N_f=4).
#   2. PT2 ON AT EVERY RUNG (`NUQU_PT2_MAX_CORE` = max_core), not capped below the ladder top.
#      290832 capped PT2 at 16k–64k, which put every PT2 point in the PRE-collapse
#      "exploration" basin — where a PT2 extrapolation demonstrably LIES (L=2 cross-check:
#      +19 MeV/site vs the deep variational answer). Audit P0-2 wants a defensible
#      extrapolated value WITH error bars at every L, and that needs >=3 PT2 rungs in the
#      POST-collapse basin. Measured (local, C++ PT2 path): the EN external space costs only
#      ~17 B/ext above the solve's own peak at L=3 (3.5M ext -> +55 MB), so PT2-everywhere is
#      affordable — the old "~150 GB at 1M core" figure was the pure-Python path.
#   Everything else (bare frame, dim=3, filling 1.0, warm-grow deep solve, ladder_start=1000,
#   NUQU_PHASE0_RUNS=32, 11 geometric rungs) matches 290832 so the n_b delta is the ONLY signal.
#
# ARMS
#   nb3  -> the new headline. L={2,3,4,5} x seed={0,1,2}. Three independent warm-grow
#           trajectories per L give the seed-spread term of the T2 error bar.
#   nb2  -> a PAIRED historical arm at IDENTICAL settings (L={2,3,4}, seed 0). 290832 cannot
#           serve this role: it has one seed and no post-collapse PT2, so no paired extrapolated
#           Delta(n_b=2->3) can be formed from it. Keeps the comparison switch alive per CLAUDE.md
#           ("keep the previous path alive as a runtime flag") — the n_b=2->3 shift in the
#           HEADLINE baseline is itself a publishable number.
#
# Deeper than 290832 where it is cheap: L=3 to 1M (was 512k), L=4 to 512k (was 256k), L=5 to
# 256k (reached 64k). `max_rung_seconds` — not max_core — is the real stop: a rung that blows the
# budget ends the ladder, and the per-rung incremental save keeps every rung already finished.
#
# COST NOTE. Locally measured at small core: n_b=3 costs the SAME as n_b=2 (the H term list is
# n_b-independent — only the per-mode Fock dimension grows), rising to ~1.8x only at the deepest
# rungs where high boson levels get populated. So 290832's wall-clocks (L=2 2.5 h, L=3 7 h,
# L=4 13 h, L=5 9 h) are the right order for the per-shard budget.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_nb3_baseline.sh test   # 1 cheap L=3 n_b=3 shard to 32k, PT2 every rung
#     sh submit_nb3_baseline.sh nb3    # the 12-shard headline arm
#     sh submit_nb3_baseline.sh nb2    # the 3-shard paired historical arm
#     sh submit_nb3_baseline.sh all    # both arms (default)
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-nb3base"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
RUNNER="./run_frame_shard.sh"

# args to run_frame_shard.sh: L seed campaign frame A filling max_core ladder_mode
#                             frame_runs orbopt_cycles phase0_core max_rung_seconds
# (A=1 is ignored — `filling 1.0` sets A = sites.)
ARGS='$(L) $(SEED) '"${BASE}"'-nb$(NB) bare 1 1.0 $(MAXCORE) independent 4 3 1000 $(MAXRUNGSEC)'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=$(NB) NUQU_N_RUNGS=11 NUQU_PT2_MAX_CORE=$(PT2CAP) NUQU_PHASE0_RUNS=32'

# ---- per-L sizing -----------------------------------------------------------------------
# L  MAXCORE   PT2CAP    MEM   CPUS  MAXRUNGSEC
#    (E_var)   (=MAXCORE: PT2 on every rung)
# mem = measured 290832 peaks (24-46 GB) + the EN map (~100 B/ext, worst case ~600M ext at
# L=4/5) + headroom. Right-sized, not 2x — over-requesting fragments the pool and inflates
# fair-share (HPC_WORKFLOW s6).
row() { printf '%s %s %s %s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"; }
sizing_for_L() {   # $1=L -> "MAXCORE PT2CAP MEM CPUS MAXRUNGSEC"
  case "$1" in
    2) echo "1024000 1024000 64G  16 14400" ;;
    3) echo "1024000 1024000 128G 16 21600" ;;
    4) echo "512000  512000  192G 24 21600" ;;
    5) echo "256000  256000  192G 24 21600" ;;
    *) echo "ERROR unknown L=$1" >&2; exit 1 ;;
  esac
}

submit_grid() {   # $1=armname  $2=grid file  $3=JobPrio
  arm="$1"; grid="$2"; prio="$3"
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
request_disk            = 10G
JobPrio                 = ${prio}
Output                  = campaign_${BASE}/logs/${arm}_nb\$(NB)_L\$(L)_s\$(SEED).out
Error                   = campaign_${BASE}/logs/${arm}_nb\$(NB)_L\$(L)_s\$(SEED).err
Log                     = campaign_${BASE}/logs/campaign.log
queue NB,L,SEED,MAXCORE,PT2CAP,MEM,CPUS,MAXRUNGSEC from ${grid}
EOF
  condor_submit "campaign_${BASE}/${arm}.sub"
  echo "${arm} arm  CAMPAIGN=${BASE}  jobs=$(wc -l < "$grid")"
}

mkdir -p "campaign_${BASE}/logs"

# ============================== smoke ====================================================
if [ "$MODE" = "test" ]; then
  G="campaign_${BASE}/smoke.txt"
  # L=3, n_b=3, shallow (32k) but PT2 on EVERY rung — exercises exactly the new setting and
  # reports n_ext growth so the deep PT2 caps can be re-checked from real cluster numbers.
  row 3 3 0 32000 32000 32G 16 7200 > "$G"
  submit_grid smoke "$G" 30
  echo "SMOKE: expect ~10-20 min. Then check:"
  echo "  condor_history <cluster> -af ExitCode MemoryUsage RemoteWallClockTime"
  echo "  grep n_ext campaign_${BASE}-nb3/shards/bare_L3d3_f1.0_s0.json   # PT2 ran on every rung"
  exit 0
fi

# ============================ nb3 arm (the headline) ====================================
if [ "$MODE" = "nb3" ] || [ "$MODE" = "all" ]; then
  G="campaign_${BASE}/grid_nb3.txt"; : > "$G"
  for L in 2 3 4 5; do
    set -- $(sizing_for_L "$L")
    for S in 0 1 2; do row 3 "$L" "$S" "$1" "$2" "$3" "$4" "$5" >> "$G"; done
  done
  submit_grid nb3 "$G" 25
  echo "  -> headline: bare TrimCI, dim=3, filling 1.0, n_b=3, L=2..5 x seed{0,1,2}, PT2 every rung"
fi

# ==================== nb2 arm (paired historical comparison) =============================
if [ "$MODE" = "nb2" ] || [ "$MODE" = "all" ]; then
  G="campaign_${BASE}/grid_nb2.txt"; : > "$G"
  for L in 2 3 4; do
    set -- $(sizing_for_L "$L")
    row 2 "$L" 0 "$1" "$2" "$3" "$4" "$5" >> "$G"
  done
  submit_grid nb2 "$G" 20
  echo "  -> paired historical arm: same settings at n_b=2 => a clean Delta(n_b=2->3) per L"
fi

echo
echo "Retrieve:  rsync -az hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_${BASE}-nb3/shards/ \\"
echo "               data/classical/\$(date +%F)/bare_baseline_nb3_<cluster>/"
