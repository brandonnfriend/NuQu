#!/bin/sh
# DEEP-SOLVE THREAD SCALING + detect_cpus CHECK (2026-09-24).
#
# WHY. qis Condor never exported _CONDOR_REQUEST_CPUS, so every earlier deep-solve job ran at
# 2 OMP threads whatever request_cpus said (292477 at "16/24", 293959 at "16" ...). Their
# ~1.5 busy cores were 75% of 2, NOT 10% of 16, so how far the deep rungs scale is unmeasured.
# run_frame_shard.sh now resolves the real allocation (detect_cpus). This campaign checks:
#   1. the fix worked: shard manifest threads.NUQU_CPUS_RESOLVED == OMP_NUM_THREADS == CPUS,
#      and condor RemoteUserCpu / wall rises with CPUS;
#   2. how the per-rung wall scales with threads -> the request_cpus for the L>=3 grid.
# SAME SOLVE at every CPUS (no levers, seed 0, deep-solve): only the thread count differs, so
# E_var must agree to FP-order noise (~1e-8 MeV) and wall_s per rung is directly comparable.
#
# GRID (6 shards):  L=2 A=5 to 256k at CPUS {2,4,8,16};  L=3 A=10 to 64k at CPUS {4,16}.
# Run from $REPO/hpc/detsvsL/ on ssh hep-submit:   sh submit_thread_scaling.sh
set -eu
BASE="$(date +%Y%m%d-%H%M%S)-thr"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
ARGS='$(L) 0 '"${BASE}"'-c$(CPUS) bare $(A) none $(MAXCORE) independent 4 3 1000 14400'
ENVSTR='NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=3 NUQU_N_RUNGS=$(NR) NUQU_PT2_MAX_CORE=$(MAXCORE) NUQU_PHASE0_RUNS=32 NUQU_PHASE0_SEED_STRIDE=1000'
mkdir -p "campaign_${BASE}/logs"
G="campaign_${BASE}/grid.txt"
cat > "$G" <<ROWS
2 5 256000 9 2 24G
2 5 256000 9 4 24G
2 5 256000 9 8 24G
2 5 256000 9 16 24G
3 10 64000 7 4 64G
3 10 64000 7 16 64G
ROWS
cat > "campaign_${BASE}/thr.sub" <<SUB
Executable              = ./run_frame_shard.sh
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
Output                  = campaign_${BASE}/logs/L\$(L)_A\$(A)_c\$(CPUS).out
Error                   = campaign_${BASE}/logs/L\$(L)_A\$(A)_c\$(CPUS).err
Log                     = campaign_${BASE}/logs/campaign.log
queue L,A,MAXCORE,NR,CPUS,MEM from ${G}
SUB
condor_submit "campaign_${BASE}/thr.sub"
echo "thr  CAMPAIGN=${BASE}  jobs=$(wc -l < "$G")"
echo "check: python - <<< shard manifest['extra']['threads'] + condor_history -af RequestCpus RemoteUserCpu RemoteWallClockTime"
