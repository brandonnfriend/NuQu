#!/bin/sh
# Large-L VARIATIONAL UPPER BOUND baseline (the "bar to beat"): bare-frame TrimCI at SMALL core for
# L=6..10, filling 1.0, n_b=2. TrimCI's E_var is a rigorous Ritz upper bound at ANY system size, so
# even a tiny selected-CI core gives a verified (loose) classical bound — cheap, and a concrete
# baseline other methods must beat. This is the "convergence is universal" half of the classical
# story (complements the deep-solve L=2..5 convergence in submit_vertexfix_baseline.sh).
#
# Small core + fork-ensemble (NOT the 1M deep-solve): light memory, fast. E_var is the reported
# bound; PT2 is a diagnostic estimate only (not variational).
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/detsvsL
#   sh submit_bare_bigL.sh test   # 1 L=6 shard (validate the small-core large-L path)
#   sh submit_bare_bigL.sh        # L=6..10
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-bareBigL"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"

# small-core, fork-ensemble bare solve (no deep-solve): ladder 1000,2000; 32-seed ensemble; PT2 on.
ENV='NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_B=2 NUQU_N_RUNGS=2 NUQU_PT2_MAX_CORE=2000 NUQU_PHASE0_RUNS=32'
# args: L seed campaign frame A filling max_core ladder_mode runs cycles phase0_core maxrungsec
ARGS_TAIL='0 '"${BASE}"' bare 1 1.0 2000 independent 16 3 500 7200'

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = 6 ${ARGS_TAIL}
environment             = "${ENV}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 32G
request_disk            = 8G
Output                  = ${DIR}/logs/test_L6.out
Error                   = ${DIR}/logs/test_L6.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (bare L=6 small-core; expect a finite E_var upper bound)"
  exit 0
fi

SH="$DIR/shards.txt"; : > "$SH"
printf '6 32G\n7 32G\n8 48G\n9 64G\n10 96G\n' > "$SH"
cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_shard.sh
arguments               = \$(L) ${ARGS_TAIL}
environment             = "${ENV}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 18
Output                  = ${DIR}/logs/bare_L\$(L).out
Error                   = ${DIR}/logs/bare_L\$(L).err
Log                     = ${DIR}/logs/campaign.log
queue L,MEM from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  jobs=$(wc -l < "$SH")  (bare small-core upper bounds: L=6,7,8,9,10)"
echo "pull -> data/classical/<date>/bare_bigL/ ; then re-run misc.make_classical_baseline_figure with both dirs"
