#!/bin/sh
# Architecture A-vs-B resource comparison campaign (task 34): bare `fock` walk vs
# Gaussian-squeezed `fock_squeezed@r*` walk, resource-vs-n_b at fixed L, both frames
# through the SAME pauli_lcu encoder. One shard per L.
#
# The verdict is read OFF-CLUSTER by overlaying each frame's accuracy-required n_b
# (classical framed vs bare ⟨n⟩ / truncation-vs-N_f): B wins iff its smaller required
# n_b beats its ~1.27× Λ cost at fixed n_b. L=2 3D is the pipeline-validation baseline
# (classically the squeeze does NOT compact at L=2 — expect ≈ neutral/worse); L=3 3D is
# the decisive size (framed ⟨n⟩ 0.012 < bare 0.022, 15-25× fewer dets).
#
# squeeze_r = classical analytic_squeeze median r* (computed off-cluster):
#   L=2 d=3 -> 0.2109 ,  L=3 d=3 -> 0.2543.
#
# Usage (from $REPO/hpc/quantum/ on the pinned submit node, after reconciling to the
# campaign branch — see submit_overnight.sh header):
#   sh submit_frame_AB.sh test     # ONE smoke job (validate the new module on a qis node)
#   sh submit_frame_AB.sh          # the real grid (L=2 + L=3)
set -eu
MODE="${1:-run}"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_AB.sh
arguments               = 2 2 0.2109 ${CAMPAIGN} test
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_AB.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/AB_test.out
Error                   = ${DIR}/logs/AB_test.err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = 8G
request_disk            = 10G
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  (smoke test — 1 job; check ${DIR}/logs/AB_test.out)"
  exit 0
fi

# "L  n_b_values('+')  squeeze_r  mem".  pauli_lcu fock is feasible to n_b~4 (dies at >=5);
# n_b 2-4 covers the operating range. Atomic per-(frame,n_b) save recovers any deep OOM.
PLAN="
2 2+3+4+5 0.2109 16G
3 2+3+4 0.2543 32G
"

: > "$DIR/shards.txt"
echo "$PLAN" | while read -r L NB R MEM; do
  [ -z "${L:-}" ] && continue
  echo "$L $NB $R $MEM" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_frame_AB.sh
arguments               = \$(L) \$(NB) \$(R) ${CAMPAIGN} run
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_frame_AB.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_AB.out
Error                   = ${DIR}/logs/L\$(L)_AB.err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 10G
periodic_remove         = (JobStatus == 2) && (time() - JobCurrentStartDate > 21600)
queue L,NB,R,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "pull: rsync -az hep:${REPO:-/nfs_scratch/bfriend3/NuQu/NuQu}/hpc/quantum/${DIR}/shards/ <local>/"
