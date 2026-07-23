#!/bin/sh
# Mint a unique rundir + submit the dets-vs-L job to the pinned schedd.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit == login01,
# the per-node schedd — submit + query from the same node):
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL && sh submit_detsvsL.sh [test|full]
#
# The job self-provisions its whole env on the compute node (see run_detsvsL.sh) —
# nothing is pre-built on the login node, nothing new is written to shared storage.
# qis-pinned: those nodes mount /nfs_scratch + have outbound internet + a compiler
# (verified by probe cluster 289774). test = ~30s shakedown; full = L=2,3 3D cores->50k.
set -eu
MODE="${1:-full}"
case "$MODE" in
  test) CPUS=2; MEM=8G;  DISK=8G  ;;
  full) CPUS=4; MEM=24G; DISK=10G ;;
  *) echo "usage: $0 [test|full]" >&2; exit 1 ;;
esac

while :; do
  DATE="$(date +%Y%m%d-%H%M%S)"; RND="$(tr -dc a-z0-9 </dev/urandom | head -c3)"
  RUNDIR="rundir-${DATE}-${RND}"; [ ! -d "$RUNDIR" ] && break
done
mkdir "$RUNDIR"; echo "using $RUNDIR (mode=$MODE)"

cat > "$RUNDIR/submit" <<EOF
Executable              = ./run_detsvsL.sh
arguments               = ${MODE}
Log                     = ${RUNDIR}/condor.log
Output                  = ${RUNDIR}/stdout
Error                   = ${RUNDIR}/stderr
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_detsvsL.sh,${RUNDIR}
transfer_output_files   = ${RUNDIR}
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}
request_disk            = ${DISK}
Queue
EOF

condor_submit "$RUNDIR/submit"
