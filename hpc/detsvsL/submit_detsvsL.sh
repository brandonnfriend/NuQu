#!/bin/sh
# Mint a unique rundir, write its Condor submit file, submit to the pinned schedd.
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit == login01).
# The schedd is per-login-node, so submit + query MUST use the same pinned node.
#
#     ssh hep-submit
#     cd /nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL && sh submit_detsvsL.sh
#
# Pattern: self-contained rundir (extends the validated smoke test). The rundir is
# transferred both ways; run_detsvsL.sh reads code+venv from the shared /nfs_scratch
# checkout and writes outputs back into the rundir (and durably into data/classical/).
set -eu

while :; do
  DATE="$(date +%Y%m%d-%H%M%S)"
  RND="$(tr -dc A-Za-z0-9 < /dev/urandom | head -c3)"
  RUNDIR="rundir-${DATE}-${RND}"
  [ ! -d "$RUNDIR" ] && break
done
mkdir "$RUNDIR"
echo "using $RUNDIR"

cat > "$RUNDIR/submit" <<EOF
Executable              = ./run_detsvsL.sh
Log                     = ${RUNDIR}/condor.log
Output                  = ${RUNDIR}/stdout
Error                   = ${RUNDIR}/stderr
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_detsvsL.sh,${RUNDIR}
transfer_output_files   = ${RUNDIR}

# Pin to qis nodes: they mount the shared /nfs_scratch (so the venv + repo are
# reachable) and share the arch the .so was built for. Explicit OR avoids regexp
# escaping pitfalls in the heredoc.
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu") || (Machine == "qis4.hep.wisc.edu")

request_cpus            = 4
request_memory          = 24G
request_disk            = 4G
Queue
EOF

condor_submit "$RUNDIR/submit"
