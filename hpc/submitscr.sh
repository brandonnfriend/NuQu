#!/bin/sh
# submitscr.sh — mint a unique rundir, write its submit file, submit to Condor.
# Run from this hpc/ dir on the pinned submit node (ssh hep-submit).
set -eu

while :; do
  DATE="$(date +%Y%m%d-%H%M%S)"
  RND="$(tr -dc A-Za-z0-9 < /dev/urandom | head -c3)"
  RUNDIR="rundir-${DATE}-${RND}"
  [ ! -d "${RUNDIR}" ] && break
done
mkdir "${RUNDIR}"
echo "using ${RUNDIR}"

cat > "${RUNDIR}/submit" <<EOF
Executable              = ./run.sh
Log                     = ${RUNDIR}/condor.log
Output                  = ${RUNDIR}/stdout
Error                   = ${RUNDIR}/stderr
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run.sh,hello.py,${RUNDIR}
transfer_output_files   = ${RUNDIR}
request_disk            = 200M
request_memory          = 512M
Queue
EOF

condor_submit "${RUNDIR}/submit"
