#!/bin/sh
# run.sh — Condor executable for the NuQu plumbing smoke test.
# Proves: (1) code transferred to a compute node, (2) it ran there,
# (3) outputs transfer back inside the rundir.
cd rundir* || { echo "ERROR: no rundir in sandbox" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-1}"
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus"

# Always-succeeds proof that we ran on a compute node (shell only).
{
  echo "hostname : $(hostname)"
  echo "date     : $(date)"
  echo "uname    : $(uname -a)"
  echo "pwd      : $(pwd)"
  echo "cpus     : $cpus"
} > hostinfo.txt

# Best-effort proof that Python works on the node (stdlib only, no venv needed).
if command -v python3 >/dev/null 2>&1; then
  python3 ../hello.py
elif command -v python >/dev/null 2>&1; then
  python ../hello.py
else
  echo "no python interpreter on $(hostname)" > python_missing.txt
fi
