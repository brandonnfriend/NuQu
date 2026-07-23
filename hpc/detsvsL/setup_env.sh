#!/bin/sh
# One-time server provisioning for the classical dets-vs-L HPC run.
#
# Run ONCE on the shared HEP checkout, on a LOGIN node (ssh hep) — NOT inside a
# Condor job:
#     ssh hep
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git pull
#     sh hpc/detsvsL/setup_env.sh
#
# Creates .venv (uv, Python 3.10), installs the minimal pinned runtime
# (hpc/detsvsL/requirements-hpc.txt), builds the mixed_ci C++ hot path for Linux,
# and self-tests the classical solver end to end. The venv lives on the shared
# /nfs_scratch checkout, so qis execute nodes read it directly — no per-job
# runtime transfer.
set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
echo "[setup] repo : $REPO"
echo "[setup] host : $(hostname)"

# 1. uv must be on PATH. (If missing: `pip install --user uv`, or a module load.)
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv not found on PATH. Install it (pip install --user uv) and re-run." >&2
    exit 1
}

# 2. venv (Python 3.10). uv fetches a managed standalone CPython if none present.
uv venv --python 3.10 "$REPO/.venv"

# 3. minimal pinned runtime — see requirements-hpc.txt (NO pyLIQTR/jax/netket).
VIRTUAL_ENV="$REPO/.venv" uv pip install -r "$REPO/hpc/detsvsL/requirements-hpc.txt"

# 4. build the mixed_ci C++ hot path (Linux build; generic x86-64, no -march=native
#    so the .so runs on any x86-64 execute node — see build_mixed_ci.sh).
PY="$REPO/.venv/bin/python" bash "$REPO/classical/trimci/backend_fork/build_mixed_ci.sh"

# 5. self-test: import the C++ module + run a tiny solve. This exercises the exact
#    path the Condor job uses (openfermion Hamiltonian build -> mixed_ci C++
#    connections -> scipy eigsh -> PT2 -> extrapolate -> save).
"$REPO/.venv/bin/python" - <<PY
import os, sys
sys.path.insert(0, os.path.join("classical", "trimci", "backend_fork"))
import mixed_ci  # noqa: F401  -- must be built for this (Linux) node
print("[setup] mixed_ci C++ backend import OK")
from classical.trimci.run_cpp import dets_vs_L_at_fixed_accuracy
r = dets_vs_L_at_fixed_accuracy(
    dim=1, L_values=(2,), A=2, n_b=1, max_core=200, n_rungs=2, ladder_start=100,
    max_rung_seconds=20, eps_persite_targets=(1.0,), n_runs=2, hpc=True,
    label="_setup_selftest", verbose=False)
print("[setup] self-test RAN OK; E_inf/site =", r["per_L"][0]["E_inf_per_site"])
for ext in (".json", ".png"):
    p = os.path.join("data", "classical", "_setup_selftest" + ext)
    if os.path.exists(p):
        os.remove(p)
PY

echo "[setup] DONE — venv + C++ backend ready at $REPO/.venv"
