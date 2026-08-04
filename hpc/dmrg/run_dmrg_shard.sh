#!/bin/sh
# Condor executable for ONE block2-DMRG isospectrality-reference shard.
# args: $1=L  $2=A  $3=campaign  $4=N_f  $5=bond_dims(csv)  $6=n_sweeps_per
#
# Self-provisions per-sandbox (a shared NFS uv dir corrupts under concurrency), pip-
# installs block2 (no C++ build -- unlike the frame shard), then runs
# hpc/dmrg/run_dmrg_shard.py which DMRGs the bare H over the chi schedule and saves
# E-vs-chi with INCREMENTAL per-chi saving (a shard that times out at the top chi keeps
# everything it finished).
#
# UNLIKE the frame fork-ensemble path, block2 is ONE multithreaded process: it WANTS all
# the cores (OpenMP/MKL), so threads are set to $cpus here, not pinned to 1.
set -u
L="$1"; A="$2"; CAMPAIGN="$3"; N_F="${4:-4}"; BOND_DIMS="${5:-100,200,400,800}"
NSWEEPS="${6:-6}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/hpc/dmrg/run_dmrg_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

cpus="${_CONDOR_REQUEST_CPUS:-4}"
# block2 = one multithreaded process -> give it every requested core.
export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus" \
       NUMEXPR_NUM_THREADS="$cpus" MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[dmrg-shard] host=$(hostname) L=$L A=$A N_f=$N_F bond_dims=$BOND_DIMS cpus=$cpus"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "ERROR: uv install failed" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }
uv python install 3.10 >/dev/null 2>&1
uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; exit 1; }
PY="$SANDBOX/venv/bin/python"
VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q -r "$REPO/hpc/dmrg/requirements-hpc.txt" \
    || { echo "ERROR: pip install failed" >&2; exit 1; }

OUTDIR="$REPO/hpc/dmrg/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$REPO"
OUT="$OUTDIR/dmrg_L${L}_A${A}_Nf${N_F}.json"

"$PY" -m hpc.dmrg.run_dmrg_shard --L "$L" --dim 3 --A "$A" --N_f "$N_F" --n_b 2 \
    --bond-dims "$BOND_DIMS" --n-sweeps-per "$NSWEEPS" --out "$OUT"
status=$?
echo "[dmrg-shard] done status=$status -> $OUT"
exit "$status"
