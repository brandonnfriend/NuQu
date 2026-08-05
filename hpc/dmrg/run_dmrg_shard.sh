#!/bin/sh
# Condor executable for ONE block2-DMRG isospectrality-reference shard.
# args: $1=L  $2=A  $3=campaign  $4=N_f  $5=bond_dims(csv)  $6=n_sweeps_per  $7=cpus
#
# Self-provisions per-sandbox (a shared NFS uv dir corrupts under concurrency), pip-
# installs block2 + the mkl runtime (no C++ build -- unlike the frame shard), then runs
# hpc/dmrg/run_dmrg_shard.py which DMRGs the bare H over the chi schedule and saves
# E-vs-chi with INCREMENTAL per-chi saving (a shard that times out at the top chi keeps
# everything it finished).
#
# UNLIKE the frame fork-ensemble path, block2 is ONE multithreaded process: it WANTS all
# the cores (OpenMP/MKL), so threads are set to $cpus here, not pinned to 1. cpus is
# passed EXPLICITLY (arg $7) because Condor doesn't reliably export _CONDOR_REQUEST_CPUS.
set -u
L="$1"; A="$2"; CAMPAIGN="$3"; N_F="${4:-4}"; BOND_DIMS="${5:-100,200,400,800}"
NSWEEPS="${6:-6}"; cpus="${7:-${_CONDOR_REQUEST_CPUS:-4}}"; MAXCHISEC="${8:-}"
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/hpc/dmrg/run_dmrg_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit 1; }

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
# block2's wheel bundles only AVX2/AVX512 MKL kernels, NOT libmkl_def (verified by
# inspecting the wheel). On the AMD (Zen4) qis nodes MKL classifies the CPU as generic
# "type 0" and DEMANDS the def kernel -- symlinking def->avx2 only earns "MKL type 5 not
# suitable for type 0 processor". Instead LD_PRELOAD the COMPLETE pip mkl==2021.4 runtime
# (block2's own pinned dep, already installed): it ships a real type-0 libmkl_def that IS
# suitable for AMD, and preloading its libmkl_rt makes block2 resolve every MKL call
# against that one consistent install -- no missing def, no undefined-symbol mix from
# grafting. Generic kernel on AMD is slower but correct, which is all a reference DMRG needs.
export MKL_ENABLE_INSTRUCTIONS=AVX2
MKLRT="$(find "$SANDBOX/venv" -name 'libmkl_rt.so*' -not -path '*block2.libs*' -print -quit 2>/dev/null)"
if [ -n "$MKLRT" ]; then
  export LD_PRELOAD="$MKLRT${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "[dmrg-shard] LD_PRELOAD complete mkl: $MKLRT"
else
  echo "WARN: pip libmkl_rt.so.1 not found for preload" >&2
fi

OUTDIR="$REPO/hpc/dmrg/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$REPO"
OUT="$OUTDIR/dmrg_L${L}_A${A}_Nf${N_F}.json"

MAXCHI_ARG=""; [ -n "$MAXCHISEC" ] && MAXCHI_ARG="--max-chi-seconds $MAXCHISEC"
"$PY" -m hpc.dmrg.run_dmrg_shard --L "$L" --dim 3 --A "$A" --N_f "$N_F" --n_b 2 \
    --bond-dims "$BOND_DIMS" --n-sweeps-per "$NSWEEPS" $MAXCHI_ARG --out "$OUT"
status=$?
echo "[dmrg-shard] done status=$status -> $OUT"
exit "$status"
