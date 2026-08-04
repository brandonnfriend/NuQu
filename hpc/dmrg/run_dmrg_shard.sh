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
NSWEEPS="${6:-6}"; cpus="${7:-${_CONDOR_REQUEST_CPUS:-4}}"
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
# block2's wheel bundles libmkl_avx2.so.1 + libmkl_avx512.so.1 but NOT libmkl_def.so.1
# (verified by inspecting the wheel). MKL ALWAYS loads libmkl_def as its base kernel
# dispatcher -- even with AVX2 forced -- so on the AMD (Zen4) qis nodes it reaches for the
# missing def and dies. The MKL kernel libs all export the SAME symbol set (per-ISA
# implementations), so symlink the absent def to block2's OWN bundled avx2 kernel: MKL's
# def load then succeeds with a Zen4-valid, ABI-matched kernel from block2's exact build.
# (Dropping pip's real libmkl_def in instead fails -- its build mismatches block2's
# hash-renamed vendored core -> "undefined symbol".) Keep AVX2 as the extra ISA too.
export MKL_ENABLE_INSTRUCTIONS=AVX2
B2LIBS="$(find "$SANDBOX/venv" -type d -name 'block2.libs' -print -quit 2>/dev/null)"
if [ -n "$B2LIBS" ] && [ -e "$B2LIBS/libmkl_avx2.so.1" ]; then
  ln -sf libmkl_avx2.so.1 "$B2LIBS/libmkl_def.so.1"
  echo "[dmrg-shard] libmkl_def.so.1 -> libmkl_avx2.so.1 (block2 build)"
else
  echo "WARN: block2.libs or bundled libmkl_avx2.so.1 not found ('$B2LIBS')" >&2
fi

OUTDIR="$REPO/hpc/dmrg/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$REPO"
OUT="$OUTDIR/dmrg_L${L}_A${A}_Nf${N_F}.json"

"$PY" -m hpc.dmrg.run_dmrg_shard --L "$L" --dim 3 --A "$A" --N_f "$N_F" --n_b 2 \
    --bond-dims "$BOND_DIMS" --n-sweeps-per "$NSWEEPS" --out "$OUT"
status=$?
echo "[dmrg-shard] done status=$status -> $OUT"
exit "$status"
