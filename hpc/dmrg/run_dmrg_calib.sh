#!/bin/sh
# Condor executable for ONE block2-DMRG CALIBRATION shard (classical_next_studies A1, 2026-10-08).
# args: $1=L  $2=dim  $3=A  $4=n_b  $5=bond_dims(PLUS-separated)  $6=campaign
#       $7=n_sweeps_per  $8=cpus  $9=max_chi_seconds
#
# Same job as run_dmrg_arealaw.sh (DMRG on the bare H over a chi schedule, E/discarded weight
# saved after every chi) at the TrimCI sweep's cutoff: n_b is an argument and N_f = 2**n_b, so
# the shard diagonalises EXACTLY the Hamiltonian the bare A-sweep shards ran (verified
# term-by-term locally: build_from_eft(2,3,2,N_f=8) == build_from_eft(2,3,3)). The shard JSON
# is what misc/compare_dmrg_trimci.py reads against the TrimCI E_inf.
#
# Provisioning follows run_frame_shard.sh (C4): pinned uv + CPython copied from $NUQU_TOOLS,
# the wheelhouse for the shared deps, PyPI (with retries) only for block2 + mkl, which the
# wheelhouse does not carry. Any provisioning failure exits 3 (re-queued by the submit file).
#
# block2 is ONE multithreaded process: threads = the cpu allocation (passed explicitly, Condor
# does not reliably export _CONDOR_REQUEST_CPUS), and the block2 driver itself is told the
# same number (--n-threads; the area-law shards ran 16-cpu jobs on a 4-thread driver).
set -u
PROVISION_FAIL=3
L="$1"; DIM="$2"; A="$3"; NB="$4"; BOND_DIMS="$(echo "$5" | tr '+' ',')"
CAMPAIGN="$6"; NSWEEPS="${7:-6}"; cpus="${8:-${_CONDOR_REQUEST_CPUS:-4}}"; MAXCHISEC="${9:-}"
N_F=$(( 1 << NB ))
REPO=/nfs_scratch/bfriend3/NuQu/NuQu
SANDBOX="$(pwd)"
[ -r "$REPO/hpc/dmrg/run_dmrg_shard.py" ] || { echo "ERROR: cannot read repo at $REPO" >&2; exit "$PROVISION_FAIL"; }

export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus" \
       NUMEXPR_NUM_THREADS="$cpus" MPLBACKEND=Agg
export HOME="$SANDBOX" UV_INSTALL_DIR="$SANDBOX/uvbin" \
       UV_CACHE_DIR="$SANDBOX/uvcache" UV_PYTHON_INSTALL_DIR="$SANDBOX/uvpy"
export PATH="$UV_INSTALL_DIR:$SANDBOX/.local/bin:$PATH"

echo "[dmrg-calib] host=$(hostname) L=$L dim=$DIM A=$A n_b=$NB N_f=$N_F bond_dims=$BOND_DIMS cpus=$cpus"

TOOLS="${NUQU_TOOLS:-/nfs_scratch/bfriend3/NuQu/tools}"   # see hpc/detsvsL/provision_tools.sh
UV_PIN="${NUQU_UV_VERSION:-0.12.23}"
REQ="$REPO/hpc/dmrg/requirements-hpc.txt"
# retry / provision_uv / provision_python are copied VERBATIM from hpc/detsvsL/run_frame_shard.sh
# (tests/test_dmrg_calib.py checks they have not drifted).
retry() {
  n="$1"; shift; i=1
  while :; do
    "$@" && return 0
    [ "$i" -ge "$n" ] && return 1
    echo "[shard] attempt $i/$n failed: $*  (retrying)" >&2
    sleep $(( i * 20 )); i=$(( i + 1 ))
  done
}
provision_uv() {
  mkdir -p "$UV_INSTALL_DIR"
  if [ -x "$TOOLS/uv" ] && cp "$TOOLS/uv" "$UV_INSTALL_DIR/uv" 2>/dev/null; then
    UV_SRC="tools"
  else
    # download THEN run (a `curl | sh` pipeline reports sh's status, so a failed curl
    # looked like success and only the later `uv not on PATH` check caught it: 294238)
    retry 3 sh -c "curl -LsSf --connect-timeout 20 --max-time 300 -o '$SANDBOX/uv-install.sh' https://astral.sh/uv/$UV_PIN/install.sh && sh '$SANDBOX/uv-install.sh' >/dev/null 2>&1" \
      || { echo "ERROR: uv install failed" >&2; return 1; }
    UV_SRC="download"
  fi
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; return 1; }
}
provision_python() {
  mkdir -p "$UV_PYTHON_INSTALL_DIR"
  if [ -d "$TOOLS/uvpy" ] && ls "$TOOLS/uvpy" 2>/dev/null | grep -q '^cpython-3\.10' \
     && cp -r "$TOOLS/uvpy/." "$UV_PYTHON_INSTALL_DIR/" 2>/dev/null; then
    PY_SRC="tools"; export UV_PYTHON_DOWNLOADS=never
  else
    rm -rf "$UV_PYTHON_INSTALL_DIR"; mkdir -p "$UV_PYTHON_INSTALL_DIR"
    retry 3 uv python install 3.10 || { echo "ERROR: uv python install failed" >&2; return 1; }
    PY_SRC="download"
  fi
  uv venv --python 3.10 "$SANDBOX/venv" >/dev/null 2>&1 || { echo "ERROR: uv venv failed" >&2; return 1; }
}
# deps: the shared pins (numpy/scipy/openfermion/cirq) come from the wheelhouse when it is
# there; block2 + mkl are not in it, so the full requirement set is resolved against PyPI
# with retries, --find-links keeping the wheelhouse wheels local.
provision_wheels() {
  WHL_SRC="pypi"
  if [ -r "$TOOLS/wheels/requirements.lock" ] && VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q --no-index \
       --find-links "$TOOLS/wheels" -r "$TOOLS/wheels/requirements.lock" >/dev/null 2>&1; then
    WHL_SRC="tools+pypi"
  fi
  retry 3 env VIRTUAL_ENV="$SANDBOX/venv" uv pip install -q --find-links "$TOOLS/wheels" -r "$REQ" \
    || { echo "ERROR: pip install failed" >&2; return 1; }
}
provision_uv      || exit "$PROVISION_FAIL"
provision_python  || exit "$PROVISION_FAIL"
provision_wheels  || exit "$PROVISION_FAIL"
PY="$SANDBOX/venv/bin/python"
echo "[dmrg-calib] provisioned uv=$UV_SRC python=$PY_SRC wheels=$WHL_SRC ($("$PY" --version 2>&1), uv $(uv --version 2>&1 | head -1))"

# MKL on the AMD qis nodes: preload the complete pip mkl runtime (see run_dmrg_arealaw.sh for why).
export MKL_ENABLE_INSTRUCTIONS=AVX2
MKLRT="$(find "$SANDBOX/venv" -name 'libmkl_rt.so*' -not -path '*block2.libs*' -print -quit 2>/dev/null)"
if [ -n "$MKLRT" ]; then
  export LD_PRELOAD="$MKLRT${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "[dmrg-calib] LD_PRELOAD complete mkl: $MKLRT"
else
  echo "WARN: pip libmkl_rt.so.1 not found for preload" >&2
fi

OUTDIR="$REPO/hpc/dmrg/campaign_${CAMPAIGN}/shards"
mkdir -p "$OUTDIR"
export PYTHONPATH="$REPO"
OUT="$OUTDIR/dmrg_L${L}d${DIM}_A${A}_Nf${N_F}.json"

MAXCHI_ARG=""; [ -n "$MAXCHISEC" ] && MAXCHI_ARG="--max-chi-seconds $MAXCHISEC"
"$PY" -m hpc.dmrg.run_dmrg_shard --L "$L" --dim "$DIM" --A "$A" --n_b "$NB" --n-threads "$cpus" \
    --bond-dims "$BOND_DIMS" --n-sweeps-per "$NSWEEPS" $MAXCHI_ARG --out "$OUT"
status=$?
echo "[dmrg-calib] done status=$status -> $OUT"
exit "$status"
