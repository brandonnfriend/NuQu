#!/usr/bin/env bash
# Build the standalone mixed_ci pybind module (Tier-2 connections C++ port).
# Run from anywhere; uses the NuQu venv's python + pybind11.
#
# Portable across macOS (laptop) and Linux (HPC): the venv python is found
# relative to this script by default (override with PY=/path/to/python), and the
# Darwin-only `-undefined dynamic_lookup` linker flag is added only on macOS.
# NOTE: no `-march=native` — the .so stays a generic x86-64 build so it runs on
# heterogeneous Condor execute nodes (avoids "Illegal instruction" traps).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"                 # backend_fork -> trimci -> classical -> repo
PY="${PY:-$REPO/.venv/bin/python}"

PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# macOS extension modules need undefined python symbols resolved at load time;
# Linux allows undefined symbols in shared objects by default, so omit the flag.
LINK_FLAGS=()
if [ "$(uname -s)" = "Darwin" ]; then
    LINK_FLAGS+=(-undefined dynamic_lookup)
fi

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I"$PYBIND_INC" -I"$PY_INC" \
    "$HERE/mixed_ci_pybind.cpp" \
    -o "$HERE/mixed_ci${EXT}" \
    "${LINK_FLAGS[@]}"

echo "built $HERE/mixed_ci${EXT}  (python: $PY)"
