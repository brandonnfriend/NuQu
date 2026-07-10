#!/usr/bin/env bash
# Build the standalone mixed_ci pybind module (Tier-2 connections C++ port).
# Run from anywhere; uses the NuQu venv's python + pybind11.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/brandonfriend/Desktop/Projects/NuQu/.venv/bin/python}"

PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I"$PYBIND_INC" -I"$PY_INC" \
    "$HERE/mixed_ci_pybind.cpp" \
    -o "$HERE/mixed_ci${EXT}" \
    -undefined dynamic_lookup

echo "built $HERE/mixed_ci${EXT}"
