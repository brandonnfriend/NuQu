"""
Validate the Tier-2 C++ connections port (mixed_ci.hpp) against the Python
reference (classical/trimci/hij.connections), exhaustively over toy sectors.

Build first:  bash classical/trimci/backend_fork/build_mixed_ci.sh
Run:          .venv/bin/python classical/trimci/backend_fork/test_mixed_ci_cpp.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # for `import mixed_ci`
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # project root

from classical.trimci import build_from_eft, enumerate_basis
from classical.trimci.hij import connections as py_connections


def _serialize_terms(H):
    """MixedH.terms -> [(complex coeff, [(mode,action)...] ferm, [...] bos)]."""
    out = []
    for t in H.terms:
        out.append((complex(t.coeff),
                    [(int(m), int(a)) for (m, a) in t.ferm_ops],
                    [(int(m), int(a)) for (m, a) in t.bos_ops]))
    return out


def _py_conn_dict(H, state):
    """Python connections as {(ferm, bos_tuple): complex}, pruned like the C++."""
    return {(s.ferm, tuple(s.bos)): v for s, v in py_connections(H, state).items()}


def _cpp_conn_dict(prov, state):
    """C++ connections as {(ferm, bos_tuple): complex}."""
    raw = prov.connections(int(state.ferm), [int(x) for x in state.bos])
    return {(int(f), tuple(b)): complex(v) for ((f, b), v) in raw}


def validate(L, dim, n_b, A, tol=1e-12):
    import mixed_ci  # the compiled standalone module
    H = build_from_eft(L, dim, n_b)
    prov = mixed_ci.MixedProvider(_serialize_terms(H), H.N_f)
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A)

    max_err = 0.0
    mism_states = 0
    for st in basis:
        py = _py_conn_dict(H, st)
        cpp = _cpp_conn_dict(prov, st)
        if set(py) != set(cpp):
            mism_states += 1
            continue
        for k in py:
            max_err = max(max_err, abs(py[k] - cpp[k]))
    assert mism_states == 0, f"{mism_states} states with mismatched neighbor SETS"
    assert max_err < tol, f"value mismatch {max_err:.2e}"

    # spot-check the diagonal too
    st0 = basis[len(basis) // 3]
    d_cpp = complex(prov.diagonal(int(st0.ferm), [int(x) for x in st0.bos]))
    d_py = _py_conn_dict(H, st0).get((st0.ferm, tuple(st0.bos)), 0.0)
    assert abs(d_cpp - d_py) < tol, "diagonal mismatch"
    print(f"  L={L},dim={dim},n_b={n_b},A={A}: {len(basis)} states, "
          f"neighbor sets identical, max|dValue|={max_err:.2e}, diagonal OK")


def main():
    print("=" * 64)
    print("  Tier-2 C++ connections port vs Python reference (exhaustive)")
    print("=" * 64)
    validate(1, 1, 2, 1)   # 256 states, all Weinberg-Tomozawa (complex)
    validate(1, 1, 2, 2)   # 384 states, A=2 fermion sector
    validate(2, 1, 1, 1)   # 512 states, gradient + H_AV + H_WT (full coupling)
    validate(2, 1, 2, 1)   # 32768 states, larger N_f
    print("=" * 64)
    print("  C++ PORT MATCHES PYTHON EXACTLY")
    print("=" * 64)


if __name__ == "__main__":
    main()
