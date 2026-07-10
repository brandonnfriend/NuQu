"""
Validate the C++ Epstein-Nesbet PT2 pass-1 port (MixedProvider.pt2_accumulate,
wired through backend.cpp_pt2_external) against the pure-Python reference
(pt2.epstein_nesbet_pt2 with use_cpp=False), over toy sectors.

The two engines share the same re-diagonalization of V, so they must agree on
every field of the PT2 result (dE_pt2, E_var, E_pt2, n_ext, n_intruder,
E_var_rayleigh) to floating-point round-off. This exercises:
  * the coherent external accumulation A_a = sum_j <a|H|j> c_j (complex),
  * the diagonal-only H_aa evaluation (diagonal_fast / diagonal_only), and
  * the internal Rayleigh quotient <psi|H|psi>.

Build first:  bash classical/trimci/backend_fork/build_mixed_ci.sh
Run:          .venv/bin/python classical/trimci/backend_fork/test_pt2_cpp.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # for `import mixed_ci`
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # project root

from classical.trimci import build_from_eft, enumerate_basis
from classical.trimci.backend import cpp_available
from classical.trimci.pt2 import epstein_nesbet_pt2


def _subset_core(H, A, frac=0.5):
    """A deterministic subset of the A-nucleon sector to use as the variational
    space V — proper subset so there IS an external space to perturb over."""
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec=A)
    k = max(1, int(len(basis) * frac))
    return basis[:k], len(basis)


def _cmp(label, a, b, tol, failures):
    ok = abs(a - b) <= tol * (1.0 + abs(b))
    status = "ok  " if ok else "FAIL"
    print(f"    [{status}] {label}: cpp={a:.10g} py={b:.10g} |d|={abs(a - b):.2e}")
    if not ok:
        failures.append(f"{label}: cpp={a} py={b}")


def validate(L, dim, n_b, A, frac, tol, failures):
    H = build_from_eft(L, dim, n_b)
    states, n_full = _subset_core(H, A, frac)

    # Same re-diagonalization inside both calls (default dense diag_fn), so any
    # difference is purely the pass-1 engine.
    py = epstein_nesbet_pt2(H, states, use_cpp=False)
    cpp = epstein_nesbet_pt2(H, states, use_cpp=True)

    print(f"  L={L},dim={dim},n_b={n_b},A={A}: |V|={len(states)}/{n_full}, "
          f"n_ext(py)={py['n_ext']}")
    _cmp("dE_pt2", cpp["dE_pt2"], py["dE_pt2"], tol, failures)
    _cmp("E_var", cpp["E_var"], py["E_var"], tol, failures)
    _cmp("E_pt2", cpp["E_pt2"], py["E_pt2"], tol, failures)
    _cmp("E_var_rayleigh", cpp["E_var_rayleigh"], py["E_var_rayleigh"], tol, failures)
    if cpp["n_ext"] != py["n_ext"]:
        failures.append(f"n_ext mismatch cpp={cpp['n_ext']} py={py['n_ext']}")
        print(f"    [FAIL] n_ext: cpp={cpp['n_ext']} py={py['n_ext']}")
    else:
        print(f"    [ok  ] n_ext: {cpp['n_ext']}")
    if cpp["n_intruder"] != py["n_intruder"]:
        failures.append(f"n_intruder mismatch cpp={cpp['n_intruder']} py={py['n_intruder']}")
        print(f"    [FAIL] n_intruder: cpp={cpp['n_intruder']} py={py['n_intruder']}")
    else:
        print(f"    [ok  ] n_intruder: {cpp['n_intruder']}")


def main():
    if not cpp_available():
        # cpp_available() also checks the official Davidson; the PT2 port itself
        # only needs mixed_ci, but the re-diag default is dense Python so this is
        # just an informational note.
        print("  note: official TrimCI Davidson not present; using dense re-diag.")

    print("=" * 66)
    print("  C++ EN-PT2 pass-1 port vs pure-Python reference")
    print("=" * 66)
    failures = []
    # complex (Weinberg-Tomozawa) sectors + full chiral coupling; a couple of
    # core fractions incl. the full-basis (no external -> dE_pt2 ~ 0) edge.
    validate(1, 1, 2, 1, frac=0.5, tol=1e-9, failures=failures)
    validate(1, 1, 2, 2, frac=0.4, tol=1e-9, failures=failures)
    validate(2, 1, 1, 1, frac=0.5, tol=1e-9, failures=failures)
    # n_b=2 sector is 32768; keep |V| under build_dense's 6000 dense re-diag cap.
    validate(2, 1, 2, 1, frac=0.15, tol=1e-9, failures=failures)
    validate(1, 1, 2, 1, frac=1.0, tol=1e-9, failures=failures)  # V = full sector
    print("=" * 66)
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        print("=" * 66)
        raise SystemExit(1)
    print("  C++ PT2 PORT MATCHES PYTHON EXACTLY")
    print("=" * 66)


if __name__ == "__main__":
    main()
