"""Reproduce SEVERAL independent Watson source quantities, not one aggregate T-count
(codex trotter_comparison_status_2026-08-24 §2).

Watson et al. (arXiv:2312.05344) Table IX tabulates ONE dynamical-pion point — the crossing time on
a 10×10×10 lattice, A=40 nucleons, a_L=2.2 fm, E_kin=10 MeV, total error 0.1, p=1:

    T-gate count = 1.3e42 ,  2-qubit depth = 6.0e36 ,  qubits = 99,000 ,  n_b = 33–39.

(The A-scaling is only a plot, Fig 11 — no exact tabulated values.) So the strong cross-check is to
reproduce MULTIPLE INDEPENDENT quantities of that one point — the T-count, the cutoff n_b, and the
qubit count each depend on different parts of the transcription, so agreeing on all three constrains
the model far more than one aggregate total.

Run: python -m pytest -q tests/test_trotter_watson_reproduction.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.trotter_theory.trotter_exact import crossing_time_cost, watson_qubits

# Watson Table IX, dynamical-pion crossing-time point (arXiv:2312.05344, p47):
WATSON = {"L": 10, "A": 40, "eps_total": 0.1, "T": 1.3e42, "depth": 6.0e36,
          "qubits": 99000, "n_b_lo": 33, "n_b_hi": 39}


def test_qubit_count_reproduces_99000():
    """Watson's 99,000 = boson register 3·L³·n_b at n_b=33 — exact qubit-accounting reproduction."""
    assert watson_qubits(10, 33, include_fermions=False) == 99000
    # with the fermion register the total is only ~4% higher (bosons dominate):
    assert 99000 < watson_qubits(10, 33, include_fermions=True) < 1.05 * 99000


def test_cutoff_nb_reproduces_watson_range():
    """The native Lemma-5 cutoff lands in/near Watson's reported n_b=33–39 across A=1..40 (the code
    is ~+1 conservative at the high-A end — within the documented ±1-n_b sensitivity)."""
    nbs = {A: crossing_time_cost(10, A, eps_total=0.1)["n_b"] for A in (1, 10, 40)}
    assert all(33 <= nb <= 41 for nb in nbs.values()), nbs        # within ±1-2 of Watson's window
    assert nbs[1] < nbs[40]                                       # grows with fermion number (log A)
    assert nbs[40] <= WATSON["n_b_hi"] + 1                        # ≤ 40 (Watson 39, +1 conservative)


def test_tgate_count_reproduces_table_ix():
    """T-count at Watson's own cutoff (n_b=39) reproduces 1.3e42 within the documented factor
    (loose worst-case commutator bound; ~1.24×)."""
    T = crossing_time_cost(10, 40, eps_total=0.1, n_b_override=39)["total_T"]
    assert 1.0e42 <= T <= 2.0e42, f"{T:.3e} outside [1e42, 2e42]"
    assert abs(T / WATSON["T"] - 1.24) < 0.15                     # pin the ~1.24× reproduction factor


def test_three_quantities_are_independent_crosschecks():
    """Sanity that the three reproduced quantities probe different parts of the model:
    n_b (cutoff/Lemma-5), qubits (∝L³·n_b geometry), T (Ξ commutator sum × steps × per-step)."""
    o = crossing_time_cost(10, 40, eps_total=0.1)                 # native cutoff
    assert o["n_b"] >= 33 and o["total_T"] > 1e42 and watson_qubits(10, o["n_b"]) > 99000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("all Watson-reproduction checks passed")
