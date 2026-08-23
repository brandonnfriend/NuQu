"""Tests for the warm-start state-preparation cost model (`src_PI/estimation/state_prep_cost.py`).

Pins the Low-2018 QROAM scaling (sweet-spot √(D·b), serial-D, ancilla growth), the Berry 7×
synthesis reduction, and the headline claim that state prep is sub-dominant to the QPE walk.

Run: `python -m pytest -q tests/test_state_prep_cost.py`
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_PI.estimation.state_prep_cost import (
    qroam_state_prep_cost, gaussian_prep_cost, state_prep_cost,
    BERRY_SYNTHESIS_SPEEDUP, T_PER_TOFFOLI,
)


def test_qroam_sweet_spot_scales_as_sqrt_Db():
    """T-optimal QROAM Toffoli ≈ 2√(D·b) (Low 2018), and beats both extremes."""
    D, b = 4096, 16
    opt = qroam_state_prep_cost(D, b=b)                      # auto k = √(D/b)
    assert opt["lambda_qr"] == 16, opt["lambda_qr"]         # √(4096/16)=16
    assert abs(opt["toffoli"] - 2 * math.sqrt(D * b)) / (2 * math.sqrt(D * b)) < 0.10
    serial = qroam_state_prep_cost(D, b=b, lambda_qr=1)     # Toffoli ≈ D
    assert serial["toffoli"] >= D - 1
    assert opt["toffoli"] < serial["toffoli"]              # sweet spot is cheaper in T
    assert serial["ancilla_total"] < opt["ancilla_total"]  # ...at higher ancilla for the sweet spot


def test_ancilla_matches_doc_estimate():
    """D=1000, b=17 → a_prep ~ O(100), exceeding the QPE phase register m~12-24 (total_costs §3a)."""
    r = qroam_state_prep_cost(1000, b=17)
    assert 80 <= r["ancilla_total"] <= 200, r["ancilla_total"]


def test_serial_min_width_knob():
    """λ_QR=1 is the min-width serial QROM: no dirty copies, ancilla = b + address only."""
    r = qroam_state_prep_cost(1000, b=17, lambda_qr=1)
    assert r["ancilla_dirty"] == 0
    assert r["ancilla_total"] == 17 + math.ceil(math.log2(1000))


def test_berry_synthesis_reduces_toffoli():
    plain = qroam_state_prep_cost(10000, b=16)["toffoli"]
    berry = qroam_state_prep_cost(10000, b=16, berry_synthesis=True)["toffoli"]
    assert abs(berry - plain / BERRY_SYNTHESIS_SPEEDUP) <= 1
    assert T_PER_TOFFOLI == 4


def test_gaussian_prep_is_cheap_and_ancilla_free():
    """Per-mode squeeze: modest rotation count, zero extra ancilla (in-place)."""
    g = gaussian_prep_cost(n_bos_modes=3000, N_f=4)         # ~L=10 mode count
    assert g["ancilla"] == 0
    assert g["t"] < 1e7                                     # ≪ walk T ~1e16


def test_state_prep_subdominant_to_walk():
    """Total warm-start prep ≪ one QPE walk-query block (the sub-dominance headline)."""
    walk_T_one_query = 3.27e8                               # L=10 anchor walk_T
    coherent_query_T = 9.28e15                              # L=10 anchor coherent-query T
    sp = state_prep_cost(D=10000, n_bos_modes=3000, N_f=4)
    assert sp["a_prep"] > 0
    # even ×(a few hundred repetitions) the prep stays far below the coherent-query T:
    assert sp["T_prep"] * 500 < coherent_query_T
    assert sp["T_prep"] < 1000 * walk_T_one_query


if __name__ == "__main__":
    for fn in (test_qroam_sweet_spot_scales_as_sqrt_Db, test_ancilla_matches_doc_estimate,
               test_serial_min_width_knob, test_berry_synthesis_reduces_toffoli,
               test_gaussian_prep_is_cheap_and_ancilla_free, test_state_prep_subdominant_to_walk):
        fn()
    print("all state_prep_cost checks passed")
