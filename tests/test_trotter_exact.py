"""Tests for the exact Watson Theorem-64 Trotter model (task 11).

The load-bearing test is that the model reproduces Watson Table IX (1.3e42 T-gates
at L=10, A=40) to the correct order of magnitude with NO free coefficient.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.trotter_theory.trotter_exact import (
    crossing_time_cost, dynamical_pion_xi, qpe_bits, qpe_cost,
)


def test_reproduces_table_ix_order_of_magnitude():
    """Crossing-time at Watson's reported n_b=39 must land on 10^42 (target 1.3e42).
    No fitting — this is the exact Theorem-64 bound."""
    o = crossing_time_cost(10, 40, eps_total=0.1, n_b_override=39, budget_split=3)
    assert 5e41 <= o['total_T'] <= 5e42, o['total_T']       # same OOM as 1.3e42 (we get ~1.6e42)


def test_xi_is_wt_dominated():
    """Ξ is ~100% the WT–WT (Lemma 78) term at dynamical-pion scale."""
    p = get_physical_parameters()
    Xi, terms = dynamical_pion_xi(10, 40, 3.7e7, 1.4e10, p, return_breakdown=True)
    assert terms['WT_WT'] / Xi > 0.999


def test_low_nb_is_vastly_cheaper():
    """Costing Trotter in the reduced n_b=6 Hilbert space is many OOM cheaper than
    Watson's Lemma-5 n_b — the 'give Trotter the low-occupation benefit too' lever."""
    high = qpe_cost(6, 100, dE=1.0)                         # Lemma-5 n_b (~44)
    low = qpe_cost(6, 100, dE=1.0, n_b_override=6)
    assert low['total_T'] < high['total_T']
    assert high['total_T'] / low['total_T'] > 1e10         # dramatic
    assert low['n_b'] == 6 and high['n_b'] > 30


def test_qpe_bits():
    assert qpe_bits(1.0, 140.0) == 8                        # ceil(log2(140)) = 8


def test_no_free_coefficient_signature():
    """The exact model's step count is r = t^2 Xi/(2 eps_prod) — no 'Cp'. Sanity:
    doubling A (which enters Xi linearly via eta) ~doubles r at fixed n_b."""
    a1 = qpe_cost(4, 20, dE=1.0, n_b_override=8)
    a2 = qpe_cost(4, 40, dE=1.0, n_b_override=8)
    assert 1.8 < a2['r'] / a1['r'] < 2.2                   # eta-linear in Xi (WT-WT)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
    print(f"test_trotter_exact: {len(fns)} passed")
