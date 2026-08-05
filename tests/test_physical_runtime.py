"""Tests for the logical->physical translation (task 30).

Load-bearing: reproduce Beverland 2211.07629's Ru-catalyst chemistry anchor
(~1 month runtime at the (ns, 1e-4) node) — validates the closed-form model.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.estimation.hardware import HardwareProfile, translate_to_physical


def test_beverland_chemistry_anchor():
    """Q_alg=2740, M_T=4.1e11, (ns,1e-4) -> Beverland reports ~1 month, ~1.9M qubits."""
    bev = HardwareProfile(name='beverland_ns', p=1e-4, t_cycle_s=0.4e-6)
    r = translate_to_physical(2740, 4.1e11, profile=bev)
    days = r.runtime_throughput_s / (3600 * 24)
    assert 20 <= days <= 45, days                 # ~1 month
    assert 13 <= r.d <= 21, r.d                    # Beverland's d-range
    assert 1e6 <= r.physical_qubits <= 6e6, r.physical_qubits   # ~few million


def test_distance_and_runtime_grow_with_T():
    a = translate_to_physical(1000, 1e10)
    b = translate_to_physical(1000, 1e14)
    assert b.d >= a.d
    assert b.runtime_throughput_s > a.runtime_throughput_s


def test_runtime_is_C_d_tcycle():
    r = translate_to_physical(500, 1e12)
    assert math.isclose(r.runtime_throughput_s,
                        r.logical_cycles * r.d * 1e-6, rel_tol=1e-9)   # default t_cycle=1us


def test_cultivation_not_self_sufficient_at_huge_M():
    """Our QPE T-counts (~1e17) far exceed cultivation's per-T error budget -> flagged."""
    r = translate_to_physical(22020, 5.03e17)
    assert r.cultivation_self_sufficient is False
    assert r.per_T_error_needed < 1e-19            # eps/3M


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
    print(f"test_physical_runtime: {len(fns)} passed")
