"""
Tests for the standalone total-QPE-cost computation (Phase E).
"""

import json
import math
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.estimation.qpe_cost import (
    DEFAULT_DELTA_E_MEV,
    compute_total_qpe_cost,
    total_qpe_t_count,
    walk_queries,
)


def _make_sweep_file(tmpdir, results, metadata=None):
    path = os.path.join(tmpdir, 'sweep.json')
    data = {'metadata': metadata or {'L': 2, 'dim': 3}, 'results': results}
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def test_walk_queries_formula():
    """N_walk = √2·π·Λ / ΔE."""
    lam, dE = 1.97e5, 1.0
    expected = math.sqrt(2.0) * math.pi * lam / dE
    assert abs(walk_queries(lam, dE) - expected) < 1e-6


def test_total_qpe_t_count_formula():
    t_step, lam, dE = 6.85e6, 1.97e5, 1.0
    expected = t_step * math.sqrt(2.0) * math.pi * lam / dE
    assert abs(total_qpe_t_count(t_step, lam, dE) - expected) < 1e-3


def test_compute_writes_fields_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            {'A': 1, 'Total_T_Count': 6.85e6, 'Physical_Lambda': 1.97e5},
            {'A': 2, 'Total_T_Count': 2.06e6, 'Physical_Lambda': 5.36e5},
        ]
        path = _make_sweep_file(tmp, results)
        data = compute_total_qpe_cost(path, delta_E=1.0)

        for r in data['results']:
            assert 'QPE_Walk_Queries' in r
            assert 'QPE_Total_T_Count' in r
            exp_nq = math.sqrt(2.0) * math.pi * r['Physical_Lambda'] / 1.0
            assert abs(r['QPE_Walk_Queries'] - exp_nq) < 1e-3
            assert abs(r['QPE_Total_T_Count'] - r['Total_T_Count'] * exp_nq) < 1.0
        assert data['metadata']['delta_E_MeV'] == 1.0

        # Verify it was written to disk too.
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk['results'][0]['QPE_Total_T_Count'] == data['results'][0]['QPE_Total_T_Count']


def test_idempotent():
    """Running twice converges to the same numbers (recomputed from invariants)."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [{'A': 1, 'Total_T_Count': 6.85e6, 'Physical_Lambda': 1.97e5}]
        path = _make_sweep_file(tmp, results)
        d1 = compute_total_qpe_cost(path, delta_E=1.0)
        v1 = d1['results'][0]['QPE_Total_T_Count']
        d2 = compute_total_qpe_cost(path, delta_E=1.0)
        v2 = d2['results'][0]['QPE_Total_T_Count']
        assert v1 == v2


def test_delta_e_scaling():
    """Halving ΔE doubles the total cost."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [{'A': 1, 'Total_T_Count': 1e6, 'Physical_Lambda': 1e5}]
        path = _make_sweep_file(tmp, results)
        d_full = compute_total_qpe_cost(path, delta_E=1.0, write=False)
        d_half = compute_total_qpe_cost(path, delta_E=0.5, write=False)
        ratio = d_half['results'][0]['QPE_Total_T_Count'] / d_full['results'][0]['QPE_Total_T_Count']
        assert abs(ratio - 2.0) < 1e-9


def test_skips_entries_missing_fields():
    """Entries without Total_T_Count / Physical_Lambda are skipped, not crashed on."""
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            {'A': 1, 'Total_T_Count': 1e6, 'Physical_Lambda': 1e5},
            {'A': 2},  # legacy / malformed entry — no T or Λ
        ]
        path = _make_sweep_file(tmp, results)
        data = compute_total_qpe_cost(path, delta_E=1.0)
        assert 'QPE_Total_T_Count' in data['results'][0]
        assert 'QPE_Total_T_Count' not in data['results'][1]


def test_default_delta_e_is_one_mev():
    assert DEFAULT_DELTA_E_MEV == 1.0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll qpe_cost tests passed.")
