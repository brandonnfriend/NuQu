"""Tests for the memoized Jordan-Wigner transform (task 34, I2 — the L⁶-wall fix).

The cache must be a pure performance optimization: cache-ON resource numbers are
required to be *bit-identical* to cache-OFF (raw `jordan_wigner`). These tests
pin that invariant, plus the memo mechanics.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openfermion import FermionOperator, jordan_wigner

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.sparse_oracle.jw_cache import (
    clear_jw_cache, jordan_wigner_cached, jw_cache_info,
)
from src_PI.hamiltonians.core.EFTParameters import (
    estimate_boson_cutoff, get_physical_parameters,
)
from src_PI.utils.Config import Config

_METRICS = ('Physical_Lambda', 'Walk_T_Count', 'Walk_Clifford_Count', 'Logical_Qubits')


def _eval_L2():
    """Run the Fock+sparse resource estimate at a small, fast point."""
    L, dim, A = 2, 3, 10
    params = get_physical_parameters()
    config = Config(pion_basis='fock', block_encoder='sparse',
                    boson_cutoff_method='heuristic')
    n_b, pi_max, _ = estimate_boson_cutoff(L, dim, A, params, epsilon_cut=0.1,
                                           E_bound=10.0 * A,
                                           boson_cutoff_method='heuristic')
    nd = evaluate_resources(L, dim, n_b, pi_max, params, config)
    return {k: nd[k] for k in _METRICS}


def test_cache_bit_identical_to_raw_jordan_wigner():
    """The whole point: the memo must not change any reported number."""
    prev = os.environ.get('NUQU_DISABLE_JW_CACHE')
    try:
        os.environ['NUQU_DISABLE_JW_CACHE'] = '1'      # raw jordan_wigner
        off = _eval_L2()

        os.environ['NUQU_DISABLE_JW_CACHE'] = '0'      # memoized
        clear_jw_cache()
        on = _eval_L2()
    finally:
        if prev is None:
            os.environ.pop('NUQU_DISABLE_JW_CACHE', None)
        else:
            os.environ['NUQU_DISABLE_JW_CACHE'] = prev

    for k in _METRICS:
        assert off[k] == on[k], f"cache changed {k}: {off[k]} != {on[k]}"


def test_cached_matches_raw_on_a_single_operator():
    """`jordan_wigner_cached` returns exactly what `jordan_wigner` would."""
    prev = os.environ.get('NUQU_DISABLE_JW_CACHE')
    os.environ.pop('NUQU_DISABLE_JW_CACHE', None)       # ensure enabled
    try:
        clear_jw_cache()
        # A long-range hopping term (the expensive, long-Z-string kind).
        op = FermionOperator('7^ 1', 1.3) + FermionOperator('1^ 7', 1.3)
        raw = jordan_wigner(op)
        cached = jordan_wigner_cached(op)
        assert cached.terms == raw.terms
        # Second call is a hit: same object, cache size unchanged.
        size_after_first = jw_cache_info()['size']
        again = jordan_wigner_cached(op)
        assert again is cached
        assert jw_cache_info()['size'] == size_after_first
    finally:
        if prev is not None:
            os.environ['NUQU_DISABLE_JW_CACHE'] = prev


def test_disable_env_bypasses_memo():
    prev = os.environ.get('NUQU_DISABLE_JW_CACHE')
    try:
        os.environ['NUQU_DISABLE_JW_CACHE'] = '1'
        clear_jw_cache()
        jordan_wigner_cached(FermionOperator('3^ 0', 1.0))
        assert jw_cache_info()['size'] == 0        # nothing memoized when disabled
        assert jw_cache_info()['disabled'] is True
    finally:
        if prev is None:
            os.environ.pop('NUQU_DISABLE_JW_CACHE', None)
        else:
            os.environ['NUQU_DISABLE_JW_CACHE'] = prev


if __name__ == '__main__':
    test_cache_bit_identical_to_raw_jordan_wigner()
    test_cached_matches_raw_on_a_single_operator()
    test_disable_env_bypasses_memo()
    print("test_jw_cache: all passed")
