"""Memoized Jordan-Wigner transform for the sparse-oracle fermion sector.

**Why this exists (the L⁶ wall).** Profiling the Fock+sparse path (task 34, I2)
showed `jordan_wigner` is ~99% of a point's wall-clock, and it is called
*redundantly*:

  * **Within one point (2×):** `SparseStrategy.estimate` calls both
    `compute_native_lambda` and `estimate_sparse_resources`, and each of those
    JW-transforms `mh.fermion_part` once and every `mixed_term.fermion_factor`
    once — so the identical ~O(num_sites) JW transforms run twice per point.
  * **Across an A-sweep (up to ~25×):** the fermion sector (`fermion_part`,
    the mixed-term `fermion_factor`s) depends only on `(L, dim, n_b)`, NOT on A.
    With `boson_cutoff_method='tong'` the per-site register `n_b` is A-flat, so
    every A in the sweep rebuilds and re-JW-transforms the *same* operators.

The individual JW cost is `O(L^{2·dim−1})` per long-range term (OpenFermion
inserts a Z-string spanning every qubit between the two operator indices), so
this redundancy is the dominant, and easily-removed, cost.

**What this does.** Memoize `jordan_wigner` keyed on the FermionOperator's
canonical term signature. Bit-identical output (it *is* `jordan_wigner`, just
not recomputed). Measured effect: within-point 2× immediately; a full
constant-n_b A-sweep collapses to a single unique computation.

**Escape hatch / validation.** Set `NUQU_DISABLE_JW_CACHE=1` to bypass the memo
entirely (calls raw `jordan_wigner`) — used by the test to prove the cached and
uncached resource numbers are identical, and available if a caller is ever
suspected of mutating a returned operator.

**Caller contract.** The returned `QubitOperator` is shared by reference; callers
in the sparse path only read `.terms` (1-norm, Pauli weights) and must not mutate
it in place. All current callers comply.
"""

import os

from openfermion import jordan_wigner as _jordan_wigner


def _cache_disabled():
    return os.environ.get('NUQU_DISABLE_JW_CACHE', '') not in ('', '0', 'false', 'False')


# Module-level memo. Keyed by the operator's sorted (term, coeff) signature,
# which is hashable and uniquely identifies the FermionOperator. A short-lived
# process (one HPC shard = one L) never accumulates unboundedly; call
# `clear_jw_cache()` to reset within a long-running process if needed.
_JW_CACHE = {}


def _signature(fermion_op):
    """Hashable canonical key for a FermionOperator.

    `terms` is `{term_tuple: coeff}`; sorting by the (unique) term tuples never
    needs to compare the complex coeffs, so the result is a well-defined,
    hashable key. Two operators with identical terms transform identically, so
    this is exactly the right cache key. Cheap (`O(T log T)`) next to the JW
    transform it guards (`O(T · string_length)`).
    """
    return tuple(sorted(fermion_op.terms.items()))


def jordan_wigner_cached(fermion_op):
    """`jordan_wigner(fermion_op)`, memoized on the operator signature.

    Falls back to the raw transform when `NUQU_DISABLE_JW_CACHE` is set.
    """
    if _cache_disabled():
        return _jordan_wigner(fermion_op)
    key = _signature(fermion_op)
    q = _JW_CACHE.get(key)
    if q is None:
        q = _jordan_wigner(fermion_op)
        _JW_CACHE[key] = q
    return q


def clear_jw_cache():
    """Drop all memoized transforms (e.g. between very different sweeps in one
    long-running process, to cap memory)."""
    _JW_CACHE.clear()


def jw_cache_info():
    """Diagnostic: current number of distinct memoized operators."""
    return {'size': len(_JW_CACHE), 'disabled': _cache_disabled()}
