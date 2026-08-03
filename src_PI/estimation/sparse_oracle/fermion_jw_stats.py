"""Analytic Jordan-Wigner statistics for the sparse-oracle fermion sector
(task 34, I2 deeper fix).

The sparse encoder never needs the JW *operator*; it needs four scalar
functionals of it:

  * `one_norm`        — Σ|coeff| over non-identity Pauli strings (feeds Λ),
  * `identity_coeff`  — the identity coefficient (the classical shift),
  * `n_pauli_terms`   — number of distinct non-identity Pauli strings (LCU L_eff),
  * `total_weight`    — Σ (Pauli weight) over those strings (feeds the T-count,
                        `fermion_T = 4·total_weight`, `Clifford = 8·total_weight`).

The JW-cache (`jw_cache.py`) already removed the *redundant* transforms; what
remains is the cost of openfermion **materializing** each Pauli string, which is
`O(string_length) = O(L^{dim−1})` per long-range term. Profiling (task 34) showed
the *bilinear* mixed-term factors (`a_p† a_q`) are the larger half of that
residual, and their JW is fully closed-form — so we can read off all four
functionals in `O(#terms)`, never building a Z-string.

**Exact single-bilinear rule.** For `a_i† a_j` (i≠j), with `lo=min(i,j)`,
`hi=max(i,j)`, the JW image is four Pauli strings, each with the same support —
Pauli at `lo`, a Z-fill on `(lo, hi)`, Pauli at `hi` — hence weight `hi−lo+1`:

    a_i† a_j = ¼ (P_lo^a ⊗ Z_(lo,hi) ⊗ P_hi^b)   summed over the XX, YY, XY, YX pieces

with coefficients (creation index decides the sign of the imaginary pair):

    i < j (creation at lo):  XX:+¼  YY:+¼  XY:+¼i  YX:−¼i
    i > j (creation at hi):  XX:+¼  YY:+¼  XY:−¼i  YX:+¼i

Distinct bilinears with the same `{lo,hi}` and Pauli pattern combine (their
coeffs add — this is how the Hermitian pair `a_p†a_q + a_q†a_p` collapses to the
real `½(XX+YY)`); different `{lo,hi}` never collide. Accumulating by
`(lo, hi, pattern)` therefore reproduces openfermion's *combined* operator
exactly, and the four functionals fall out of that small dict.

Quartic terms (the on-site contacts, `a†a†aa`) are NOT handled here — they have
16-string images that collide with the bilinear Z-strings, so exactness would
need the full combination. `fermion_jw_stats` falls back to the cached openfermion
transform for any operator that isn't purely bilinear.

**Validation / escape hatch.** `NUQU_ANALYTIC_BILINEAR_JW=0` forces the openfermion
path for every operator. The test suite asserts the analytic and openfermion
functionals are bit-identical on the real mixed factors (L=2/3/4) and on random
bilinear operators; `verify_against_openfermion` is the reusable checker.
"""

import os

from src_PI.estimation.sparse_oracle.jw_cache import jordan_wigner_cached

_TOL = 1e-12


def _analytic_disabled():
    return os.environ.get('NUQU_ANALYTIC_BILINEAR_JW', '') in ('0', 'false', 'False')


def _bilinear_contrib(i, j, coeff):
    """The four `(lo, hi, pattern) -> coeff` contributions of `coeff · a_i† a_j`
    (i != j). `pattern` is the ordered pair of Pauli letters at (lo, hi)."""
    if i < j:
        lo, hi = i, j
        imag = 1.0j
    else:
        lo, hi = j, i
        imag = -1.0j
    q = coeff / 4.0
    return (
        ((lo, hi, 'XX'), q),
        ((lo, hi, 'YY'), q),
        ((lo, hi, 'XY'), q * imag),
        ((lo, hi, 'YX'), -q * imag),
    )


def _as_one_body(term):
    """Classify a FermionOperator term key as a handled one-body form:
      * `('hop', i, j)` — hopping bilinear `a_i† a_j`, i != j;
      * `('num', i)`    — number operator `a_i† a_i`;
      * `None`          — anything else (pair-creation `a†a†`, quartic contact,
                          …) ⇒ caller falls back to openfermion.

    `term` is a tuple of `(mode, action)` with action 1 = creation, 0 = annih.
    """
    if len(term) != 2:
        return None
    (i, ai), (j, aj) = term
    if ai == 1 and aj == 0:
        return ('num', i) if i == j else ('hop', i, j)
    return None


def bilinear_jw_stats(fermion_op):
    """Analytic JW functionals for a FermionOperator whose every term is a
    one-body hopping bilinear (`a_i† a_j`, i != j) or number operator (`a_i† a_i`).

    Returns `{one_norm, identity_coeff, n_pauli_terms, total_weight}`, or None if
    any term is a higher form (⇒ fall back to the cached openfermion transform).

    Number operators contribute `n_i = (I − Z_i)/2`: `+coeff/2` to the identity and
    a single-qubit `−coeff/2 · Z_i` (weight 1). Their identity parts accumulate
    (as they do under openfermion), and their `Z_i` strings never collide with the
    two-endpoint XX/YY/XY/YX strings of the hopping bilinears.
    """
    acc = {}
    identity = 0j
    for term, coeff in fermion_op.terms.items():
        kind = _as_one_body(term)
        if kind is None:
            return None                       # higher form -> fall back
        c = complex(coeff)
        if kind[0] == 'num':
            i = kind[1]
            identity += c / 2.0
            key = (i, i, 'Z')
            acc[key] = acc.get(key, 0j) - c / 2.0
        else:                                 # ('hop', i, j)
            for key, contrib in _bilinear_contrib(kind[1], kind[2], c):
                acc[key] = acc.get(key, 0j) + contrib

    one_norm = 0.0
    total_weight = 0
    n_terms = 0
    for (lo, hi, _pat), c in acc.items():
        if abs(c) <= _TOL:
            continue                          # coeffs cancelled to zero
        one_norm += abs(c)
        total_weight += (hi - lo + 1)         # (i,i,'Z') gives weight 1
        n_terms += 1
    return {'one_norm': one_norm, 'identity_coeff': identity.real,
            'n_pauli_terms': n_terms, 'total_weight': total_weight}


def _openfermion_stats(fermion_op):
    """The same four functionals via the cached openfermion transform (reference
    path; correct for any FermionOperator including quartic contacts)."""
    if not fermion_op.terms:
        return {'one_norm': 0.0, 'identity_coeff': 0.0,
                'n_pauli_terms': 0, 'total_weight': 0}
    q = jordan_wigner_cached(fermion_op)
    one_norm = 0.0
    total_weight = 0
    n_terms = 0
    identity_coeff = 0.0
    for term, coeff in q.terms.items():
        if term == ():
            identity_coeff = complex(coeff).real
            continue
        one_norm += abs(coeff)
        total_weight += len(term)
        n_terms += 1
    return {'one_norm': one_norm, 'identity_coeff': identity_coeff,
            'n_pauli_terms': n_terms, 'total_weight': total_weight}


def fermion_jw_stats(fermion_op):
    """JW functionals `{one_norm, identity_coeff, n_pauli_terms, total_weight}`
    of a FermionOperator, analytic for purely-bilinear operators and via the
    cached openfermion transform otherwise (or when `NUQU_ANALYTIC_BILINEAR_JW=0`).
    """
    if not _analytic_disabled():
        stats = bilinear_jw_stats(fermion_op)
        if stats is not None:
            return stats
    return _openfermion_stats(fermion_op)


def verify_against_openfermion(fermion_op, tol=1e-9):
    """Return `(ok, analytic, reference)` — the analytic vs openfermion functionals
    for a purely-bilinear operator. `ok` is False (with analytic=None) if the
    operator isn't bilinear. Used by the test suite as the bit-exactness gate."""
    analytic = bilinear_jw_stats(fermion_op)
    reference = _openfermion_stats(fermion_op)
    if analytic is None:
        return False, None, reference
    ok = (
        abs(analytic['one_norm'] - reference['one_norm']) <= tol
        and abs(analytic['identity_coeff'] - reference['identity_coeff']) <= tol
        and analytic['n_pauli_terms'] == reference['n_pauli_terms']
        and analytic['total_weight'] == reference['total_weight']
    )
    return ok, analytic, reference
