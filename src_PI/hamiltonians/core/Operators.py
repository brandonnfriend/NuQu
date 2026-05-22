"""
Shared low-level operator primitives used by both amplitude- and Fock-basis
pion encodings.

Basis-specific operators (Pi_Squared_Operator, Gradient_Squared_Operator,
Momentum_Squared_Operator) have moved to
src_PI/hamiltonians/core/pion_basis/amplitude.py — they encoded the
amplitude-basis-specific P, Q, Pp, Qp parameters that don't exist in the
Fock basis.

What stays here:
- Nucleon_Transition_JW: Jordan-Wigner-mapped (a†_α a_β + h.c.) for one
  site, cached. Used by both bases for the fermionic coupling structure.
- Dict-accumulator helpers (_add, _z_key, _to_qubit_op): used by amplitude.py
  to build dense Pauli sums efficiently. Kept here so they're accessible
  to any future basis module.
"""

from openfermion import QubitOperator, FermionOperator, jordan_wigner

from src_PI.utils.LatticeGeometry import site_to_nucleon_qubit


# --- Dict-accumulator helpers ----------------------------------------------
# Used by the amplitude-basis Pauli-string builders to avoid the per-term
# QubitOperator parse + dict-merge overhead of `H += QubitOperator(...)`.


def _z_key(idx):
    """Canonical single-Z term tuple."""
    return ((idx, 'Z'),)


def _add(terms_dict, key, coeff):
    """Accumulate coeff into terms_dict[key] (dict.get keeps it O(1))."""
    terms_dict[key] = terms_dict.get(key, 0.0) + coeff


def _to_qubit_op(terms_dict):
    """Wrap an accumulated terms-dict into a QubitOperator (no per-term overhead)."""
    op = QubitOperator()
    op.terms = terms_dict
    return op


# --- Nucleon transitions ---------------------------------------------------
# JW result depends only on (site_id, mode_alpha, mode_beta, n_b), all hashable.
# Within a single Hamiltonian build, H_WT_Logic calls each (site, alpha, beta)
# 6x (once per epsilon-tensor row) and H_axial_vector up to dim*3 = 9x per
# site, so the cache saves a meaningful chunk of jordan_wigner work.
#
# The cache is module-level and never cleared. Callers must NOT mutate the
# returned QubitOperator (in code today all use sites pre-multiply, so this is
# fine — we return a fresh copy to be defensive).
_NUCLEON_TRANSITION_CACHE = {}


def _nucleon_transition_jw_uncached(site_id, mode_alpha, mode_beta, n_b):
    idx_alpha = site_to_nucleon_qubit(site_id, mode_alpha, n_b)
    idx_beta = site_to_nucleon_qubit(site_id, mode_beta, n_b)
    f_op = (FermionOperator(f'{idx_alpha}^ {idx_beta}')
            + FermionOperator(f'{idx_beta}^ {idx_alpha}'))
    return jordan_wigner(f_op)


def Nucleon_Transition_JW(site_id, mode_alpha, mode_beta, n_b):
    """Jordan-Wigner mapped Pauli string for (a^dagger_alpha a_beta + h.c.)"""
    key = (site_id, mode_alpha, mode_beta, n_b)
    cached = _NUCLEON_TRANSITION_CACHE.get(key)
    if cached is None:
        cached = _nucleon_transition_jw_uncached(*key)
        _NUCLEON_TRANSITION_CACHE[key] = cached
    # Return a shallow copy of the terms dict so callers can safely *= or mutate
    # without poisoning the cache.
    op = QubitOperator()
    op.terms = dict(cached.terms)
    return op
