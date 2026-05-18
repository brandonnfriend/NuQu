from functools import lru_cache
from openfermion import QubitOperator, FermionOperator, jordan_wigner
from src_PI.utils.LatticeGeometry import site_to_nucleon_qubit, site_to_pion_qubit


# --- Dict-accumulator helpers ----------------------------------------------
# Building a QubitOperator via repeated `H += QubitOperator(f'Z{i}', c)` parses
# a string AND merges dicts on every term. We instead accumulate into a plain
# dict keyed by canonical Pauli tuples and assign at the end.
#
# For the openfermion canonical form, Pauli-tuple keys must be sorted by qubit
# index ascending and have no duplicate indices.


def _z_key(idx):
    """Canonical single-Z term tuple."""
    return ((idx, 'Z'),)


def _zz_key(idx_a, idx_b):
    """Canonical Z_a Z_b term tuple (assumes a != b; collapse Z*Z=I if equal)."""
    if idx_a < idx_b:
        return ((idx_a, 'Z'), (idx_b, 'Z'))
    return ((idx_b, 'Z'), (idx_a, 'Z'))


def _add(terms_dict, key, coeff):
    """Accumulate coeff into terms_dict[key] (dict.get keeps it O(1))."""
    terms_dict[key] = terms_dict.get(key, 0.0) + coeff


def _to_qubit_op(terms_dict):
    """Wrap an accumulated terms-dict into a QubitOperator (no per-term overhead)."""
    op = QubitOperator()
    op.terms = terms_dict
    return op


def Pi_Squared_Operator(site_id, pion_species, n_b, P, Q):
    """Implements Eq. (71): pi^2 expressed in Pauli Z operators."""
    terms = {}
    _add(terms, (), P * P)
    two_PQ = 2.0 * P * Q
    Q_sq = Q * Q

    # Term 2: 2PQ * sum(2^m * Z_m)
    indices = [site_to_pion_qubit(site_id, pion_species, m, n_b) for m in range(n_b)]
    for m, idx in enumerate(indices):
        _add(terms, _z_key(idx), two_PQ * (1 << m))

    # Term 3: Q^2 * sum_{m, m'} 2^(m+m') * Z_m Z_m'
    for m in range(n_b):
        idx_m = indices[m]
        pow_m = 1 << m
        for mp in range(n_b):
            idx_mp = indices[mp]
            coeff = Q_sq * pow_m * (1 << mp)
            if idx_m == idx_mp:
                _add(terms, (), coeff)
            else:
                _add(terms, _zz_key(idx_m, idx_mp), coeff)
    return _to_qubit_op(terms)


def Gradient_Squared_Operator(site_x, site_y, pion_species, n_b, Q, a_L):
    """Implements Eq. (72): ((pi(y) - pi(x)) / a_L)^2 for any adjacent site pair."""
    terms = {}
    inv_aL2 = 1.0 / (a_L * a_L)
    Q_sq = Q * Q

    idx_x = [site_to_pion_qubit(site_x, pion_species, m, n_b) for m in range(n_b)]
    idx_y = [site_to_pion_qubit(site_y, pion_species, m, n_b) for m in range(n_b)]

    for m in range(n_b):
        ix_m = idx_x[m]
        iy_m = idx_y[m]
        pow_m = 1 << m
        for n in range(n_b):
            ix_n = idx_x[n]
            iy_n = idx_y[n]
            common_coeff = Q_sq * pow_m * (1 << n) * inv_aL2

            # site x internal
            if ix_m == ix_n:
                _add(terms, (), common_coeff)
            else:
                _add(terms, _zz_key(ix_m, ix_n), common_coeff)

            # site y internal
            if iy_m == iy_n:
                _add(terms, (), common_coeff)
            else:
                _add(terms, _zz_key(iy_m, iy_n), common_coeff)

            # cross term -2 * pi(x) * pi(y)  (sites differ so indices differ)
            _add(terms, _zz_key(ix_m, iy_n), -2.0 * common_coeff)
    return _to_qubit_op(terms)


def Momentum_Squared_Operator(site_id, pion_species, n_b, Pp, Qp):
    """Implements the Pi^2 (conjugate momentum) operator in its diagonal basis."""
    terms = {}
    _add(terms, (), Pp * Pp)
    two_PpQp = 2.0 * Pp * Qp
    Qp_sq = Qp * Qp

    indices = [site_to_pion_qubit(site_id, pion_species, m, n_b) for m in range(n_b)]
    for m, idx in enumerate(indices):
        _add(terms, _z_key(idx), two_PpQp * (1 << m))

    for m in range(n_b):
        idx_m = indices[m]
        pow_m = 1 << m
        for mp in range(n_b):
            idx_mp = indices[mp]
            coeff = Qp_sq * pow_m * (1 << mp)
            if idx_m == idx_mp:
                _add(terms, (), coeff)
            else:
                _add(terms, _zz_key(idx_m, idx_mp), coeff)
    return _to_qubit_op(terms)


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
