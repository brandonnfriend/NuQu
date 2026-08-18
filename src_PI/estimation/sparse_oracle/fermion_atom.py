"""
Fermion atom for the compiled sparse full-bundle (C1 step 2).

The sparse encoder's fermion factors — the static nucleon `fermion_part` and
the nucleon bilinear `fermion_factor` of each `MixedTerm` (H_AV / H_WT) — are
block-encoded through pyLIQTR's off-the-shelf **PauliLCU** encoder over the
Jordan-Wigner image, exactly the genuinely circuit-level path
`estimators.py::_estimate_one` already uses for the whole PauliLCU pipeline.

This **replaces the fermion LOWER bound** of the analytical proxy
(`resources.py`), which charged `4·total_weight` T per JW string — omitting the
PauliLCU PREP/SELECT overhead entirely (e.g. 32 T vs. the genuine 836 T for a
single H_AV bilinear at L=2). The atom's cost is now a real
`estimate_resources(QubitizedWalkOperator(·))` number, and its rescale factor
`α = encoding.alpha` is the Pauli 1-norm — identical to
`fermion_jw_stats(fermion_op)['one_norm']`, so it drops straight into the
global α_tot invariant.

**Reality.** After the vertex fix each `fermion_factor` is a *Hermitian*
χ-channel contraction `Σ_αβ χ_αβ a†_α a_β` (χ = σ_S ⊗ τ_I Hermitian), so its
JW image has purely real Pauli coefficients (verified across L=2 dim=1/2/3):
the imaginary XY/YX pieces of `a†_α a_β` and `a†_β a_α` cancel. We nonetheless
**assert** reality when packing the pauli_dict — a non-Hermitian factor (e.g.
under a future fermion encoding) must fail loudly rather than have its phase
silently dropped by `float(coeff.real)`.
"""

import os

from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources

from src_PI.estimation.instances import MyCustomHamiltonian
from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
from src_PI.estimation.sparse_oracle.jw_cache import jordan_wigner_cached

_IMAG_TOL = 1e-9


def fermion_pauli_dict(fermion_op):
    """JW-map `fermion_op` and pack a `{pauli_string: real_coeff}` dict.

    Raises `ValueError` if any JW Pauli coefficient carries a non-negligible
    imaginary part (a non-Hermitian factor) — we never silently drop a phase.
    The identity Pauli (from number operators) is included with key `"I"`, as
    `MyCustomHamiltonian` expects; it contributes to the classical shift, not Λ.
    """
    q = jordan_wigner_cached(fermion_op)
    pauli_dict = {}
    for term, coeff in q.terms.items():
        c = complex(coeff)
        if abs(c.imag) > _IMAG_TOL:
            raise ValueError(
                "fermion atom expects a Hermitian factor (real JW image); got "
                f"Pauli {term} with imaginary coeff {c.imag:.3e}. A non-Hermitian "
                "fermion factor must be handled explicitly, not real-projected."
            )
        p_string = " ".join(f"{op}{idx}" for idx, op in term) if term else "I"
        pauli_dict[p_string] = c.real
    return pauli_dict


def fermion_atom_instance(fermion_op):
    """`MyCustomHamiltonian` over `jordan_wigner_cached(fermion_op)` (reality-checked)."""
    return MyCustomHamiltonian(fermion_pauli_dict(fermion_op))


def fermion_atom_encoding(fermion_op):
    """Off-the-shelf pyLIQTR PauliLCU `BlockEncoding` for the fermion factor.

    `encoding.alpha` is the Pauli 1-norm (== `fermion_jw_stats['one_norm']`,
    excluding the identity), the fermion atom's contribution to α_tot.
    """
    return getEncoding(VALID_ENCODINGS.PauliLCU)(fermion_atom_instance(fermion_op))


def _pauli_dict_key(pauli_dict):
    """Hashable, order-independent cache key for a pauli_dict (rounded coeffs)."""
    return frozenset((p, round(c, 12)) for p, c in pauli_dict.items())


# Cache the (expensive, ~100 ms) PauliLCU walk estimate keyed on the pauli_dict.
# Many χ-channel bilinears across a bundle share the same JW structure, so this
# collapses O(#mixed terms) estimator calls to O(#distinct factor shapes). Set
# NUQU_DISABLE_PYLIQTR_CACHE=1 to bypass (matches estimators.py).
_WALK_COST_CACHE = {}


def fermion_atom_walk_cost(fermion_op):
    """Genuine circuit-level PauliLCU walk cost of the fermion factor.

    Returns `{T, Clifford, LogicalQubits, alpha}`. Cached on the pauli_dict, so
    the many identical χ-channel bilinears across a bundle cost one estimate.
    Returns a zero-cost, α=0 entry for an empty operator (no fermion sector).
    """
    pauli_dict = fermion_pauli_dict(fermion_op)
    if not pauli_dict:
        return {'T': 0, 'Clifford': 0, 'LogicalQubits': 0, 'alpha': 0.0}
    cache_disabled = os.environ.get('NUQU_DISABLE_PYLIQTR_CACHE', '') == '1'
    key = None if cache_disabled else _pauli_dict_key(pauli_dict)
    if key is not None and key in _WALK_COST_CACHE:
        return dict(_WALK_COST_CACHE[key])
    encoding = getEncoding(VALID_ENCODINGS.PauliLCU)(MyCustomHamiltonian(pauli_dict))
    walk = QubitizedWalkOperator(encoding)
    res = estimate_resources(walk)
    cost = {
        'T': res.get('T', 0),
        'Clifford': res.get('Clifford', 0),
        'LogicalQubits': res.get('LogicalQubits', 0),
        'alpha': encoding.alpha,
    }
    if key is not None:
        _WALK_COST_CACHE[key] = cost
    return dict(cost)


def fermion_atom_alpha(fermion_op):
    """Pauli 1-norm of `jordan_wigner(fermion_op)` (== fermion_jw_stats one_norm)."""
    return fermion_jw_stats(fermion_op)['one_norm']
