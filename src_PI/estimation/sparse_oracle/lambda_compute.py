"""
Native-algebra Λ for the sparse-oracle block encoder.

Walks the `MixedHamiltonian` structure term-by-term and computes the
sparse-oracle LCU 1-norm without ever expanding the bosonic sector to
Pauli strings. Formula (task 26 §"Normalization & Λ threading", Liu
2025 Eq. 17 / LOBE Eq. 17):

    λ_term_boson = 2 · ⌈log₂(s_term)⌉ · max_{i,j} |A_{ij}|

where `s_term` is the row sparsity of the monomial and `max|A_{ij}|` is
its largest matrix element on the truncated boson register. For a P-
ladder monomial (e.g. `â†·â·â`), row sparsity is bounded by `2^P` and
the largest matrix element is bounded by `(N_f − 1)^(P/2)` on a register
of N_f = 2^n_b states.

For pure-fermion contributions (static nucleon, the fermion factor of a
mixed term), the strategy goes through pyLIQTR's PauliLCU encoder per
task 26 line 102, so we use the Pauli 1-norm of the Jordan-Wigner
mapped operator. The Gilyén product rule (LOBE §2.4 Lemma 30) gives
the rescaling for a fermion × boson mixed term:

    λ_mixed_term = λ_fermion_factor · λ_boson_factor.

The identity coefficient (the zero-point `m_π/2` per pion mode, plus
any pure constants from the fermion side) is tracked separately as
`identity_shift` — it shifts the measured energy and does not enter Λ.

**Caveat (C1):** the per-monomial Λ formulas above are reasonable upper
bounds for the true sparse-oracle 1-norm. The final per-term rescaling
factor reported by the live oracle (C2+) may be tighter; this helper
gives the conservative number for early plumbing and a sanity-check
ceiling on what the encoder needs to beat.
"""

import math

import numpy as np
from openfermion import jordan_wigner

from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian


def _qubit_op_one_norm(qubit_op):
    """Sum of |coeff| across the QubitOperator's non-identity terms."""
    total = 0.0
    for term, coeff in qubit_op.terms.items():
        if term == ():
            continue
        total += abs(coeff)
    return total


def _qubit_op_identity_coeff(qubit_op):
    """The identity-term coefficient (real part)."""
    coeff = qubit_op.terms.get((), 0.0)
    return complex(coeff).real


def _sparse_lambda_for_boson_monomial(monomial, n_b):
    """Sparse-oracle λ for a single bosonic-ladder monomial.

    `monomial` is the OpenFermion `BosonOperator` term tuple, e.g.
    `((3, 1), (3, 0))` for `b_3^† b_3` (one creation, one annihilation
    on mode 3 — a "1+1" length-2 monomial).

    Returns `2 · ⌈log₂(2^P)⌉ · (N_f − 1)^(P/2)` where `P = len(monomial)`
    is the total ladder count. The empty tuple `()` is the identity and
    returns `1.0` (no sparsity overhead — the identity is sparsity-1).
    """
    if monomial == ():
        return 1.0
    P = len(monomial)
    sparsity_pow2 = 1 << P                       # 2^P
    log2_S = math.ceil(math.log2(sparsity_pow2)) # == P (already a power of 2)
    N_f = 1 << n_b
    max_amp = (N_f - 1) ** (P / 2.0)
    return 2.0 * log2_S * max_amp


def _sparse_lambda_for_boson_op(boson_op, n_b):
    """Sparse-oracle Λ contribution for a `BosonOperator`.

    Sums `|coeff| · λ_term` over the operator's terms; the identity
    coefficient (if any) is tracked separately as the classical shift.
    """
    lambda_total = 0.0
    identity_shift = 0.0
    for monomial, coeff in boson_op.terms.items():
        if monomial == ():
            identity_shift += complex(coeff).real
            continue
        lambda_term = _sparse_lambda_for_boson_monomial(monomial, n_b)
        lambda_total += abs(coeff) * lambda_term
    return lambda_total, identity_shift


def _pauli_lambda_for_fermion_op(fermion_op):
    """Pauli 1-norm of `jordan_wigner(fermion_op)`, plus identity shift.

    The fermion side of a sparse / LOBE encoder still goes through
    pyLIQTR's PauliLCU per task 26 line 102, so the per-term λ here is
    the standard Pauli 1-norm (sum of |coeff| over non-identity Paulis).
    """
    if not fermion_op.terms:
        return 0.0, 0.0
    q = jordan_wigner(fermion_op)
    return _qubit_op_one_norm(q), _qubit_op_identity_coeff(q)


def compute_native_lambda(mixed_hamiltonian, n_b):
    """Compute the sparse-oracle Λ for a `MixedHamiltonian`.

    Walks `fermion_part`, `boson_part`, and `mixed_terms` and applies
    the appropriate per-term rescaling factor (Pauli 1-norm on the
    fermion side, sparse-oracle factor on the boson side, Gilyén
    product on mixed terms).

    Args:
        mixed_hamiltonian: `MixedHamiltonian` from `fock_native.py`.
        n_b: bits per pion species (determines `N_f = 2^n_b`).

    Returns:
        dict with:
          - 'physical_lambda':  total Λ (float).
          - 'identity_shift':   total classical energy shift (real float).
          - 'per_part_lambdas': dict breakdown by ('fermion',
            'boson_sparse', 'mixed_sparse').
          - 'per_part_identity_shifts': dict breakdown of identity shifts.
    """
    if not isinstance(mixed_hamiltonian, MixedHamiltonian):
        raise TypeError(
            "compute_native_lambda expects a MixedHamiltonian, "
            f"got {type(mixed_hamiltonian).__name__}"
        )
    mh = mixed_hamiltonian

    # Pure-fermion contribution: JW + Pauli 1-norm.
    lambda_fermion, id_fermion = _pauli_lambda_for_fermion_op(mh.fermion_part)

    # Pure-boson contribution: per-monomial sparse-oracle λ.
    lambda_boson, id_boson = _sparse_lambda_for_boson_op(mh.boson_part, n_b)

    # Mixed contribution: Gilyén product over (fermion · boson) factors.
    lambda_mixed = 0.0
    for mt in mh.mixed_terms:
        lam_f, _ = _pauli_lambda_for_fermion_op(mt.fermion_factor)
        lam_b, _ = _sparse_lambda_for_boson_op(mt.boson_factor, n_b)
        # Mixed terms carry no pure-identity contribution by construction
        # (fermion factor has no identity term after JW of a bilinear, and
        # boson factor for H_AV/H_WT is a polynomial in (â + â†), never
        # the identity).
        lambda_mixed += abs(mt.coeff) * lam_f * lam_b

    return {
        'physical_lambda': lambda_fermion + lambda_boson + lambda_mixed,
        'identity_shift':  id_fermion + id_boson,
        'per_part_lambdas': {
            'fermion':       lambda_fermion,
            'boson_sparse':  lambda_boson,
            'mixed_sparse':  lambda_mixed,
        },
        'per_part_identity_shifts': {
            'fermion': id_fermion,
            'boson':   id_boson,
        },
    }
