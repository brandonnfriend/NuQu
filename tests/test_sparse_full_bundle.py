"""
Validation gates for the compiled sparse full-bundle block encoding (C1).

Built up step-by-step alongside the implementation, in the gated order of
`docs/sparse_full_bundle_design.md` §6:

  Step 1 (this commit) — `SparseBosonMonomialBlockEncoding`:
    * §6.2 per-atom block-matrix sim: `α_l·⟨0|U_l|0⟩` vs the exact truncated
      monomial matrix, over EVERY monomial shape that appears in the real
      `MixedHamiltonian` (linear â/â†, number op, two-mode products), n_b=2,3.
    * α invariant (local): each monomial's `α = ∏_m α_m` equals
      `lambda_compute._monomial_max_amplitude` — the per-atom piece of the
      global α_tot invariant (§6.1).
    * the pyLIQTR `BlockEncoding` subclass decomposes and costs through
      `estimate_resources(QubitizedWalkOperator(be))`.

  Steps 2-5 add: fermion-atom vs PauliLCU (§6.3), α_tot invariant + scaled-toy
  assembly sim (§6.1/§6.5), compiled-vs-analytical A/B (§6.4).

Run: `python -m pytest tests/test_sparse_full_bundle.py -q`
"""

import math

import numpy as np
import pytest

from src_PI.estimation.sparse_oracle.boson_monomial_encoding import (
    BosonMonomialProblemInstance,
    SparseBosonMonomialBlockEncoding,
    build_boson_monomial_circuit,
    extracted_monomial_block,
    monomial_alpha,
    monomial_reference_matrix,
)
from src_PI.estimation.sparse_oracle.lambda_compute import _monomial_max_amplitude


# Every distinct boson-monomial shape that appears in a real MixedHamiltonian
# (verified by enumerating build_native_mixed_hamiltonian(2,1,2,params)):
#   boson_part: single-mode 2-factor {aa, a†a†, a†a, aa†}, two-mode 2-factor.
#   mixed-term boson factors: single-mode 1-factor {a, a†}, two-mode 2-factor.
_MONOMIAL_SHAPES = {
    'a_linear':        ((0, 0),),
    'adag_linear':     ((0, 1),),
    'aa_down2':        ((0, 0), (0, 0)),
    'adagadag_up2':    ((0, 1), (0, 1)),
    'number_adag_a':   ((0, 1), (0, 0)),
    'number_a_adag':   ((0, 0), (0, 1)),
    'twomode_a0_a1':   ((0, 0), (1, 0)),
    'twomode_ad0_a1':  ((0, 1), (1, 0)),
    'twomode_a0_ad1':  ((0, 0), (1, 1)),
    'twomode_ad0_ad1': ((0, 1), (1, 1)),
}

_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Step 1 §6.2 — per-atom block-matrix simulation                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('name,monomial', list(_MONOMIAL_SHAPES.items()))
@pytest.mark.parametrize('n_b', [2, 3])
def test_boson_monomial_block_matches_exact(name, monomial, n_b):
    """α·⟨0|U|0⟩ reproduces the exact truncated monomial matrix to ~machine eps."""
    circuit, amps, mregs, alpha = build_boson_monomial_circuit(monomial, n_b)
    extracted = extracted_monomial_block(circuit, amps, mregs, n_b, alpha)
    reference = monomial_reference_matrix(monomial, n_b)
    err = np.linalg.norm(extracted - reference)
    assert err < _TOL, f"{name} (n_b={n_b}): ||extracted - exact||_F = {err:.3e}"


@pytest.mark.parametrize('name,monomial', list(_MONOMIAL_SHAPES.items()))
@pytest.mark.parametrize('n_b', [2, 3])
def test_boson_monomial_alpha_equals_lambda_compute(name, monomial, n_b):
    """Per-atom α = ∏_m α_m equals lambda_compute's _monomial_max_amplitude.

    This is the per-monomial piece of the global α_tot invariant (§6.1): the
    encoder's rescale factor is *exactly* the number `compute_native_lambda`
    sums, so no α mismatch can enter the bundle.
    """
    a_enc = monomial_alpha(monomial, n_b)
    a_lc = _monomial_max_amplitude(monomial, n_b)
    assert abs(a_enc - a_lc) < 1e-12, (
        f"{name} (n_b={n_b}): encoder α={a_enc} != lambda_compute α={a_lc}"
    )


def test_boson_monomial_build_returns_alpha_product():
    """build_boson_monomial_circuit's α equals monomial_alpha (the ∏_m α_m)."""
    for monomial in _MONOMIAL_SHAPES.values():
        _c, _a, _m, alpha = build_boson_monomial_circuit(monomial, 2)
        assert abs(alpha - monomial_alpha(monomial, 2)) < 1e-12


# --------------------------------------------------------------------------- #
# Step 1 — pyLIQTR BlockEncoding subclass decomposes + costs                   #
# --------------------------------------------------------------------------- #


def test_boson_monomial_block_encoding_estimates():
    """The pyLIQTR BlockEncoding subclass runs through estimate_resources and
    reports α equal to the problem instance's α (no rescale drift)."""
    from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
    from pyLIQTR.utils.resource_analysis import estimate_resources

    for monomial in (
        ((0, 1), (0, 0)),       # number op (single mode, no shift)
        ((0, 0),),              # linear (single mode, shift)
        ((0, 1), (1, 0)),       # two-mode product
    ):
        pi = BosonMonomialProblemInstance(monomial, 2)
        be = SparseBosonMonomialBlockEncoding(pi)
        walk = QubitizedWalkOperator(be)
        res = estimate_resources(walk)
        assert res['T'] > 0
        assert res['LogicalQubits'] >= 2
        # the encoder's α is the exact d=1 sparse-oracle rescale
        assert abs(pi.get_alpha() - _monomial_max_amplitude(monomial, 2)) < 1e-12


def test_boson_monomial_number_op_has_no_shift():
    """The number operator â†â is diagonal (Δn=0) → no AddK in the decomposition;
    a linear â carries a Δn=∓1 shift → one AddK. Structural guard on the
    per-mode shift logic."""
    from src_PI.estimation.sparse_oracle.boson_monomial_encoding import (
        single_mode_monomial_matrix, _column_shift_and_values,
    )
    # number operator: diagonal, Δn=0
    M_num = single_mode_monomial_matrix((1, 0), 2)
    delta_num, _ = _column_shift_and_values(M_num)
    assert delta_num == 0
    # annihilation: Δn=-1
    M_a = single_mode_monomial_matrix((0,), 2)
    delta_a, _ = _column_shift_and_values(M_a)
    assert delta_a == -1
    # creation: Δn=+1
    M_ad = single_mode_monomial_matrix((1,), 2)
    delta_ad, _ = _column_shift_and_values(M_ad)
    assert delta_ad == +1


# --------------------------------------------------------------------------- #
# Step 2 §6.3 — fermion atom via off-the-shelf PauliLCU                        #
# --------------------------------------------------------------------------- #


def _first_mixed_fermion_factor(L=2, dim=1, n_b=2):
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    mh = build_native_mixed_hamiltonian(L, dim, n_b, get_physical_parameters())
    return mh.mixed_terms[0].fermion_factor


def test_fermion_atom_alpha_equals_pauli_one_norm():
    """The fermion atom's α (== encoding.alpha) equals fermion_jw_stats' Pauli
    1-norm — its exact contribution to the global α_tot invariant."""
    from src_PI.estimation.sparse_oracle.fermion_atom import (
        fermion_atom_encoding, fermion_atom_alpha,
    )
    ff = _first_mixed_fermion_factor()
    enc = fermion_atom_encoding(ff)
    assert abs(enc.alpha - fermion_atom_alpha(ff)) < 1e-12


def test_fermion_atom_cost_is_genuine_not_lower_bound():
    """The genuine PauliLCU walk cost strictly exceeds the retired `4·weight`
    lower-bound proxy — demonstrating the fermion floor is gone."""
    from src_PI.estimation.sparse_oracle.fermion_atom import fermion_atom_walk_cost
    from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
    ff = _first_mixed_fermion_factor()
    genuine = fermion_atom_walk_cost(ff)
    proxy_T = 4 * fermion_jw_stats(ff)['total_weight']
    assert genuine['T'] > proxy_T, (
        f"genuine PauliLCU T={genuine['T']} should exceed proxy floor {proxy_T}"
    )
    assert genuine['LogicalQubits'] >= 4


def test_fermion_atom_matches_standalone_pauli_lcu_term_for_term():
    """The fermion atom's encoding is bit-for-bit the standalone PauliLCU path
    (`estimators._ham_to_pyliqtr_instance` → getEncoding) on the same JW image:
    same α and same estimate_resources T/Clifford/LogicalQubits."""
    from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
    from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
    from pyLIQTR.utils.resource_analysis import estimate_resources
    from src_PI.estimation.estimators import _ham_to_pyliqtr_instance
    from src_PI.estimation.sparse_oracle.jw_cache import jordan_wigner_cached
    from src_PI.estimation.sparse_oracle.fermion_atom import fermion_atom_walk_cost

    ff = _first_mixed_fermion_factor()
    # standalone reference path (identical to the whole-Hamiltonian PauliLCU code)
    ref_inst = _ham_to_pyliqtr_instance(jordan_wigner_cached(ff))
    ref_enc = getEncoding(VALID_ENCODINGS.PauliLCU)(ref_inst)
    ref = estimate_resources(QubitizedWalkOperator(ref_enc))

    atom = fermion_atom_walk_cost(ff)
    assert abs(atom['alpha'] - ref_enc.alpha) < 1e-12
    assert atom['T'] == ref['T']
    assert atom['Clifford'] == ref['Clifford']
    assert atom['LogicalQubits'] == ref['LogicalQubits']


def test_fermion_atom_rejects_non_hermitian_factor():
    """A non-Hermitian fermion factor (complex JW image) must raise, never have
    its phase silently real-projected."""
    from openfermion import FermionOperator
    from src_PI.estimation.sparse_oracle.fermion_atom import fermion_pauli_dict
    # a†_0 a_1 alone is non-Hermitian → JW has ±i/2 XY/YX pieces
    with pytest.raises(ValueError, match="Hermitian"):
        fermion_pauli_dict(FermionOperator('0^ 1'))


def test_fermion_atom_empty_operator_is_zero():
    from openfermion import FermionOperator
    from src_PI.estimation.sparse_oracle.fermion_atom import fermion_atom_walk_cost
    cost = fermion_atom_walk_cost(FermionOperator())
    assert cost == {'T': 0, 'Clifford': 0, 'LogicalQubits': 0, 'alpha': 0.0}


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
