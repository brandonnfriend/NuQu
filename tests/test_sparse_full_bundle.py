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


# --------------------------------------------------------------------------- #
# Step 3 §6.1 — α_tot invariant on real bundles                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('L,dim', [(2, 1), (2, 2), (2, 3)])
def test_bundle_alpha_tot_invariant(L, dim):
    """be.alpha == compute_native_lambda(mh, n_b)['physical_lambda'] to ~machine
    precision — the block-encoding subnormalization matches the Λ every Λ/N_walk
    downstream uses. Retires the α half of the reflection/α correctness risk."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        FullBundleProblemInstance, SparseFullBundleBlockEncoding,
    )
    n_b = 2
    mh = build_native_mixed_hamiltonian(L, dim, n_b, get_physical_parameters())
    pi = FullBundleProblemInstance(mh, n_b, num_sites=L ** dim)
    be = SparseFullBundleBlockEncoding(pi)
    lam = compute_native_lambda(mh, n_b)['physical_lambda']
    assert abs(be.alpha - lam) <= 1e-9 * max(1.0, lam), (
        f"L={L} dim={dim}: α_tot={be.alpha} != physical_lambda={lam}")


def test_bundle_keeps_imaginary_boson_coefficients():
    """Regression: H_WT's conjugate-momentum Π pieces give *imaginary* boson-factor
    coefficients. Every mixed term must be kept (not real-projected to zero), so
    the atom count matches len(mixed_terms) and α_tot is exact."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.bundle_encoding import extract_atoms
    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    atoms = extract_atoms(mh, 2)
    n_mixed = sum(1 for a in atoms if a.kind == 'mixed')
    assert n_mixed == len(mh.mixed_terms) == 15, (
        f"kept {n_mixed} mixed atoms of {len(mh.mixed_terms)} — imaginary "
        "boson coeffs were dropped")


def test_bundle_per_part_weights_match_compute_native_lambda():
    """Per-kind Σ weight equals compute_native_lambda's per_part_lambdas."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
    from src_PI.estimation.sparse_oracle.bundle_encoding import extract_atoms
    mh = build_native_mixed_hamiltonian(2, 2, 2, get_physical_parameters())
    atoms = extract_atoms(mh, 2)
    mine = {'boson': 0.0, 'fermion': 0.0, 'mixed': 0.0}
    for a in atoms:
        mine[a.kind] += a.weight
    ref = compute_native_lambda(mh, 2)['per_part_lambdas']
    assert abs(mine['boson'] - ref['boson_sparse']) < 1e-9
    assert abs(mine['mixed'] - ref['mixed_sparse']) < 1e-9
    assert abs(mine['fermion'] - ref['fermion']) < 1e-9


# --------------------------------------------------------------------------- #
# Step 3 §6.5 — toy assembly sim (retires the reflection-subspace risk)       #
# --------------------------------------------------------------------------- #


def _toy_bundle(boson_part, fermion_part, mode_to_qubits, n_b=1,
                num_sites=1, num_pion_species=1):
    from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        FullBundleProblemInstance, SparseFullBundleBlockEncoding,
    )
    mh = MixedHamiltonian(boson_part=boson_part, fermion_part=fermion_part,
                          mode_to_qubits=mode_to_qubits)
    pi = FullBundleProblemInstance(mh, n_b, num_sites=num_sites,
                                   num_pion_species=num_pion_species)
    return SparseFullBundleBlockEncoding(pi)


def _assembly_error(be):
    import numpy as np
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        extracted_bundle_block, bundle_reference_matrix,
    )
    return float(np.linalg.norm(extracted_bundle_block(be) - bundle_reference_matrix(be)))


def test_toy_assembly_boson_only_mixed_phases():
    """α_tot·⟨0|_flag U|0⟩_flag = H for a boson-only toy with real ± and imaginary
    phases — validates PREP magnitudes, D_phase (real+imag), dispatch, reflection
    subspace, operator assembly."""
    from openfermion import BosonOperator, FermionOperator
    bp = (BosonOperator('0', 1.0) + BosonOperator('0^', -1.0)
          + BosonOperator('0^ 0', 1.0j))
    be = _toy_bundle(bp, FermionOperator(), {0: [0]})
    assert _assembly_error(be) < 1e-9


def test_toy_assembly_heterogeneous_boson_and_fermion():
    """Heterogeneous toy: a 2-mode boson atom (K=2 → multi-qubit shared ancilla,
    A_atom=2), an imaginary-phase number op, AND a fermion atom (dilation). All
    block-flag qubits are reflected; the encoded operator is exact."""
    from openfermion import BosonOperator, FermionOperator
    bp = BosonOperator('0 1', 0.7) + BosonOperator('0^ 0', -0.5j)
    fp = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    be = _toy_bundle(bp, fp, {0: [2], 1: [3]})
    # confirm the multi-qubit shared ancilla actually arises
    assert be.PI._A_atom >= 2
    assert _assembly_error(be) < 1e-9


def test_toy_all_flag_qubits_in_selection_registers():
    """Every block-flag qubit (outer select + shared atom ancilla) is in
    selection_registers, so QubitizedWalkOperator reflects about exactly the
    flagged block (design risk #1). No flag qubit hides in junk."""
    from openfermion import BosonOperator, FermionOperator
    bp = BosonOperator('0 1', 0.7) + BosonOperator('0^ 0', -0.5j)
    fp = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    be = _toy_bundle(bp, fp, {0: [2], 1: [3]})
    sel_bits = sum(r.total_bits() for r in be.selection_registers)
    assert sel_bits == be.PI._b_out + be.PI._A_atom


# --------------------------------------------------------------------------- #
# Step 3 — compiled walk cost runs end-to-end                                 #
# --------------------------------------------------------------------------- #


def test_bundle_estimate_resources_runs():
    """estimate_resources(QubitizedWalkOperator(be)) returns a finite compiled
    T-count on the real L=2 dim=1 bundle (uses _t_complexity_, not the sim)."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        FullBundleProblemInstance, SparseFullBundleBlockEncoding,
    )
    from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
    from pyLIQTR.utils.resource_analysis import estimate_resources
    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    pi = FullBundleProblemInstance(mh, 2, num_sites=2)
    be = SparseFullBundleBlockEncoding(pi)
    res = estimate_resources(QubitizedWalkOperator(be))
    assert res['T'] > 0 and res['Clifford'] > 0
    assert res['LogicalQubits'] >= pi._w_flag + pi._w_sys - 8   # ≈ within junk slack


# --------------------------------------------------------------------------- #
# Step 4 §6.4 — compiled-vs-analytical A/B                                     #
# --------------------------------------------------------------------------- #


def test_compiled_vs_analytical_ab():
    """The compiled bundle and the analytical proxy are directly comparable:
    same α_tot, and the compiled Walk_T lands in a sane band around the proxy
    (between its boson-ceiling and fermion-floor). The ratio + per-kind
    breakdown is the 'cost of honest compilation' deliverable."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
    from src_PI.estimation.sparse_oracle.resources import compiled_vs_analytical

    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    ab = compiled_vs_analytical(mh, 2, num_sites=2)
    c = ab['compiled']
    # genuine compiled numbers are positive
    assert c['Walk_T_Count'] > 0 and c['Walk_Clifford_Count'] > 0
    # same subnormalization as the analytical path (compute_native_lambda)
    lam = compute_native_lambda(mh, 2)['physical_lambda']
    assert abs(c['Physical_Lambda'] - lam) <= 1e-9 * max(1.0, lam)
    # compiled lands within a small factor of the proxy (no order-of-magnitude
    # surprise — the proxy was a reasonable, if mixed-bound, estimate)
    assert 0.2 < ab['ratio'] < 5.0
    # the breakdown accounts for every atom
    bk = c['compiled_breakdown']
    assert sum(v['count'] for v in bk['per_kind'].values()) == bk['n_atoms']


# --------------------------------------------------------------------------- #
# KNOWN DEFECT (quantum-algorithms review, 2026-08-18) — walk validity         #
# --------------------------------------------------------------------------- #
#
# The d=1 per-mode atoms encode non-Hermitian monomials (â, â† alone), so the
# bundle block encoding U is NOT Hermitian (‖U−U†‖ ≫ 0). pyLIQTR's
# QubitizedWalkOperator is a SINGLE-reflection walk W = (2Π−I)·U, which
# qubitizes only a Hermitian U. Hence estimate_resources(QubitizedWalkOperator)
# costs an object that does NOT run QPE correctly. The block encoding
# (α_tot·⟨0|U|0⟩ = H) and α_tot are exact; the *walk* is not yet valid.
#
# Fix = Hermitize the atoms (re-pair conjugate monomials c·m + c̄·m† into
# Hermitian d=2 encoders; Hermitian dilation for diagonal n̂; fermion PauliLCU
# atoms are already Hermitian). α_tot is preserved under re-pairing. This test
# is the regression gate for that fix: it is expected to FAIL until Hermitized,
# and strict=True flips the suite red the moment it starts passing.


@pytest.mark.xfail(reason="d=1 atoms non-Hermitian -> single-reflection walk not "
                          "a valid qubitization; Hermitization pending", strict=True)
def test_bundle_walk_qubitizes_hermitian_H():
    """W = (2Π−I)·U must have the qubitization spectrum e^{±i·arccos(E_k/α)}."""
    import cirq
    import numpy as np
    from openfermion import BosonOperator
    from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
    from src_PI.estimation.sparse_oracle.bundle_encoding import (
        FullBundleProblemInstance, SparseFullBundleBlockEncoding,
        bundle_reference_matrix,
    )
    # Hermitian toy: H = (â + â†) + 0.5 n̂ on one n_b=2 mode.
    bp = BosonOperator('0', 1.0) + BosonOperator('0^', 1.0) + BosonOperator('0^ 0', 0.5)
    mh = MixedHamiltonian(boson_part=bp, mode_to_qubits={0: [0, 1]})
    pi = FullBundleProblemInstance(mh, 2, num_sites=0, num_pion_species=0)
    be = SparseFullBundleBlockEncoding(pi)
    alpha = pi._alpha
    flag = [cirq.NamedQubit(f'f{i}') for i in range(pi._w_flag)]
    sysq = [cirq.NamedQubit(f's{i}') for i in range(pi._w_sys)]
    U = cirq.Circuit(be.decompose_from_registers(context=None, flag=flag, system=sysq)
                     ).unitary(qubit_order=flag + sysq)
    dim_sys = 1 << pi._w_sys
    Pi = np.diag([1.0] * dim_sys + [0.0] * (len(U) - dim_sys))
    W = (2 * Pi - np.eye(len(U))) @ U
    wphases = np.angle(np.linalg.eigvals(W))
    E = np.linalg.eigvalsh(bundle_reference_matrix(be))
    for th in np.arccos(np.clip(E / alpha, -1, 1)):
        dist = np.min(np.abs((wphases - th + np.pi) % (2 * np.pi) - np.pi))
        assert dist < 1e-6, f"qubitization phase {th:.4f} absent from walk spectrum"


# --------------------------------------------------------------------------- #
# Step 5 — Config switch + cache + full-pipeline dispatch                      #
# --------------------------------------------------------------------------- #


def test_config_sparse_oracle_mode_axis():
    from src_PI.utils.Config import Config
    assert Config().sparse_oracle_mode == 'analytical'          # default unchanged
    c = Config(block_encoder='sparse', sparse_oracle_mode='compiled')
    assert c.sparse_oracle_mode == 'compiled'
    assert Config.from_dict(c.to_dict()).sparse_oracle_mode == 'compiled'
    with pytest.raises(ValueError, match='sparse_oracle_mode'):
        Config(sparse_oracle_mode='nope')


def _sparse_estimate(mode):
    """Run the full sparse pipeline at L=2 dim=1 n_b=2 in the given oracle mode."""
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.block_encoders.sparse import SparseStrategy
    from src_PI.utils.Config import Config
    cfg = Config(pion_basis='fock', block_encoder='sparse', sparse_oracle_mode=mode)
    bundle, _q, ns = build_eft_hamiltonian(
        2, 1, 2, pi_max=0.0, params=get_physical_parameters(), config=cfg)
    return SparseStrategy().estimate(bundle, ns, 2, cfg)


def test_sparse_switch_dispatches_both_modes():
    """The 'compiled' switch routes SparseStrategy to the walk-VALID Hermitian
    bundle; 'analytical' keeps the proxy. The Hermitian Λ is *tighter* than the
    proxy's per-monomial 1-norm (edge colouring), and both walk numbers are
    positive."""
    a = _sparse_estimate('analytical')
    c = _sparse_estimate('compiled')
    assert a['sparse_oracle_mode'] == 'analytical'
    assert c['sparse_oracle_mode'] == 'compiled'
    # Hermitian Λ ≤ analytical Λ (tighter), both positive
    assert 0 < c['Physical_Lambda'] <= a['Physical_Lambda'] + 1e-6 * a['Physical_Lambda']
    assert a['Walk_T_Count'] > 0 and c['Walk_T_Count'] > 0
    assert a['Walk_T_Count'] != c['Walk_T_Count']


def test_compiled_estimate_cache_returns_identical():
    """The compiled walk estimate is cached on an A-independent bundle signature."""
    from src_PI.hamiltonians.core.pion_basis.fock_native import (
        build_native_mixed_hamiltonian,
    )
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.sparse_oracle.resources import (
        estimate_sparse_resources_compiled,
    )
    mh = build_native_mixed_hamiltonian(2, 1, 2, get_physical_parameters())
    r1 = estimate_sparse_resources_compiled(mh, 2, 2)
    r2 = estimate_sparse_resources_compiled(mh, 2, 2)
    assert r1['Walk_T_Count'] == r2['Walk_T_Count']
    assert r1['Logical_Qubits'] == r2['Logical_Qubits']


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
