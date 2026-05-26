"""
Sub-phase C1 tests for the sparse-oracle Λ helper and SparseStrategy
dispatch. These cover the native-algebra Λ walker without invoking the
yet-to-be-built block-encoding circuit (raised separately as
NotImplementedError per the C1 contract).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openfermion import BosonOperator, FermionOperator

from src_PI.estimation.block_encoders import get_block_encoder
from src_PI.estimation.sparse_oracle import compute_native_lambda
from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian, MixedTerm
from src_PI.utils.Config import Config


# --- Λ helper directly --------------------------------------------------


def test_lambda_of_empty_mixed_hamiltonian_is_zero():
    mh = MixedHamiltonian()
    data = compute_native_lambda(mh, n_b=3)
    assert data['physical_lambda'] == 0.0
    assert data['identity_shift'] == 0.0


def test_lambda_picks_up_identity_shift_from_boson_part():
    mh = MixedHamiltonian(boson_part=BosonOperator('', 7.5))
    data = compute_native_lambda(mh, n_b=3)
    assert data['physical_lambda'] == 0.0
    assert abs(data['identity_shift'] - 7.5) < 1e-12


def test_lambda_for_single_ladder_monomial():
    """For `c · â_0` at n_b=3 (N_f=8): λ = 2 · 1 · √7 ≈ 5.29."""
    import math
    mh = MixedHamiltonian(boson_part=BosonOperator('0', 1.0))
    data = compute_native_lambda(mh, n_b=3)
    expected = 2.0 * 1.0 * math.sqrt(7.0)  # P=1, log2_S=1, max_amp=√(N_f-1)
    assert abs(data['physical_lambda'] - expected) < 1e-9


def test_lambda_for_number_operator():
    """For `m_π · â_0^† â_0` (P=2): λ = 2 · 2 · 7 = 28; coeff scaling applies."""
    m_pi = 135.0
    mh = MixedHamiltonian(boson_part=BosonOperator('0^ 0', m_pi))
    data = compute_native_lambda(mh, n_b=3)
    expected = m_pi * 2.0 * 2.0 * 7.0  # 2·log₂(4)·(N_f-1)^1
    assert abs(data['physical_lambda'] - expected) < 1e-6


def test_lambda_pure_fermion_uses_pauli_one_norm():
    """`c · a_0^† a_1 + h.c.` → JW gives 4 Pauli terms with |c|=0.5 each → 1-norm = 2|c|."""
    c = 3.0
    f = c * (FermionOperator('0^ 1') + FermionOperator('1^ 0'))
    mh = MixedHamiltonian(fermion_part=f)
    data = compute_native_lambda(mh, n_b=3)
    # JW of `c · (a_0^ a_1 + a_1^ a_0)` is `c/2 · (X_0 X_1 + Y_0 Y_1)` → 1-norm = c.
    assert abs(data['physical_lambda'] - abs(c)) < 1e-9


def test_mixed_term_lambda_is_product_of_factors():
    """Gilyén product: λ_mixed = |c| · λ_f · λ_b."""
    import math
    c = 2.5
    F = FermionOperator('0^ 1') + FermionOperator('1^ 0')  # JW 1-norm = 1
    B = BosonOperator('0')                                 # λ = 2·1·√7
    mh = MixedHamiltonian(mixed_terms=[MixedTerm(coeff=c, fermion_factor=F, boson_factor=B)])
    data = compute_native_lambda(mh, n_b=3)
    expected = abs(c) * 1.0 * (2.0 * 1.0 * math.sqrt(7.0))
    assert abs(data['physical_lambda'] - expected) < 1e-9


# --- End-to-end via SparseStrategy --------------------------------------


def _build_native_bundle(L=2, dim=3, n_b=3):
    """Build the EFT in native Fock algebra (block_encoder='sparse')."""
    config = Config(pion_basis='fock', block_encoder='sparse')
    params = get_physical_parameters()
    return build_eft_hamiltonian(L, dim, n_b, pi_max=0.0, params=params, config=config)


def test_sparse_strategy_computes_lambda_then_raises():
    """SparseStrategy.estimate computes Λ + identity_shift, then raises NotImplementedError."""
    bundle, num_sites, _ = _build_native_bundle()
    strat = get_block_encoder('sparse')
    try:
        strat.estimate(bundle, num_sites, n_b=3, config=None)
    except NotImplementedError as e:
        msg = str(e)
        assert 'sparse-oracle' in msg
        assert 'C2' in msg or 'circuit construction' in msg
    else:
        raise AssertionError(
            "SparseStrategy.estimate should raise NotImplementedError in C1"
        )


def test_sparse_strategy_rejects_pauli_bundle():
    """If a bundle has algebra='pauli', sparse strategy refuses to dispatch."""
    config = Config(pion_basis='fock', block_encoder='pauli_lcu')
    params = get_physical_parameters()
    bundle, num_sites, _ = build_eft_hamiltonian(
        L=2, dim=3, n_b=3, pi_max=0.0, params=params, config=config
    )
    strat = get_block_encoder('sparse')
    try:
        strat.estimate(bundle, num_sites, n_b=3, config=config)
    except TypeError as e:
        assert 'fermion_boson' in str(e)
    else:
        raise AssertionError(
            "SparseStrategy.estimate should raise TypeError on a Pauli bundle"
        )


def test_full_eft_native_lambda_is_finite_and_positive():
    """Smoke test on the full L=2 dim=3 n_b=3 EFT (the regression case).

    Notes on identity shift: the native path captures the TRUE classical
    shift (boson zero-point + JW(fermion)'s identity coeff). It does NOT
    include the binary-expansion artifacts that show up in PauliLCU's
    identity_shift (e.g. `_number_op_register` contributes
    `Σ_k 2^k/2 = (N_f-1)/2` of identity per mode, which cancels against
    non-identity Z-strings on the working subspace). So
    `native.identity_shift << PauliLCU.identity_shift` is expected at our
    sizes, and not a bug. See `04_refactor_execution_log.md` §6.
    """
    bundle, _, _ = _build_native_bundle()
    mh = bundle.sub_hamiltonians[0].operator
    data = compute_native_lambda(mh, n_b=3)
    assert data['physical_lambda'] > 0.0
    parts = data['per_part_lambdas']
    # All three sectors should contribute non-trivially at this size.
    assert parts['fermion'] > 0.0,      "static nucleon (fermion) should contribute"
    assert parts['boson_sparse'] > 0.0, "H_pion_free (boson) should contribute"
    assert parts['mixed_sparse'] > 0.0, "H_AV + H_WT (mixed) should contribute"
    # Boson identity = sum of m_π/2 per pion mode (num_sites × 3 species × m_π/2).
    expected_boson_id = 8 * 3 * 135.0 / 2.0  # L=2 dim=3 → 8 sites
    assert abs(data['per_part_identity_shifts']['boson'] - expected_boson_id) < 1e-9


def test_native_identity_shift_does_not_include_binary_expansion_artifacts():
    """Native identity_shift captures the *true* classical shift only.

    At L=2 dim=3 n_b=3 the binary expansion of `n̂` per mode contributes
    `(N_f − 1)/2 · m_π = 3.5 · 135 = 472.5` of *artifact* identity to
    PauliLCU's normalized shift on top of the genuine `m_π/2` zero-point.
    Across 24 modes that's ~11340 of additional Pauli-identity that does
    NOT belong in the native shift. Verify the native shift sits well
    below PauliLCU's by at least that margin (sanity).
    """
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe

    bundle_n, _, _ = _build_native_bundle()
    mh = bundle_n.sub_hamiltonians[0].operator
    native = compute_native_lambda(mh, n_b=3)

    config_p = Config(pion_basis='fock', block_encoder='pauli_lcu')
    params = get_physical_parameters()
    bundle_p, _, _ = build_eft_hamiltonian(
        L=2, dim=3, n_b=3, pi_max=0.0, params=params, config=config_p
    )
    pauli_id = float(normalize_for_qpe(bundle_p)['identity_shift'].real)

    expected_artifact_floor = 24 * 135 * 3.5  # 11340
    assert pauli_id - native['identity_shift'] > expected_artifact_floor


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll sparse-oracle C1 Λ tests passed.")
