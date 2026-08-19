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


def test_compute_native_lambda_per_sector_values():
    """compute_native_lambda's per-sector rescale (n_b=3, N_f=8): boson d=1 max
    amplitude (â=√7, n̂=7·m_π), fermion Pauli 1-norm, and the Gilyén mixed
    product λ_mixed = |c|·λ_f·λ_b — plus the per-monomial max-amplitude values."""
    import math
    from src_PI.estimation.sparse_oracle.lambda_compute import (
        _sparse_lambda_for_boson_monomial,
    )
    # boson: â → √7
    assert abs(compute_native_lambda(
        MixedHamiltonian(boson_part=BosonOperator('0', 1.0)), 3)['physical_lambda']
        - math.sqrt(7.0)) < 1e-9
    # boson: n̂ (diagonal) → 7·m_π
    assert abs(compute_native_lambda(
        MixedHamiltonian(boson_part=BosonOperator('0^ 0', 135.0)), 3)['physical_lambda']
        - 135.0 * 7.0) < 1e-6
    # fermion: c·(a0†a1 + h.c.) → Pauli 1-norm = c
    assert abs(compute_native_lambda(
        MixedHamiltonian(fermion_part=3.0 * (FermionOperator('0^ 1') + FermionOperator('1^ 0'))),
        3)['physical_lambda'] - 3.0) < 1e-9
    # mixed: Gilyén product |c|·λ_f·λ_b = 2.5·1·√7
    mt = MixedTerm(coeff=2.5, fermion_factor=FermionOperator('0^ 1') + FermionOperator('1^ 0'),
                   boson_factor=BosonOperator('0'))
    assert abs(compute_native_lambda(MixedHamiltonian(mixed_terms=[mt]), 3)['physical_lambda']
               - 2.5 * math.sqrt(7.0)) < 1e-9
    # per-monomial max amplitudes: â†â† → √42, two-mode â0â1 → 7, identity → 1
    assert abs(_sparse_lambda_for_boson_monomial(((0, 1), (0, 1)), 3) - math.sqrt(42.0)) < 1e-9
    assert abs(_sparse_lambda_for_boson_monomial(((0, 0), (1, 0)), 3) - 7.0) < 1e-9
    assert _sparse_lambda_for_boson_monomial((), 3) == 1.0


# --- End-to-end via SparseStrategy --------------------------------------


def _build_native_bundle(L=2, dim=3, n_b=3):
    """Build the EFT in native Fock algebra (block_encoder='sparse')."""
    config = Config(pion_basis='fock', block_encoder='sparse')
    params = get_physical_parameters()
    return build_eft_hamiltonian(L, dim, n_b, pi_max=0.0, params=params, config=config)


def test_sparse_strategy_returns_full_resource_dict():
    """C3d.1: SparseStrategy now returns a finished resource dict for the full bundle."""
    bundle, _q_count, num_sites = _build_native_bundle()
    strat = get_block_encoder('sparse')
    result = strat.estimate(bundle, num_sites, n_b=3, config=None)
    # Mandatory shape — orchestrator + sweep + plot read these keys.
    for key in ('Walk_T_Count', 'Walk_Clifford_Count', 'Logical_Qubits',
                'Physical_Lambda', 'Per_Sub_Walk', 'walk_mode',
                'identity_shift', 'physical_lambda'):
        assert key in result, f"missing key in sparse return dict: {key}"
    assert result['Walk_T_Count'] > 0
    assert result['Logical_Qubits'] > 0
    assert result['Physical_Lambda'] > 0


def test_sparse_strategy_breakdown_counts_lcu_summands_correctly():
    """The breakdown's L_eff should equal boson + fermion + mixed term counts."""
    bundle, _q_count, num_sites = _build_native_bundle()
    strat = get_block_encoder('sparse')
    result = strat.estimate(bundle, num_sites, n_b=3, config=None)
    bd = result['Sparse_Breakdown']
    assert bd['L_eff'] == bd['boson_terms'] + bd['fermion_terms'] + bd['mixed_terms']
    # All three sectors contribute at full L=2 dim=3 n_b=3.
    assert bd['boson_terms'] > 0
    assert bd['fermion_terms'] > 0
    assert bd['mixed_terms'] > 0


def test_sparse_strategy_rejects_pauli_bundle():
    """If a bundle has algebra='pauli', sparse strategy refuses to dispatch."""
    config = Config(pion_basis='fock', block_encoder='pauli_lcu')
    params = get_physical_parameters()
    bundle, _q_count, num_sites = build_eft_hamiltonian(
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
    bundle, _q_count, _num_sites = _build_native_bundle()
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

    bundle_n, _q_count, _num_sites = _build_native_bundle()
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
