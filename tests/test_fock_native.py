"""
Smoke tests for the native-algebra Fock builder (Phase B).

These check that `build_eft_hamiltonian` with `block_encoder='sparse'` or
`block_encoder='lobe'` produces a well-formed `MixedHamiltonian` carried in
a `SubHamiltonian` tagged `algebra='fermion_boson'`. The PauliLCU path is
exercised by the Phase 0 / Phase A regression baselines elsewhere; this
file just covers the new code path.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openfermion import BosonOperator, FermionOperator

from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.hamiltonians.core.HamiltonianBundle import HamiltonianBundle
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian, MixedTerm
from src_PI.hamiltonians.core.SubHamiltonian import SubHamiltonian
from src_PI.utils.Config import Config


def _build_native(L=2, dim=3, n_b=3, encoder='sparse'):
    """Build the EFT through the native-Fock dispatch."""
    config = Config(pion_basis='fock', block_encoder=encoder)
    params = get_physical_parameters()
    bundle, q_count, num_sites = build_eft_hamiltonian(
        L, dim, n_b, pi_max=0.0, params=params, config=config
    )
    return bundle, q_count, num_sites


def test_native_dispatch_for_sparse():
    """Fock + 'sparse' should land in the native fermion_boson path."""
    bundle, _, _ = _build_native(encoder='sparse')
    assert isinstance(bundle, HamiltonianBundle)
    assert len(bundle) == 1
    sh = bundle.sub_hamiltonians[0]
    assert isinstance(sh, SubHamiltonian)
    assert sh.name == 'fock'
    assert sh.algebra == 'fermion_boson'
    assert isinstance(sh.operator, MixedHamiltonian)


def test_native_dispatch_for_lobe():
    """Fock + 'lobe' should also land in the native fermion_boson path."""
    bundle, _, _ = _build_native(encoder='lobe')
    sh = bundle.sub_hamiltonians[0]
    assert sh.algebra == 'fermion_boson'
    assert isinstance(sh.operator, MixedHamiltonian)


def test_native_mixed_hamiltonian_structure():
    """MixedHamiltonian should have non-empty fermion/boson/mixed parts."""
    bundle, _, _ = _build_native()
    mh = bundle.sub_hamiltonians[0].operator
    assert isinstance(mh, MixedHamiltonian)
    assert isinstance(mh.fermion_part, FermionOperator)
    assert isinstance(mh.boson_part, BosonOperator)
    assert isinstance(mh.mode_to_qubits, dict)
    assert isinstance(mh.mixed_terms, list)

    # Static nucleon (Free_Hopping + Free_Onsite + HC + HCI2) contributes.
    assert len(mh.fermion_part.terms) > 0, "fermion_part should carry static nucleon FermionOps"

    # H_pion_free contributes (n̂ per mode + gradient + identity zero-point).
    assert len(mh.boson_part.terms) > 0, "boson_part should carry H_pion_free BosonOps"

    # H_AV + H_WT contribute MixedTerms.
    assert len(mh.mixed_terms) > 0, "mixed_terms should carry H_AV / H_WT contributions"
    for mt in mh.mixed_terms:
        assert isinstance(mt, MixedTerm)
        assert isinstance(mt.fermion_factor, FermionOperator)
        assert isinstance(mt.boson_factor, BosonOperator)
        assert len(mt.fermion_factor.terms) > 0
        assert len(mt.boson_factor.terms) > 0


def test_native_mode_indexing():
    """mode_to_qubits should cover every (site, species) global mode."""
    L, dim, n_b = 2, 3, 3
    bundle, _, num_sites = _build_native(L=L, dim=dim, n_b=n_b)
    mh = bundle.sub_hamiltonians[0].operator
    expected_modes = num_sites * 3
    assert len(mh.mode_to_qubits) == expected_modes
    # Each mode maps to exactly n_b backing qubits.
    for mode, qubits in mh.mode_to_qubits.items():
        assert isinstance(mode, int)
        assert len(qubits) == n_b


def test_pauli_path_still_pauli_for_pauli_lcu():
    """Regression: Fock + 'pauli_lcu' must still produce the legacy Pauli path."""
    config = Config(pion_basis='fock', block_encoder='pauli_lcu')
    params = get_physical_parameters()
    bundle, _, _ = build_eft_hamiltonian(
        L=2, dim=3, n_b=3, pi_max=0.0, params=params, config=config
    )
    sh = bundle.sub_hamiltonians[0]
    assert sh.algebra == 'pauli'
    # Should be a QubitOperator (eager Pauli expansion, unchanged from Phase A).
    from openfermion import QubitOperator
    assert isinstance(sh.operator, QubitOperator)


if __name__ == '__main__':
    # Bare-bones runner for `python tests/test_fock_native.py`.
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"running {name} ...", end=' ', flush=True)
            fn()
            print("PASS")
    print("\nAll fock_native smoke tests passed.")
