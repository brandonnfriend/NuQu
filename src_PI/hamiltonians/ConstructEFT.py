from openfermion import jordan_wigner

from src_PI.hamiltonians.core import pion_basis
from src_PI.hamiltonians.core.HamiltonianBundle import HamiltonianBundle
from src_PI.hamiltonians.core.StaticTerms import Static_Nucleon_Hamiltonian
from src_PI.hamiltonians.core.SubHamiltonian import SubHamiltonian
from src_PI.utils.LatticeGeometry import total_qubits, get_total_sites


# Block encoders that consume native-algebra (BosonOperator + FermionOperator)
# inputs in the Fock basis. PauliLCU goes through the existing eager-Pauli
# path; sparse and LOBE route to the parallel native builder.
_NATIVE_FOCK_ENCODERS = frozenset({'sparse', 'lobe'})


def _use_native_fock_path(config):
    """True iff the (basis, encoder) combination should build native algebra.

    Currently: Fock basis + (sparse | lobe). Amplitude basis is Pauli-native
    so it always uses the existing path regardless of encoder.
    """
    return (config.pion_basis == 'fock'
            and config.block_encoder in _NATIVE_FOCK_ENCODERS)


def build_eft_hamiltonian(L, dim, n_b, pi_max, params, config):
    """Constructs the full EFT HamiltonianBundle for D dimensions.

    Routes per (config.pion_basis, config.block_encoder):
      * amplitude + any encoder: existing amplitude path (algebra='pauli').
      * fock + 'pauli_lcu':      existing fock.py path (algebra='pauli').
      * fock + 'sparse'/'lobe':  new fock_native.py path
                                 (algebra='fermion_boson', `MixedHamiltonian`
                                  payload).

    The static-nucleon sector is added in both cases:
      * Pauli path: Jordan-Wigner first, fold the resulting QubitOperator
        into the first sub-Hamiltonian.
      * Native path: keep as a FermionOperator and fold into
        `MixedHamiltonian.fermion_part` without applying JW (pseudocode
        demand — fermion ladder ops stay native until the encoder
        boundary).

    Args:
        L (int): lattice side length.
        dim (int): spatial dimension.
        n_b (int): number of qubits per pion species per site. In the
            Fock basis this is n_q = log₂(N_f); in the amplitude basis
            it is the field-amplitude binary width.
        pi_max (float): amplitude-basis field cutoff. Ignored for Fock.
        params (dict): physical parameters from EFTParameters.
        config (Config): pipeline config object. Selects pion_basis,
            block_encoder, and walk_mode.

    Returns:
        bundle (HamiltonianBundle): the assembled bundle.
        q_count (int): total qubit count of the lattice register.
        num_sites (int): total number of lattice sites.
    """
    num_sites = get_total_sites(L, dim)
    q_count = total_qubits(L, dim, n_b)

    # Static Nucleon Sector (basis-independent; lives on nucleon qubits).
    H_static_f = Static_Nucleon_Hamiltonian(
        params['h'], params['C'], params['CI'], L, dim, n_b
    )

    if _use_native_fock_path(config):
        # Native Fock path: fermion stays as FermionOperator, boson as
        # BosonOperator, mixed terms unmultiplied.
        from src_PI.hamiltonians.core.pion_basis import fock_native

        mh = fock_native.build_native_mixed_hamiltonian(L, dim, n_b, params)
        # Fold the static-nucleon FermionOperator into the MixedHamiltonian's
        # fermion_part without applying Jordan-Wigner.
        mh.fermion_part = mh.fermion_part + H_static_f
        sub_h = SubHamiltonian(
            name='fock', operator=mh, algebra='fermion_boson'
        )
        metadata = {
            'pion_basis': 'fock',
            'block_encoder': config.block_encoder,
            'L': L, 'dim': dim, 'n_b': n_b, 'num_sites': num_sites,
        }
        bundle = HamiltonianBundle(
            [sub_h], walk_mode=config.walk_mode, metadata=metadata
        )
        return bundle, q_count, num_sites

    # Pauli path (current behavior, unchanged from Phase A).
    H_static_q = jordan_wigner(H_static_f)
    basis_module = pion_basis.get(config.pion_basis)
    sub_hamiltonians = basis_module.Full_Dynamical_Pion_Hamiltonian(
        L, dim, n_b, pi_max, params
    )
    name_first, op_first = sub_hamiltonians[0]
    sub_hamiltonians[0] = (name_first, H_static_q + op_first)

    metadata = {
        'pion_basis': basis_module.BASIS_NAME,
        'L': L,
        'dim': dim,
        'n_b': n_b,
        'num_sites': num_sites,
    }
    bundle = HamiltonianBundle(
        sub_hamiltonians,
        walk_mode=config.walk_mode,
        metadata=metadata,
    )
    return bundle, q_count, num_sites
