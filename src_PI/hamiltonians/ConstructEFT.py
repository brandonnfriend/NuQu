from openfermion import jordan_wigner

from src_PI.hamiltonians.core import pion_basis
from src_PI.hamiltonians.core.HamiltonianBundle import HamiltonianBundle
from src_PI.hamiltonians.core.StaticTerms import Static_Nucleon_Hamiltonian
from src_PI.utils.LatticeGeometry import total_qubits, get_total_sites


def build_eft_hamiltonian(L, dim, n_b, pi_max, params, config):
    """Constructs the full EFT HamiltonianBundle for D dimensions.

    Routes to amplitude- or Fock-basis pion construction via `config.pion_basis`.
    The static-nucleon sector is added once and absorbed into the first
    sub-Hamiltonian (typically 'pos_dyn' for amplitude or 'fock' for Fock).

    Args:
        L (int): lattice side length.
        dim (int): spatial dimension.
        n_b (int): number of qubits per pion species per site. In the
            Fock basis this is n_q = log₂(N_f); in the amplitude basis
            it is the field-amplitude binary width.
        pi_max (float): amplitude-basis field cutoff. Ignored for Fock.
        params (dict): physical parameters from EFTParameters.
        config (Config): pipeline config object. Selects pion_basis and
            walk_mode.

    Returns:
        bundle (HamiltonianBundle): the assembled bundle.
        q_count (int): total qubit count of the lattice register.
        num_sites (int): total number of lattice sites.
    """
    num_sites = get_total_sites(L, dim)
    q_count = total_qubits(L, dim, n_b)

    # 1. Build Static Nucleon Sector (basis-independent; lives on nucleon qubits)
    H_static_f = Static_Nucleon_Hamiltonian(
        params['h'], params['C'], params['CI'], L, dim, n_b
    )
    H_static_q = jordan_wigner(H_static_f)

    # 2. Build Dynamical Pion Sector via the basis module
    basis_module = pion_basis.get(config.pion_basis)
    sub_hamiltonians = basis_module.Full_Dynamical_Pion_Hamiltonian(
        L, dim, n_b, pi_max, params
    )

    # 3. Fold the static term into the first sub-Hamiltonian.
    #    The static term commutes with pion ops (acts on disjoint qubits),
    #    so absorbing it into the first walk doesn't change semantics.
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
