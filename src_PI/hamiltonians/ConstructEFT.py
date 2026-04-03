from openfermion import jordan_wigner
from src_PI.hamiltonians.core.DynamicalPion import Full_Dynamical_Pion_Hamiltonian
from src_PI.hamiltonians.core.StaticTerms import Static_Nucleon_Hamiltonian
from src_PI.utils.LatticeGeometry import total_qubits, get_total_sites

def build_eft_hamiltonian(L, dim, n_b, pi_max, params):
    """Constructs the full EFT Hamiltonian for D-dimensions."""
    num_sites = get_total_sites(L, dim)
    q_count = total_qubits(L, dim, n_b)
    
    # 1. Build Static Nucleon Sector
    H_static_f = Static_Nucleon_Hamiltonian(
        params['h'], params['C'], params['CI'], L, dim, n_b
    )
    H_static_q = jordan_wigner(H_static_f)
    
    # 2. Build Dynamical Pion Sector
    H_pos_dyn, H_mom = Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, pi_max, params)
    
    # Combine all position-basis terms
    H_pos = H_static_q + H_pos_dyn
    
    return H_pos, H_mom, q_count, num_sites