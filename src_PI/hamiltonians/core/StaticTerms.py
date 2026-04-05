from openfermion import FermionOperator, normal_ordered
import numpy as np

# Use the dimension-agnostic geometry functions
from src_PI.utils.LatticeGeometry import site_to_nucleon_qubit, get_total_sites, get_neighbors

# Constants
fermionic_modes = [(0,0), (0,1), (1,0), (1,1)] # (spin, isospin)
tau_mats = [
    np.array([[0, 1], [1, 0]], dtype=complex),    # tau_1
    np.array([[0, -1j], [1j, 0]], dtype=complex), # tau_2
    np.array([[1, 0], [0, -1]], dtype=complex)    # tau_3
]

# --- Density Operator Helpers ---

def rho(site_id, n_b):
    """Total nucleon density at a site: sum_{alpha} a_alpha^dag a_alpha"""
    res = FermionOperator()
    for mode in fermionic_modes:
        idx = site_to_nucleon_qubit(site_id, mode, n_b)
        res += FermionOperator(f'{idx}^ {idx}')
    return res

def rho_I(I_idx, site_id, n_b):
    """Isospin density at a site: sum_{alpha,beta} a_alpha^dag [tau_I]_alpha,beta a_beta"""
    mat = tau_mats[I_idx - 1]
    res = FermionOperator()
    for s in [0, 1]: 
        for b in [0, 1]: 
            for c in [0, 1]:
                coeff = mat[b, c]
                if abs(coeff) > 1e-9:
                    idx_b = site_to_nucleon_qubit(site_id, (s, b), n_b)
                    idx_c = site_to_nucleon_qubit(site_id, (s, c), n_b)
                    res += FermionOperator(f'{idx_b}^ {idx_c}', coeff)
    return res

# --- Hamiltonian Components ---

def Free_Hopping(h, L, dim, n_b):
    """Kinetic hopping between adjacent sites (Agnostic to dimension)"""
    H_hop = FermionOperator()
    num_sites = get_total_sites(L, dim)
    
    for i in range(num_sites):
        neighbors = get_neighbors(i, L, dim)
        for j in neighbors:
            if j > i: # Enforce j > i to avoid double-counting the undirected bond
                for mode in fermionic_modes:
                    idx_i = site_to_nucleon_qubit(i, mode, n_b)
                    idx_j = site_to_nucleon_qubit(j, mode, n_b)
                    H_hop += FermionOperator(f'{idx_i}^ {idx_j}') + FermionOperator(f'{idx_j}^ {idx_i}')
    return -h * H_hop    

def Free_Onsite(h, L, dim, n_b):
    """On-site kinetic energy compensation. Coefficient scales as 2 * dim * h"""
    H_N = FermionOperator()
    num_sites = get_total_sites(L, dim)
    
    for site in range(num_sites):
        for mode in fermionic_modes:
            idx = site_to_nucleon_qubit(site, mode, n_b)
            H_N += FermionOperator(f'{idx}^ {idx}')
            
    # Corrected scaling: 2h in 1D, 4h in 2D, 6h in 3D
    return (2 * dim * h) * H_N 

def HC(C, L, dim, n_b):
    """Nucleon-nucleon contact term (Eq. 54)"""
    H = FermionOperator()
    num_sites = get_total_sites(L, dim)
    for site in range(num_sites):
        r = rho(site, n_b)
        H += r * r
    return normal_ordered(H * (C / 2.0))

def HCI2(CI, L, dim, n_b):
    """Isospin-dependent contact term (Eq. 55)"""
    H = FermionOperator()
    num_sites = get_total_sites(L, dim)
    for site in range(num_sites):
        for I in [1, 2, 3]:
            rI = rho_I(I, site, n_b)
            H += rI * rI
    return normal_ordered(H * (CI / 2.0))

def Static_Nucleon_Hamiltonian(h, C, CI, L, dim, n_b):
    """Combines all static nucleon terms for D-dimensional Dynamical Pion EFT"""
    H_free = Free_Hopping(h, L, dim, n_b) + Free_Onsite(h, L, dim, n_b)
    H_contact = HC(C, L, dim, n_b) + HCI2(CI, L, dim, n_b)
    return H_free + H_contact