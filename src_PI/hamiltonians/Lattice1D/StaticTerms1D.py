from openfermion import FermionOperator, normal_ordered
import numpy as np
from src_PI.utils.utils import site_to_nucleon_qubit_1D

"""
TODO: MULTI-DIMENSIONAL FERMIONIC PHYSICS
1. High-D Hopping: Generalize 'Free_Hopping_1D' to iterate over neighbors in 
   N-dimensions. This will significantly increase Jordan-Wigner string 
   weights as hopping bypasses large pion-qubit blocks.
2. Contact Scaling: Verify that Contact Terms (C, CI) are correctly normalized 
   when the coordination number increases (z=2 in 1D, z=4 in 2D, z=6 in 3D).
3. Normal Ordering: Optimize the 'normal_ordered' calls for higher-site 
   counts to avoid memory bottlenecks during Hamiltonian assembly.
"""

# Constants
fermionic_modes = [(0,0), (0,1), (1,0), (1,1)] # (spin, isospin)
tau_mats = [
    np.array([[0, 1], [1, 0]], dtype=complex),    # tau_1
    np.array([[0, -1j], [1j, 0]], dtype=complex), # tau_2
    np.array([[1, 0], [0, -1]], dtype=complex)    # tau_3
]

# --- Density Operator Helpers (Native n_b awareness) ---

def rho(site_id, n_b):
    """Total nucleon density at a site: sum_{alpha} a_alpha^dag a_alpha"""
    res = FermionOperator()
    for mode in fermionic_modes:
        idx = site_to_nucleon_qubit_1D(site_id, mode, n_b)
        res += FermionOperator(f'{idx}^ {idx}')
    return res

def rho_I(I_idx, site_id, n_b):
    """Isospin density at a site: sum_{alpha,beta} a_alpha^dag [tau_I]_alpha,beta a_beta"""
    # I_idx is 1, 2, or 3. Mapping to 0, 1, 2 for list index.
    mat = tau_mats[I_idx - 1]
    res = FermionOperator()
    for s in [0, 1]: # Spin indices
        for b in [0, 1]: # Isospin indices
            for c in [0, 1]:
                coeff = mat[b, c]
                if abs(coeff) > 1e-9:
                    idx_b = site_to_nucleon_qubit_1D(site_id, (s, b), n_b)
                    idx_c = site_to_nucleon_qubit_1D(site_id, (s, c), n_b)
                    res += FermionOperator(f'{idx_b}^ {idx_c}', coeff)
    return res

# --- Hamiltonian Components ---

def Free_Hopping_1D(h, num_sites, n_b):
    H_hop = FermionOperator()
    for i in range(num_sites - 1):
        for mode in fermionic_modes:
            idx_i = site_to_nucleon_qubit_1D(i, mode, n_b)
            idx_j = site_to_nucleon_qubit_1D(i + 1, mode, n_b)
            H_hop += FermionOperator(f'{idx_i}^ {idx_j}') + FermionOperator(f'{idx_j}^ {idx_i}')
    return -h * H_hop    

def Free_Onsite_1D(h, num_sites, n_b):
    H_N = FermionOperator()
    for site in range(num_sites):
        for mode in fermionic_modes:
            idx = site_to_nucleon_qubit_1D(site, mode, n_b)
            H_N += FermionOperator(f'{idx}^ {idx}')
    return 6 * h * H_N 

def HC_1D(C, num_sites, n_b):
    """Nucleon-nucleon contact term (Eq. 54)"""
    H = FermionOperator()
    for site in range(num_sites):
        r = rho(site, n_b)
        H += r * r
    return normal_ordered(H * (C / 2.0))

def HCI2_1D(CI, num_sites, n_b):
    """Isospin-dependent contact term (Eq. 55)"""
    H = FermionOperator()
    for site in range(num_sites):
        for I in [1, 2, 3]:
            rI = rho_I(I, site, n_b)
            H += rI * rI
    return normal_ordered(H * (CI / 2.0))

def Static_Nucleon_Hamiltonian_1D(h, C, CI, num_sites, n_b):
    """Combines all static nucleon terms for Dynamical Pion EFT"""
    H_free = Free_Hopping_1D(h, num_sites, n_b) + Free_Onsite_1D(h, num_sites, n_b)
    H_contact = HC_1D(C, num_sites, n_b) + HCI2_1D(CI, num_sites, n_b)
    return H_free + H_contact