import numpy as np
from openfermion import QubitOperator, FermionOperator, jordan_wigner



# Pauli matrices as 2x2 arrays
sigma_mats = [
    np.eye(2, dtype=complex),                  # 0: Identity
    np.array([[0, 1], [1, 0]], dtype=complex),    # 1: Sigma_x
    np.array([[0, -1j], [1j, 0]], dtype=complex), # 2: Sigma_y
    np.array([[1, 0], [0, -1]], dtype=complex)    # 3: Sigma_z
]

def calculate_chiral_coeff(mode_a, mode_b, iso_idx, spin_idx):
    """
    mode_a/b are tuples (spin_idx, iso_idx) where 0=up/proton, 1=down/neutron.
    iso_idx: 1, 2, 3 (pion species / tau index)
    spin_idx: 1, 2, 3 (gradient direction / sigma index)
    """
    s_a, i_a = mode_a
    s_b, i_b = mode_b
    
    # <isospin_a | tau_I | isospin_b>
    iso_val = sigma_mats[iso_idx][i_a, i_b]
    # <spin_a | sigma_S | spin_b>
    spin_val = sigma_mats[spin_idx][s_a, s_b]
    
    return iso_val * spin_val

def get_P_Q(pi_max, n_b):
    delta_pi = (2 * pi_max) / (2**n_b - 1)
    P = -pi_max + (delta_pi / 2) * (2**n_b - 1)
    Q = -delta_pi / 2
    # Note: These definitions come from Eq 29/71. 
    # In some papers Q is simply delta_pi; let's stick to the paper's P and Q.
    return P, Q

def get_Pp_Qp(pi_max, n_b, a_L):
    """
    Calculates P' and Q' for the Conjugate Momentum encoding (Eq 32-33).
    """
    # From Eq 32
    delta_pi = (2 * pi_max) / (2**n_b - 1)
    delta_Pi = (2 * np.pi) / (a_L**3 * delta_pi * (2**n_b))
    Pi_max = np.pi / (a_L**3 * delta_pi)
    
    # Eq 33/74
    Pp = -Pi_max + (delta_Pi / 2) * (2**n_b - 1)
    Qp = -delta_Pi / 2
    return Pp, Qp

def site_to_nucleon_qubit_1D(site_id, fermionic_mode, n_b):
    # Each site block size is 4 (nucleons) + 3 * n_b (pions)
    stride = 4 + 3 * n_b
    if fermionic_mode == (0,0):
        return site_id * stride + 0
    elif fermionic_mode == (0,1):
        return site_id * stride + 1
    elif fermionic_mode == (1,0):
        return site_id * stride + 2
    elif fermionic_mode == (1,1):
        return site_id * stride + 3
    else:
        raise ValueError("Invalid fermionic mode")

def site_to_pion_qubit_1D(site_id, pion_species, bit_id, n_b):
    """
    pion_species: 0, 1, 2 (corresponding to I=1, 2, 3)
    bit_id: 0 to n_b-1
    """
    stride = 4 + 3 * n_b
    # Nucleons take first 4 slots, pions take the rest
    return site_id * stride + 4 + (pion_species * n_b) + bit_id

def total_qubits_1D(num_sites, n_b):
    return num_sites * (4 + 3 * n_b)

def fm_to_mev(fm):
    return fm * 197.327

def mev_to_fm(mev):
    return mev / 197.327