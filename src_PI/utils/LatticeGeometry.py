# src_PI/utils/LatticeGeometry.py

import numpy as np

def get_total_sites(L, dim):
    """Returns the total number of sites in an L^dim lattice."""
    return L**dim

def index_to_coord(index, L, dim):
    """
    Converts a flat 1D index into (x, y, z, ...) coordinates.
    Standard row-major extraction.
    """
    coords = []
    temp = index
    for _ in range(dim):
        coords.append(temp % L)
        temp //= L
    return tuple(coords)

def coord_to_index(coords, L):
    """
    Converts (x, y, z, ...) coordinates back to a flat 1D index.
    """
    idx = 0
    for i, val in enumerate(coords):
        idx += val * (L**i)
    return idx

def get_neighbors(index, L, dim):
    """
    Returns a list of valid neighbor indices for a given site index
    using Open Boundary Conditions (OBC).
    """
    coords = index_to_coord(index, L, dim)
    neighbors = []
    
    # Check each dimension (x, y, z, etc.)
    for d in range(dim):
        # Forward neighbor (if not at the upper open boundary)
        if coords[d] < L - 1:
            neighbors.append(index + L**d)
        
        # Backward neighbor (if not at the lower open boundary)
        if coords[d] > 0:
            neighbors.append(index - L**d)
            
    return neighbors

def site_to_nucleon_qubit(site_id, fermionic_mode, n_b):
    """
    General mapping for any dimension.
    fermionic_mode: (spin, isospin) where 0=up/p, 1=down/n
    """
    stride = 4 + 3 * n_b
    # Nucleon slots are 0, 1, 2, 3 within the site block.
    # Mathematically mapping (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    mode_offset = fermionic_mode[0] * 2 + fermionic_mode[1]
    return site_id * stride + mode_offset

def site_to_pion_qubit(site_id, pion_species, bit_id, n_b):
    """
    pion_species: 0, 1, 2 (corresponding to I=1, 2, 3)
    bit_id: 0 to n_b-1
    """
    stride = 4 + 3 * n_b
    return site_id * stride + 4 + (pion_species * n_b) + bit_id

def total_qubits(L, dim, n_b):
    """
    Calculates total qubits across the entire lattice.
    """
    num_sites = get_total_sites(L, dim)
    return num_sites * (4 + 3 * n_b)
