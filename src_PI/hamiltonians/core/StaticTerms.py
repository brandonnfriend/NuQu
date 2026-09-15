from openfermion import FermionOperator, normal_ordered
import numpy as np

# Use the dimension-agnostic geometry functions
from src_PI.utils.LatticeGeometry import site_to_nucleon_qubit, get_total_sites, get_neighbors


# ---------------------------------------------------------------------------
#  Contact-term ordering convention  (comparison switch, CLAUDE.md discipline)
# ---------------------------------------------------------------------------
# Watson 2026 DEFINES the contact terms with the Wick (normal-ordering)
# prescription:   H_C = (C/2) Σ_x :ρ²(x):  (Eq. 54),
#                 H_CI2 = (C_I²/2) Σ_x Σ_I :ρ_I²(x):  (Eq. 55),
# and his Eq. (60) expansion of Σ_I ρ_I² is consistent with Eq. (55) (verified
# term-by-term). His Eq. (59) drops the σ≠σ' restriction, but that is a typo:
# one page earlier he writes the IDENTICAL structure for pionless EFT with the
# restriction intact in BOTH the definition (Eq. 38) and the encoding (Eq. 48).
#
# THE BUG (found 2026-09-14). This module used to call OpenFermion's
# `normal_ordered`, which is an ALGEBRA-PRESERVING REWRITE — it applies the
# anticommutation relations to move creation operators left, keeping every
# contraction term, so `normal_ordered(X) == X` as operators. It is NOT the
# `: :` prescription, which is a linear map that DISCARDS the contractions.
# The two differ by exactly (Wick's theorem):
#
#     ρ²  =  :ρ²:  +  ρ ,          Σ_I ρ_I²  =  :Σ_I ρ_I²:  +  3ρ
#
# so the un-prescribed Hamiltonian carries a spurious ONE-BODY self-interaction
#
#     H_unordered − H_wick = (C/2 + 3·C_I²/2)·N̂  ≡  c·N̂ ,   c = −23.3725 MeV
#
# i.e. a lone nucleon on an empty lattice feels the contact interaction with
# ITSELF. Within a fixed-A sector this is the c-number c·A: eigenvectors,
# determinant selection, PT2, σ, compactness, every energy DIFFERENCE and the
# binding energy are all exactly invariant, and legacy energies convert with
#     E_wick = E_unordered − c·A = E_unordered + 23.3725·A  [MeV].
# See `contact_self_energy_shift` below and `docs/theory_latex_reference.md` V1.
#
# `wick_ordered=True` (default) is the corrected, Watson-matching Hamiltonian.
# `wick_ordered=False` reproduces every pre-2026-09-14 number bit-identically
# and is kept so the published legacy data stays regenerable.
WICK_ORDERED_CONTACTS_DEFAULT = True


def contact_self_energy_shift(A, params):
    """`c·A` [MeV] — the spurious contact self-energy carried by the
    un-Wick-ordered convention in a sector with `A` nucleons.

    Exact and independent of L, dim, n_b, the pion basis and the frame: the
    difference operator is `c·N̂` on the fermion sector alone, and `N̂ = A` there.

        E_wick(A) = E_unordered(A) − contact_self_energy_shift(A, params)

    (the shift is NEGATIVE, so the corrected energies are HIGHER).
    """
    c = params['C'] / 2.0 + 3.0 * params['CI'] / 2.0
    return c * A


def _wick(op):
    """The `: :` prescription for a product of two fermionic bilinears.

    Normal-orders the product and keeps only the fully-uncontracted (degree-4)
    part — by Wick's theorem exactly what `: :` retains. Verified against the
    explicit `Σ_{σ≠σ'} N_σ N_σ'` form and against Watson's Eq. (60).
    """
    out = FermionOperator()
    for term, coeff in normal_ordered(op).terms.items():
        if len(term) == 4:
            out += FermionOperator(term, coeff)
    return out

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

def HC(C, L, dim, n_b, wick_ordered=WICK_ORDERED_CONTACTS_DEFAULT):
    """Nucleon-nucleon contact term, Watson Eq. (54): `(C/2) Σ_x :ρ²(x):`.

    `wick_ordered=False` is the legacy (pre-2026-09-14) un-prescribed `ρ²`,
    which carries a `(C/2)·N̂` self-interaction. See the module note.
    """
    H = FermionOperator()
    num_sites = get_total_sites(L, dim)
    for site in range(num_sites):
        r = rho(site, n_b)
        rr = r * r
        H += _wick(rr) if wick_ordered else normal_ordered(rr)
    return normal_ordered(H * (C / 2.0))

def HCI2(CI, L, dim, n_b, wick_ordered=WICK_ORDERED_CONTACTS_DEFAULT):
    """Isospin-dependent contact term, Watson Eq. (55): `(C_I²/2) Σ_x Σ_I :ρ_I²(x):`.

    `wick_ordered=False` is the legacy un-prescribed `Σ_I ρ_I²`, which carries a
    `(3·C_I²/2)·N̂` self-interaction. See the module note.
    """
    H = FermionOperator()
    num_sites = get_total_sites(L, dim)
    for site in range(num_sites):
        S = FermionOperator()
        for I in [1, 2, 3]:
            rI = rho_I(I, site, n_b)
            S += rI * rI
        H += _wick(S) if wick_ordered else normal_ordered(S)
    return normal_ordered(H * (CI / 2.0))

def Static_Nucleon_Hamiltonian(h, C, CI, L, dim, n_b,
                               wick_ordered=WICK_ORDERED_CONTACTS_DEFAULT):
    """Combines all static nucleon terms for D-dimensional Dynamical Pion EFT.

    `wick_ordered=True` (default) matches Watson's definitions (Eqs. 54/55);
    `False` reproduces the pre-2026-09-14 numbers bit-identically.
    """
    H_free = Free_Hopping(h, L, dim, n_b) + Free_Onsite(h, L, dim, n_b)
    H_contact = (HC(C, L, dim, n_b, wick_ordered)
                 + HCI2(CI, L, dim, n_b, wick_ordered))
    return H_free + H_contact