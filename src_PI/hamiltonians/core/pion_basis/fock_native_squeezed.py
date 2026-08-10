"""
Gaussian-squeezed NATIVE Fock builder — Architecture B on the analytical pipeline.

The native-algebra (`MixedHamiltonian`) counterpart of `fock_squeezed.py`: it applies
the SAME exactly-isospectral Gaussian squeeze (`c_π→e^{r}c_π`, `c_Π→e^{−r}c_Π` per mode,
degree-≤2 ⇒ a pure coefficient rescale) but emits the unmultiplied fermion/boson factors
the `sparse` (and `lobe`) encoders consume — so the squeezed walk runs through the
**analytical Gilyén-Lemma-30 per-term aggregation** (`sparse_oracle/`) that scales to
L=10, instead of the circuit-building `pauli_lcu` path that `fock_squeezed.py` is capped
to. This is what makes the bare-vs-squeeze resource comparison production-scale.

Mirrors `fock_native.py` term-for-term; the squeeze enters exactly as in
`fock_squeezed.py`:
  * gradient   ∝ c_π²    → × e^{2r}   (built with the squeezed c_π)
  * axial (AV) ∝ c_π     → × e^{ r}   (scale each MixedTerm coeff)
  * WT         ∝ c_π·c_Π → invariant  (reuse `fock_native.H_WT_native` unchanged)
  * free-local  → explicit `d·(2n̂+1) + o·(â²+â†²)` with the squeezed coefficients
                  (the collapsed `m_π·n̂` shortcut drops the off-diagonal the squeeze
                  switches on).

At `r=0` the output MixedHamiltonian is identical to `fock_native`'s (o→0, 2d→m_π) —
the built-in self-consistency check (tested). `r` is read from `params['squeeze_r']`.
Selected when `config.pion_basis == 'fock_squeezed'` and the encoder is sparse/lobe
(dispatch in `ConstructEFT._use_native_fock_path`); `fock_squeezed@pauli_lcu` still uses
the Pauli path in `fock_squeezed.py`.
"""

import numpy as np
from openfermion import BosonOperator, FermionOperator

from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian, MixedTerm
from src_PI.hamiltonians.core.pion_basis import fock_native
from src_PI.utils.LatticeGeometry import get_total_sites, index_to_coord


BASIS_NAME = 'fock_native_squeezed'


def get_squeeze_r(params):
    """Squeeze amplitude `r` (0 ⇒ identical to the bare `fock_native` builder)."""
    return float(params.get('squeeze_r', 0.0))


def _squeezed_coefficients(params, dim, r):
    """Bare `(c_π, c_Π)` rescaled by the squeeze: `(e^{r}·c_π, e^{−r}·c_Π)`."""
    c_pi, c_Pi = fock_native._basis_coefficients(params, dim)
    return np.exp(r) * c_pi, np.exp(-r) * c_Pi


def H_pion_free_native_squeezed(L, dim, n_b, params, r):
    """Squeezed pure-boson `H_pion_free` as a `BosonOperator` over global modes.

    Local `(a_L^d/2)(Π̂²+m²π̂²) = d·(2n̂+1) + o·(â²+â†²)` built explicitly (so the
    off-diagonal survives), gradient `∝ c_π²` scaled by the squeezed `c_π` (= e^{2r}).
    At r=0: `o=0`, `2d=m_π` ⇒ reduces to `fock_native.H_pion_free_native`.
    """
    c_pi, c_Pi = _squeezed_coefficients(params, dim, r)
    m_pi = params['m_pi']
    a_L = params['a_L']
    aL_d = a_L ** dim
    d = (aL_d / 2.0) * (m_pi ** 2 * c_pi ** 2 + c_Pi ** 2)   # diagonal weight
    o = (aL_d / 2.0) * (m_pi ** 2 * c_pi ** 2 - c_Pi ** 2)   # off-diagonal weight
    grad_factor = (a_L ** (dim - 2)) / 2.0 * (c_pi ** 2)     # e^{2r}·(bare grad_factor)
    num_sites = get_total_sites(L, dim)

    H_b = BosonOperator()
    # local: d·(2n̂+1) + o·(â²+â†²) per (site, species)
    for x in range(num_sites):
        for I in fock_native._PION_SPECIES:
            m = fock_native._global_mode(x, I)
            H_b += 2.0 * d * BosonOperator(f'{m}^ {m}')
            H_b += BosonOperator('', d)
            if abs(o) > 1e-15:
                H_b += o * (BosonOperator(f'{m} {m}') + BosonOperator(f'{m}^ {m}^'))
    # gradient: (a_L^{d-2}/2)·c_π_sq²·(b̂_y − b̂_x)² over adjacent (x, y) pairs
    for x in range(num_sites):
        coords = index_to_coord(x, L, dim)
        for dd in range(dim):
            if coords[dd] >= L - 1:
                continue
            site_next = x + L ** dd
            for I in fock_native._PION_SPECIES:
                mx = fock_native._global_mode(x, I)
                my = fock_native._global_mode(site_next, I)
                diff = fock_native._b_x_global(my) - fock_native._b_x_global(mx)
                H_b += grad_factor * diff * diff
    return H_b


def H_axial_vector_native_squeezed(L, dim, n_b, params, r):
    """H_AV mixed-terms, squeezed: each bare MixedTerm coeff × e^{r} (H_AV ∝ c_π,
    with the c_π carried in the coeff)."""
    s = float(np.exp(r))
    return [MixedTerm(coeff=t.coeff * s, fermion_factor=t.fermion_factor,
                      boson_factor=t.boson_factor)
            for t in fock_native.H_axial_vector_native(L, dim, n_b, params)]


def H_WT_native_squeezed(L, dim, n_b, params, r):
    """H_WT mixed-terms, squeezed: INVARIANT (∝ c_π·c_Π = e^{r}·e^{−r}); reuse bare."""
    return fock_native.H_WT_native(L, dim, n_b, params)


def build_native_mixed_hamiltonian(L, dim, n_b, params):
    """Assemble the squeezed native MixedHamiltonian. `r` from `params['squeeze_r']`.
    Static-nucleon FermionOperator is folded in by `ConstructEFT` (no JW), as for
    `fock_native`."""
    r = get_squeeze_r(params)
    num_sites = get_total_sites(L, dim)
    return MixedHamiltonian(
        fermion_part=FermionOperator(),
        boson_part=H_pion_free_native_squeezed(L, dim, n_b, params, r),
        mode_to_qubits=fock_native._build_mode_to_qubits(num_sites, n_b),
        mixed_terms=H_axial_vector_native_squeezed(L, dim, n_b, params, r)
                   + H_WT_native_squeezed(L, dim, n_b, params, r),
    )
