"""
Gaussian-squeezed Fock basis — Architecture B of the frame study (task 34).

This is the `fock` basis walked in the **squeeze frame** `H_sq = U†HU`, so QPE on
this Hamiltonian returns `spec(H_sq) = spec(H)` (the squeeze is a canonical, exactly
isospectral transform) while the walk register enjoys a smaller boson Fock cutoff —
the certified `n_b` win (`docs/frame_on_quantum_side.md` §2). It is a "comparison
switch" per `CLAUDE.md`: the bare `fock` path stays intact; this slots alongside it
and is selected by `config.pion_basis='fock_squeezed'`.

**How the squeeze enters (degree-≤2 ⇒ a pure coefficient rescale).** The single-mode
squeeze `U=exp[½ r(â²−â†²)]` acts on the conjugate field pair by a reciprocal
quadrature rescale (φ=0):

    U†π̂U = e^{ r}·π̂ ,   U†Π̂U = e^{−r}·Π̂        (canonical: [π̂,Π̂] preserved)

Because every EFT term is a polynomial of degree ≤ 2 in the boson operators, `U†HU`
is just `H` with `c_π → e^{r}·c_π` and `c_Π → e^{−r}·c_Π` per mode — no truncation,
no Franck-Condon series (contrast the LF frame). Term by term:

    * gradient   ∝ c_π²      → × e^{2r}     (reuse `fock.H_pion_free_gradient`)
    * axial (AV) ∝ c_π       → × e^{ r}     (reuse `fock.H_axial_vector`)
    * WT         ∝ c_π·c_Π   → × e^{0}=1    (INVARIANT — reuse `fock.H_WT_Logic`)
    * free-local ∝ (e^{−2r}Π̂² + m²e^{2r}π̂²)  → built EXPLICITLY here

The free-**local** term is the one structural change. `fock.H_pion_free_local` uses
the collapsed shortcut `m·(n̂+½)`, valid only when the π̂²/Π̂² weights balance (r=0).
Under the squeeze they no longer balance, so the collapse fails and the term carries
off-diagonal `(â²+â†²)` pieces — exactly the terms `fock.py` warns it drops when
`ω_0≠m_π`. We therefore build `(a_L^d/2)(Π̂²+m²π̂²)` explicitly from the ladder
algebra, with the squeezed coefficients, so those pieces are included.

At `r=0` this collapses back to `fock`'s output **exactly** (tested) — the built-in
self-consistency check. Isospectrality for `r≠0` holds by construction (canonical
transform) and is verified numerically at converged `N_f` (tested); on a finite Fock
register the match is cutoff-limited, same as the classical `frame.isospectral_check`.

`r` is read from `params['squeeze_r']` (scalar, uniform per mode — the default
per-mode squeeze regime, [[feedback_frame_default_permode]]). A per-mode `r_m` array
is a future extension; a scalar is exact-isospectral regardless and is the value the
classical `analytic_squeeze`/`optimize_squeeze` return for the (translation-symmetric)
bulk. Only the `pauli_lcu` encoder is wired (the native/sparse path keys on
`pion_basis=='fock'`); extending the sparse encoder to the squeezed basis is a
follow-up.
"""

import numpy as np
from openfermion import BosonOperator, QubitOperator

from src_PI.hamiltonians.core.pion_basis import fock
from src_PI.utils.LatticeGeometry import get_total_sites


BASIS_NAME = 'fock_squeezed'


def get_squeeze_r(params):
    """Squeeze amplitude `r` for this run (0 ⇒ identical to the bare `fock` basis)."""
    return float(params.get('squeeze_r', 0.0))


def _squeezed_coefficients(params, dim, r):
    """Bare `(c_π, c_Π)` rescaled by the squeeze: `(e^{r}·c_π, e^{−r}·c_Π)`."""
    c_pi, c_Pi = fock._basis_coefficients(params, dim)
    return np.exp(r) * c_pi, np.exp(-r) * c_Pi


def H_pion_free_local(L, dim, n_b, params, r=None):
    """Local free-pion `(a_L^d/2)·(Π̂² + m_π²·π̂²)` with the squeezed coefficients.

    Expanding `Π̂ = i·c_Π(â†−â)`, `π̂ = c_π(â+â†)` and using `[â,â†]=1` on the number
    operator gives the exact split into a normal-ordered DIAGONAL and an OFF-DIAGONAL
    pair term:

        (a_L^d/2)(Π̂²+m²π̂²) = d·(2n̂+1) + o·(â²+â†²),
            d = (a_L^d/2)(m²c_π² + c_Π²),   o = (a_L^d/2)(m²c_π² − c_Π²).

    The diagonal is built from `fock._number_op_register` (the exact truncated number
    operator — NO top-Fock-state defect, unlike an unnormal-ordered `ââ†`); the
    off-diagonal `â²+â†²` — which the squeeze switches on and `fock.H_pion_free_local`
    drops — is built from the ladder templates.

    At r=0 the coefficients balance (`m²c_π²=c_Π²`): `o=0`, `2d=m_π`, so this reduces
    to `m_π·(n̂+½)` — exactly `fock.H_pion_free_local` (tested).
    """
    if r is None:
        r = get_squeeze_r(params)
    c_pi, c_Pi = _squeezed_coefficients(params, dim, r)
    m_pi = params['m_pi']
    aL_d = params['a_L'] ** dim
    num_sites = get_total_sites(L, dim)

    d = (aL_d / 2.0) * (m_pi ** 2 * c_pi ** 2 + c_Pi ** 2)   # diagonal weight
    o = (aL_d / 2.0) * (m_pi ** 2 * c_pi ** 2 - c_Pi ** 2)   # off-diagonal weight
    offdiag_b = BosonOperator('0 0') + BosonOperator('0^ 0^')  # â² + â†²

    H = QubitOperator()
    for x in range(num_sites):
        for I in fock._PION_SPECIES:
            # d·(2n̂+1)
            H += 2.0 * d * fock._number_op_register(x, I, n_b)
            H += QubitOperator((), d)
            # o·(â²+â†²)
            if abs(o) > 1e-15:
                mode_qubits = {0: fock._site_to_pion_qubits(x, I, n_b)}
                H += o * fock._bosonop_to_qubitop(offdiag_b, n_b, mode_qubits)
    return fock._drop_imag_noise(H)


def H_pion_free(L, dim, n_b, params, r=None):
    """Total free-pion in the squeeze frame: explicit local + e^{2r}·gradient."""
    if r is None:
        r = get_squeeze_r(params)
    grad = np.exp(2.0 * r) * fock.H_pion_free_gradient(L, dim, n_b, params)
    return H_pion_free_local(L, dim, n_b, params, r) + grad


def H_axial_vector(L, dim, n_b, params, r=None):
    """Axial-vector coupling in the squeeze frame: `e^{r}·` the bare AV (∝ c_π)."""
    if r is None:
        r = get_squeeze_r(params)
    return np.exp(r) * fock.H_axial_vector(L, dim, n_b, params)


def H_WT_Logic(L, dim, n_b, params, r=None):
    """Weinberg-Tomozawa in the squeeze frame: INVARIANT (∝ c_π·c_Π = e^{r}·e^{−r})."""
    return fock.H_WT_Logic(L, dim, n_b, params)


def Full_Dynamical_Pion_Hamiltonian(L, dim, n_b, pi_max, params):
    """Build the squeezed dynamical-pion sector. `pi_max` accepted and ignored
    (Fock signature compatibility). `r` from `params['squeeze_r']` (0 ⇒ bare fock).

    Returns `[('fock_squeezed', H_full)]` — one sub-Hamiltonian, one walk.
    """
    r = get_squeeze_r(params)
    H_full = (H_pion_free(L, dim, n_b, params, r)
              + H_axial_vector(L, dim, n_b, params, r)
              + H_WT_Logic(L, dim, n_b, params, r))
    return [('fock_squeezed', H_full)]
