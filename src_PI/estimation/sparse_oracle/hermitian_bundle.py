"""
Hermitian full-bundle assembly (C1 walk-validity rebuild, sub-step 3).

Re-groups a `MixedHamiltonian` into **Hermitian** atoms so the LCU block
encoding `U = PREP†·SELECT·PREP` is Hermitian and pyLIQTR's single-reflection
walk `W = (2Π−I)·U` qubitizes it (the defect the non-Hermitian d=1 bundle had —
see `tests/test_sparse_full_bundle.py::test_bundle_walk_qubitizes_hermitian_H`).

Re-pairing (robust, no monomial-key matching)
---------------------------------------------
* **Boson**: group `boson_part` monomials by their *touched-mode set*. Each group
  is Hermitian (a monomial and its conjugate share a mode-set), so its summed
  matrix `M_group` is Hermitian → one Hermitian atom.
* **Fermion**: the whole static `fermion_part` (Hermitian JW image) → one atom.
* **Mixed**: each `MixedTerm` `mt.coeff·F⊗B` (F, B, mt.coeff all Hermitian/real
  after the vertex fix) → one Hermitian atom.

Each atom's operator `M_l` is Hermitian, so its contraction dilation
`[[M_l/α_l, √(I−M_l²/α_l²)], [√, −M_l/α_l]]` is Hermitian and self-inverse. LCU
with a Hermitian SELECT (`Σ_l |l⟩⟨l|⊗B_l`, real ± signs kept as ±B_l) and PREP
over `√(α_l/α_tot)` gives a Hermitian `U`. No `D_phase` — every phase lives
inside the (already-Hermitian) atom operator.

`α_l` for a boson atom is the **matching-dilation** subnormalization (edge-
coloured 1-norm; `hermitian_boson_encoding.build_hermitian_boson_be`), which is
*tighter* than the per-monomial sum `compute_native_lambda` uses — so the
Hermitian bundle's `α_tot = Σ_l α_l` is a valid, tighter Λ.

This module provides the exact simulation + walk-qubitization validation. The
sparse compiled cost (matching-dilation `_t_complexity_`) is layered on top.
"""

import math
from collections import defaultdict

import numpy as np

from src_PI.estimation.sparse_oracle.bundle_encoding import (
    _contraction_dilation,
    _embed_operator,
    _fermion_dense,
)
from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
    _householder,
    build_hermitian_boson_be,
    monomial_flat_matrix,
)
from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian

_TOL = 1e-12


def _boson_group_matrix(monos_coeffs, modes, n_b):
    """Summed flattened matrix of the boson monomials on `modes` (sorted)."""
    N = (1 << n_b) ** len(modes)
    M = np.zeros((N, N), dtype=complex)
    for mono, coeff in monos_coeffs:
        prod, _m = monomial_flat_matrix(mono, n_b)
        M += complex(coeff) * prod
    return M


class HermitianAtom:
    """A Hermitian LCU atom: operator `M` (on `support` global qubits), rescale
    `alpha`, and a real sign (folded as ±B in the Hermitian SELECT)."""

    __slots__ = ('kind', 'M', 'alpha', 'sign', 'support')

    def __init__(self, kind, M, alpha, support, sign=1.0):
        self.kind = kind
        self.M = M
        self.alpha = float(alpha)
        self.sign = float(sign)
        self.support = list(support)

    @property
    def weight(self):
        return self.alpha


def extract_hermitian_atoms(mh, n_b, mode_to_qubits):
    """Group `mh` into Hermitian atoms (see module docstring)."""
    atoms = []

    # --- boson: group by touched-mode set ---
    groups = defaultdict(list)
    for mono, coeff in mh.boson_part.terms.items():
        if mono == () or abs(complex(coeff)) <= _TOL:
            continue
        modes = tuple(sorted({m for m, _a in mono}))
        groups[modes].append((mono, coeff))
    for modes, monos in groups.items():
        M = _boson_group_matrix(monos, modes, n_b)
        if np.abs(M).max() <= _TOL:
            continue
        _U, alpha, _N = build_hermitian_boson_be(M)
        support = [q for m in modes for q in mode_to_qubits[m]]
        atoms.append(HermitianAtom('boson', M, alpha, support))

    # --- fermion: whole static nucleon part ---
    if len(mh.fermion_part.terms) > 0:
        stats = fermion_jw_stats(mh.fermion_part)
        if stats['one_norm'] > _TOL:
            M, support = _fermion_dense(mh.fermion_part)
            atoms.append(HermitianAtom('fermion', M, stats['one_norm'], support))

    # --- mixed: each MixedTerm as one Hermitian atom ---
    for mt in mh.mixed_terms:
        c = complex(mt.coeff)
        if abs(c) <= _TOL:
            continue
        fstats = fermion_jw_stats(mt.fermion_factor)
        alpha_f = fstats['one_norm']
        # boson factor: group by mode-set, sum into one Hermitian boson matrix.
        b_groups = defaultdict(list)
        for mono, bc in mt.boson_factor.terms.items():
            if mono == () or abs(complex(bc)) <= _TOL:
                continue
            b_groups[tuple(sorted({m for m, _a in mono}))].append((mono, bc))
        if alpha_f <= _TOL or not b_groups:
            continue
        # single mode-set for our H_WT factors; handle the general case by summing
        # the (disjoint-support) boson groups into one operator via embed.
        b_modes = sorted({m for ms in b_groups for m in ms})
        Fdense, fsupport = _fermion_dense(mt.fermion_factor)
        bsupport = [q for m in b_modes for q in mode_to_qubits[m]]
        # Bdense on the boson support (sum of the mode-set groups, embedded).
        Bdim = 1 << len(bsupport)
        Bdense = np.zeros((Bdim, Bdim), dtype=complex)
        bpos = {q: i for i, q in enumerate(bsupport)}
        alpha_b = 0.0
        for ms, monos in b_groups.items():
            Mg = _boson_group_matrix(monos, ms, n_b)
            _U, a_g, _N = build_hermitian_boson_be(Mg)
            alpha_b += a_g
            gsupport = [bpos[q] for m in ms for q in mode_to_qubits[m]]
            Bdense += _embed_operator(Mg, gsupport, len(bsupport))
        support = list(fsupport) + list(bsupport)
        # M_l = |c| * (F ⊗ B) with the sign carried separately (Hermitian ±B).
        M = np.kron(Fdense, Bdense) * abs(c)
        alpha = abs(c) * alpha_f * alpha_b
        atoms.append(HermitianAtom('mixed', M, alpha, support,
                                    sign=(-1.0 if c.real < 0 else 1.0)))
    return atoms


def build_hermitian_bundle_sim(mh, n_b, mode_to_qubits, w_sys):
    """Build the Hermitian bundle block-encoding unitary (exact sim).

    Returns `(U, alpha_tot, w_sys)`. `U` acts on `flag = (LCU-select ⊕ 1 dilation
    ancilla)` (MSBs) then the `w_sys`-qubit system, with
    `α_tot·⟨0|_flag U|0⟩_flag = H` and `U = U†` (⇒ the walk qubitizes)."""
    if not isinstance(mh, MixedHamiltonian):
        raise TypeError("expected a MixedHamiltonian")
    atoms = extract_hermitian_atoms(mh, n_b, mode_to_qubits)
    alpha_tot = sum(a.alpha for a in atoms)
    b_out = max(1, int(math.ceil(math.log2(max(1, len(atoms))))))
    L = 1 << b_out
    d = 1 << (1 + w_sys)                        # 1 dilation ancilla + system

    # SELECT = block-diag over the LCU-select register of per-atom dilations
    # (embedded onto the system), padded with identity on unused branches.
    sel = np.zeros((L * d, L * d), dtype=complex)
    for k, atom in enumerate(atoms):
        M_full = _embed_operator(atom.M, atom.support, w_sys)
        B = _contraction_dilation(atom.sign * M_full, atom.alpha)
        sel[k * d:(k + 1) * d, k * d:(k + 1) * d] = B
    for k in range(len(atoms), L):
        sel[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.eye(d)

    amps = np.zeros(L)
    for k, atom in enumerate(atoms):
        amps[k] = math.sqrt(atom.alpha / alpha_tot)
    prep = _householder(amps)
    P = np.kron(prep, np.eye(d))
    U = P.conj().T @ sel @ P
    return U, alpha_tot, w_sys


def hermitian_bundle_reference(mh, n_b, mode_to_qubits, w_sys):
    """Exact `H = Σ_l (±)M_l` on the `w_sys` system register."""
    atoms = extract_hermitian_atoms(mh, n_b, mode_to_qubits)
    H = np.zeros((1 << w_sys, 1 << w_sys), dtype=complex)
    for atom in atoms:
        H += atom.sign * _embed_operator(atom.M, atom.support, w_sys)
    return H
