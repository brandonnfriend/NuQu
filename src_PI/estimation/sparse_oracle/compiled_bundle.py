"""
Compiled Hermitian bundle composite (C1 sparse P0-2).

Codex's audit found the Hermitian bundle's cost came from a hand-assembled
`_t_complexity_` — the block encoding had no executable `decompose_from_registers`.
This module assembles the **compiled** single-mode boson atoms
(`matching_dilation.atom_dilation_ops`) + the fermion PauliLCU into a real,
decomposable outer LCU composite `PREP · SELECT · PREP†`, so `cirq.unitary`
extraction reproduces `α_tot·⟨0|_flag U|0⟩ = H` on small instances and the walk
qubitizes — and the cost is traversed from the actual circuit.

Registers (all block-flag qubits in one `flag` register, per the reflection):
  * `outer_sel`  — b_out qubits selecting the atom (outer LCU),
  * `inner_sel`  — the atoms' shared inner-LCU select (diagonal + matchings),
  * `b_dil`      — the shared 1-qubit dilation ancilla,
  * `system`     — the target (nucleon + pion registers).

Scope: the **dominant sectors** — single-mode boson atoms (~88% of Λ) + the
static-fermion PauliLCU (~11%). Two-mode boson atoms (non-power-of-two Δ) are the
remaining P0-1 piece and are not yet dispatched here (a bundle containing them
raises, so nothing is silently dropped).

Verification path: like the atoms, the outer PREP here is an explicit unitary
(so `cirq.unitary` can check the composite exactly on small instances). For the
large production bundle the explicit PREP is replaced by alias sampling for cost
(a documented cost-form swap), exactly as QROM would replace the per-edge
rotations at large n_b.
"""

import math

import cirq
import numpy as np

from src_PI.estimation.sparse_oracle.matching_dilation import (
    atom_dilation_ops,
    two_mode_atom_dilation_ops,
)
from src_PI.estimation.sparse_oracle.hermitian_bundle import (
    extract_hermitian_atoms,
)
from src_PI.estimation.sparse_oracle.bundle_encoding import (
    _contraction_dilation,
    _embed_operator,
    _fermion_dense,
)

_TOL = 1e-12


def _prep_unitary(weights):
    """Householder |0⟩ → Σ_l √(w_l/Σw)|l⟩ on ceil(log2 len) qubits; returns (gate, b)."""
    n = len(weights)
    b = max(1, int(math.ceil(math.log2(max(1, n)))))
    dim = 1 << b
    v = np.zeros(dim)
    v[:n] = np.sqrt(np.asarray(weights, float) / float(sum(weights)))
    e0 = np.zeros(dim)
    e0[0] = 1.0
    u = e0 - v
    nu = np.linalg.norm(u)
    P = (np.eye(dim) if nu < _TOL
         else np.eye(dim) - 2.0 * np.outer(u / nu, u / nu))
    return cirq.MatrixGate(P.astype(complex), name='oPREP'), b


def _atom_inner_bits(atom, n_b):
    """Inner-LCU select width for a single-mode boson atom (log2 #components)."""
    from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
        _split_into_components,
    )
    diag, matchings = _split_into_components(atom.M)
    n_comp = (1 if np.abs(diag).max() > _TOL else 0) + len(matchings)
    return max(1, int(math.ceil(math.log2(max(1, n_comp)))))


def _dominant_atoms(mh, n_b, mode_to_qubits):
    """The single-mode + two-mode boson atoms + the static fermion atom (dense M).

    Single- and two-mode boson atoms are dispatched to the compiled matching-
    dilation; the fermion atom uses a dense contraction dilation here (toy-
    verifiable), which the production cost swaps for the PauliLCU encoder. Raises
    on boson atoms spanning >2 modes (none arise in this Hamiltonian)."""
    atoms = extract_hermitian_atoms(mh, n_b, mode_to_qubits, need_dense=True)
    kept = []
    for a in atoms:
        if a.kind == 'boson' and len(a.support) not in (n_b, 2 * n_b):
            raise NotImplementedError(
                f"boson atom on {len(a.support) // n_b} modes not compiled")
        if a.kind == 'mixed':
            raise NotImplementedError("mixed atom compile not yet wired")
        kept.append(a)
    return kept


def compiled_bundle_ops(outer_sel, inner_sel, b_dil, sys_qubits,
                        atoms, mode_to_qubits, n_b):
    """Yield the compiled composite `PREP · SELECT · PREP†` over the atoms.

    Heterogeneous dispatch: a single-mode **boson** atom contributes its compiled
    `atom_dilation_ops` on its mode's system qubits; the **fermion** atom
    contributes the Hermitian contraction dilation of its (dense) `M` on its
    support — both controlled by `outer_sel == l`, sharing `b_dil`.
    `α_tot·⟨0|_flag U|0⟩ = Σ_l M_l`."""
    weights = [a.alpha for a in atoms]
    prep, b_out = _prep_unitary(weights)
    osel = list(outer_sel)
    assert len(osel) == b_out
    yield prep.on(*osel)
    for l, atom in enumerate(atoms):
        cvals = [(l >> (b_out - 1 - i)) & 1 for i in range(b_out)]
        if atom.kind == 'boson':
            support = [sys_qubits[q] for q in atom.support]
            if len(atom.support) == 2 * n_b:           # two-mode (H_WT / gradient)
                atom_ops = two_mode_atom_dilation_ops(
                    list(inner_sel)[:1], b_dil, support, atom.M, n_b)
            else:                                       # single-mode
                isel = list(inner_sel)[:_atom_inner_bits(atom, n_b)]
                atom_ops = atom_dilation_ops(isel, b_dil, support, atom.M)
        else:                                          # fermion (dense dilation)
            support = [sys_qubits[q] for q in atom.support]
            B = _contraction_dilation(atom.M, atom.alpha)   # Hermitian self-inverse
            atom_ops = [cirq.MatrixGate(B, name='Bferm').on(b_dil, *support)]
        for op in atom_ops:
            yield op.controlled_by(*osel, control_values=cvals)
    yield cirq.inverse(prep).on(*osel)


def compiled_bundle_widths(mh, n_b, mode_to_qubits):
    """`(b_out, b_inner, w_sys, atoms)` for the compiled composite."""
    atoms = _dominant_atoms(mh, n_b, mode_to_qubits)
    b_out = max(1, int(math.ceil(math.log2(max(1, len(atoms))))))
    b_inner = max((_atom_inner_bits(a, n_b) for a in atoms
                   if a.kind == 'boson' and len(a.support) == n_b), default=1)
    w_sys = 1 + max((max(a.support) for a in atoms if a.support), default=0)
    return b_out, b_inner, w_sys, atoms


def extract_compiled_bundle(mh, n_b, mode_to_qubits):
    """Build the compiled composite on named qubits; return `(U, α_tot, block)`."""
    b_out, b_inner, w_sys, atoms = compiled_bundle_widths(mh, n_b, mode_to_qubits)
    alpha_tot = sum(a.alpha for a in atoms)
    osel = [cirq.NamedQubit(f'o{i}') for i in range(b_out)]
    isel = [cirq.NamedQubit(f'i{i}') for i in range(b_inner)]
    b_dil = cirq.NamedQubit('b_dil')
    sysq = [cirq.NamedQubit(f's{i}') for i in range(w_sys)]
    circ = cirq.Circuit(compiled_bundle_ops(
        osel, isel, b_dil, sysq, atoms, mode_to_qubits, n_b))
    flag = [*osel, *isel, b_dil]
    U = circ.unitary(qubit_order=[*flag, *sysq])
    N = 1 << w_sys
    return U, alpha_tot, U[:N, :N] * alpha_tot


def compiled_bundle_reference(mh, n_b, mode_to_qubits):
    """Exact Σ_l M_l embedded on the system register (for the composite gate)."""
    _b_out, _b_inner, w_sys, atoms = compiled_bundle_widths(mh, n_b, mode_to_qubits)
    H = np.zeros((1 << w_sys, 1 << w_sys), dtype=complex)
    for a in atoms:
        H += _embed_operator(a.M, a.support, w_sys)
    return H
