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

This module provides the exact simulation + walk-qubitization validation (the
*ideal* construction). The layered `_t_complexity_` is a primitive-based **cost
model**, NOT a compiled decomposition — see `SparseHermitianBundleBlockEncoding`
for the (Codex-audited) list of what it omits; use PauliLCU for publication.
"""

import math
from collections import defaultdict

import cirq
import numpy as np
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
    ProgrammableRotationGateArray,
)
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

from src_PI.estimation.sparse_oracle.bundle_encoding import (
    _contraction_dilation,
    _embed_operator,
    _fermion_dense,
)
from src_PI.estimation.sparse_oracle.hermitian_boson_encoding import (
    _householder,
    _split_into_components,
    build_hermitian_boson_be,
    monomial_flat_matrix,
)
from src_PI.estimation.sparse_oracle.fermion_atom import fermion_atom_encoding
from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
from pyLIQTR.utils.resource_analysis import pylqt_t_complexity

_TOL = 1e-12
_KAPPA = 8
_PREP_EPS = 1e-3


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
    `alpha`, a real sign (folded as ±B in the Hermitian SELECT), and a `payload`
    of kind-specific data for the compiled cost roll-up."""

    __slots__ = ('kind', 'M', 'alpha', 'sign', 'support', 'payload')

    def __init__(self, kind, M, alpha, support, sign=1.0, payload=None):
        self.kind = kind
        self.M = M                                  # None on the cost path
        self.alpha = float(alpha)
        self.sign = float(sign)
        self.support = list(support) if support is not None else None
        self.payload = payload or {}

    @property
    def weight(self):
        return self.alpha


def extract_hermitian_atoms(mh, n_b, mode_to_qubits, need_dense=False):
    """Group `mh` into Hermitian atoms (see module docstring).

    `need_dense=False` (the default, used by the **cost** path) computes only
    each atom's `α` and cost payload — it never materialises the (potentially
    huge) dense operator of a fermion/mixed atom, whose JW image can span the
    whole interleaved register. `need_dense=True` (the **sim** path, tiny toys
    only) also builds `atom.M` / `atom.support` for `build_hermitian_bundle_sim`.
    """
    atoms = []

    # --- boson: group by touched-mode set (matrices are small: ≤ few modes) ---
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
        atoms.append(HermitianAtom('boson', M, alpha, support,
                                   payload={'n_bits': len(support)}))

    # --- fermion: whole static nucleon part (α via 1-norm; dense only for sim) ---
    if len(mh.fermion_part.terms) > 0:
        stats = fermion_jw_stats(mh.fermion_part)
        if stats['one_norm'] > _TOL:
            M, support = (_fermion_dense(mh.fermion_part) if need_dense
                          else (None, None))
            atoms.append(HermitianAtom('fermion', M, stats['one_norm'], support,
                                       payload={'fermion_op': mh.fermion_part}))

    # --- mixed: each MixedTerm as one Hermitian atom ---
    for mt in mh.mixed_terms:
        c = complex(mt.coeff)
        if abs(c) <= _TOL:
            continue
        alpha_f = fermion_jw_stats(mt.fermion_factor)['one_norm']
        # boson factor: group by mode-set (small matrices).
        b_groups = defaultdict(list)
        for mono, bc in mt.boson_factor.terms.items():
            if mono == () or abs(complex(bc)) <= _TOL:
                continue
            b_groups[tuple(sorted({m for m, _a in mono}))].append((mono, bc))
        if alpha_f <= _TOL or not b_groups:
            continue
        alpha_b = 0.0
        boson_group_bits = []
        boson_group_mats = []
        for ms, monos in b_groups.items():
            Mg = _boson_group_matrix(monos, ms, n_b)
            _U, a_g, _N = build_hermitian_boson_be(Mg)
            alpha_b += a_g
            boson_group_bits.append(len({q for m in ms for q in mode_to_qubits[m]}))
            boson_group_mats.append(Mg)
        alpha = abs(c) * alpha_f * alpha_b

        M = support = None
        if need_dense:                              # tiny toys only
            b_modes = sorted({m for ms in b_groups for m in ms})
            Fdense, fsupport = _fermion_dense(mt.fermion_factor)
            bsupport = [q for m in b_modes for q in mode_to_qubits[m]]
            bpos = {q: i for i, q in enumerate(bsupport)}
            Bdense = np.zeros((1 << len(bsupport), 1 << len(bsupport)), dtype=complex)
            for (ms, monos), Mg in zip(b_groups.items(), boson_group_mats):
                gsupport = [bpos[q] for m in ms for q in mode_to_qubits[m]]
                Bdense += _embed_operator(Mg, gsupport, len(bsupport))
            M = np.kron(Fdense, Bdense) * abs(c)
            support = list(fsupport) + list(bsupport)

        atoms.append(HermitianAtom(
            'mixed', M, alpha, support,
            sign=(-1.0 if c.real < 0 else 1.0),
            payload={'fermion_factor': mt.fermion_factor,
                     'boson_group_mats': boson_group_mats,
                     'boson_group_bits': boson_group_bits}))
    return atoms


def build_hermitian_bundle_sim(mh, n_b, mode_to_qubits, w_sys):
    """Build the Hermitian bundle block-encoding unitary (exact sim).

    Returns `(U, alpha_tot, w_sys)`. `U` acts on `flag = (LCU-select ⊕ 1 dilation
    ancilla)` (MSBs) then the `w_sys`-qubit system, with
    `α_tot·⟨0|_flag U|0⟩_flag = H` and `U = U†` (⇒ the walk qubitizes)."""
    if not isinstance(mh, MixedHamiltonian):
        raise TypeError("expected a MixedHamiltonian")
    atoms = extract_hermitian_atoms(mh, n_b, mode_to_qubits, need_dense=True)
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
    atoms = extract_hermitian_atoms(mh, n_b, mode_to_qubits, need_dense=True)
    H = np.zeros((1 << w_sys, 1 << w_sys), dtype=complex)
    for atom in atoms:
        H += atom.sign * _embed_operator(atom.M, atom.support, w_sys)
    return H


# --------------------------------------------------------------------------- #
# Compiled cost roll-up (matching-dilation as real Qualtran bloqs)            #
# --------------------------------------------------------------------------- #


def _rotation_array_tc(n_entries, kappa=_KAPPA):
    """`ProgrammableRotationGateArray` cost for an `n_entries`-angle oracle.

    `kappa` = rotation-synthesis bit precision (from the P0-4 error budget).
    Uses a *varied non-zero* angle table — an all-zero table is costed as free
    by Qualtran (no rotations), which would massively undercount the amplitude
    oracle (the walk-T driver via rotation synthesis)."""
    n = max(2, int(n_entries))
    table = tuple(1 + (i % 251) for i in range(n))
    return ProgrammableRotationGateArray(table, kappa=kappa,
                                         rotation_gate=cirq.Y).t_complexity()


def _alias_prep_tc(n_items, prob_eps=_PREP_EPS):
    if n_items < 2:
        return TComplexity()
    return StatePreparationAliasSampling.from_lcu_probs(
        [1.0] * n_items, probability_epsilon=prob_eps).t_complexity()


def _and_ladder_tc(n_items):
    toff = 2 * max(0, n_items - 1)
    return TComplexity(t=4 * toff, clifford=9 * toff)


def _boson_matrix_cost(M, n_bits, kappa=_KAPPA, prob_eps=_PREP_EPS):
    """Compiled cost of the matching-dilation encoder of Hermitian `M`.

    Per component (diagonal + matchings): two amplitude oracles (the `M_k/α`
    values and the diagonal `√(I−M_k²/α²)`) over the `2^n_bits`-entry register,
    plus one conditional shift (`AddK`) per matching; an inner alias PREP over
    the components. All real Qualtran-bloq `.t_complexity()` — no floors/ceilings.
    `kappa`/`prob_eps` come from the P0-4 error budget.
    """
    diag, matchings = _split_into_components(M)
    n_entries = 1 << n_bits
    n_comp = (1 if np.abs(diag).max() > _TOL else 0) + len(matchings)
    n_comp = max(1, n_comp)
    total = 2 * _alias_prep_tc(n_comp, prob_eps)          # inner LCU PREP + PREP†
    if np.abs(diag).max() > _TOL:
        total = total + _rotation_array_tc(n_entries, kappa)   # diagonal oracle
    for Mk in matchings:
        total = total + 2 * _rotation_array_tc(n_entries, kappa)  # M_k/α + √ oracles
        shift = _shift_of_matching(Mk)
        total = total + AddK(bitsize=max(1, n_bits), k=max(1, shift % (1 << n_bits)),
                             signed=False).t_complexity()
    return total


def _shift_of_matching(Mk):
    nz = np.argwhere(np.abs(Mk) > _TOL)
    return abs(int(nz[0][0] - nz[0][1])) if len(nz) else 1


def _atom_cost(atom, kappa=_KAPPA, prob_eps=_PREP_EPS):
    if atom.kind == 'boson':
        return _boson_matrix_cost(atom.M, atom.payload['n_bits'], kappa, prob_eps)
    if atom.kind == 'fermion':
        return pylqt_t_complexity(fermion_atom_encoding(atom.payload['fermion_op']))
    if atom.kind == 'mixed':
        cost = pylqt_t_complexity(fermion_atom_encoding(atom.payload['fermion_factor']))
        mats = atom.payload['boson_group_mats']
        bits = atom.payload['boson_group_bits']
        for Mg, nb in zip(mats, bits):
            cost = cost + _boson_matrix_cost(Mg, nb, kappa, prob_eps)
        return cost
    raise ValueError(f"unknown atom kind {atom.kind!r}")


def hermitian_bundle_t_complexity(atoms, kappa=_KAPPA, prob_eps=_PREP_EPS):
    """Cost-MODEL `_t_complexity_` of the Hermitian bundle: 2·outer-PREP + AND-ladder
    + Σ atom costs. Leaf costs are real Qualtran bloqs, but the composition is
    hand-asserted (not compiled) and omits controls/matching predicates/boundary/
    phase — an OPTIMISTIC estimate. `kappa`/`prob_eps` from the P0-4 error budget
    (default to the module constants). See SparseHermitianBundleBlockEncoding."""
    total = 2 * _alias_prep_tc(len(atoms), prob_eps) + _and_ladder_tc(len(atoms))
    for atom in atoms:
        total = total + _atom_cost(atom, kappa, prob_eps)
    return total


# --------------------------------------------------------------------------- #
# pyLIQTR wrapper + valid-walk resource estimate                              #
# --------------------------------------------------------------------------- #

import qualtran.bloqs.mcmt  # noqa: E402,F401  (pyLIQTR reflection needs it)
from typing import Tuple  # noqa: E402
from qualtran._infra.data_types import QAny  # noqa: E402
from qualtran._infra.registers import Register, Signature  # noqa: E402
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding  # noqa: E402
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance  # noqa: E402
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator  # noqa: E402
from pyLIQTR.utils.resource_analysis import estimate_resources  # noqa: E402


class HermitianBundlePI(ProblemInstance):
    """ProblemInstance for the Hermitian bundle: atoms, α_tot, register widths."""

    def __init__(self, mh, n_b, num_sites, num_pion_species=3):
        self._mh = mh
        self._n_b = int(n_b)
        self._atoms = extract_hermitian_atoms(mh, n_b, mh.mode_to_qubits)
        self._alpha = sum(a.alpha for a in self._atoms)
        self._b_out = max(1, int(math.ceil(math.log2(max(1, len(self._atoms))))))
        # per-atom internal flag ancilla: inner LCU select over components + dil.
        def _atom_flag(a):
            if a.kind == 'boson':
                _d, matchings = _split_into_components(a.M)
                ncomp = max(1, len(matchings) + 1)
                return int(math.ceil(math.log2(max(2, ncomp)))) + 1
            if a.kind == 'fermion':
                return _fermion_flag(a.payload['fermion_op'])
            return _fermion_flag(a.payload['fermion_factor']) + 3
        self._A_atom = max((_atom_flag(a) for a in self._atoms), default=1)
        self._w_flag = self._b_out + self._A_atom
        self._w_sys = 4 * int(num_sites) + num_pion_species * int(num_sites) * self._n_b

    @property
    def alpha(self):
        return self._alpha

    @property
    def atoms(self):
        return self._atoms

    def n_qubits(self):
        return self._w_flag + self._w_sys

    def get_alpha(self, **kwargs):
        return self._alpha

    def __str__(self):
        return f"HermitianBundlePI(atoms={len(self._atoms)}, alpha={self._alpha:.4f})"


def _fermion_flag(fermion_op):
    n = fermion_jw_stats(fermion_op)['n_pauli_terms']
    return int(math.ceil(math.log2(max(2, n))))


class SparseHermitianBundleBlockEncoding(BlockEncoding):
    """Hermitian bundle carrying a primitive-based **cost MODEL** `_t_complexity_`.

    ⚠ **NOT a compiled block encoding (Codex audit 2026-08-18).** This class has
    **no** `decompose_from_registers` / call graph — `estimate_resources` simply
    reads the declared `_t_complexity_`, it does not compile or verify a
    PREP–SELECT–PREP† circuit. The *ideal* Hermitian construction is validated
    (it qubitizes — dense toy sims in the test suite), but the reported `Walk_T`
    is a **hand-assembled cost model, not a compiler-derived count**, and is
    OPTIMISTIC: it omits the matching endpoint/direction predicates + boundary
    (no-wrap) logic (an `AddK` is charged where a real matching permutation is
    needed), the coherent heterogeneous SELECT controls (per-atom cost charged
    *uncontrolled*), the actual (nonuniform) PREP weights and folded phases, and
    the real amplitude/√ angle tables (synthetic tables are used). **Use the
    PauliLCU anchor for publication numbers** until an executable decomposable
    composite lands (`compiled_bundle`). A declared `junk` register keeps
    `LogicalQubits` from silently dropping the qalloc'd temporaries (heuristic,
    not compiler-scheduled). **Precision is no longer hard-coded** — `kappa`
    (rotation-synthesis bits) and `prob_eps` (alias-PREP precision) come from the
    P0-4 error budget (`precision_budget`), derived from the target ΔE."""

    def __init__(self, problem_instance, control_val=None,
                 kappa=_KAPPA, prob_eps=_PREP_EPS, **kwargs):
        if not isinstance(problem_instance, HermitianBundlePI):
            raise TypeError("requires a HermitianBundlePI")
        super().__init__(problem_instance, control_val=control_val, **kwargs)
        self._encoding_type = None
        self._kappa = kappa            # rotation-synthesis bits (P0-4 budget)
        self._prob_eps = prob_eps      # alias-PREP precision (P0-4 budget)

    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('flag', QAny(self.PI._w_flag)),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('system', QAny(self.PI._w_sys)),)

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        j = self._peak_junk_width()
        return (Register('junk', QAny(j)),) if j > 0 else ()

    @property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers,
                          *self.target_registers])

    def _peak_junk_width(self):
        """Peak internal ancilla: outer alias-PREP temporaries + one atom's inner
        QROM-load (kappa) / inner alias PREP. Atoms fire one at a time, so the
        per-atom junk is a max, not a sum."""
        n = len(self.PI.atoms)
        outer = 0
        if n >= 2:
            sp = StatePreparationAliasSampling.from_lcu_probs(
                [1.0] * n, probability_epsilon=self._prob_eps)
            outer = sum(r.bitsize for r in sp.signature) - self.PI._b_out
        return max(0, outer) + self._kappa     # + one QROM-load kappa batch

    def _t_complexity_(self) -> TComplexity:
        return hermitian_bundle_t_complexity(
            self.PI.atoms, self._kappa, self._prob_eps)


def estimate_hermitian_sparse_resources(mh, n_b, num_sites, num_pion_species=3,
                                        delta_E=1.0, qpe_fraction=0.5):
    """Primitive-based COST-MODEL estimate for the Hermitian sparse bundle.

    Returns `{Walk_T_Count, Walk_Clifford_Count, Logical_Qubits, Physical_Lambda,
    n_atoms, budget}`. **Precision is now derived from the target energy accuracy
    `delta_E` (MeV), not hard-coded** (P0-4): the block-encoding error budget sets
    the alias-PREP precision (`probability_epsilon`) and the rotation-synthesis
    precision (`kappa` bits + `circuit_precision`) — see `precision_budget`. The
    `Physical_Lambda` (tighter Hermitian α_tot) IS a valid subnormalization.

    The *ideal* Hermitian construction qubitizes (validated on toys); the
    `Walk_T_Count` remains a cost model, **not compiler-derived** (the executable
    composite lives in `compiled_bundle`), so it is not a headline publication
    number — use the PauliLCU anchor. But it is now precision-consistent."""
    from src_PI.estimation.sparse_oracle.precision_budget import qpe_error_budget

    pi = HermitianBundlePI(mh, n_b, num_sites, num_pion_species)
    budget = qpe_error_budget(float(pi.alpha), delta_E, qpe_fraction)
    be = SparseHermitianBundleBlockEncoding(
        pi, kappa=budget['kappa'], prob_eps=budget['probability_epsilon'])
    # rotation synthesis to the budgeted total circuit precision (pyLIQTR divides
    # it across the rotations); the alias PREP precision is baked into _t_complexity_.
    res = estimate_resources(
        QubitizedWalkOperator(be), circuit_precision=budget['circuit_precision'])
    return {
        'Walk_T_Count': int(res['T']),
        'Walk_Clifford_Count': int(res['Clifford']),
        'Logical_Qubits': int(res['LogicalQubits']),
        'Physical_Lambda': float(be.alpha),
        'n_atoms': len(pi.atoms),
        'budget': budget,
    }
