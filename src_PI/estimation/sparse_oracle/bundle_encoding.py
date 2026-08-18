"""
Compiled sparse full-bundle block encoding (C1 step 3).

Assembles the per-atom encoders of steps 1-2 into a single pyLIQTR
`BlockEncoding` for the whole `MixedHamiltonian`, as a linear combination of
unitaries (LCU) over *atoms*:

    H = Σ_l c_l M_l ,   atom l ∈ { boson monomial, static fermion, mixed term }

where each `M_l` has an exact block encoding `B_l` with
`α_l·⟨0|_anc B_l|0⟩_anc = M_l` (α_l ≥ 0). Folding the signs `s_l = sign(c_l)`
into the outer SELECT and the magnitudes into the PREP:

    U = PREP†_outer · SELECT · D_sign · PREP_outer ,
    PREP_outer : |0⟩_sel ↦ Σ_l √(w_l/α_tot) |l⟩ ,   w_l = |c_l|·α_l ,
    D_sign     : |l⟩ ↦ s_l |l⟩ ,
    SELECT     : |l⟩ |0⟩_anc |ψ⟩ ↦ |l⟩ B_l(|0⟩_anc |ψ⟩) ,

gives `α_tot·⟨0|_{sel,anc} U |0⟩_{sel,anc} = H`, with

    α_tot = Σ_l w_l = Σ_l |c_l|·α_l = compute_native_lambda(mh, n_b)['physical_lambda'].

That identity is the **α_tot invariant** (design §2, §6.1) — it holds by
construction because each atom's α_l is the same d=1 sparse-oracle / Pauli
1-norm factor `compute_native_lambda` sums.

**Reflection subspace / block-flag qubit placement.**
`QubitizedWalkOperator.decompose_from_registers` reflects about *all*
`selection_registers` qubits being |0⟩ and leaves `junk_registers` unreflected
(assumed to return to |0⟩). The flagged block of this encoding is
`sel=0 ∧ atom_anc=0`, so **both the outer selection AND the shared atom
ancilla are placed in `selection_registers`** (one `flag` register) — every
block-flag qubit is reflected, none hides in junk. The toy assembly sim
(`tests/test_sparse_full_bundle.py`) validates `α_tot·⟨0|_flag U|0⟩_flag = H`
directly.

**⚠ KNOWN DEFECT — the walk is not yet a valid qubitization (Hermitization
pending; quantum-algorithms review 2026-08-18).** The d=1 per-mode atoms encode
*non-Hermitian* monomials (`â`, `â†` alone), so the assembled block encoding `U`
is NOT Hermitian (`‖U−U†‖ ≫ 0`). pyLIQTR's `QubitizedWalkOperator` is a
*single-reflection* walk `W = (2Π−I)·U`, which qubitizes only a Hermitian `U`;
for this non-Hermitian `U` the walk spectrum lacks the qubitization phases
`e^{±i·arccos(E_k/α)}` (regression test
`test_bundle_walk_qubitizes_hermitian_H`, currently xfail). **What IS exact and
validated:** the block encoding `α_tot·⟨0|U|0⟩ = H` and `α_tot` (= Λ). **What is
NOT yet valid:** the *walk*, so `estimate_resources(QubitizedWalkOperator(be))`
is a **block-encoding-level cost estimate**, not a genuine QPE walk cost, until
the atoms are Hermitized (re-pair `c·m + c̄·m†` into Hermitian d=2 encoders; the
fermion PauliLCU atoms are already Hermitian; α_tot is preserved under
re-pairing).

**Cost roll-up (still an improvement over the mixed-bound proxy).**
`_t_complexity_` is a roll-up of real sub-bloq `.t_complexity()` values —
alias-sampling PREP (linear-T), per-atom block-encoding costs (boson
`SparseBosonMonomialBlockEncoding`, fermion off-the-shelf PauliLCU), the
unary-dispatch AND-ladder — with no hand-inserted `P·single_ladder` ceiling or
`4·weight` floor. Two known undercounts (per-atom SELECT charged *uncontrolled*;
`LogicalQubits` junk width estimated) push it optimistically; and the
Hermitization overhead above is not yet added. Treat the number as an optimistic
block-encoding-level estimate, not a final compiled walk cost.

Status (C1 step 3): cost path (α_tot + `_t_complexity_` + signature) +
`decompose_from_registers` ideal-sim for the toy assembly gate. The toy sim
validates the *novel* outer block-encoding assembly (PREP + phase fold +
dispatch + shared-ancilla uncompute + block-flag placement) for boson + fermion
atoms; mixed-atom operator sim and walk Hermiticity are open (see the test file).
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import cirq
import numpy as np
import qualtran as qt
import qualtran.bloqs.mcmt  # noqa: F401  (pyLIQTR reflection needs it preloaded)

from qualtran._infra.data_types import QAny
from qualtran._infra.registers import Register, Signature
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance
from pyLIQTR.utils.resource_analysis import pylqt_t_complexity

from src_PI.estimation.sparse_oracle.boson_monomial_encoding import (
    BosonMonomialProblemInstance,
    SparseBosonMonomialBlockEncoding,
    build_boson_monomial_circuit,
    monomial_alpha,
    monomial_mode_groups,
    monomial_reference_matrix,
)
from src_PI.estimation.sparse_oracle.fermion_atom import fermion_atom_encoding
from src_PI.estimation.sparse_oracle.fermion_jw_stats import fermion_jw_stats
from src_PI.estimation.sparse_oracle.jw_cache import jordan_wigner_cached
from src_PI.estimation.sparse_oracle.lambda_compute import (
    _monomial_max_amplitude,
    compute_native_lambda,
)
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian

_PREP_EPS = 1e-3          # alias-sampling coefficient precision
_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Atom extraction                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class BundleAtom:
    """One LCU summand of the bundle.

    kind    : 'boson' | 'fermion' | 'mixed'.
    coeff   : c_l (complex — the pion coupling injects imaginary weights, e.g.
              H_WT's conjugate momentum Π = i(â†−â)/√2; the phase e^{i·arg c_l}
              folds into the outer SELECT `D_phase`, the magnitude into PREP).
    alpha   : α_l ≥ 0 — the block-encoding rescale (d=1 max / Pauli 1-norm /
              Gilyén product), the factor compute_native_lambda sums (via |·|).
    n_flag_anc : width of this atom's block-flag ancilla (∈ selection_registers).
    payload : kind-specific data for the cost roll-up + sim.
    """
    kind: str
    coeff: complex
    alpha: float
    n_flag_anc: int
    payload: dict = field(default_factory=dict)

    @property
    def weight(self):
        return abs(self.coeff) * self.alpha

    @property
    def phase(self):
        """Unit-modulus e^{i·arg(c_l)} folded into D_phase (1.0 if coeff≈0)."""
        m = abs(self.coeff)
        return 1.0 + 0j if m <= _TOL else self.coeff / m

    @property
    def is_phased(self):
        """True if the phase is not +1 (needs a D_phase entry)."""
        return abs(self.phase - 1.0) > 1e-12


def _boson_flag_anc(monomial):
    """Block-flag ancilla width for a boson monomial atom = #touched modes."""
    return len(monomial_mode_groups(monomial))


def _fermion_flag_anc(fermion_op):
    """PauliLCU block-flag width = ceil(log2(#non-identity Pauli strings))."""
    n = fermion_jw_stats(fermion_op)['n_pauli_terms']
    return int(math.ceil(math.log2(max(2, n))))


def _boson_op_lcu(boson_op, n_b):
    """(α_b, list[(monomial, coeff)], flag_width) for an LCU over a BosonOperator.

    α_b = Σ_q |coeff_q|·max-amplitude(monomial_q) — the boson-factor rescale used
    by compute_native_lambda; identity monomials are excluded (classical shift).
    Coefficients are kept **complex** (H_WT's Π pieces are imaginary); the
    magnitude sets α_b, the phase folds into the inner SELECT. flag_width =
    (inner-select bits over the monomials) + (max #modes over them).
    """
    terms = [(m, complex(c)) for m, c in boson_op.terms.items() if m != ()]
    alpha_b = sum(abs(c) * _monomial_max_amplitude(m, n_b) for m, c in terms)
    if terms:
        inner_sel = int(math.ceil(math.log2(max(2, len(terms)))))
        max_modes = max(len(monomial_mode_groups(m)) for m, _ in terms)
    else:
        inner_sel = max_modes = 0
    return alpha_b, terms, inner_sel + max_modes


def extract_atoms(mh, n_b):
    """Decompose a `MixedHamiltonian` into `BundleAtom`s (per-MixedTerm compressed).

    Boson monomials → one atom each; the whole static `fermion_part` → one atom;
    each `MixedTerm` → one atom (Gilyén product α = α_fermion·α_boson). Matches
    `compute_native_lambda`'s term walk exactly, so Σ_l w_l = physical_lambda.
    """
    atoms = []

    # Pure-boson atoms (free-pion; real coeffs, but kept complex for generality).
    for monomial, coeff in mh.boson_part.terms.items():
        if monomial == ():
            continue                                   # identity → classical shift
        c = complex(coeff)
        if abs(c) <= _TOL:
            continue
        atoms.append(BundleAtom(
            kind='boson',
            coeff=c,
            alpha=monomial_alpha(monomial, n_b),
            n_flag_anc=_boson_flag_anc(monomial),
            payload={'monomial': monomial, 'n_b': n_b},
        ))

    # Static-fermion atom (whole fermion_part as one PauliLCU).
    if len(mh.fermion_part.terms) > 0:
        fstats = fermion_jw_stats(mh.fermion_part)
        if fstats['one_norm'] > _TOL:
            atoms.append(BundleAtom(
                kind='fermion',
                coeff=1.0 + 0j,                 # sign/phase lives inside the Paulis
                alpha=fstats['one_norm'],
                n_flag_anc=_fermion_flag_anc(mh.fermion_part),
                payload={'fermion_op': mh.fermion_part},
            ))

    # Mixed atoms (H_AV / H_WT): one Gilyén-product atom per MixedTerm. The
    # fermion/boson factors carry their own internal (possibly imaginary) phases
    # inside BE_F / BE_B; the outer atom coeff is mt.coeff, α = α_f·α_b (moduli).
    for mt in mh.mixed_terms:
        fstats = fermion_jw_stats(mt.fermion_factor)
        alpha_f = fstats['one_norm']
        alpha_b, b_terms, boson_flag = _boson_op_lcu(mt.boson_factor, n_b)
        c = complex(mt.coeff)
        if alpha_f <= _TOL or alpha_b <= _TOL or abs(c) <= _TOL:
            continue
        atoms.append(BundleAtom(
            kind='mixed',
            coeff=c,
            alpha=alpha_f * alpha_b,
            n_flag_anc=_fermion_flag_anc(mt.fermion_factor) + boson_flag,
            payload={
                'fermion_factor': mt.fermion_factor,
                'boson_terms': b_terms,
                'n_b': n_b,
            },
        ))

    return atoms


# --------------------------------------------------------------------------- #
# Cost roll-up helpers                                                        #
# --------------------------------------------------------------------------- #


def _boson_monomial_block_cost(monomial, n_b):
    be = SparseBosonMonomialBlockEncoding(BosonMonomialProblemInstance(monomial, n_b))
    return be._t_complexity_()


def _fermion_block_cost(fermion_op):
    """Block-encoding (PREP_F + SELECT_F) cost of the PauliLCU over JW(fermion_op)."""
    return pylqt_t_complexity(fermion_atom_encoding(fermion_op))


def _alias_prep_cost(weights):
    """`StatePreparationAliasSampling` cost for the given non-negative weights."""
    if len(weights) < 2:
        return TComplexity()
    probs = list(np.asarray(weights, dtype=float))
    sp = StatePreparationAliasSampling.from_lcu_probs(
        probs, probability_epsilon=_PREP_EPS)
    return sp.t_complexity()


def _and_ladder_cost(n_items):
    """Unary-iteration AND-ladder for an n-way controlled dispatch: ~2(n-1) Toffoli.

    A Toffoli is 4 T + Clifford; we roll up via the T-count (the Clifford part is
    subdominant and tracked loosely)."""
    toffolis = 2 * max(0, n_items - 1)
    return TComplexity(t=4 * toffolis, clifford=9 * toffolis)


def _atom_select_cost(atom):
    """Block-encoding cost of one atom's controlled `B_l` (uncontrolled leaf cost;
    the dispatch control is charged once in the AND-ladder)."""
    if atom.kind == 'boson':
        return _boson_monomial_block_cost(atom.payload['monomial'], atom.payload['n_b'])
    if atom.kind == 'fermion':
        return _fermion_block_cost(atom.payload['fermion_op'])
    if atom.kind == 'mixed':
        n_b = atom.payload['n_b']
        cost = _fermion_block_cost(atom.payload['fermion_factor'])
        b_terms = atom.payload['boson_terms']
        # inner boson LCU: alias PREP over the monomial weights + Σ monomial blocks.
        b_weights = [abs(c) * _monomial_max_amplitude(m, n_b) for m, c in b_terms]
        cost = cost + 2 * _alias_prep_cost(b_weights)
        for m, _c in b_terms:
            cost = cost + _boson_monomial_block_cost(m, n_b)
        cost = cost + _and_ladder_cost(len(b_terms))
        return cost
    raise ValueError(f"unknown atom kind {atom.kind!r}")


# --------------------------------------------------------------------------- #
# ProblemInstance + BlockEncoding                                             #
# --------------------------------------------------------------------------- #


class FullBundleProblemInstance(ProblemInstance):
    """Carries the atom list, α_tot, and register widths for the full bundle."""

    def __init__(self, mh, n_b, num_sites, num_pion_species=3):
        if not isinstance(mh, MixedHamiltonian):
            raise TypeError("FullBundleProblemInstance expects a MixedHamiltonian")
        self._mh = mh
        self._n_b = int(n_b)
        self._num_sites = int(num_sites)
        self._mode_to_qubits = dict(mh.mode_to_qubits)
        self._atoms = extract_atoms(mh, n_b)
        self._alpha = sum(a.weight for a in self._atoms)
        # register widths
        self._b_out = int(math.ceil(math.log2(max(2, len(self._atoms)))))
        self._A_atom = max((a.n_flag_anc for a in self._atoms), default=1)
        self._w_flag = self._b_out + self._A_atom
        # System width: the standard interleaved layout is 4·S nucleon + (species·
        # S·n_b) boson qubits; take the max with the actual atom supports so a
        # hand-built toy MixedHamiltonian sizes correctly too.
        std = 4 * self._num_sites + num_pion_species * self._num_sites * self._n_b
        self._w_sys = max(std, _max_support_index(mh, self._mode_to_qubits) + 1)

    @property
    def alpha(self):
        return self._alpha

    @property
    def atoms(self):
        return self._atoms

    @property
    def n_b(self):
        return self._n_b

    @property
    def mode_to_qubits(self):
        return self._mode_to_qubits

    def n_qubits(self):
        return self._w_flag + self._w_sys

    def get_alpha(self, **kwargs):
        return self._alpha

    def __str__(self):
        return (f"FullBundleProblemInstance(atoms={len(self._atoms)}, "
                f"alpha={self._alpha:.4f})")


class SparseFullBundleBlockEncoding(BlockEncoding):
    """Compiled sparse block encoding of a whole `MixedHamiltonian`.

    `selection_registers` = one `flag` register (b_out outer-select + A_atom
    shared atom ancilla — every block-flag qubit, so the walk reflection is
    correct by construction). `target_registers` = the `system` register.
    `alpha` = α_tot = compute_native_lambda physical_lambda.
    """

    def __init__(self, problem_instance, control_val=None, **kwargs):
        if not isinstance(problem_instance, FullBundleProblemInstance):
            raise TypeError(
                "SparseFullBundleBlockEncoding requires a FullBundleProblemInstance")
        super().__init__(problem_instance, control_val=control_val, **kwargs)
        self._encoding_type = None

    # --- register signature --------------------------------------------

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
        # Peak internal junk: alias-sampling temporaries (dominant) + one atom's
        # QROM-load kappa. Declared so LogicalQubits is not silently undercounted.
        j = self._peak_junk_width()
        return (Register('junk', QAny(j)),) if j > 0 else ()

    @property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers,
                          *self.target_registers])

    def _peak_junk_width(self):
        n_atoms = len(self.PI.atoms)
        if n_atoms < 2:
            return 8
        probs = [max(a.weight, 1e-18) for a in self.PI.atoms]
        sp = StatePreparationAliasSampling.from_lcu_probs(
            probs, probability_epsilon=_PREP_EPS)
        alias_junk = sum(r.bitsize for r in sp.signature) - self.PI._b_out
        return max(0, alias_junk) + 8            # +8 for a QROM-load kappa batch

    # --- cost -----------------------------------------------------------

    def _t_complexity_(self) -> TComplexity:
        atoms = self.PI.atoms
        weights = [a.weight for a in atoms]
        total = 2 * _alias_prep_cost(weights)             # PREP + PREP†
        total = total + _and_ladder_cost(len(atoms))      # unary dispatch
        for atom in atoms:
            total = total + _atom_select_cost(atom)
        # D_phase is a diagonal Clifford/phase on sel per phased atom (real sign
        # = Z; imaginary = S); Clifford-only, tracked loosely.
        n_phased = sum(1 for a in atoms if a.is_phased)
        total = total + TComplexity(clifford=n_phased)
        return total

    # --- ideal-sim decomposition (toy assembly gate; NOT the cost path) --

    def decompose_from_registers(self, *, context=None, **quregs):
        """Ideal LCU assembly for the toy sim: PREP → D_phase → SELECT → PREP†.

        Uses *ideal* primitives (exact PREP/phase MatrixGates; step-1 boson
        encoders for boson atoms; 1-ancilla contraction dilations for fermion
        atoms) so `cirq.unitary` reproduces `α_tot·⟨0|_flag U|0⟩_flag = H` to
        machine precision on a tiny instance — validating the novel outer
        assembly (magnitudes, phase fold, unary dispatch, shared-ancilla
        return-to-|0⟩, reflection subspace). The realistic bloq costs live in
        `_t_complexity_`; this method is never used for costing (mixed atoms
        raise — they're covered by the α_tot invariant + steps 1-2)."""
        pi = self.PI
        flag = list(quregs['flag'])
        sysq = list(quregs['system'])
        b_out, A_atom = pi._b_out, pi._A_atom
        sel = flag[:b_out]
        atom_anc = flag[b_out:b_out + A_atom]
        atoms = pi.atoms

        amps = [math.sqrt(a.weight / pi._alpha) for a in atoms]
        prep = _householder_prep(amps, b_out)
        yield cirq.MatrixGate(prep, name='PREP').on(*sel)

        dphase = np.array([a.phase for a in atoms])
        if np.any(np.abs(dphase - 1.0) > 1e-12):
            diag = np.ones(1 << b_out, dtype=complex)
            diag[:len(atoms)] = dphase
            yield cirq.MatrixGate(np.diag(diag), name='Dphase').on(*sel)

        for l, atom in enumerate(atoms):
            U, anc_w, support = _atom_sim_unitary(atom, pi.mode_to_qubits)
            targets = list(atom_anc[:anc_w]) + [sysq[g] for g in support]
            ctrl_vals = [(l >> (b_out - 1 - k)) & 1 for k in range(b_out)]
            yield cirq.ControlledGate(
                cirq.MatrixGate(U, name=f'B{l}'),
                control_values=ctrl_vals,
            ).on(*sel, *targets)

        yield cirq.MatrixGate(prep.conj().T, name='PREPdag').on(*sel)


# --------------------------------------------------------------------------- #
# Ideal-sim helpers (toy assembly gate)                                       #
# --------------------------------------------------------------------------- #


def _max_support_index(mh, mode_to_qubits):
    """Largest global qubit index used by any atom (boson modes + fermion terms)."""
    m = 0
    for qs in mode_to_qubits.values():
        if qs:
            m = max(m, max(qs))
    ops = [mh.fermion_part] + [mt.fermion_factor for mt in mh.mixed_terms]
    for op in ops:
        for term in op.terms:
            for idx, _act in term:
                m = max(m, idx)
    return m


def _householder_prep(amps, b_out):
    """Real symmetric-orthogonal PREP with first column = padded, normalized amps.

    A Householder reflection mapping |0⟩ to the (unit) amplitude vector; its own
    inverse, so PREP† = PREPᵀ = PREP."""
    dim = 1 << b_out
    v = np.zeros(dim)
    v[:len(amps)] = amps
    nrm = np.linalg.norm(v)
    if nrm < _TOL:
        return np.eye(dim)
    v = v / nrm
    e0 = np.zeros(dim)
    e0[0] = 1.0
    u = e0 - v
    nu = np.linalg.norm(u)
    if nu < _TOL:                                    # v already ≈ |0⟩
        return np.eye(dim)
    u = u / nu
    return np.eye(dim) - 2.0 * np.outer(u, u)


_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _fermion_dense(fermion_op):
    """Dense `jordan_wigner(fermion_op)` matrix + its sorted qubit support.

    Support = every qubit any Pauli string touches (incl. JW Z-fill); position i
    (MSB-first in the returned matrix) ↔ support[i]."""
    q = jordan_wigner_cached(fermion_op)
    support = sorted({g for term in q.terms for g, _op in term})
    if not support:
        # pure identity → 1×1 scalar acting on no qubits
        coeff = sum(complex(c) for t, c in q.terms.items())
        return np.array([[coeff]], dtype=complex), []
    pos = {g: i for i, g in enumerate(support)}
    s = len(support)
    M = np.zeros((1 << s, 1 << s), dtype=complex)
    for term, coeff in q.terms.items():
        letters = ['I'] * s
        for g, op in term:
            letters[pos[g]] = op
        mat = np.array([[1.0 + 0j]])
        for ch in letters:
            mat = np.kron(mat, _PAULI[ch])
        M += complex(coeff) * mat
    return M, support


def _contraction_dilation(M, alpha):
    """1-ancilla block encoding of `M/alpha`: U = [[A, Dc],[Dr, −A†]], A = M/α.

    Unitary iff ‖A‖ ≤ 1 (α ≥ ‖M‖, true for α = Pauli 1-norm ≥ spectral norm).
    `⟨0|_anc U|0⟩_anc = A`, so `α·⟨0|U|0⟩ = M`. Ancilla is the MSB qubit."""
    A = np.asarray(M, dtype=complex) / alpha
    s = A.shape[0]
    ident = np.eye(s, dtype=complex)

    def _psd_sqrt(P):
        w, V = np.linalg.eigh((P + P.conj().T) / 2.0)
        w = np.clip(w.real, 0.0, None)
        return (V * np.sqrt(w)) @ V.conj().T

    Dc = _psd_sqrt(ident - A @ A.conj().T)
    Dr = _psd_sqrt(ident - A.conj().T @ A)
    return np.block([[A, Dc], [Dr, -A.conj().T]])


def _atom_sim_unitary(atom, mode_to_qubits):
    """`(U, anc_width, support_global)` — the atom's ideal block-encoding unitary.

    `α_l·⟨0|_{anc_width} U |0⟩ = M_l` on `support_global` (MSB = first). Boson
    atoms use the validated step-1 encoder (K amp ancilla); fermion atoms use a
    1-ancilla dilation. Mixed atoms raise — see `decompose_from_registers`."""
    if atom.kind == 'boson':
        mono, n_b = atom.payload['monomial'], atom.payload['n_b']
        circ, amp_qubits, mode_regs, _alpha = build_boson_monomial_circuit(mono, n_b)
        order = list(amp_qubits) + [q for reg in mode_regs for q in reg]
        U = circ.unitary(qubit_order=order)
        modes = [m for m, _act in monomial_mode_groups(mono)]
        support = [g for m in modes for g in mode_to_qubits[m]]
        return U, len(amp_qubits), support
    if atom.kind == 'fermion':
        M, support = _fermion_dense(atom.payload['fermion_op'])
        return _contraction_dilation(M, atom.alpha), 1, support
    raise NotImplementedError(
        "mixed-atom ideal sim is intentionally unimplemented — the outer "
        "assembly is validated by boson+fermion toys; mixed atoms are covered "
        "by the α_tot invariant (all mixed terms) + steps 1-2.")


def _embed_operator(M, support, N):
    """Embed operator `M` (on `support` qubit positions, MSB-first) into the full
    `N`-qubit register (identity elsewhere). O(2^N·2^k); toy-only."""
    k = len(support)
    dim = 1 << N
    out = np.zeros((dim, dim), dtype=complex)
    if k == 0:
        return complex(M[0, 0]) * np.eye(dim, dtype=complex)
    for i in range(dim):
        bits = [(i >> (N - 1 - p)) & 1 for p in range(N)]
        sup_in = 0
        for t in range(k):
            sup_in = (sup_in << 1) | bits[support[t]]
        for sup_out in range(1 << k):
            val = M[sup_out, sup_in]
            if abs(val) < 1e-15:
                continue
            bj = list(bits)
            for t in range(k):
                bj[support[t]] = (sup_out >> (k - 1 - t)) & 1
            j = 0
            for p in range(N):
                j = (j << 1) | bj[p]
            out[j, i] += val
    return out


def extracted_bundle_block(be):
    """`α_tot·⟨0|_flag U|0⟩_flag` — the operator the (ideal-sim) bundle encodes.

    Tiny instances only (builds the full `cirq.unitary`). Compare to
    `bundle_reference_matrix(be)`."""
    pi = be.PI
    flag = [cirq.NamedQubit(f'flag_{i}') for i in range(pi._w_flag)]
    sysq = [cirq.NamedQubit(f'sys_{i}') for i in range(pi._w_sys)]
    circuit = cirq.Circuit(
        be.decompose_from_registers(context=None, flag=flag, system=sysq))
    U = circuit.unitary(qubit_order=flag + sysq)
    dim_sys = 1 << pi._w_sys
    return U[:dim_sys, :dim_sys] * pi._alpha        # flag are MSBs → |0⟩ block


def bundle_reference_matrix(be):
    """Exact `H = Σ_l c_l M_l` on the system register (for the toy assembly gate)."""
    pi = be.PI
    H = np.zeros((1 << pi._w_sys, 1 << pi._w_sys), dtype=complex)
    for atom in pi.atoms:
        if atom.kind == 'boson':
            M = monomial_reference_matrix(atom.payload['monomial'], atom.payload['n_b'])
            modes = [m for m, _a in monomial_mode_groups(atom.payload['monomial'])]
            support = [g for m in modes for g in pi.mode_to_qubits[m]]
        elif atom.kind == 'fermion':
            M, support = _fermion_dense(atom.payload['fermion_op'])
        else:
            raise NotImplementedError("bundle_reference_matrix: mixed atoms are "
                                      "not part of the toy sim (see _atom_sim_unitary)")
        H += atom.coeff * _embed_operator(M, support, pi._w_sys)
    return H
