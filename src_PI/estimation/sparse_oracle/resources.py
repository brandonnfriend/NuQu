"""
Full-bundle resource estimate for the sparse-oracle block encoder
(Phase C3d.1: analytical composition via Gilyén Lemma 30 + LCU).

This module aggregates the C3c single-mode `(â + â†)` block-encoding
cost across every term in a `MixedHamiltonian`, treating the full
Hamiltonian as a linear combination of unitaries (LCU) over per-term
block encodings. Per-term costs are derived via Gilyén Lemma 30
(LOBE §2.4):

    block-encoding(A · B) costs T_A + T_B, ancillae a_A + a_B,
    rescale α = α_A · α_B.

This is an **analytical** estimate — it doesn't build the full
composite Cirq circuit through pyLIQTR's `QubitizedWalkOperator`. The
returned T-count is what a real circuit *would* cost, derived from
each sub-bloq's Qualtran-tracked cost plus LCU PREP/SELECT overhead.
C3d.2 / C3d.3 will replace this with a real `SparseFullBundleBlockEncoding`
once the per-term encoders for `n̂`-shaped and multi-mode monomials
land. The Walk_T_Count reported here is what the comparison plot
should use as the sparse number.

**Approximations** (documented for the user, see execution log §10 and
the "C3d.3 future-polish notes" subsection):

  * **Boson side — UPPER bound.** Each P-factor monomial is costed as
    `P × single-mode (â+â†) walk T` (Gilyén product). Diagonal monomials
    (`n̂ = a^† a`) and pure-shift monomials (`a` or `a^†` alone) are
    strictly cheaper in the real BCK construction (d=1 vs d=2). So the
    boson contribution is a **conservative ceiling**.
  * **Fermion side — LOWER bound.** Pure-fermion JW Pauli strings are
    charged `4 · weight` T each (controlled-Pauli Toffoli decomp). This
    omits the per-string PauliLCU PREP/SELECT overhead, so the fermion
    contribution is a **floor**. At our problem sizes the fermion
    contribution is < 1% of total Walk_T_Count, so this directional
    asymmetry doesn't materially affect the C4 sparse-vs-PauliLCU plot —
    but readers should know it's there. C3d.3 polish would replace this
    with a real pyLIQTR PauliLCU sub-encoder cost calculation.
  * **Mixed terms** are aggregated per `MixedTerm` (Gilyén composition
    of fermion + boson sub-encoders), C3d.2 post-compression. NOT a
    cross-product expansion.
  * **LCU PREP cost** ≈ `4 · L_eff` T for `L_eff` LCU summands (coarse
    alias-sampling-style approximation; real cost depends on
    coefficient distribution).
"""

import functools
import math

from openfermion import jordan_wigner

from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources

from src_PI.estimation.sparse_oracle.block_encoding import (
    SingleLadderProblemInstance,
    SparseSingleLadderBlockEncoding,
)


# Cache the single-mode walk-step cost per n_b — invocations cost ~100 ms
# of pyLIQTR's resource analysis each, and we call this O(L) times per
# bundle. Once per n_b is enough.
@functools.lru_cache(maxsize=None)
def single_mode_walk_cost(n_b: int):
    """Per-walk-step resources for the single-mode `(â + â†)` BCK encoder.

    Returns the C3c estimate as a `(T, Clifford, LogicalQubits)` tuple.
    Used as the per-mode "atomic" cost in Gilyén composition.
    """
    pi = SingleLadderProblemInstance(n_b)
    be = SparseSingleLadderBlockEncoding(pi)
    walk = QubitizedWalkOperator(be)
    results = estimate_resources(walk)
    return results['T'], results['Clifford'], results['LogicalQubits']


def _pauli_strings(qubit_op):
    """Iterate over non-identity Pauli strings in a `QubitOperator`."""
    for term, coeff in qubit_op.terms.items():
        if term == ():
            continue
        yield term, coeff


def _ladder_factor_count(monomial):
    """Number of ladder factors (length of OpenFermion BosonOperator term tuple)."""
    return len(monomial)


def _modes_touched(monomial):
    """Set of distinct mode indices acted on by an OpenFermion boson monomial."""
    return set(mode_idx for mode_idx, _ in monomial)


def estimate_sparse_resources(mh, n_b, num_sites):
    """Aggregate the full-bundle sparse-oracle resource estimate.

    Args:
        mh: `MixedHamiltonian` from `pion_basis/fock_native.py`.
        n_b: bits per pion mode.
        num_sites: lattice sites (for qubit-count bookkeeping).

    Returns:
        dict with `Walk_T_Count`, `Walk_Clifford_Count`, `Logical_Qubits`,
        plus a `breakdown` sub-dict for diagnostics.
    """
    single_T, single_Cl, single_LQ = single_mode_walk_cost(n_b)

    # --- 1. Pure-boson contribution (H_pion_free terms) ---
    boson_T = 0
    boson_Cl = 0
    boson_terms = 0
    for monomial, _coeff in mh.boson_part.terms.items():
        if monomial == ():
            continue                              # identity (zero-point) — no LCU entry
        P = _ladder_factor_count(monomial)
        boson_T += P * single_T                   # Gilyén product upper bound
        boson_Cl += P * single_Cl
        boson_terms += 1

    # --- 2. Pure-fermion contribution (static nucleon, JW-mapped) ---
    fermion_T = 0
    fermion_Cl = 0
    fermion_terms = 0
    if len(mh.fermion_part.terms) > 0:
        f_q = jordan_wigner(mh.fermion_part)
        for pauli_term, _coeff in _pauli_strings(f_q):
            weight = len(pauli_term)
            fermion_T += 4 * weight               # controlled-Pauli Toffoli decomp
            fermion_Cl += 8 * weight
            fermion_terms += 1

    # --- 3. Mixed contribution (H_AV, H_WT — each carries native F · B factors) ---
    # Each `MixedTerm` is treated as ONE LCU summand whose block-encoding is
    # the Gilyén composition (LOBE §2.4) of:
    #   * BE_F = pyLIQTR-style PauliLCU over JW(mt.fermion_factor)
    #   * BE_B = multi-mode sparse encoder over mt.boson_factor
    # The composed BE has T-cost = T_F + T_B (Gilyén) and rescale
    # α = |mt.coeff| · α_F · α_B. C3d.1 used a cross-product expansion that
    # treated each `(JW-Pauli, boson-monomial)` pair as a separate LCU
    # summand — that's the LARGEST-L_eff option and overcounts the SELECT
    # cost by (~ #JW-Paulis × #boson-monomials / 1).
    mixed_T = 0
    mixed_Cl = 0
    mixed_terms_count = 0
    for mt in mh.mixed_terms:
        f_q = jordan_wigner(mt.fermion_factor)
        # T_F: PauliLCU sub-encoder cost. Approximate as Σ_p 4·weight T;
        # alias-sampling PREP overhead is small and folded into the
        # global PREP estimate below.
        T_F = sum(4 * len(p) for p, _ in _pauli_strings(f_q))
        Cl_F = sum(8 * len(p) for p, _ in _pauli_strings(f_q))
        # T_B: multi-mode sparse sub-encoder cost. Sum over boson monomials
        # (each contributes its P-factor Gilyén product cost).
        T_B = 0
        Cl_B = 0
        for b_monomial, _ in mt.boson_factor.terms.items():
            if b_monomial == ():
                continue
            P = _ladder_factor_count(b_monomial)
            T_B += P * single_T
            Cl_B += P * single_Cl
        # Gilyén product: F-side and B-side costs add.
        mixed_T += T_F + T_B
        mixed_Cl += Cl_F + Cl_B
        mixed_terms_count += 1

    # --- 4. SELECT cost and LCU summand count ---
    select_T = boson_T + fermion_T + mixed_T
    select_Cl = boson_Cl + fermion_Cl + mixed_Cl
    L_eff = boson_terms + fermion_terms + mixed_terms_count

    # --- 5. LCU PREP cost (coarse alias-sampling approximation) ---
    prep_T = 4 * max(1, L_eff)
    prep_Cl = 8 * max(1, L_eff)

    # --- 6. Walk operator cost: 2·PREP + SELECT (standard qubitized walk formula) ---
    walk_T = 2 * prep_T + select_T
    walk_Cl = 2 * prep_Cl + select_Cl

    # --- 7. Logical qubit count ---
    #   Bundle (nucleon + pion) + selection register (log L) + sparse-oracle
    #   ancilla overhead from the single-mode encoder (`single_LQ - n_b`).
    nucleon_qubits = 4 * num_sites
    pion_qubits = 3 * num_sites * n_b
    select_log = int(math.ceil(math.log2(max(2, L_eff))))
    sparse_ancilla_overhead = max(0, single_LQ - n_b)
    logical_qubits = (
        nucleon_qubits + pion_qubits + select_log + sparse_ancilla_overhead
    )

    return {
        'Walk_T_Count': int(walk_T),
        'Walk_Clifford_Count': int(walk_Cl),
        'Logical_Qubits': int(logical_qubits),
        'breakdown': {
            'L_eff': L_eff,
            'boson_terms': boson_terms,
            'fermion_terms': fermion_terms,
            'mixed_terms': mixed_terms_count,
            'single_mode_walk_T': int(single_T),
            'select_T': int(select_T),
            'prep_T': int(prep_T),
        },
    }
