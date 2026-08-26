"""Controlled-sum LCU walk — EXPERIMENTAL cost roll-up for the split-oracle (NOT VALIDATED).

⚠️ STATUS (codex audit `02_quantum_resources/amplitude_combined_walk_audit_2026-08-20.md`):
this is an **experimental cost roll-up, not a validated block encoding or QPE walk**, and
its amplitude numbers are **not publication-grade**. It picks the right *target*
architecture (a controlled-sum LCU) but does NOT build it, and it does NOT fix the operator:
the six Weinberg–Tomozawa ε-terms remain inside one `pos_dyn` PauliLCU that treats π and Π
as simultaneously diagonal (impossible — each ε-term needs a *different* species in the
momentum basis). Adding a scalar `wt_basis_change_t` T-charge does not change the operator
PauliLCU block-encodes, so the amplitude resource number still does not correspond to the
Watson amplitude Hamiltonian. **Do not use amplitude `combined_lcu` for watson/ns QPE totals,
amplitude-vs-Fock comparisons, or any paper claim.** The paper's anchor is Fock/PauliLCU; the
valid controlled species-selective sum encoding is FUTURE WORK (see the audit's 7-step repair).

The amplitude basis splits the Hamiltonian into `H_pos` (position/field basis) and
`H_mom` (Π², momentum basis). The LEGACY pipeline qubitized each independently, summed
the two walk T-counts, and added a flat QFT charge (`walk_composition='split_sum'`).
That is **not** a QPE algorithm for `H_pos + H_mom`: applying W_pos then W_mom N_walk
times realizes (W_pos·W_mom)^N_walk, whose eigenphases are unrelated to the eigenvalues
of the sum, and no QFT-between-independent-walks changes that (codex audit P0-4 /
release-blocker 3).

This module TARGETS the right general construction — a single block encoding of the sum
via a controlled-sum LCU, then ONE qubitized walk — but implements only its numerical cost
roll-up (no PREP/SELECT/flag/walk is constructed, no projected block is verified, the WT
species-selective branches are not extracted, and controlled-branch overhead is omitted):

    B_H = PREP · SELECT · PREP†     (one LCU control register over the k sub-oracles)
      PREP   : Σ_i √(λ_i/λ) |i⟩ ⊗ (sub-oracle i's own PREPARE)          λ = Σ_i λ_i
      SELECT : controlled on |i⟩, apply sub-oracle i's SELECT, wrapped in its
               basis change — for the momentum branch and the six species-selective
               WT ε-terms this is QFT · SELECT_i · QFT† applied *inside* SELECT.
    W_H = R · B_H                   (one reflection about |0⟩ on the combined ancilla)

QPE then makes N_walk = π·λ/ΔE queries to the SINGLE walk W_H. λ (= Σ_i λ_i) and
N_walk are unchanged from the legacy path (they were already right); what changes is the
per-step T-count (one combined walk, not two summed), the logical-qubit count (adds the
LCU control register + QFT workspace the real walk needs), and the QFT is now a coherent
part of the step rather than an external charge.

This is only a COST ROLL-UP, and an incomplete one. A defensible compositional estimate
would have to be *derived from a validated small executable composite* (verified to project
to the target amplitude Hamiltonian, Hermitian, clean ancillae, correct qubitization
eigenphases) and include every controlled operation — none of which is done here. It also
sums UNCONTROLLED sub-encoding costs for what must be LCU-controlled SELECTs, undercounts the
two-sided outer PREP, and uses a QFT/`δ_WT`/reflection cost with no derived precision contract.
So the amplitude output is an experimental placeholder, not a publication number. The paper
uses the Fock/PauliLCU anchor; the valid controlled species-selective sum encoding is future
work (codex `02_quantum_resources/amplitude_combined_walk_audit_2026-08-20.md`).
"""

import math

# --- composition-piece precision knobs (documented; subdominant to SELECT) ------ #
# k-way PREPARE rotation synthesis precision (Ross–Selinger single-rotation T-cost).
_ROT_SYNTH_EPS = 1e-10
# Species-selective WT basis change: the six ε-terms (H_WT_Logic) each need a distinct
# species in the momentum basis, so no single basis diagonalizes them (Watson Eqs.
# 102–104). Modeled as controlled QFT pairs on individual species registers, amortized
# to `_WT_SPECIES_QFT_PAIRS` per walk step (one per pion species). This basis-change LCU
# incurs a controlled-Trotter error `δ_WT` on top of the block-encoding precision — it is
# a COST MODEL for the species-selective sequence, not a compiled term-controlled circuit.
_WT_SPECIES_QFT_PAIRS = 3          # 3 pion species cycled through by the 6 ε-terms


def _rotation_synth_t(eps=_ROT_SYNTH_EPS):
    """Ross–Selinger T-count for one arbitrary single-qubit rotation to precision `eps`."""
    return int(math.ceil(1.15 * math.log2(1.0 / eps) + 9.2))


def _reflection_t(n_ancilla):
    """T-count of the walk reflection R = about |0⟩ on `n_ancilla` qubits — an
    (n_ancilla−1)-controlled Z ≈ (n_ancilla−1) Toffoli (~4 T each). Subdominant."""
    return 4 * max(0, n_ancilla - 1)


def _qft_t_per_register(n_b):
    """Per-register QFT T-cost (matches EstimateResources.calculate_qft_cost)."""
    if n_b <= 1:
        return 0
    return int(8 * n_b * math.log2(n_b))


def wt_basis_change_t(L, dim, n_b):
    """Species-selective basis-change T-cost for the six WT ε-terms (Watson Eqs.
    102–104), modeled as `_WT_SPECIES_QFT_PAIRS` QFT+IQFT pairs, each on one pion
    species' `L**dim` registers. Cost model with controlled-Trotter precision δ_WT."""
    t_per = _qft_t_per_register(n_b)
    sites = L ** dim
    return _WT_SPECIES_QFT_PAIRS * 2 * sites * t_per     # ×2 = QFT + IQFT


def compose_combined_walk(per_sub, momentum_qft_t, L, dim, n_b):
    """Compose the QPE-valid single walk W_H from ≥2 split-oracle sub-encodings.

    `per_sub`: list of dicts with the pyLIQTR-derived BLOCK-ENCODING (SELECT+PREPARE)
    costs per sub-oracle: {'name', 'T_enc', 'Clifford_enc', 'qubits_enc', 'n_system',
    'alpha'}.  `momentum_qft_t`: the QFT+IQFT T-cost for the momentum branch
    (EstimateResources.calculate_qft_cost). Returns the combined-walk dict.
    """
    k = len(per_sub)
    if k < 2:
        raise ValueError("compose_combined_walk needs ≥2 sub-oracles (the split case)")

    n_lcu_control = int(math.ceil(math.log2(k)))            # 1 for k=2
    n_qft_workspace = n_b                                   # phase-gradient reg (reusable)

    # Block-encoding pieces: Σ SELECT+PREPARE over sub-oracles + basis changes + k-way PREP.
    sum_enc_t = sum(int(e['T_enc']) for e in per_sub)
    sum_enc_clifford = sum(int(e['Clifford_enc']) for e in per_sub)
    basis_change_t = int(momentum_qft_t) + wt_basis_change_t(L, dim, n_b)
    prep_lcu_t = (k - 1) * _rotation_synth_t()             # k-way coefficient PREPARE

    # Ancilla: the sub-oracles' block-encoding ancillas are reused (only one branch is
    # live per LCU value) → max, plus the LCU control register and QFT workspace.
    n_system = max(int(e['n_system']) for e in per_sub)
    n_ancilla_sub = max(int(e['qubits_enc']) - int(e['n_system']) for e in per_sub)
    n_ancilla_combined = n_ancilla_sub + n_lcu_control + n_qft_workspace

    be_t = sum_enc_t + basis_change_t + prep_lcu_t
    refl_t = _reflection_t(n_ancilla_combined)
    walk_t = be_t + refl_t
    walk_clifford = sum_enc_clifford                       # basis-change/reflection are ~Clifford-light
    logical_qubits = n_system + n_ancilla_combined + 1     # +1 walk reflection qubit

    return {
        'Walk_T_Count': walk_t,
        'Walk_Clifford_Count': walk_clifford,
        'Logical_Qubits': logical_qubits,
        'walk_composition': 'combined_lcu',
        'composition_components': {
            'sum_select_prepare_T': sum_enc_t,
            'momentum_qft_T': int(momentum_qft_t),
            'wt_species_basis_change_T': wt_basis_change_t(L, dim, n_b),
            'lcu_prepare_T': prep_lcu_t,
            'reflection_T': refl_t,
            'n_lcu_control': n_lcu_control,
            'n_qft_workspace': n_qft_workspace,
            'n_system': n_system,
            'n_ancilla_sub_max': n_ancilla_sub,
        },
        # Honest labels for downstream tables (N6 claim discipline).
        'composition_label': (
            'EXPERIMENTAL cost roll-up — NOT a validated block encoding. H_WT is still '
            'mis-represented in pos_dyn (π and Π both diagonal), no combined circuit is '
            'built or projected-block-checked, and controlled-branch overhead is omitted. '
            'Amplitude numbers are not publication-grade; the valid construction is future '
            'work (codex amplitude_combined_walk_audit_2026-08-20). Fock/PauliLCU is the anchor.'
        ),
    }
