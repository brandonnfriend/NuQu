"""
STEP 7 — quantum-side payoff of the classical frame (task 33; feeds task 14).

The winning classical frame's unitary `U` (squeeze ∘ displace ∘ orbital-rotation, all
from `frame.py`) IS the QPE **warm start**: prepare `U|ref⟩` (ref = boson vacuum ⊗
Hartree-Fock determinant) instead of a cold `|ref⟩`. Because `U†HU` has a COMPACT ground
state (the whole point of task 33), `U|ref⟩` has high overlap with the true ground state.
Two quantum payoffs, both read off the same frame:

  1. **Warm-start overlap p₀** — QPE succeeds with prob p₀ per run ⇒ ~1/p₀ repetitions.
     p₀ = |⟨ref|U†|g⟩|² = |⟨ref|framed_g⟩|² = the weight of the framed ground state on its
     dominant determinant = `max_i |c_i|²` of the framed TrimCI core. The frame's gain is
     `p₀_framed / p₀_bare` — directly measured at realistic size from the cores.
  2. **Boson cutoff Λ (= N_f) reduction** — the frame lowers ⟨n⟩ (mean boson number), so a
     smaller Fock cutoff suffices ⇒ fewer boson qubits (`n_b = ⌈log₂ N_f⌉` per mode) and,
     in the Tong-et-al. regime, poly-Λ → polylog-Λ.

Against these gains, the frame's **state-prep circuit** is a ONE-TIME additive cost:
per-mode squeeze = N parallel single-mode squeezers (O(1) depth); displacement = one
(conditional) displacement/mode; orbital rotation = an n_ferm Givens network (O(n²) gates,
O(n) depth). Compared to the QPE walk — `N_walk = √2·π·Λ/ΔE` steps, each `Total_T_Count`
T-gates, times ~1/p₀ repetitions — the state prep is negligible. So the net is: pay a tiny
one-time state-prep, buy a 1/p₀-fewer-repetitions × fewer-qubits QPE.

Costs use the standard Ross–Selinger single-qubit-rotation synthesis
`T_rot(ε) ≈ 3.07·log₂(1/ε) + 9.7` T-gates; all assumptions are explicit arguments.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
#  Warm-start overlap p₀ (measured from a TrimCI core)
# ---------------------------------------------------------------------------

def warmstart_overlap(coeffs):
    """QPE warm-start overlap `p₀ = max_i |c_i|² / Σ_i |c_i|²` from a ground-state core
    (the weight on the dominant determinant = the natural single-determinant reference
    the frame maps the state onto). Higher = better warm start = ~1/p₀ fewer QPE runs."""
    p = [abs(c) ** 2 for c in coeffs]
    s = sum(p)
    return (max(p) / s) if s > 0 else 0.0


def mean_boson_number(coeffs, bos_arr):
    """⟨n⟩ = Σ_i |c_i|² Σ_m n_{i,m} — mean total boson occupation of a core (drives the
    Fock-cutoff N_f the block encoder needs)."""
    import numpy as np
    c = np.asarray(coeffs, dtype=complex)
    w = np.abs(c) ** 2
    w = w / w.sum() if w.sum() > 0 else w
    return float(np.sum(w * np.asarray(bos_arr).sum(axis=1)))


# ---------------------------------------------------------------------------
#  State-prep circuit cost of the frame U (one-time, additive)
# ---------------------------------------------------------------------------

def t_rot(eps_synth=1e-10):
    """Ross–Selinger T-count for one arbitrary single-qubit rotation to precision ε."""
    return 3.067 * math.log2(1.0 / eps_synth) + 9.678


def stateprep_tcount(n_bos, n_ferm, n_b, *, squeeze=True, displace=True, orbital=False,
                     eps_synth=1e-10):
    """One-time T-count to prepare the warm start `U|ref⟩`, summed over the enabled
    frame layers (assumptions are explicit):
      * squeeze   — N_bos single-mode squeezers, ~n_b² rotations each (a quadrature
                    scaling on an n_b-qubit register; per-mode ⇒ O(1) depth);
      * displace  — N_bos (conditional) displacements, ~n_b rotations each (a quadrature
                    shift; O(1) depth);
      * orbital   — an n_ferm Givens network, n_ferm(n_ferm−1)/2 rotations (O(n_ferm) depth).
    Returns `{T_squeeze, T_displace, T_orbital, T_total, depth_layers}`. Rotation count ×
    `t_rot(ε)`. Order-of-magnitude by construction — the point is that `T_total ≪` the walk."""
    tr = t_rot(eps_synth)
    t_sq = (n_bos * n_b * n_b * tr) if squeeze else 0.0
    t_dp = (n_bos * n_b * tr) if displace else 0.0
    t_or = (n_ferm * (n_ferm - 1) / 2 * tr) if orbital else 0.0
    depth = (1 if (squeeze or displace) else 0) + (n_ferm if orbital else 0)
    return {"T_squeeze": t_sq, "T_displace": t_dp, "T_orbital": t_or,
            "T_total": t_sq + t_dp + t_or, "depth_layers": depth}


# ---------------------------------------------------------------------------
#  The payoff: fold p₀ + cutoff into the QPE cost (feeds task 14)
# ---------------------------------------------------------------------------

def qpe_payoff(*, p0_bare, p0_frame, mean_n_bare, mean_n_frame,
               t_step, physical_lambda, n_bos, n_ferm, n_b,
               delta_E=1.0, frame_layers=("squeeze",), eps_synth=1e-10):
    """Combine the measured frame quantities into the QPE-cost impact (task 14 row).

    Cold QPE total ≈ `T_step · N_walk · 1/p₀` (walk queries `N_walk=√2·π·Λ/ΔE`, times
    ~1/p₀ repetitions to hit the ground state). The frame changes:
      * repetitions: `1/p₀_bare → 1/p₀_frame`  ⇒  factor `p₀_frame/p₀_bare` fewer runs;
      * boson qubits: `n_b_bare → ⌈log₂(⟨n⟩_frame + margin)⌉` per mode (cutoff shrinks with ⟨n⟩);
    at a ONE-TIME state-prep cost `T_prep` (this module). Returns the factors + the
    net QPE-T ratio and the (tiny) prep overhead. `physical_lambda`, `t_step` are that
    system's block-encoding Λ and per-walk-step T-count (from a sweep JSON)."""
    from src_PI.estimation.qpe_cost import walk_queries
    n_walk = walk_queries(physical_lambda, delta_E)
    qpe_cold = t_step * n_walk / max(p0_bare, 1e-15)
    qpe_warm = t_step * n_walk / max(p0_frame, 1e-15)
    prep = stateprep_tcount(n_bos, n_ferm, n_b,
                            squeeze=("squeeze" in frame_layers),
                            displace=("displace" in frame_layers),
                            orbital=("orbital" in frame_layers), eps_synth=eps_synth)
    # boson-qubit saving: cutoff ~ ⟨n⟩ with a Poisson-ish margin (⟨n⟩ + 5√(⟨n⟩+1))
    def n_b_needed(mean_n):
        cutoff = mean_n + 5.0 * math.sqrt(mean_n + 1.0)
        return max(1, math.ceil(math.log2(cutoff + 1)))
    return {
        "N_walk": n_walk,
        "p0_gain": p0_frame / max(p0_bare, 1e-15),
        "repetition_factor": p0_bare / max(p0_frame, 1e-15),   # <1 ⇒ fewer runs
        "qpe_T_cold": qpe_cold, "qpe_T_warm": qpe_warm,
        "qpe_T_ratio": qpe_warm / max(qpe_cold, 1e-300),        # incl. the 1/p₀ change
        "T_prep": prep["T_total"],
        "prep_vs_walk": prep["T_total"] / max(qpe_warm, 1e-300),
        "n_b_bare": n_b, "n_b_frame": n_b_needed(mean_n_frame),
        "boson_qubit_saving_per_mode": n_b - n_b_needed(mean_n_frame),
    }
