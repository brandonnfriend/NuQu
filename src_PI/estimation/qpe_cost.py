"""
Standalone total-QPE-cost computation (Phase E of the block-encoder refactor).

The total T-cost of a qubitized QPE run is the per-walk-step T-count times
the number of walk-operator queries:

    QPE_Total_T_Count = Total_T_Count · N_walk
    N_walk            = √2 · π · Λ / ΔE          (Eq. 9; Λ = Physical_Lambda)

This used to be computed inside `plot_sweep_data.py` at plot time. Per the
pseudocode (`human_knowledge/pseudocode/Block Encoders.md` step 7) the
computation now lives here as a standalone, file-in/file-out function so:
  * `run_nucleon_sweep.py` calls it once after each sweep and the totals are
    saved into the JSON;
  * `plot_sweep_data.py` reads the precomputed totals instead of recomputing;
  * it can be re-applied to *legacy* sweep files that predate the field.

`ΔE` is the QPE energy-precision target in MeV. The Watson reference report
uses 1 MeV, which is the default here.

**This formula is encoder-agnostic.** It comes from qubitization QPE
(Babbush et al. 2018, PRX 8 041015, arXiv:1805.03662 — reference [11] of the
NuQu final report, Eq. 9): the walk operator is `W(H) = e^{i·arccos(H/λ)}`,
so its eigenphases are `±arccos(E_k/λ)` and QPE resolves `E_k` to precision
ΔE with `O(λ/ΔE)` walk queries. Here `λ` is the **block-encoding
subnormalization** — the factor s.t. `⟨0|U|0⟩ = H/λ` — NOT specifically a
Pauli 1-norm. Babbush's own phrasing is "λ is a parameter *closely related
to* the induced 1-norm." It equals the Pauli 1-norm only for the PauliLCU
encoder; for the sparse-oracle encoder λ is the BCK subnormalization
(`Σ_l |c_l|·α_l`, what `sparse_oracle.compute_native_lambda` returns). So
this function works for every encoder: it reads each sweep's own
`Physical_Lambda` (= that encoder's λ) and plugs it into the same formula
with the same √2·π constant (a QPE-protocol property, encoder-independent).
A sparse vs PauliLCU comparison via `QPE_Total_T_Count` is therefore
apples-to-apples even though the two λ values differ for the same H.

CLI:
    python -m src_PI.estimation.qpe_cost <sweep.json> [--delta-e 1.0]
"""

import argparse
import json
import math


DEFAULT_DELTA_E_MEV = 1.0

# The qubitized-walk query prefactor `C` in `N_walk = C·λ/ΔE`.
#
#   √2·π ≈ 4.443  — Babbush 2018 Eq. 26 UPPER bound. The extra √2 over the
#                   Heisenberg constant is an EQUAL-ERROR-BUDGET split: Babbush
#                   sets phase-estimation variance = gate-synthesis variance and
#                   gives each ΔE/√2 (his Eqs. 23–25). This is the historical
#                   default; it is what the committed r3 raw shards used, so it
#                   stays the module default for shard reproducibility.
#   π ≈ 3.142     — SOTA/Heisenberg-optimal single-window constant. Our cost
#                   model budgets synthesis error SEPARATELY (`eps_be`, via
#                   `circuit_precision`), so keeping Babbush's √2 double-counts
#                   the synthesis budget. Dropping it (synthesis ≪ PE error)
#                   gives π — a 1.41× reduction that keeps the upper-bound
#                   confidence factor. This is the ADOPTED HEADLINE constant
#                   (applied at the reporting layer: `make_headline_resource_figure`).
#   π/2 ≈ 1.571   — 1σ information floor (drops the confidence factor too); not used.
#
# See `claude/research/total_costs/00_literature_review.md` for the full provenance.
WALK_QUERY_CONSTANT_BABBUSH_UB = math.sqrt(2.0) * math.pi   # 4.443 — Eq. 26 upper bound
WALK_QUERY_CONSTANT_HEISENBERG = math.pi                    # 3.142 — adopted headline
WALK_QUERY_CONSTANT = WALK_QUERY_CONSTANT_BABBUSH_UB        # module default (raw shards)


def walk_queries(physical_lambda, delta_E=DEFAULT_DELTA_E_MEV,
                 constant=WALK_QUERY_CONSTANT):
    """N_walk = C · Λ / ΔE — the qubitized-walk query count for QPE.

    `C` defaults to `√2·π` (`WALK_QUERY_CONSTANT_BABBUSH_UB`, Babbush 2018
    Eq. 26 upper bound — the constant the committed shards were generated with).
    Pass `constant=WALK_QUERY_CONSTANT_HEISENBERG` (= π) for the adopted-headline
    value: a 1.41× tightening justified because our model budgets synthesis error
    separately, so Babbush's equal-split √2 would double-count it.
    """
    return (constant * physical_lambda) / delta_E


def total_qpe_t_count(total_t_count, physical_lambda, delta_E=DEFAULT_DELTA_E_MEV,
                      constant=WALK_QUERY_CONSTANT):
    """Total QPE T-cost = per-step T-count · N_walk."""
    return total_t_count * walk_queries(physical_lambda, delta_E, constant)


# GSEE overlap/success crossover: amplitude-amplified binary search (∝1/√p0) beats naive
# min-of-samples (∝1/p0) only for p0 ≲ this; above it, sampling's smaller prefactor wins
# (Berry et al. 2024, arXiv:2409.11748, p.4). A warm start's value is keeping p0 large — few
# repetitions AND on the cheap sampling side — not the 1/√p0 amplification.
OVERLAP_CROSSOVER_P0 = 0.003


def overlap_repetition_factor(p0, confidence=0.99, branch="sampling",
                              crossover=OVERLAP_CROSSOVER_P0):
    """GSEE repetition factor `R` = how many (state-prep + one QPE window) shots are needed, as a
    multiplier on the per-window coherent-query T-count. `p0 = |⟨g|ψ_init⟩|²` is the warm-start
    success probability (from `frame_qpe.warmstart_fidelity`).

    Reports TWO explicitly-named operational metrics (warmstart audit §1 — the paper must pick and
    name one, not mix them):
      * **`R` (fixed-confidence, EXACT Bernoulli)** — the smallest `R` with `(1−p0)^R ≤ δ`
        (`δ=1−confidence`), i.e. `R = ⌈ln δ / ln(1−p0)⌉`. This is the exact "≥1 ground projection
        in R shots" count. (The old `⌈ln(1/δ)/p0⌉` was the `1−p ≤ e^{−p}` APPROXIMATION — only tight
        for small p0; it over-counted at high p0.)
      * **`expected_shots = 1/p0`** — the mean shots to first success. A DIFFERENT metric; the
        warm/cold ratio here is the continuous `p0_warm/p0_cold`.

    `branch='binary'` is an EXPERIMENTAL sensitivity cartoon (`⌈ln(1/δ)/√(crossover·p0)⌉`, calibrated
    to touch sampling at `p0=crossover`) — it is NOT the transcribed Berry amplitude-estimation
    formula and is NEVER auto-selected. For the high overlaps we measure, only exact sampling is
    relevant; use `branch='sampling'` (default). Implement the real Berry/Lin–Tong algorithm + its
    oracle/register costs before quoting a filtering speedup.

    Returns `{R, branch, R_sampling, expected_shots, R_binary_experimental, ae_register_qubits,
    p0, confidence}`.
    """
    if not (0.0 < p0 <= 1.0):
        raise ValueError(f"p0 must be in (0, 1], got {p0}")
    delta = 1.0 - confidence
    if p0 >= 1.0:
        r_sampling = 1                                   # certain success in one shot
    else:
        r_sampling = max(1, math.ceil(math.log(delta) / math.log1p(-p0)))   # EXACT Bernoulli
    expected_shots = 1.0 / p0
    r_binary = math.ceil(math.log(1.0 / delta) / math.sqrt(crossover * p0))  # experimental cartoon
    ae_qubits = max(1, math.ceil(math.log2(1.0 / math.sqrt(p0))))
    if branch == "sampling":
        chosen, R, ae = "sampling", r_sampling, 0
    elif branch == "binary":                             # opt-in, experimental only
        chosen, R, ae = "binary", r_binary, ae_qubits
    else:
        raise ValueError(f"branch must be 'sampling' (default) or 'binary' (experimental), got {branch}")
    return {"R": int(R), "branch": chosen, "R_sampling": int(r_sampling),
            "expected_shots": float(expected_shots), "R_binary_experimental": int(r_binary),
            "ae_register_qubits": int(ae), "p0": p0, "confidence": confidence}


def qpe_phase_register_qubits(physical_lambda, eps_qpe=DEFAULT_DELTA_E_MEV,
                              constant=WALK_QUERY_CONSTANT):
    """`m` = number of QPE PHASE-REGISTER ancilla qubits (Babbush 2018).

        m = ⌈log₂( C·Λ / (2·ε_qpe) )⌉ = ⌈log₂( N_walk / 2 )⌉        (log base 2)

    Babbush et al. 2018 (PRX 8 041015, arXiv:1805.03662): standard qubitized
    QPE uses `m` phase-register ancilla whose largest controlled walk power is
    `W^(2^(m-1))`, so the total number of walk applications is `N_walk ≈ 2^m`,
    i.e. `2^m ≈ √2·π·Λ/ε_qpe` (their Eq. 26 upper bound). Since our
    `walk_queries` already returns that `N_walk`, `m = ⌈log₂(N_walk/2)⌉` ties
    the ancilla count to the SAME N_walk we report — there is no independent
    constant to get out of sync. This is the **QPE phase register**, which the
    walk/block-encoding logical-qubit count EXCLUDES; add it to the total.

    `ε_qpe` is the phase-estimation resolution share of the total ΔE budget (in
    our optimizer, `eps_qpe ≈ 0.96 MeV`, not the full 1 MeV); passing it keeps
    `m` consistent with the reported `N_walk`. `Λ` is the block-encoding
    subnormalization (Pauli 1-norm for PauliLCU). `constant` follows the same
    N_walk prefactor switch as `walk_queries`.
    """
    n_walk = walk_queries(physical_lambda, eps_qpe, constant)
    return qpe_phase_register_qubits_from_nwalk(n_walk)


def qpe_phase_register_qubits_from_nwalk(n_walk):
    """`m = ⌈log₂(N_walk/2)⌉` from an already-computed `N_walk`.

    Use this when a record already stores `QPE_Walk_Queries` so `m` is tied to
    the exact walk count reported (independent of which prefactor produced it).
    """
    if n_walk <= 0:
        raise ValueError("n_walk must be positive")
    return max(1, math.ceil(math.log2(n_walk / 2.0)))


def total_logical_qubits(walk_register_qubits, qpe_phase_qubits,
                         state_prep_ancilla=0):
    """Peak logical width of the full GSEE run (one walk register + coexisting ancilla).

        total = walk_register + max(m_QPE, a_stateprep)

    The `max` (not the sum) is the ancilla-REUSE argument: initial-state
    preparation runs to completion and its workspace is reset BEFORE the QPE
    phase register is ever populated, so the two never coexist. Whichever needs
    more qubits sets the peak; the smaller one is a subset of the larger's
    lifetime. (This is conservative: state prep can additionally borrow the
    idle block-encoding ancilla already inside the walk register, so the true
    peak may be lower.) The walk-register count itself already includes the
    system register + block-encoding ancilla.

    `state_prep_ancilla=0` (default) recovers the walk-register + QPE-phase-only
    total until the state-prep cost is modeled (see the `total_costs` plan).
    Magic-state factories and routing are PHYSICAL-layer overheads and are NOT
    part of this logical width — they belong to the runtime module.
    """
    return int(walk_register_qubits) + max(int(qpe_phase_qubits),
                                            int(state_prep_ancilla))


def recommended_n_b_from_occupation(mean_n_per_mode, margin_sigmas=5.0):
    """Fock register size `n_b` (bits/mode) from a measured mean per-mode boson
    occupation `⟨n⟩` — an explicitly HEURISTIC Poisson-margin ESTIMATE (per the
    codex publication-readiness audit), NOT a certified cutoff.

    The cutoff keeps a Poisson-ish safety margin above the mean —
    `N_cut = ⟨n⟩ + margin_sigmas·√(⟨n⟩+1)` — and `n_b = ⌈log₂(N_cut + 1)⌉`. For the
    verified near-vacuum ground state (`⟨n⟩≈0.045`) this lands at a small cutoff.

    SUPERSEDED for cutoff DETERMINATION by the empirical convergence study
    (`misc/run_nb_convergence.py`: energy / weighted-tail convergence) — use that
    to actually set n_b, and treat this Poisson-margin formula only as a quick
    frame->QPE bridge estimate.

    The single source of truth for the frame->QPE bridge's boson-cutoff reduction
    (task 34, I1 seam a): a frame that lowers ⟨n⟩ needs a smaller Fock cutoff.
    `frame_qpe.qpe_payoff` calls this so the bridge and the sweep agree on the map.
    """
    cutoff = mean_n_per_mode + margin_sigmas * math.sqrt(mean_n_per_mode + 1.0)
    return max(1, math.ceil(math.log2(cutoff + 1.0)))


def compute_total_qpe_cost(filepath, delta_E=DEFAULT_DELTA_E_MEV, write=True):
    """Read a sweep JSON, compute the total QPE cost per result entry, write back.

    For each entry in `data['results']` adds:
      * 'QPE_Walk_Queries'   = √2·π·Λ / ΔE
      * 'QPE_Total_T_Count'  = Total_T_Count · QPE_Walk_Queries

    Records the `ΔE` used under `data['metadata']['delta_E_MeV']`. Idempotent:
    re-running recomputes from `Total_T_Count` and `Physical_Lambda`, which the
    function never mutates, so repeated calls converge to the same values.

    Args:
        filepath: path to a sweep JSON (the `save_sweep_data` schema).
        delta_E: QPE energy precision in MeV (default 1.0).
        write: if True, write the updated JSON back to `filepath` (indent=4,
            matching `DataIO.save_sweep_data`). If False, just return the dict.

    Returns:
        The updated data dict.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    for r in results:
        t_step = r.get('Total_T_Count')
        lam = r.get('Physical_Lambda')
        if t_step is None or lam is None:
            # Entry predates the Total_T_Count / Physical_Lambda fields; skip
            # rather than guess.
            continue
        nq = walk_queries(lam, delta_E)
        r['QPE_Walk_Queries'] = nq
        r['QPE_Total_T_Count'] = t_step * nq

    data.setdefault('metadata', {})['delta_E_MeV'] = delta_E

    if write:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Compute total QPE T-cost (N_walk · T_per_step) for a sweep JSON, '
                    'writing QPE_Total_T_Count back into the file.'
    )
    parser.add_argument('filepath', help='Path to a sweep JSON file.')
    parser.add_argument('--delta-e', type=float, default=DEFAULT_DELTA_E_MEV,
                        help=f'QPE energy precision ΔE in MeV (default {DEFAULT_DELTA_E_MEV}).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and print but do not write the file back.')
    args = parser.parse_args()

    data = compute_total_qpe_cost(
        args.filepath, delta_E=args.delta_e, write=not args.dry_run
    )
    results = data.get('results', [])
    written = 'NOT written (dry run)' if args.dry_run else 'written back'
    print(f"QPE total cost for {len(results)} entries ({written}, ΔE={args.delta_e} MeV):")
    for r in results[:8]:
        if 'QPE_Total_T_Count' in r:
            print(f"  A={r.get('A')!s:>4}  n_b={r.get('n_b')!s:>3}  "
                  f"T_step={r['Total_T_Count']:.3e}  Λ={r['Physical_Lambda']:.3e}  "
                  f"QPE_total={r['QPE_Total_T_Count']:.3e}")
    if len(results) > 8:
        print(f"  ... ({len(results) - 8} more)")


if __name__ == '__main__':
    main()
