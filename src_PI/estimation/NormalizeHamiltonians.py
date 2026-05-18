from openfermion import QubitOperator


def get_hamiltonian_stats(H):
    """
    Extracts the Identity coefficient and the LCU lambda
    (sum of absolute values of non-identity coefficients).
    """
    id_coeff = 0.0
    lcu_lambda = 0.0
    for term, coeff in H.terms.items():
        if term == ():
            id_coeff += coeff
        else:
            lcu_lambda += abs(coeff)
    return id_coeff, lcu_lambda


# Λ-coupled noise floor for the normalized Hamiltonian. Empirically, the raw
# H_pos has float-cancellation terms whose normalized magnitudes wall at
# ~1.45e-8 (verified by structural diff across A at fixed n_b: the "extra"
# terms in higher-Λ runs all land in [1.10e-8, 1.45e-8] regardless of n_b).
# Smallest real-physics terms sit above 1e-7. Pruning at 2e-8 in normalized
# units cleanly separates them. The equivalent raw cutoff is
#     |c_raw| < 2e-8 * Δ = 5e-8 * Λ
# i.e. it scales with Λ, so the same *physical* noise content is removed
# regardless of which A is being normalized → structurally A-invariant H_norm
# at fixed n_b, which is what makes the pyLIQTR structural cache hit reliably.
_NOISE_FLOOR_NORM = 2e-8


def normalize_for_qpe(H_pos, H_mom, safety_factor=2.5):
    """
    Performs full normalization for Qubitized Phase Estimation.
    1. Shifts: Removes Identity terms (tracked as classical offsets).
    2. Scales: Divides by Delta (safety_factor * Lambda) to fit eigenvalues in [0, 0.5].
    3. Prunes float-cancellation noise (Λ-coupled threshold).

    pyLIQTR's structural cost is A-invariant at fixed n_b (verified by running
    on the unnormalized H: identical T/Clifford/LogQ across A). The ~0.15%
    A-dependence seen in the un-pruned normalized run is not physics — it is
    openfermion's hardcoded 1e-12 zero_tolerance interacting with the per-A Δ
    rescaling, which flips a few hundred near-threshold terms in/out of the
    LCU. The Λ-coupled prune below eliminates that artifact.
    """
    id_pos, lam_pos = get_hamiltonian_stats(H_pos)
    id_mom, lam_mom = get_hamiltonian_stats(H_mom)

    total_physical_lambda = lam_pos + lam_mom
    total_identity_shift = id_pos + id_mom

    # Define Delta (Spectral Range). Delta ~ safety_factor * Lambda
    delta = safety_factor * total_physical_lambda

    # Λ-coupled raw threshold: prune in normalized space at _NOISE_FLOOR_NORM,
    # equivalent to |c_raw| < _NOISE_FLOOR_NORM * Δ. Single dict pass avoids
    # the per-term QubitOperator(...) parse + __iadd__ merge of the old
    # `H_norm += QubitOperator(t, c/delta)` pattern.
    raw_thresh = _NOISE_FLOOR_NORM * delta

    pos_terms = {}
    for term, coeff in H_pos.terms.items():
        if term == ():
            continue
        if abs(coeff) >= raw_thresh:
            pos_terms[term] = coeff / delta

    mom_terms = {}
    for term, coeff in H_mom.terms.items():
        if term == ():
            continue
        if abs(coeff) >= raw_thresh:
            mom_terms[term] = coeff / delta

    H_pos_norm = QubitOperator()
    H_pos_norm.terms = pos_terms
    H_mom_norm = QubitOperator()
    H_mom_norm.terms = mom_terms

    return {
        'H_pos_norm': H_pos_norm,
        'H_mom_norm': H_mom_norm,
        'delta': delta,
        'identity_shift': total_identity_shift,
        'physical_lambda': total_physical_lambda,
    }
