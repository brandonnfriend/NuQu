from openfermion import QubitOperator

from src_PI.hamiltonians.core.HamiltonianBundle import HamiltonianBundle


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
# amplitude-basis H_pos has float-cancellation terms whose normalized magnitudes
# wall at ~1.45e-8 (verified by structural diff across A at fixed n_b: the
# "extra" terms in higher-Λ runs all land in [1.10e-8, 1.45e-8] regardless of
# n_b). Smallest real-physics terms sit above 1e-7. Pruning at 2e-8 in
# normalized units cleanly separates them. The equivalent raw cutoff is
#     |c_raw| < 2e-8 * Δ = 5e-8 * Λ
# i.e. it scales with Λ, so the same *physical* noise content is removed
# regardless of which A is being normalized → structurally A-invariant H_norm
# at fixed n_b, which is what makes the pyLIQTR structural cache hit reliably.
_NOISE_FLOOR_NORM = 2e-8


def normalize_for_qpe(bundle, safety_factor=2.5):
    """
    Performs full normalization for Qubitized Phase Estimation on every
    sub-Hamiltonian in the bundle.

    1. Shifts: Removes Identity terms (tracked as classical offsets).
    2. Scales: Divides by Delta (safety_factor * total Lambda) to fit
       eigenvalues in [0, 0.5].
    3. Prunes float-cancellation noise (Λ-coupled threshold).

    The same Δ is applied to every sub-Hamiltonian so their walks share a
    common spectral scale at QPE time. Per-sub-Hamiltonian Λ contributions
    are tracked for diagnostics.

    Returns a dict:
        'sub_hamiltonians': list of (name, normalized H) tuples in the
            same order as the input bundle.
        'sub_lambdas': list of (name, λ) tuples (per sub-Hamiltonian
            unnormalized Λ).
        'sub_identity_shifts': list of (name, shift) tuples.
        'delta': global Δ used for normalization.
        'identity_shift': total identity shift across all sub-Hamiltonians.
        'physical_lambda': sum of per-sub-Hamiltonian Λs.
        'walk_mode': passed through from the bundle.
    """
    if not isinstance(bundle, HamiltonianBundle):
        raise TypeError(
            f"normalize_for_qpe now expects a HamiltonianBundle, got {type(bundle).__name__}"
        )

    # Pass 1: collect Lambdas and identity shifts across the whole bundle.
    sub_lambdas = []
    sub_identity_shifts = []
    for name, H in bundle.sub_hamiltonians:
        id_coeff, lcu_lambda = get_hamiltonian_stats(H)
        sub_lambdas.append((name, lcu_lambda))
        sub_identity_shifts.append((name, id_coeff))

    total_physical_lambda = sum(lam for _, lam in sub_lambdas)
    total_identity_shift = sum(shift for _, shift in sub_identity_shifts)

    # Δ for the whole bundle. Eigenvalues of sum of walks lie in
    # ±total_physical_lambda; we pad by safety_factor.
    delta = safety_factor * total_physical_lambda
    if delta == 0:
        # Empty Hamiltonian — return zeros without dividing.
        normalized = [(name, QubitOperator()) for name, _ in bundle.sub_hamiltonians]
        return {
            'sub_hamiltonians': normalized,
            'sub_lambdas': sub_lambdas,
            'sub_identity_shifts': sub_identity_shifts,
            'delta': 0.0,
            'identity_shift': total_identity_shift,
            'physical_lambda': 0.0,
            'walk_mode': bundle.walk_mode,
        }

    raw_thresh = _NOISE_FLOOR_NORM * delta

    # Pass 2: normalize each sub-Hamiltonian with the shared Δ.
    normalized = []
    for name, H in bundle.sub_hamiltonians:
        new_terms = {}
        for term, coeff in H.terms.items():
            if term == ():
                continue
            if abs(coeff) >= raw_thresh:
                new_terms[term] = coeff / delta
        H_norm = QubitOperator()
        H_norm.terms = new_terms
        normalized.append((name, H_norm))

    return {
        'sub_hamiltonians': normalized,
        'sub_lambdas': sub_lambdas,
        'sub_identity_shifts': sub_identity_shifts,
        'delta': delta,
        'identity_shift': total_identity_shift,
        'physical_lambda': total_physical_lambda,
        'walk_mode': bundle.walk_mode,
    }
