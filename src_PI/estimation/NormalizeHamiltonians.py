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


# MACHINE-NOISE prune floor (RELATIVE to the largest coefficient). RATIONALE
# (2026-08-21, codex quantum_round2 audit): the OLD floor was Λ-coupled
# (`|c_raw| < 2e-8·Δ = 5e-8·Λ`), designed for amplitude-basis float-cancellation
# terms (which cluster ~1.45e-8 normalized). But the FOCK Hamiltonian (the paper
# anchor) has NO such noise cluster — its raw coefficients form a clean continuum
# from ~6e-3 MeV up to ~850 MeV with nothing below 1e-4 MeV. A Λ-coupled floor
# therefore grows with the system and, at large Λ (L=10, high n_b), cuts REAL
# small-coefficient physics — an UNCONTROLLED Hamiltonian truncation (e.g. L=10
# n_b=2 discarded 3465 MeV of one-norm). Fix: prune only genuine floating-point
# zeros — terms below `_PRUNE_REL_FLOOR × max|c|` (machine-eps relative to the
# largest term). For the clean Fock construction this removes NOTHING, so the
# resource estimate is of the EXACT target Hamiltonian at every L/n_b. (Amplitude,
# now retired/experimental, would need its own noise handling if ever revived.)
_PRUNE_REL_FLOOR = 1e-12


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

    # Machine-noise prune floor RELATIVE to the largest raw coefficient (removes only
    # genuine floating-point zeros; the clean Fock construction has none, so nothing is
    # removed and the estimate is of the exact target Hamiltonian). Not Λ-coupled.
    max_abs = 0.0
    for _name, H in bundle.sub_hamiltonians:
        for term, coeff in H.terms.items():
            if term != () and abs(coeff) > max_abs:
                max_abs = abs(coeff)
    raw_thresh = _PRUNE_REL_FLOOR * max_abs

    # Pass 2: normalize each sub-Hamiltonian with the shared Δ. ACCUMULATE the discarded
    # coefficient one-norm (audit issue 2): pruning is a systematic Hamiltonian
    # perturbation bounded by the SUM of |removed coefficients| (RAW MeV energy units).
    # With the machine-noise floor this is ~0 for the Fock anchor at every L/n_b.
    normalized = []
    discarded_one_norm = 0.0            # Σ|c_raw| over pruned terms, in MeV
    discarded_count = 0
    for name, H in bundle.sub_hamiltonians:
        new_terms = {}
        for term, coeff in H.terms.items():
            if term == ():
                continue
            if abs(coeff) >= raw_thresh:
                new_terms[term] = coeff / delta
            else:
                discarded_one_norm += abs(coeff)     # raw energy removed
                discarded_count += 1
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
        # pruning provenance (audit issue 2): the removed Pauli one-norm as an energy
        # error bound, its count, and the raw threshold used.
        'pruned_one_norm_MeV': discarded_one_norm,
        'pruned_term_count': discarded_count,
        'prune_threshold_raw': raw_thresh,
    }
