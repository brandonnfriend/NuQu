import math
import numpy as np


def get_physical_parameters():
    """
    Returns the physical constants for the Dynamical Pion EFT in MeV.
    Derived from a_L = 2.2 fm, using standard conversions.
    """
    # 1. Base Constants
    hc = 197.327            # Conversion factor: MeV * fm
    a_L_fm = 2.2            # fm
    a_L = a_L_fm / hc       # MeV^-1

    # 2. Table I Constants
    M_N = 938.0             # Nucleon mass (MeV)
    m_pi = 135.0            # Pion mass (MeV)
    f_pi = 93.0             # Pion decay constant (MeV)
    g_A = 1.26              # Axial coupling

    # 3. Derived Hamiltonian Coefficients
    # Nucleon kinetic hopping parameter: 1 / (2 * M * a_L^2)
    h_hop = 1.0 / (2.0 * M_N * (a_L**2))

    # Contact terms from Table IV (calculated at a_L^-1 = 100 MeV)
    C = -51.9425            # MeV
    CI = 1.7325             # MeV

    return {
        'h': h_hop,
        'C': C,
        'CI': CI,
        'a_L': a_L,
        'm_pi': m_pi,
        'g_A': g_A,
        'f_pi': f_pi,
        'M_N': M_N,
        # HO basis frequency for the Fock encoding. Default = m_π so the
        # free-pion local term (a_L^d/2)·(Π² + m_π² π²) collapses exactly
        # to m_π·(â†â + ½). Override if you want a lattice-effective ω_0;
        # the collapse becomes approximate.
        'm_0': m_pi,
    }


def calculate_dynamic_cutoffs(L, dim, A_nucleons, params, epsilon_cut=0.1, E_bound=140.0):
    """
    Calculates the dynamic pion field cutoffs and required qubits (n_b)
    based on Lemma 5 of the paper.

    Used for the **amplitude-basis** path with the energy-bound ('energy_bound')
    cutoff method. The Nyquist-Shannon amplitude path uses calculate_ns_cutoffs()
    and the Fock-basis path uses estimate_boson_cutoff(), both below.
    """
    a_L = params['a_L']
    m_pi = params['m_pi']
    f_pi = params['f_pi']
    g_A = params['g_A']
    C = params['C']
    CI = params['CI']

    eta = A_nucleons
    L_vol = L ** dim  # Generalizes L^3 to handle our dimensional sweeps

    # Equation 77: A and B coefficients
    A_coeff = ((m_pi**2) * (a_L**3) / 2.0) - (1.0 / (2.0 * (f_pi**2) * a_L))
    B_coeff = ((a_L**3) / 2.0) - (a_L / (2.0 * (f_pi**2)))

    if A_coeff <= 0 or B_coeff <= 0:
        raise ValueError(f"Lattice spacing a_L={a_L} yields invalid (<=0) A or B coefficients.")

    # -----------------------------------------------------------------
    # Equation 75 & 76: Field Cutoffs
    # -----------------------------------------------------------------
    # Common prefactor for both pi_max and Pi_max: (\sqrt{3L^3 / \epsilon_cut} + 1)
    prefactor = np.sqrt(3.0 * L_vol / epsilon_cut) + 1.0

    # Common Energy + Contact term: E + 8 \eta |C| + 4 \eta |C_{I^2}|
    energy_contact_sum = E_bound + 8.0 * eta * abs(C) + 4.0 * eta * abs(CI)

    # Precompute frequently used fractions
    gA_fpi_aL_A = (3.0 * g_A) / (f_pi * a_L * A_coeff)
    gA_fpi_aL   = (3.0 * g_A) / (f_pi * a_L)
    mass_term   = (6.0 * g_A) / ((m_pi**2) * f_pi * (a_L**4))

    # pi_max calculation (Eq 75)
    sqrt_inner_pi = (energy_contact_sum / A_coeff) + \
                    (3.0 * eta * (gA_fpi_aL_A**2)) + \
                    ((9.0 * eta * (m_pi**2) * (a_L**3) / A_coeff) * (mass_term**2))

    pi_max = prefactor * (gA_fpi_aL_A + np.sqrt(sqrt_inner_pi))

    # Pi_max calculation (Eq 76)
    sqrt_inner_Pi = (energy_contact_sum / B_coeff) + \
                    ((3.0 * eta / (A_coeff * B_coeff)) * (gA_fpi_aL**2)) + \
                    ((9.0 * eta * (m_pi**2) * (a_L**3) / B_coeff) * (mass_term**2))

    Pi_max = prefactor * np.sqrt(sqrt_inner_Pi)

    # -----------------------------------------------------------------
    # Equation 78: Qubit Requirement (n_b)
    # -----------------------------------------------------------------
    # n_b = log_2( (2 a_L^3 / \pi) * Pi_max * pi_max + 1 )
    inner_term = (2.0 * (a_L**3) / np.pi) * Pi_max * pi_max + 1.0
    n_b_float = np.log2(inner_term)

    # "we choose the nearest cutoffs above these bounds to ensure n_b is an integer"
    n_b = np.ceil(n_b_float)
    n_b = int(n_b)  # Convert to integer for later use
    return n_b, pi_max, Pi_max


# -----------------------------------------------------------------------
# Per-site boson-number cutoff (shared by the Fock and NS amplitude paths)
# -----------------------------------------------------------------------
# TWO methods, selected by `boson_cutoff_method` on estimate_boson_cutoff:
#
#   'heuristic' (default) — the starter formula below. NOT a rigorous
#     derivation; intentionally conservative for small-A test runs and grows
#     slowly with A (which broadens P(n) via the H_AV/H_WT sources).
#
#   'tong' — the rigorous Tong-2022 bound. Tong et al. (arXiv:2110.06942)
#     proves that for bosonic Hamiltonians with bounded-degree polynomial
#     coupling — which includes our chiral EFT (H_AV degree 1, H_WT degree 2
#     in â, â†) — a Fock cutoff N_f = O(polylog(1/ε)) is rigorously
#     sufficient. `classical/trimci/tong_bound.py::cutoff_predictions`
#     instantiates that bound (SCS occupation + first/second-order spectral
#     tails) for our specific H + lattice geometry, and `_tong_boson_cutoff`
#     returns the certified choice max(n_b_eng, n_b_spec1) = 4-5, essentially
#     A-independent. The classical nb/N_f convergence study confirms n_b=4-5
#     are safe here. (The full Theorem-6 prefactor derivation remains the
#     paper-grade open homework in CLAUDE.md; `tong_bound` is the first-draft
#     rigorous replacement that this switch makes usable now.)
#
# The same per-site boson cutoff drives two encodings:
#   - Fock basis (Reading A): N_f = 2^n_q states, n_b = n_q qubits.
#   - NS amplitude basis (Reading B): boson cutoff N_b = 2^n_q, encoded in
#     a field-amplitude register of N_phi = 2*N_b grid points (one extra
#     qubit; see calculate_ns_cutoffs).
#
# Formula:
#     n_q = max(N_Q_MIN, ceil(N_Q_BASE + N_Q_PER_LOG_A · log2(1 + A)))
#
# Rationale for constants:
#     N_Q_MIN = 4    — Klco–Savage's "localized" single-site result
#     N_Q_BASE = 4   — same floor; n_q grows only when A pushes it up
#     N_Q_PER_LOG_A = 1   — empirically generous: doubles N_f every time A
#                           doubles. Tong's polylog scaling is much milder
#                           in principle, so this is a safety margin.
#
# Concrete values:
#     A = 1:     n_q = max(4, ceil(4 + 1·log2(2))) = max(4, 5) = 5
#     A = 10:    n_q = max(4, ceil(4 + 1·log2(11))) = max(4, ceil(7.46)) = 8
#     A = 100:   n_q = max(4, ceil(4 + 1·log2(101))) = max(4, ceil(10.66)) = 11
#
# These numbers are bigger than Tong's polylog would justify, but small
# enough that the test pipeline stays tractable. Switch to
# boson_cutoff_method='tong' for the rigorous (much smaller, A-flat) cutoff.
N_Q_MIN = 4
N_Q_BASE = 4
N_Q_PER_LOG_A = 1.0


def _tong_boson_cutoff(L, dim, A_nucleons, params):
    """Rigorous per-mode boson cutoff n_q from Tong et al. 2022, via the SCS +
    spectral-bound instantiation in ``classical/trimci/tong_bound.py``.

    Returns the *certified* choice ``max(n_b_eng, n_b_spec1)`` (tong_bound doc
    §2-3: the engineering safety cutoff OR-ed with the first-order
    Cauchy-Schwarz spectral bound — the "certified rigorous choice" the doc
    calls out). For this EFT that lands at **n_q = 4-5 across the whole physical
    A range** — near-A-independent, reflecting Tong's polylog(1/eps) scaling —
    versus the heuristic's ``log2(1+A)`` growth to ~11. The classical nb/N_f
    convergence study (``misc/run_nb_convergence.py``) confirms n_b = 4-5 are
    safe here, so this is now a measured bound, not just a bracket.

    The import is deferred (the classical solver package is heavier than this
    module needs at import time, and only the 'tong' path pulls it in).
    """
    from classical.trimci.tong_bound import cutoff_predictions
    pred = cutoff_predictions(L, dim, A_nucleons, params=params)
    return max(pred["n_b_eng"], pred["n_b_spec1"])


def estimate_boson_cutoff(L, dim, A_nucleons, params, epsilon_cut=0.1,
                          E_bound=140.0, boson_cutoff_method='heuristic'):
    """
    Returns (n_q, pi_max, Pi_max), the shared per-site boson-cutoff estimate.

    n_q is the number of qubits needed to index the per-site boson cutoff:
      - Fock-basis path uses it directly (N_f = 2^n_q states, n_b = n_q).
      - NS amplitude path treats the boson cutoff as N_b = 2^n_q and derives
        its own register size from it (see calculate_ns_cutoffs).

    ``boson_cutoff_method`` selects how n_q is chosen (a design axis, kept as a
    runtime switch so 'heuristic' vs 'tong' is a direct comparison rather than a
    replacement):
      - 'heuristic' (default): the starter formula
        ``n_q = max(N_Q_MIN, ceil(N_Q_BASE + N_Q_PER_LOG_A·log2(1+A)))`` —
        conservative, grows with A. See the module-level note for the caveat.
      - 'tong': the rigorous Tong-2022 bound (``_tong_boson_cutoff``),
        n_q = 4-5, essentially A-independent.

    pi_max and Pi_max are computed via the amplitude-basis (energy-bound)
    formula for *return-shape consistency* with calculate_dynamic_cutoffs()
    and for diagnostic comparison plots. They are NOT used in Fock-basis
    operator construction, and are NOT the NS-optimal windows.

    Args mirror calculate_dynamic_cutoffs() so the sweep dispatcher can
    call either with the same signature.
    """
    if boson_cutoff_method == 'tong':
        n_q = _tong_boson_cutoff(L, dim, A_nucleons, params)
    elif boson_cutoff_method == 'heuristic':
        # See module-level note for the rigor caveat on this heuristic.
        n_q = max(
            N_Q_MIN,
            int(math.ceil(N_Q_BASE + N_Q_PER_LOG_A * math.log2(max(1, A_nucleons) + 1)))
        )
    else:
        raise ValueError(
            f"boson_cutoff_method must be 'heuristic' or 'tong', "
            f"got {boson_cutoff_method!r}"
        )

    # Compute amplitude-basis pi_max/Pi_max for the metadata, not used in
    # Fock operator construction. Caller is free to ignore.
    try:
        _, pi_max, Pi_max = calculate_dynamic_cutoffs(
            L, dim, A_nucleons, params,
            epsilon_cut=epsilon_cut, E_bound=E_bound,
        )
    except ValueError:
        # If the amplitude formula fails (bad a_L regime), just report NaN.
        pi_max = float('nan')
        Pi_max = float('nan')

    return n_q, pi_max, Pi_max


def calculate_ns_cutoffs(L, dim, A_nucleons, params, epsilon_cut=0.1, E_bound=140.0,
                         boson_cutoff_method='heuristic'):
    """
    Returns (n_b, pi_max, Pi_max) for the **amplitude basis with the
    Nyquist-Shannon-optimal cutoff** (Reading B / Path B).

    Same field-amplitude register and operator constructions as the
    energy-bound path (calculate_dynamic_cutoffs); only the windows change.
    Instead of Watson Lemma 5's global worst-case energy bound, the field
    window F = pi_max and conjugate window K = Pi_max are set per-site from
    the boson cutoff N_b and the oscillator frequency omega_0 (Macridin 2022
    Eq. 87), giving n_b independent of L, A, E_total (at fixed N_b).

    Derivation (field-theory units, [pi,Pi] = i/a_L^d):
      - N_b = 2^n_q from estimate_boson_cutoff (the shared physics input).
      - N_phi = 2*N_b is the field register size needed to hold the first
        N_b oscillator states with O(1e-4) leakage (the "2" is the empirical
        Hermite-Gauss tail margin; see the bosonic-encodings research note).
      - n_b = ceil(log2(2*N_b)) = n_q + 1 (one qubit more than the Fock path).
      - Macridin Eq. 87 windows, ratio K/F = omega_0 (the matching condition):
            pi_max = sqrt(pi * N_phi / (2 * a_L^d * omega_0))
            Pi_max = sqrt(pi * N_phi * omega_0 / (2 * a_L^d))
        The a_L^d factors carry the field-theory normalization (Macridin's
        QM-convention formula has no a_L^d). N_phi uses the realized 2^n_b
        (>= 2*N_b after the ceil) so the windows match the power-of-2 FFT grid.

    omega_0 defaults to params['m_0'] (= m_pi). Note: when fed through
    get_Pp_Qp, the realized Pi_max equals this Pi_max up to a factor
    (2^n_b - 1)/2^n_b (one grid cell); the matching ratio K/F = omega_0 is
    enforced automatically by the existing conjugate-grid relation.
    """
    a_L = params['a_L']
    omega_0 = params['m_0']

    n_q, _, _ = estimate_boson_cutoff(
        L, dim, A_nucleons, params, epsilon_cut=epsilon_cut, E_bound=E_bound,
        boson_cutoff_method=boson_cutoff_method,
    )
    N_b = 2 ** n_q
    n_b = int(math.ceil(math.log2(2 * N_b)))  # = n_q + 1

    N_phi = 2 ** n_b  # realized field-register size (>= 2*N_b)
    aLd = a_L ** dim

    pi_max = math.sqrt(math.pi * N_phi / (2.0 * aLd * omega_0))
    Pi_max = math.sqrt(math.pi * N_phi * omega_0 / (2.0 * aLd))

    return n_b, pi_max, Pi_max


def T_cross_MeV(a_L_MeV, L, E_kin, M_N=938.0):
    """Calculates the crossing time T_cross in MeV^-1 for a given kinetic energy E_kin (in MeV) and lattice parameters."""
    return a_L_MeV * L * np.sqrt(M_N/ (2 * E_kin))
