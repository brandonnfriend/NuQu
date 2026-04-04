import math

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
        'M_N': M_N          
    }

def calculate_dynamic_cutoffs(L, dim, A_nucleons, params, epsilon_cut=0.1, E_bound=140.0):
    """
    Calculates the dynamic pion field cutoffs and required qubits (n_b) 
    based on Lemma 5 of the paper.
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
    prefactor = math.sqrt(3.0 * L_vol / epsilon_cut) + 1.0
    
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
    
    pi_max = prefactor * (gA_fpi_aL_A + math.sqrt(sqrt_inner_pi))
    
    # Pi_max calculation (Eq 76)
    sqrt_inner_Pi = (energy_contact_sum / B_coeff) + \
                    ((3.0 * eta / (A_coeff * B_coeff)) * (gA_fpi_aL**2)) + \
                    ((9.0 * eta * (m_pi**2) * (a_L**3) / B_coeff) * (mass_term**2))
                    
    Pi_max = prefactor * math.sqrt(sqrt_inner_Pi)
    
    # -----------------------------------------------------------------
    # Equation 78: Qubit Requirement (n_b)
    # -----------------------------------------------------------------
    # n_b = log_2( (2 a_L^3 / \pi) * Pi_max * pi_max + 1 )
    inner_term = (2.0 * (a_L**3) / math.pi) * Pi_max * pi_max + 1.0
    n_b_float = math.log2(inner_term)
    
    # "we choose the nearest cutoffs above these bounds to ensure n_b is an integer"
    n_b = math.ceil(n_b_float)
    
    return n_b, pi_max, Pi_max