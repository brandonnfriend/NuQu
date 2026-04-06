import numpy as np
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs, T_cross_MeV


def g(L, n_b):
    """Lemma 23 of Watson et. al 2025"""
    return (45 * n_b**2 + 114 * n_b + 76) * (L**3)

def DynamicalPion_TrotterStep_Tgates_cost(L, n_b, delta):
    """Lemma 23 of Watson et. al 2025"""
    return g(L, n_b) * (1.15 * np.log10(2 * g(L, n_b) / delta) + 9.2)

def error_product(t, L, A, params,p, Cp=1.0, E=100.0):
    params = get_physical_parameters()
    _, pi_max, Pi_max = calculate_dynamic_cutoffs(L, 3, A, params, epsilon_cut=0.1, E_bound=E)
    return Cp* (Pi_max**(p+1))* (pi_max**(p+1)) * (L**3) * (t**(p+1))

def get_Trotter_steps_cross_time(L, A, E_bound, E_kin, total_error, p=1, Cp=1.0):
    #get EFT parameters
    params = get_physical_parameters()

    #e_cut = ((total_error/3)/(2 * np.sqrt(2)))**2

    #get the dynamic cutoffs pi_max and Pi_max based on the provided parameters and error budget
    _, pi_max, Pi_max = calculate_dynamic_cutoffs(L, 3, A, params, epsilon_cut=0.1, E_bound=E_bound)

    t_cross = T_cross_MeV(params["a_L"], L, E_kin)
    return Cp * A* (Pi_max**(p+1)) * (pi_max**(p+1)) * L**3 * (t_cross**(p+1))/ (total_error/3)

def commutator_bound_WT_WT(L, A, E_bound):
    """Lemma 78 of Watson et. al 2025"""
    params = get_physical_parameters()
    f_pi = params['f_pi']
    a_L = params['a_L']
    _, pi_max, Pi_max = calculate_dynamic_cutoffs(L, 3, A, params, epsilon_cut=0.1, E_bound=E_bound)
    return 384 * (1/(4*f_pi**2))**2 *(3 * pi_max * Pi_max+ 2/(a_L**3)) * pi_max * Pi_max * A

