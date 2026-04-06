import numpy as np
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs
from src_PI.trotter_theory.trotter_theory import DynamicalPion_TrotterStep_Tgates_cost, get_Trotter_steps_cross_time

def get_total_trotter_cost(A, L, e=0.1, E_kin=10, E_bound=None, Cp=1e-3, dim=3):
    """
    Computes the total T-gate cost for Trotterization for a given nucleon number A.
    
    Args:
        A (int): Nucleon number.
        L (int): Spatial lattice extent.
        e (float): Trotter error tolerance. Defaults to 0.1.
        E_kin (float): Kinetic energy parameter. Defaults to 10.
        E_bound (float): Energy bound. If None, defaults to E_kin * A.
        Cp (float): Trotter simulation time parameter. Defaults to 1e-3.
        dim (int): Spatial dimensions. Defaults to 3.
        
    Returns:
        float: Total T-gate cost (cost per step * number of steps).
    """
    # Allow E_bound to scale with A if not explicitly provided
    if E_bound is None:
        E_bound = E_kin * A
        
    phys_params = get_physical_parameters()
    
    # Calculate cutoffs
    nb, _, _ = calculate_dynamic_cutoffs(
        L, dim, A, phys_params, epsilon_cut=0.1, E_bound=E_bound
    )
    
    # Calculate steps and cost per step
    r = get_Trotter_steps_cross_time(L, A, E_bound, E_kin, e, Cp=Cp)
    a = DynamicalPion_TrotterStep_Tgates_cost(L, nb, e / (3 * r))
    
    return a * r