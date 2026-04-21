from src_PI.estimation.EstimateResources import evaluate_resources

def run_toy_tests():
    # Don't trust these parameters! Code will run faster for these small values. 
    #Should pull parameters from EFTParameters.py .
    params = {
        'h': 1.0,       # Hopping parameter
        'C': -0.5,      # Contact term
        'CI': -0.1,     # Isospin contact term
        'a_L': 1.0,     # Lattice spacing (fermis)
        'm_pi': 0.14,   # Pion mass (GeV)
        'g_A': 1.29,    # Axial coupling
        'f_pi': 0.093   # Pion decay constant (GeV)
    }
    
    L = 2
    n_b = 2
    pi_max = 5.0 # Arbitrary cutoff for the toy field magnitude
    
    print("============================================================")
    print(" INITIATING DIMENSIONAL PIPELINE TEST")
    print("============================================================")
    
    for dim in [1, 2, 3]:
        print("\n" + "*"*60)
        print(f"   TESTING DIMENSION: {dim}D (L={L}, n_b={n_b})")
        print("*"*60)
        
        try:
            # The evaluator will print all the hardware stats internally
            norm_data = evaluate_resources(L, dim, n_b, pi_max, params)
            print(f"\n--> {dim}D Pipeline Completed Successfully! ✅")
            
        except Exception as e:
            print(f"\n--> ERROR in {dim}D Pipeline: {e} ❌")

if __name__ == "__main__":
    run_toy_tests()

from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters, calculate_dynamic_cutoffs

def verify_paper_nb():
    print("\n--- Verifying n_b Calculation vs. Paper (10x10x10 Lattice) ---")
    params = get_physical_parameters()
    L = 10
    dim = 3
    
    # The paper notes n_b varies with nucleon number (eta/A)
    # Testing for A = 2 (Deuteron-like) up to A = 20
    for A in [2, 4, 8, 16, 20]:
        n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(L, dim, A, params, epsilon_cut=0.1, E_bound=140.0)
        print(f"Nucleons A={A:2}:  n_b={n_b:2} | pi_max={pi_max:8.2f} | Pi_max={Pi_max:8.2f}")

if __name__ == "__main__":
    verify_paper_nb()