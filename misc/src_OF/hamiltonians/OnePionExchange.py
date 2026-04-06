
from misc.src_OF.hamiltonians.Operators import rho, rho_S, rho_I, Create, Annihilate, Number
from openfermion import normal_ordered
from misc.src_OF.utils import nearest_neighbor as NN
from misc.src_OF.utils.utils import site_to_qubit_1D


#todo: Test the Hamiltonian
#Write Long-range term (Equation 56 in Watson et al. 2025)

fermionic_modes = [(0,0), (0,1), (1,0), (1,1)] #spin up/down and isospin up/down

def Free_Hopping_1D(h, num_sites):
    #hoppings between nearest neighbors
    H_hop = 0
    NN_pairs = NN.get_NN_pairs_1D(num_sites)
    for pair in NN_pairs:
        for mode in fermionic_modes:
            #qubit_id for site i and mode
            qubit_id_i = site_to_qubit_1D(pair[0], mode)
            qubit_id_j = site_to_qubit_1D(pair[1], mode)
            
            H_hop += Create(qubit_id_i) * Annihilate(qubit_id_j) + Create(qubit_id_j) * Annihilate(qubit_id_i)

    return -h * H_hop    

def Free_Onsite_1D(h, num_sites):
    #free term for each site and mode
    H_N = 0
    for site in range(num_sites):
        for mode in fermionic_modes:
            qubit_id = site_to_qubit_1D(site, mode)
            H_N += Number(qubit_id) 
    return 6 * h * H_N 


def Free_Term_1D(h, num_sites):
    return Free_Hopping_1D(h, num_sites) + Free_Onsite_1D(h, num_sites)

def HC_1D(C, num_sites):
    H=0
    for site in range(num_sites):
        H+= rho(site) * rho(site)
    H*= C/2
    return normal_ordered(H)


def HCI2_1D(CI, num_sites):
    H=0
    for site in range(num_sites):
        for I in [1,2,3]:
            H+= rho_I(I, site) * rho_I(I, site)
    H*= CI/2
    return normal_ordered(H)

def HLR_1D(num_sites):
    H=0
    return normal_ordered(H)

def OnePionExchange_1D(h,C, CI, num_sites):
    H_free = Free_Term_1D(h, num_sites)
    HC = HC_1D(C, num_sites)
    HCI2 = HCI2_1D(CI, num_sites)
    #HLR = HLR_1D(num_sites)
    HLR = 0 #ignore for long range term for now
    H = H_free + HC + HCI2 + HLR
    return H


