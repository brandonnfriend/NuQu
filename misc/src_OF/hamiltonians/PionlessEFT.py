from openfermion import QubitOperator, FermionOperator, jordan_wigner
import numpy as np
from misc.src_OF.utils import nearest_neighbor as NN
from misc.src_OF.hamiltonians.Operators import Create, Annihilate, Number
from misc.src_OF.utils.utils import site_to_qubit_1D, qubit_to_site_1D, total_qubits_1D

#Assume 1D lattice for simplicity. Each site can have 4 fermionic modes corresponding to spin up/down and isospin up/down. The Hamiltonian includes a free term (hopping and on-site energy), a two-body contact interaction, and a three-body contact interaction.


#Todo:
#  Test the Hamiltonian
#  Test jordan-wigner on those 


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

def TwoBody_Contact_1D(C, num_sites):
    Hc = 0
    for site in range(num_sites):
        for mode1 in fermionic_modes:
            for mode2 in fermionic_modes:
                if mode1 != mode2: #only different modes can interact
                    qubit_id_1 = site_to_qubit_1D(site, mode1)
                    qubit_id_2 = site_to_qubit_1D(site, mode2)
                    Hc += Number(qubit_id_1) * Number(qubit_id_2)

    return 0.5 * C * Hc

def ThreeBody_Contact_1D(D, num_sites):
    H_D = 0
    for site in range(num_sites):
        for mode1 in fermionic_modes:
            for mode2 in fermionic_modes:
                for mode3 in fermionic_modes:
                    if len(set([mode1, mode2, mode3])) == 3: #only different modes can interact
                        qubit_id_1 = site_to_qubit_1D(site, mode1)
                        qubit_id_2 = site_to_qubit_1D(site, mode2)
                        qubit_id_3 = site_to_qubit_1D(site, mode3)
                        H_D += Number(qubit_id_1) * Number(qubit_id_2) * Number(qubit_id_3)

    return (1/6) * D * H_D

def PionlessEFT_Hamiltonian_1D(h, C, D, num_sites):
    H_free = Free_Term_1D(h, num_sites)
    H_2body = TwoBody_Contact_1D(C, num_sites)
    H_3body = ThreeBody_Contact_1D(D, num_sites)

    return H_free + H_2body + H_3body

def PionlessEFT_Hamiltonian_1D_jw(h, C, D, num_sites):
    H = PionlessEFT_Hamiltonian_1D(h, C, D, num_sites)
    return jordan_wigner(H)

