from openfermion import QubitOperator, FermionOperator
from src_OF.utils.utils import site_to_qubit_1D, qubit_to_site_1D
import numpy as np

fermionic_modes = [(0,0), (0,1), (1,0), (1,1)] #spin up/down and isospin up/down

def Pauli(S, qubit_id):
    if S == 1:
        return QubitOperator(f"X{qubit_id}")
    elif S == 2:
        return QubitOperator(f"Y{qubit_id}")
    elif S == 3:
        return QubitOperator(f"Z{qubit_id}")
    else:
        raise ValueError("Invalid Pauli operator")

def Pauli_matrix(S):
    if S == 1:
        return np.array([[0, 1], [1, 0]]) #X
    elif S == 2:
        return np.array([[0, -1j], [1j, 0]]) #Y
    elif S == 3:
        return np.array([[1, 0], [0, -1]]) #Z
    else:
        raise ValueError("Invalid Pauli operator")

#Ferminonic creation, annihilation, and number operators
def Create(qubit_id):
    return FermionOperator(f"{qubit_id}^")

def Annihilate(qubit_id):
    return FermionOperator(f"{qubit_id}")

def Number(qubit_id):
    return FermionOperator(f"{qubit_id}^") * FermionOperator(f"{qubit_id}")


#Fermionic Bilinear Operators (see (5)->(8) in Watson et al. 2025)
#useful for One-pion Exchange term
def rho(site_id):
    H=0
    for mode in fermionic_modes:
        qubit_id = site_to_qubit_1D(site_id, mode)
        H+= Number(qubit_id)
    return H


def rho_S(S, site_id):
    sigma_S = Pauli_matrix(S)
    H = 0

    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                Create_ab = Create(site_id, (a,b))
                Annihilate_cb = Annihilate(site_id, (c,b))
                H+= Create_ab * sigma_S[a,c] * Annihilate_cb
    return H

def rho_I(I,site_id):
    sigma_I = Pauli_matrix(I)
    H = 0

    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                Create_ab = Create(site_id, (a,b))
                Annihilate_ac = Annihilate(site_id, (a,c))
                H+= Create_ab * sigma_I[b,c] * Annihilate_ac
    return H


def rho_SI(S,I,site_id):
    sigma_S = Pauli_matrix(S)
    sigma_I = Pauli_matrix(I)
    H = 0

    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    Create_ab = Create(site_id, (a,b))
                    Annihilate_cd = Annihilate(site_id, (c,d))
                    H+= Create_ab * sigma_S[a,c] * sigma_I[b,d] * Annihilate_cd
    return H

