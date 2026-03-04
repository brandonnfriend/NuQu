from openfermion import QubitOperator, FermionOperator
from src_OF.utils.utils import site_to_qubit_1D, qubit_to_site_1D

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

#Ferminonic creation, annihilation, and number operators
def Create(qubit_id):
    return FermionOperator(f"{qubit_id}^")

def Annihilate(qubit_id):
    return FermionOperator(f"{qubit_id}")

def Number(qubit_id):
    return FermionOperator(f"{qubit_id}^") * FermionOperator(f"{qubit_id}")


#Fermionic Bilinear Operators (see (5)->(8) in Watson et al. 2025)
#useful for One-pion Exchange
def rho(site_id):
    H=0
    for mode in fermionic_modes:
        qubit_id = site_to_qubit_1D(site_id, mode)
        H+= Number(qubit_id)
    return H


def rho_S(S, site_id):
    pass

def rho_I(site_id):
    pass

def rho_SI(site_id):
    pass

