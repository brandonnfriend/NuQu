from openfermion import QubitOperator, FermionOperator
from src_OF.utils.utils import site_to_qubit_1D, qubit_to_site_1D

fermionic_modes = [(1,1), (1,-1), (-1,1), (-1,-1)] #spin up/down and isospin up/down

def Pauli_wrapper(S, qubit_id):
    if S == 'X':
        return QubitOperator(f"X{qubit_id}")
    elif S == 'Y':
        return QubitOperator(f"Y{qubit_id}")
    elif S == 'Z':
        return QubitOperator(f"Z{qubit_id}")
    else:
        raise ValueError("Invalid Pauli operator")



def Create(qubit_id):
    return FermionOperator(f"{qubit_id}^")

def Annihilate(qubit_id):
    return FermionOperator(f"{qubit_id}")

def Number(qubit_id):
    return FermionOperator(f"{qubit_id}^") * FermionOperator(f"{qubit_id}")


#Fermionic Bilinear Operators
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

