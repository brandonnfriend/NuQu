#Fermionic_mode is (spin, isospin) which can be (1,1), (1,-1), (-1,1), (-1,-1) for spin up/down and isospin up/down. So each site will be encoded into 4 qubits. The mapping can be as follows:
def site_to_qubit_1D(site_id, fermionic_mode):
    if fermionic_mode == (0,0):
        return site_id * 4 + 0
    elif fermionic_mode == (0,1):
        return site_id * 4 + 1
    elif fermionic_mode == (1,0):
        return site_id * 4 + 2
    elif fermionic_mode == (1,1):
        return site_id * 4 + 3
    else:
        raise ValueError("Invalid fermionic mode")
    
def qubit_to_site_1D(qubit_id):
    site_id = qubit_id // 4
    mode_id = qubit_id % 4
    if mode_id == 0:
        fermionic_mode = (0,0)
    elif mode_id == 1:
        fermionic_mode = (0,1)
    elif mode_id == 2:
        fermionic_mode = (1,0)
    elif mode_id == 3:
        fermionic_mode = (1,1)
    else:
        raise ValueError("Invalid qubit id")
    
    return site_id, fermionic_mode
    
def total_qubits_1D(num_sites):
    return num_sites * 4 