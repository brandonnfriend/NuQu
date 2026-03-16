import re
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance

class MyCustomHamiltonian(ProblemInstance):
    """
    Bridge class between OpenFermion QubitOperators and pyLIQTR.
    """
    def __init__(self, pauli_dict):
        self.pauli_dict = pauli_dict
        self._n_qubits = self._calculate_qubits()
        
    def _calculate_qubits(self):
        max_idx = 0
        for term in self.pauli_dict.keys():
            indices = [int(n) for n in re.findall(r'\d+', term)]
            if indices:
                max_idx = max(max_idx, max(indices))
        return max_idx + 1

    def n_qubits(self):
        return self._n_qubits

    def __str__(self):
        return "CustomPionEFT"

    def get_pauli_extension(self):
        return self.pauli_dict

    def get_alpha(self):
        return sum(abs(coeff) for coeff in self.pauli_dict.values())

    def yield_PauliLCU_Info(self, **kwargs):
        for p_str, coeff in self.pauli_dict.items():
            dense_list = ['I'] * self._n_qubits
            if p_str != "I":
                for factor in p_str.split():
                    gate = factor[0]
                    idx  = int(factor[1:])
                    dense_list[idx] = gate
            yield ("".join(dense_list), coeff)