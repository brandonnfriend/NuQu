import  numpy                  as   np
import  matplotlib.pyplot      as   plt

## Problem instance & Encoding items
##
from    pyLIQTR.ProblemInstances.getInstance                  import   getInstance
from    pyLIQTR.clam.lattice_definitions                      import   CubicLattice, SquareLattice, TriangularLattice
from pyLIQTR.ProblemInstances.LatticeInstance  import   LatticeInstance
from    pyLIQTR.BlockEncodings.getEncoding                    import   getEncoding, VALID_ENCODINGS

from openfermion import QubitOperator, FermionOperator, jordan_wigner
import numpy as np
from src_OF.utils import nearest_neighbor as NN
from src_OF.hamiltonians.Operators import Create, Annihilate, Number
from src_OF.utils.utils import site_to_qubit_1D, qubit_to_site_1D, total_qubits_1D

###
### Todo: Implement Class to for Dynamical Pion EFT Model in pyLIQTR

class DynamicalPionEFT(LatticeInstance):
    """
    This class will implement the problem instance of the  Dynamical Pion EFT nuclear Hamiltonian
    Need to implement all five Hamiltonian terms as a function of inputs
    Lattice Dimension
    Lattice Size
    Nucleon Number
    Pion Number
    """

    def __init__(self, 
                 dim=2, 
                 shape=(4, 4), 
                 Nucleons=3, 
                 Pions=2, 
                 Coeffs=(1, 1, 1, 1, 1, 1), 
                 pbcs=None, 
                 **kwargs):
        self._model_prefix = "Dynamical Pion EFT"
        self.Nucleons = Nucleons
        self.Pions = Pions
        self.Coeffs = Coeffs
        
        # Shape validation (ensuring shape length matches dim)
        if len(shape) != dim:
            raise ValueError(f"Shape must match dimension for {self._model_prefix}.")
        elif len(set(shape)) > 1:
            raise ValueError(f"Only cubic lattices currently supported for {self._model_prefix}.")

        # Dynamic pbcs handling based on dim
        if pbcs is None:
            # Default to False for all dimensions
            self.pbcs = (False,) * self.dim
        elif isinstance(pbcs, bool):
            # If user passes pbcs=True, apply to all dimensions
            self.pbcs = (pbcs,) * self.dim
        else:
            # Validate user-provided tuple/list length
            if len(pbcs) != self.dim:
                raise ValueError(f"Length of pbcs ({len(pbcs)}) must match dim ({self.dim}).")
            self.pbcs = tuple(pbcs)

        self.shape = shape
        self._M_vals = np.array(shape)

        self.pbcs = pbcs

        self._N = np.prod(shape)  

    def _generate_hamiltonian(self):
        """
        Need to implement the Hamiltonian construction based on the Dynamical Pion EFT model.
        This will involve creating FermionOperators for each term in the Hamiltonian, using the coefficients
        """
        N = self._N
        c1, c2, c3, c4, c5 = self.Coeffs[:5] # Example mapping
        ham = FermionOperator()

        for i in range(N):
            # 1. Nucleon Mass/Chemical Potential Term: c1 * N_i
            # n_i = a^\dagger_i a_i
            ham += FermionOperator(f"{i}^ {i}", c1)

            # 2. Pion Mass Term: c2 * n_pion_i
            # Note the index offset for pions
            pion_idx = i + N
            ham += FermionOperator(f"{pion_idx}^ {pion_idx}", c2)

            # 3. Interaction / Hopping Terms
            # You'll need a neighbor list based on your 'dim' and 'shape'
            neighbors = self._get_neighbors(i) 
            for j in neighbors:
                # Nucleon Hopping: c3 * (a^\dagger_i a_j + h.c.)
                ham += FermionOperator(f"{i}^ {j}", c3)
                
                # Yukawa-style Interaction (Nucleon-Pion coupling)
                # Example: c4 * n_nucleon_i * (pion_create_j + pion_annihilate_j)
                ham += FermionOperator(f"{i}^ {i} {pion_idx + (j-i)}^", c4)
                ham += FermionOperator(f"{i}^ {i} {pion_idx + (j-i)}", c4)

        return ham

    def _get_neighbors(self, index):
        """
        Calculates the 1D indices of adjacent sites for a given site index.
        Handles N-dimensional lattices and Periodic Boundary Conditions (PBCs).
        """
        neighbors = []
        # Convert 1D index to ND coordinates (e.g., 5 -> (1, 1) in a 4x4)
        coords = np.array(np.unravel_index(index, self.shape))

        for d in range(len(self.shape)):
            # Look at both directions (+1 and -1) for each dimension
            for shift in [-1, 1]:
                neighbor_coords = coords.copy()
                neighbor_coords[d] += shift

                # Handle Boundary Conditions
                if self.pbcs[d]:
                    # Wrap around using modulo
                    neighbor_coords[d] %= self.shape[d]
                    neighbors.append(int(np.ravel_multi_index(neighbor_coords, self.shape)))
                else:
                    # Only add if within bounds (Hard wall / Dirichlet)
                    if 0 <= neighbor_coords[d] < self.shape[d]:
                        neighbors.append(int(np.ravel_multi_index(neighbor_coords, self.shape)))
        
        return list(set(neighbors)) # Remove duplicates if any

    def get_qubit_hamiltonian(self):
        fermion_ham = self._generate_hamiltonian()
        return jordan_wigner(fermion_ham)
    