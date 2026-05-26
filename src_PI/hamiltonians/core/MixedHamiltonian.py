"""
MixedHamiltonian: native-algebra container for sub-Hamiltonians whose
fermion and boson factors are carried *unmultiplied*.

Used as the `operator` payload of a `SubHamiltonian` tagged with
algebra='fermion_boson'. Encoders that consume mixed fermion/boson
operators natively (LOBE, possibly sparse) read these fields directly;
encoders that require Pauli-LCU (current pyLIQTR PauliLCU) go through a
separate pre-built Pauli builder (`pion_basis/fock.py`) — this dataclass
is *not* part of the PauliLCU bit-identity contract.

Indexing conventions:
  * Fermion: global nucleon qubit indices (same as
    `Static_Nucleon_Hamiltonian` / `Nucleon_Transition_*` use).
  * Boson:   global pion-mode indices via
        mode = site * num_pion_species + species
    matching `mode_to_qubits`. Each entry of `mode_to_qubits` maps a
    global mode index to its ordered list of n_b backing qubits.

Mixed terms (`H_AV`, `H_WT`) live in `mixed_terms` as
`MixedTerm(coeff, fermion_factor, boson_factor)`. The fermion and boson
factors act on disjoint registers and therefore commute trivially — a
downstream encoder applies them as a tensor product without worrying
about ordering.
"""

from dataclasses import dataclass, field

from openfermion import BosonOperator, FermionOperator


@dataclass
class MixedTerm:
    coeff: complex
    fermion_factor: FermionOperator
    boson_factor: BosonOperator


@dataclass
class MixedHamiltonian:
    fermion_part: FermionOperator = field(default_factory=FermionOperator)
    boson_part: BosonOperator = field(default_factory=BosonOperator)
    mode_to_qubits: dict = field(default_factory=dict)
    mixed_terms: list = field(default_factory=list)  # list[MixedTerm]
