"""
HamiltonianBundle: container for one-or-more sub-Hamiltonians that together
describe a physical Hamiltonian, plus the metadata needed to know how their
walk operators compose at resource-estimation time.

Why this abstraction:
- The amplitude basis encodes the EFT as a split oracle: H_pos (position-basis
  terms) and H_mom (kinetic Π² term), connected by QFT/IQFT around each H_mom
  step. Two sub-Hamiltonians, two walks per step.
- The Fock basis encodes everything in a single shared (a, a†) algebra. One
  sub-Hamiltonian, one walk per step, no QFT.
- Future hybrid encodings (e.g. THC-style factorization) may want N walks.

Downstream code (normalize / estimate / report) should iterate over
`bundle.sub_hamiltonians` and *not* assume a particular count.

Walk-mode (`series` vs `parallel`) controls how per-walk logical qubits are
combined into a peak qubit count:
- series: walks reuse the same hardware → peak = max(walk_qubits)
- parallel: walks run simultaneously → peak = sum(walk_qubits)

Storage: `self.sub_hamiltonians` is a list of `SubHamiltonian` instances.
For backwards-compatibility, `SubHamiltonian` is tuple-unpackable as
`(name, operator)`, so existing iteration code `for name, op in bundle:`
keeps working. The constructor also accepts legacy `(name, operator)`
tuples and auto-wraps them as `SubHamiltonian(name, operator, algebra='pauli')`.
"""

from src_PI.hamiltonians.core.SubHamiltonian import SubHamiltonian


class HamiltonianBundle:
    """A list of named sub-Hamiltonians (`SubHamiltonian` objects) plus walk-mode metadata."""

    def __init__(self, sub_hamiltonians, walk_mode='series', metadata=None):
        """
        Args:
            sub_hamiltonians: list of `SubHamiltonian` instances, OR (back-compat)
                a list of `(name, operator)` tuples — tuples are auto-wrapped as
                `SubHamiltonian(name, operator, algebra='pauli')`.
            walk_mode: 'series' (default) or 'parallel'.
            metadata: optional dict of extra info (basis label, etc.) to
                preserve through the pipeline.
        """
        if walk_mode not in ('series', 'parallel'):
            raise ValueError(f"walk_mode must be 'series' or 'parallel', got {walk_mode!r}")
        wrapped = []
        for item in sub_hamiltonians:
            if isinstance(item, SubHamiltonian):
                wrapped.append(item)
            else:
                name, operator = item
                wrapped.append(SubHamiltonian(name=name, operator=operator, algebra='pauli'))
        self.sub_hamiltonians = wrapped
        self.walk_mode = walk_mode
        self.metadata = dict(metadata) if metadata else {}

    def __iter__(self):
        return iter(self.sub_hamiltonians)

    def __len__(self):
        return len(self.sub_hamiltonians)

    def names(self):
        return [sh.name for sh in self.sub_hamiltonians]

    def operators(self):
        return [sh.operator for sh in self.sub_hamiltonians]

    def algebras(self):
        """Return the algebra label of each sub-Hamiltonian, in order."""
        return [sh.algebra for sh in self.sub_hamiltonians]

    def get(self, name):
        """Lookup a sub-Hamiltonian's operator by name. Returns None if not found."""
        for sh in self.sub_hamiltonians:
            if sh.name == name:
                return sh.operator
        return None

    def combine_qubit_count(self, per_walk_counts):
        """Combine a list of per-sub-Hamiltonian qubit counts per walk_mode.

        Series walks share hardware → peak qubit need is max.
        Parallel walks run simultaneously → peak qubit need is sum.
        """
        if not per_walk_counts:
            return 0
        if self.walk_mode == 'series':
            return max(per_walk_counts)
        return sum(per_walk_counts)
