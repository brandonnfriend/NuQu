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
"""


class HamiltonianBundle:
    """A list of named (QubitOperator) sub-Hamiltonians plus walk-mode metadata."""

    def __init__(self, sub_hamiltonians, walk_mode='series', metadata=None):
        """
        Args:
            sub_hamiltonians: list of (name, QubitOperator) tuples.
                e.g. [('pos', H_pos), ('mom', H_mom)] for amplitude basis,
                     [('fock', H_full)]              for Fock basis.
            walk_mode: 'series' (default) or 'parallel'.
            metadata: optional dict of extra info (basis label, etc.) to
                preserve through the pipeline.
        """
        if walk_mode not in ('series', 'parallel'):
            raise ValueError(f"walk_mode must be 'series' or 'parallel', got {walk_mode!r}")
        self.sub_hamiltonians = list(sub_hamiltonians)
        self.walk_mode = walk_mode
        self.metadata = dict(metadata) if metadata else {}

    def __iter__(self):
        return iter(self.sub_hamiltonians)

    def __len__(self):
        return len(self.sub_hamiltonians)

    def names(self):
        return [name for name, _ in self.sub_hamiltonians]

    def operators(self):
        return [op for _, op in self.sub_hamiltonians]

    def get(self, name):
        """Lookup a sub-Hamiltonian by name. Returns None if not found."""
        for n, op in self.sub_hamiltonians:
            if n == name:
                return op
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
