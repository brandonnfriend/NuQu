"""
pion_basis: namespace-by-basis module dispatch.

Each basis module (`amplitude.py`, `fock.py`) exposes the same set of names:
- `Full_Dynamical_Pion_Hamiltonian(...)` returning a list of
  (name, QubitOperator) sub-Hamiltonian tuples.
- `H_pion_free(...)`, `H_axial_vector(...)`, `H_WT_Logic(...)`
- A `BASIS_NAME` constant for diagnostics.

Callers use `get(basis_name)` to select a module and then invoke the same
function names regardless of basis. New bases slot in by adding a new module
here; no caller code changes.
"""

from src_PI.hamiltonians.core.pion_basis import amplitude, fock


_MODULES = {
    'amplitude': amplitude,
    'fock': fock,
}


def get(basis_name):
    """Return the module implementing the given pion basis."""
    try:
        return _MODULES[basis_name]
    except KeyError:
        valid = ', '.join(sorted(_MODULES))
        raise ValueError(
            f"Unknown pion_basis {basis_name!r}; available: {valid}"
        )


def available():
    """List of valid basis names."""
    return sorted(_MODULES)
