"""
SubHamiltonian: a named operator carrying its algebra label.

The algebra label tells downstream block-encoder strategies how to interpret
`operator`:

- 'pauli'         — OpenFermion `QubitOperator`. Current default.
- 'bosonic'       — pure-boson native algebra (e.g. `BosonOperator`).
- 'fermionic'     — pure-fermion native algebra (e.g. `FermionOperator`).
- 'fermion_boson' — mixed term carried as an unmultiplied structured
                    representation (see Phase B). Block encoders that
                    support mixed modes natively (LOBE) consume this
                    directly; others (PauliLCU) expand to 'pauli' at the
                    encoder boundary.

Back-compat: `for name, op in [sub_h]:` still unpacks correctly because
`SubHamiltonian.__iter__` yields exactly `(name, operator)`. Existing
call sites that did `for name, op in bundle.sub_hamiltonians:` keep
working without changes.
"""

from dataclasses import dataclass
from typing import Any


_VALID_ALGEBRAS = ('pauli', 'bosonic', 'fermionic', 'fermion_boson')


@dataclass
class SubHamiltonian:
    name: str
    operator: Any
    algebra: str = 'pauli'

    def __post_init__(self):
        if self.algebra not in _VALID_ALGEBRAS:
            raise ValueError(
                f"algebra must be one of {_VALID_ALGEBRAS}, got {self.algebra!r}"
            )

    def __iter__(self):
        # Tuple-unpacking back-compat: `name, op = sub_h` and
        # `for name, op in [sub_h, ...]:` both keep working.
        yield self.name
        yield self.operator
