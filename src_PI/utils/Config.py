"""
Config: project-wide configuration object that threads through the pipeline.

Records all design-axis switches (pion basis, walk mode, future encoder /
fermion-encoding / cutoff-method choices) in one place. Saved into JSON
metadata so a sweep file is self-describing.

Currently supported axes:
- pion_basis: 'amplitude' or 'fock'
- walk_mode:  'series' (default) or 'parallel'

To add a new design axis (e.g. block_encoder, fermion_encoding,
cutoff_method): add a field here with a sensible default; downstream
dispatch reads `config.<axis>` at the entry point and routes to the
appropriate module. Old call sites keep working because defaults match
current behavior.
"""

from dataclasses import dataclass, asdict, field


_VALID_PION_BASES = ('amplitude', 'fock')
_VALID_WALK_MODES = ('series', 'parallel')


@dataclass
class Config:
    pion_basis: str = 'amplitude'
    walk_mode: str = 'series'

    # Future axes — add as we implement them. Defaults preserve current
    # behavior so existing call sites don't break.
    # block_encoder: str = 'pauli_lcu'
    # fermion_encoding: str = 'jw'
    # cutoff_method: str = 'lemma5'

    # Free-form extras: anything the user wants to remember about the run
    # but that doesn't drive code dispatch. Saved to JSON alongside the
    # main fields.
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pion_basis not in _VALID_PION_BASES:
            raise ValueError(
                f"pion_basis must be one of {_VALID_PION_BASES}, got {self.pion_basis!r}"
            )
        if self.walk_mode not in _VALID_WALK_MODES:
            raise ValueError(
                f"walk_mode must be one of {_VALID_WALK_MODES}, got {self.walk_mode!r}"
            )

    def to_dict(self):
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Load from a dict (e.g. from JSON metadata)."""
        if d is None:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known}
        if extras and 'extras' not in kwargs:
            kwargs['extras'] = extras
        return cls(**kwargs)
