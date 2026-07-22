"""
Config: project-wide configuration object that threads through the pipeline.

Records all design-axis switches (pion basis, walk mode, future encoder /
fermion-encoding / cutoff-method choices) in one place. Saved into JSON
metadata so a sweep file is self-describing.

Currently supported axes:
- pion_basis:    'amplitude' or 'fock'
- walk_mode:     'series' (default) or 'parallel'
- cutoff_method: 'energy_bound' (Watson Lemma 5, default) or 'ns'
                 (Nyquist-Shannon optimal). Only consulted for the
                 amplitude basis; the Fock basis derives its own cutoff.
- boson_cutoff_method: 'heuristic' (default log2(1+A) starter formula) or
                 'tong' (rigorous Tong-2022 polylog bound, n_q=4-5). Chooses
                 how the per-site boson register size n_q is set. Drives the
                 Fock basis directly and the NS amplitude register indirectly;
                 ignored by the amplitude 'energy_bound' path (Lemma 5 sets
                 its own n_b).
- block_encoder: 'pauli_lcu' (default — current pyLIQTR path),
                 'sparse' (BCK sparse-oracle, task 26),
                 or 'lobe' (Ladder-Operator Block-Encoding, task 28).
                 Selects the strategy in `src_PI/estimation/block_encoders/`.

To add a new design axis (e.g. fermion_encoding): add a field here with
a sensible default; downstream dispatch reads `config.<axis>` at the
entry point and routes to the appropriate module. Old call sites keep
working because defaults match current behavior.
"""

from dataclasses import dataclass, asdict, field


_VALID_PION_BASES = ('amplitude', 'fock')
_VALID_WALK_MODES = ('series', 'parallel')
_VALID_CUTOFF_METHODS = ('energy_bound', 'ns')
_VALID_BOSON_CUTOFF_METHODS = ('heuristic', 'tong')
_VALID_BLOCK_ENCODERS = ('pauli_lcu', 'sparse', 'lobe')


@dataclass
class Config:
    pion_basis: str = 'amplitude'
    walk_mode: str = 'series'
    # Cutoff prescription for the amplitude basis. 'energy_bound' = Watson
    # Lemma 5 (current default); 'ns' = Nyquist-Shannon optimal (Path B).
    # Ignored by the Fock basis, which derives its own cutoff.
    cutoff_method: str = 'energy_bound'
    # Per-site boson register-size method. 'heuristic' = starter log2(1+A)
    # formula (current default); 'tong' = rigorous Tong-2022 polylog bound
    # (n_q=4-5, A-flat). Drives the Fock basis directly and the NS amplitude
    # register indirectly; ignored by the amplitude 'energy_bound' path.
    boson_cutoff_method: str = 'heuristic'
    # Block-encoder strategy. Default 'pauli_lcu' preserves the current
    # behavior; 'sparse' / 'lobe' will be wired in by tasks 26 / 28.
    block_encoder: str = 'pauli_lcu'

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
        if self.cutoff_method not in _VALID_CUTOFF_METHODS:
            raise ValueError(
                f"cutoff_method must be one of {_VALID_CUTOFF_METHODS}, "
                f"got {self.cutoff_method!r}"
            )
        if self.boson_cutoff_method not in _VALID_BOSON_CUTOFF_METHODS:
            raise ValueError(
                f"boson_cutoff_method must be one of {_VALID_BOSON_CUTOFF_METHODS}, "
                f"got {self.boson_cutoff_method!r}"
            )
        if self.block_encoder not in _VALID_BLOCK_ENCODERS:
            raise ValueError(
                f"block_encoder must be one of {_VALID_BLOCK_ENCODERS}, "
                f"got {self.block_encoder!r}"
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
