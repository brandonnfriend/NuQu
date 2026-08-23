"""
Config: project-wide configuration object that threads through the pipeline.

Records all design-axis switches (pion basis, walk mode, future encoder /
fermion-encoding / cutoff-method choices) in one place. Saved into JSON
metadata so a sweep file is self-describing.

Currently supported axes:
- pion_basis:    'amplitude', 'fock', or 'fock_squeezed' (fock walked in the
                 Gaussian squeeze frame; squeeze amplitude via params['squeeze_r'])
- walk_mode:     'series' (default) or 'parallel'
- cutoff_method: 'energy_bound' (Watson Lemma 5, default) or 'ns'
                 (Nyquist-Shannon optimal). Only consulted for the
                 amplitude basis; the Fock basis derives its own cutoff.
- boson_cutoff_method: 'heuristic' (default log2(1+A) starter formula),
                 'tong' (first-draft Tong-SCS + Cauchy-Schwarz ESTIMATE, not a
                 certificate; n_q=4-5), or 'gaussian_reference_estimate' (aka
                 the deprecated alias 'tong_rigorous'): an exact-Bogoliubov
                 Gaussian-reference ESTIMATE (not rigorous/certified — the true
                 interacting-GS tail is an open theorem, see
                 codex_audit/03_cutoff), dim-general. Chooses how the per-site
                 boson register size n_q is set. Drives the Fock basis directly
                 and the NS amplitude register indirectly; ignored by the
                 amplitude 'energy_bound' path (Lemma 5 sets its own n_b).
- block_encoder: 'pauli_lcu' (default — current pyLIQTR path),
                 'sparse' (BCK sparse-oracle, task 26),
                 or 'lobe' (Ladder-Operator Block-Encoding, task 28).
                 Selects the strategy in `src_PI/estimation/block_encoders/`.
- sparse_oracle_mode: 'analytical' (default — the Gilyén+LCU proxy that mixes a
                 boson upper bound with a fermion lower bound; A/B baseline) or
                 'hermitian_cost_model' (the walk-VALID Hermitian matching-
                 dilation construction, costed by a primitive-based **cost
                 model** — NOT a compiler-derived count: the block encoding has
                 no executable decomposition yet, and the model omits coherent
                 controls / matching predicates / boundary logic / phase / a
                 precision budget, so its T is an optimistic provisional
                 estimate). Only consulted when block_encoder='sparse'. For
                 publication-grade quantum resources use the PauliLCU anchor
                 until the decomposable composite lands (Codex audit 2026-08-18).

To add a new design axis (e.g. fermion_encoding): add a field here with
a sensible default; downstream dispatch reads `config.<axis>` at the
entry point and routes to the appropriate module. Old call sites keep
working because defaults match current behavior.
"""

from dataclasses import dataclass, asdict, field


_VALID_PION_BASES = ('amplitude', 'fock', 'fock_squeezed')
_VALID_WALK_MODES = ('series', 'parallel')
_VALID_CUTOFF_METHODS = ('energy_bound', 'ns')
_VALID_BOSON_CUTOFF_METHODS = ('heuristic', 'tong', 'gaussian_reference_estimate',
                               'tong_rigorous')  # 'tong_rigorous' = deprecated alias
_VALID_BLOCK_ENCODERS = ('pauli_lcu', 'sparse', 'lobe')
_VALID_SPARSE_ORACLE_MODES = ('analytical', 'hermitian_cost_model', 'compiled')
_VALID_WALK_COMPOSITIONS = ('combined_lcu', 'split_sum')


@dataclass
class Config:
    pion_basis: str = 'amplitude'
    walk_mode: str = 'series'
    # Cutoff prescription for the amplitude basis. 'energy_bound' = Watson
    # Lemma 5 (current default); 'ns' = Nyquist-Shannon optimal (Path B).
    # Ignored by the Fock basis, which derives its own cutoff.
    cutoff_method: str = 'energy_bound'
    # Per-site boson register-size method. 'heuristic' = starter log2(1+A)
    # formula (current default); 'tong' = first-draft Tong-SCS + Cauchy-Schwarz
    # ESTIMATE (not a certificate; n_q=4-5, A-flat); 'gaussian_reference_estimate'
    # (aka deprecated 'tong_rigorous') = exact-Bogoliubov Gaussian-reference
    # ESTIMATE (not rigorous/certified; open theorem, see codex_audit/03_cutoff).
    # Drives the Fock basis directly and the NS amplitude register indirectly;
    # ignored by the amplitude 'energy_bound' path.
    boson_cutoff_method: str = 'heuristic'
    # Block-encoder strategy. Default 'pauli_lcu' preserves the current
    # behavior; 'sparse' / 'lobe' will be wired in by tasks 26 / 28.
    block_encoder: str = 'pauli_lcu'
    # Sparse-oracle costing mode (only consulted for block_encoder='sparse').
    # 'analytical' = the mixed-bound Gilyén+LCU proxy (A/B baseline, default);
    # 'hermitian_cost_model' = walk-valid Hermitian construction, hand-assembled
    # cost (optimistic); 'compiled' = genuinely compiled walk — every rotation
    # synthesized to Clifford+T at the ΔE-derived precision, scalable primitives,
    # full Hamiltonian incl. mixed atoms (compiled_resources).
    sparse_oracle_mode: str = 'analytical'
    # How the split-oracle sub-walks (amplitude basis: H_pos + H_mom) are combined.
    # ⚠️ BOTH amplitude paths are EXPERIMENTAL / not publication-grade — the paper anchor
    # is Fock/PauliLCU, and amplitude quantum totals must not be reported (codex
    # amplitude_combined_walk_audit_2026-08-20). 'combined_lcu' (default) targets the right
    # controlled-sum LCU architecture but is only an incomplete COST ROLL-UP: it does not
    # build/validate the block encoding and leaves H_WT mis-represented in pos_dyn.
    # 'split_sum' is the older invalid two-walk sum. Only consulted for the amplitude split
    # (≥2 sub-walks); the single-walk Fock/PauliLCU anchor is byte-identical under either.
    walk_composition: str = 'combined_lcu'

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
        if self.sparse_oracle_mode not in _VALID_SPARSE_ORACLE_MODES:
            raise ValueError(
                f"sparse_oracle_mode must be one of {_VALID_SPARSE_ORACLE_MODES}, "
                f"got {self.sparse_oracle_mode!r}"
            )
        if self.walk_composition not in _VALID_WALK_COMPOSITIONS:
            raise ValueError(
                f"walk_composition must be one of {_VALID_WALK_COMPOSITIONS}, "
                f"got {self.walk_composition!r}"
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
