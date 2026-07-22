"""
block_encoders: per-strategy modules dispatched by `config.block_encoder`.

Public API:
    get_block_encoder(name) -> BlockEncoderStrategy instance.

Registered strategies:
    'pauli_lcu' — current pyLIQTR PauliLCU path (default).
    'sparse'    — BCK sparse oracle (task 26, not yet implemented).
    'lobe'      — Ladder-Operator BE (task 28, not yet implemented).
"""

from src_PI.estimation.block_encoders.base import BlockEncoderStrategy
from src_PI.estimation.block_encoders.lobe import LOBEStrategy
from src_PI.estimation.block_encoders.pauli_lcu import PauliLCUStrategy
from src_PI.estimation.block_encoders.sparse import SparseStrategy


_STRATEGIES = {
    'pauli_lcu': PauliLCUStrategy,
    'sparse':    SparseStrategy,
    'lobe':      LOBEStrategy,
}


def get_block_encoder(name):
    """Return a fresh instance of the named block-encoder strategy."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown block_encoder: {name!r}. "
            f"Valid choices: {sorted(_STRATEGIES)}"
        )
    return _STRATEGIES[name]()


__all__ = [
    'BlockEncoderStrategy',
    'PauliLCUStrategy',
    'SparseStrategy',
    'LOBEStrategy',
    'get_block_encoder',
]
