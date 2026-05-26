"""
LOBE (Ladder-Operator Block-Encoding) strategy. Stub for Phase A.

The real implementation lands in Phase D of the refactor; see
`tasks/28-lobe-implementation.md` and `claude/research/block-encoders/`.
"""


class LOBEStrategy:
    name = 'lobe'

    def estimate(self, bundle, num_sites, n_b, config):
        raise NotImplementedError(
            "LOBE block encoder is not yet implemented. "
            "Track progress in tasks/28-lobe-implementation.md (Phase D)."
        )
