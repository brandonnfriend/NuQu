"""
Sparse-oracle (BCK) strategy. Stub for Phase A.

The real implementation lands in Phase C of the refactor; see
`tasks/26-sparse-oracle-fock.md` and `claude/research/block-encoders/`.
"""


class SparseStrategy:
    name = 'sparse'

    def estimate(self, bundle, num_sites, n_b, config):
        raise NotImplementedError(
            "sparse-oracle block encoder is not yet implemented. "
            "Track progress in tasks/26-sparse-oracle-fock.md (Phase C)."
        )
