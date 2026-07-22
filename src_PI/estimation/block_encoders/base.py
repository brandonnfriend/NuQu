"""
BlockEncoderStrategy: protocol for a block-encoder backend.

Each strategy takes a `HamiltonianBundle` (algebra-tagged sub-Hamiltonians)
and returns a resource-estimate dict shaped like the legacy `norm_data`
output. Keys downstream code expects:

  - 'identity_shift', 'physical_lambda', 'delta'         (normalization)
  - 'sub_lambdas', 'sub_identity_shifts'                 (per-sub diagnostics)
  - 'sub_hamiltonians'                                   (normalized operators)
  - 'walk_mode'                                          (passed through)
  - 'Walk_T_Count', 'Walk_Clifford_Count'                (walk-step resources)
  - 'Logical_Qubits', 'Physical_Lambda'                  (totals)
  - 'Per_Sub_Walk'                                       (per-sub walk results)

QFT-overhead and the final `Total_T_Count = Walk_T_Count + QFT_T_Count` are
added by the orchestrator (basis-conditional, not encoder-specific) in
`EstimateResources.evaluate_resources`.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlockEncoderStrategy(Protocol):
    name: str

    def estimate(self, bundle, num_sites: int, n_b: int, config) -> dict:
        """Run the strategy's end-to-end resource estimate for the bundle."""
        ...
