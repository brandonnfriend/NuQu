"""
sparse_oracle: BCK sparse-oracle block-encoder primitives (task 26).

Sub-phase status (see `claude/research/block-encoders/04_refactor_execution_log.md`):
  * C1 (current): native-algebra `Λ` helper + scaffold; actual block-encoding
    construction is deferred to C2.
  * C2 (next):    single-mode `(â + â†)` block encoding via Qualtran bloqs
                  (Lee-Kanno 2023 §3 construction) + classical-sim validation.
  * C3:           multi-mode products (Gilyén Lemma 30) + mixed fermion/boson
                  composition.
  * C4:           full pipeline + comparison sweep against PauliLCU / LOBE.
"""

from src_PI.estimation.sparse_oracle.lambda_compute import (
    compute_native_lambda,
)

__all__ = [
    'compute_native_lambda',
]
