"""
Sparse-oracle (BCK) block-encoder strategy.

Sub-phase status (see `claude/research/block-encoders/04_refactor_execution_log.md`):
  * **C1 (current)**: dispatch validates that the bundle is in native
    fermion-boson algebra, computes Λ + identity shift from the
    `MixedHamiltonian` via `compute_native_lambda`, prints a diagnostic
    summary, and raises `NotImplementedError` on the actual
    block-encoding step. The encoder is not yet wired through pyLIQTR.
  * **C2 (next)**: single-mode `(â + â†)` block encoding via Qualtran
    bloqs + classical-sim validation at n_b=3.
  * **C3**: multi-mode products (Gilyén Lemma 30) + mixed fermion/boson.
  * **C4**: full pipeline + comparison sweep.

Refer to `tasks/26-sparse-oracle-fock.md` and
`claude/research/block-encoders/01_pyliqtr_audit.md` §7c for the planned
construction.
"""

from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
from src_PI.hamiltonians.core.SubHamiltonian import SubHamiltonian


class SparseStrategy:
    name = 'sparse'

    def estimate(self, bundle, num_sites, n_b, config):
        # C1 validation: the bundle must be the native fermion-boson form,
        # produced by ConstructEFT's native-Fock path (Phase B).
        sub = bundle.sub_hamiltonians
        if len(sub) != 1 or not isinstance(sub[0], SubHamiltonian) \
                or sub[0].algebra != 'fermion_boson' \
                or not isinstance(sub[0].operator, MixedHamiltonian):
            raise TypeError(
                "sparse-oracle strategy expects a single SubHamiltonian "
                "with algebra='fermion_boson' carrying a MixedHamiltonian "
                "payload (Fock-native path). Got: "
                + ", ".join(
                    f"{s.name}(algebra={s.algebra!r}, "
                    f"op_type={type(s.operator).__name__})"
                    for s in sub
                )
            )

        mh = sub[0].operator

        # C1 deliverable: compute Λ + identity shift from native algebra,
        # without expanding the bosonic sector to Pauli.
        lam_data = compute_native_lambda(mh, n_b)
        physical_lambda = lam_data['physical_lambda']
        identity_shift = lam_data['identity_shift']

        print("Computing native-algebra Λ for sparse-oracle encoder (C1 stub)...")
        print(f"-> Identity (classical) shift:   {identity_shift:.4e}")
        print(f"-> Physical Lambda (total):      {physical_lambda:.4e}")
        for part, value in lam_data['per_part_lambdas'].items():
            share = (value / physical_lambda * 100.0) if physical_lambda else 0.0
            print(f"   - {part:>14}: λ = {value:.4e}  ({share:.2f}% of Λ)")

        # C2+ work: construct the BCK oracles (O_C, O_A, D_s), wrap as a
        # pyLIQTR BlockEncoding_select_prepare subclass, and run
        # estimate_resources. For now, fail loudly so callers can't
        # mistake an unimplemented encoder for a finished one.
        raise NotImplementedError(
            "sparse-oracle block-encoding circuit construction is not yet "
            "implemented. Λ has been computed from the native MixedHamiltonian "
            "(above). C2 of the refactor will land the actual oracle circuits; "
            "see tasks/26-sparse-oracle-fock.md and "
            "claude/research/block-encoders/04_refactor_execution_log.md (Phase C)."
        )
