"""
Sparse-oracle (BCK) block-encoder strategy.

Sub-phase status (see `claude/research/block-encoders/04_refactor_execution_log.md`):
  * **C1**: scaffold + native-algebra Λ helper.
  * **C2**: single-mode `(â + â†)` BCK block encoding + classical-sim
    validation at n_b ∈ {2..5}.
  * **C3a**: pyLIQTR `BlockEncoding` wrap of the single-mode encoder.
  * **C3b**: Qualtran `AddK` for realistic shift T-cost.
  * **C3c**: `ProgrammableRotationGateArray` for QROM-loaded amplitude
    oracle. BCK Õ(log N_f) asymptotic established.
  * **C3d.1 (current)**: full-bundle analytical resource estimate.
    `SparseStrategy.estimate` walks every term in the `MixedHamiltonian`,
    uses the C3c single-mode walk-step cost as the per-mode atomic
    primitive, applies Gilyén Lemma 30 multiplicatively for multi-factor
    monomials and mixed fermion/boson terms, sums into an LCU with
    standard PREP/SELECT overhead, returns a complete resource dict.
    Does *not* yet build the unified Cirq circuit — that's C3d.2/C4.
  * **C3d.2+**: replace the analytical aggregator with a real composite
    pyLIQTR `BlockEncoding` for the full bundle.

Refer to `tasks/26-sparse-oracle-fock.md` and
`claude/research/block-encoders/01_pyliqtr_audit.md` §7c.
"""

from src_PI.estimation.sparse_oracle.lambda_compute import compute_native_lambda
from src_PI.estimation.sparse_oracle.resources import estimate_sparse_resources
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
from src_PI.hamiltonians.core.SubHamiltonian import SubHamiltonian


class SparseStrategy:
    name = 'sparse'

    def estimate(self, bundle, num_sites, n_b, config):
        # Validation (C1): bundle must be the native fermion-boson form.
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

        # Λ + identity shift from native algebra (C1).
        lam_data = compute_native_lambda(mh, n_b)
        physical_lambda = lam_data['physical_lambda']
        identity_shift = lam_data['identity_shift']

        print(f"--- Sparse-oracle strategy (C3d.1: analytical full-bundle estimate) ---")
        print(f"-> Identity (classical) shift:   {identity_shift:.4e}")
        print(f"-> Physical Lambda (total):      {physical_lambda:.4e}")
        for part, value in lam_data['per_part_lambdas'].items():
            share = (value / physical_lambda * 100.0) if physical_lambda else 0.0
            print(f"   - {part:>14}: λ = {value:.4e}  ({share:.2f}% of Λ)")

        # C3d.1 deliverable: full-bundle resource estimate.
        res = estimate_sparse_resources(mh, n_b, num_sites)
        bd = res['breakdown']
        print(f"-> LCU summand count (L_eff): {bd['L_eff']} "
              f"(boson={bd['boson_terms']}, fermion={bd['fermion_terms']}, "
              f"mixed={bd['mixed_terms']})")
        print(f"-> Single-mode walk T (n_b={n_b}, C3c): {bd['single_mode_walk_T']}")
        print(f"-> SELECT T  = {bd['select_T']:.4e}")
        print(f"-> PREP T    = {bd['prep_T']:.4e}")
        print(f"-> Walk T (2·PREP + SELECT) = {res['Walk_T_Count']:.4e}")
        print(f"-> Logical qubits           = {res['Logical_Qubits']}")

        # Build a `norm_data`-shaped dict the orchestrator expects.
        return {
            'sub_hamiltonians': [(sub[0].name, mh)],
            'sub_lambdas': [(sub[0].name, physical_lambda)],
            'sub_identity_shifts': [(sub[0].name, identity_shift)],
            'identity_shift': identity_shift,
            'physical_lambda': physical_lambda,
            'delta': 0.0,                          # no Δ for sparse path (no PauliLCU normalize)
            'walk_mode': bundle.walk_mode,
            'Walk_T_Count': res['Walk_T_Count'],
            'Walk_Clifford_Count': res['Walk_Clifford_Count'],
            'Logical_Qubits': res['Logical_Qubits'],
            'Physical_Lambda': physical_lambda,
            'Per_Sub_Walk': [{
                'name': sub[0].name,
                'T': res['Walk_T_Count'],
                'Clifford': res['Walk_Clifford_Count'],
                'LogicalQubits': res['Logical_Qubits'],
                'alpha': physical_lambda,
            }],
            'Sparse_Breakdown': res['breakdown'],   # diagnostic for downstream plots
        }
