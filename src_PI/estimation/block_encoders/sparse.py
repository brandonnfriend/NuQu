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
from src_PI.estimation.sparse_oracle.hermitian_bundle import (
    estimate_hermitian_sparse_resources,
)
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

        # Λ + identity shift from native algebra. The identity (classical) shift
        # is Hermitization-invariant; the analytical path's Λ is the per-monomial
        # 1-norm, the compiled (Hermitian) path's Λ is the tighter edge-coloured α.
        lam_data = compute_native_lambda(mh, n_b)
        physical_lambda = lam_data['physical_lambda']
        identity_shift = lam_data['identity_shift']

        # Modes (sparse_oracle_mode):
        #   'analytical'           — mixed-bound Gilyén+LCU proxy (default, A/B baseline);
        #   'hermitian_cost_model' — walk-valid Hermitian construction, hand-assembled
        #                            cost (optimistic; omits synthesis/controls);
        #   'compiled'             — GENUINELY compiled walk: every rotation synthesized
        #                            to Clifford+T at the ΔE-derived precision, scalable
        #                            primitives (AddK/alias/PauliLCU, no dense MatrixGate),
        #                            full physical Hamiltonian incl. mixed atoms
        #                            (compiled_resources; Codex follow-up 2026-08-19).
        mode = getattr(config, 'sparse_oracle_mode', 'analytical')

        if mode == 'compiled':
            from src_PI.estimation.sparse_oracle.compiled_resources import (
                compiled_walk_resources,
            )
            delta_E = float(getattr(config, 'extras', {}).get('delta_E_MeV', 1.0))
            print("--- Sparse-oracle (COMPILED walk — rotations synthesized, ΔE-wired) ---")
            print(f"-> Identity (classical) shift:   {identity_shift:.4e}")
            res = compiled_walk_resources(mh, n_b, num_sites, delta_E=delta_E)
            physical_lambda = res['Physical_Lambda']
            bd = res['breakdown']
            print(f"-> Physical Lambda (Hermitian, tighter): {physical_lambda:.4e}")
            print(f"-> Atoms: {res['n_atoms']}   ΔE={delta_E} MeV   "
                  f"circ_prec={res['budget']['circuit_precision']:.2e}")
            print(f"-> Walk T (compiled, synthesized) = {res['Walk_T_Count']:.4e}  "
                  f"(PREP {bd['prep_outer_T']}, dispatch {bd['dispatch_T']}, "
                  f"SELECT {bd['select_T']}, reflection {bd['reflection_T']})")
            print(f"-> Logical qubits = {res['Logical_Qubits']}")
            walk_T = res['Walk_T_Count']
            walk_Cl = res['Walk_Clifford_Count']
            logical_q = res['Logical_Qubits']
            breakdown = {'n_atoms': res['n_atoms'], 'compiled': True,
                         'compiler_derived': True, 'budget': res['budget']}
        elif mode == 'hermitian_cost_model':
            print("--- Sparse-oracle (Hermitian COST MODEL — not compiler-derived) ---")
            print(f"-> Identity (classical) shift:   {identity_shift:.4e}")
            res = estimate_hermitian_sparse_resources(mh, n_b, num_sites)
            physical_lambda = res['Physical_Lambda']         # tighter Hermitian Λ
            print(f"-> Physical Lambda (Hermitian, tighter): {physical_lambda:.4e}")
            print(f"-> Atoms (mode-set grouped): {res['n_atoms']}")
            print(f"-> Walk T (COST MODEL, optimistic; omits control/matching/"
                  f"boundary/phase/precision) = {res['Walk_T_Count']:.4e}")
            print(f"-> Logical qubits (estimate) = {res['Logical_Qubits']}")
            walk_T = res['Walk_T_Count']
            walk_Cl = res['Walk_Clifford_Count']
            logical_q = res['Logical_Qubits']
            breakdown = {'n_atoms': res['n_atoms'], 'hermitian_cost_model': True,
                         'compiler_derived': False}
        else:
            print("--- Sparse-oracle strategy (analytical mixed-bound proxy) ---")
            print(f"-> Identity (classical) shift:   {identity_shift:.4e}")
            print(f"-> Physical Lambda (total):      {physical_lambda:.4e}")
            for part, value in lam_data['per_part_lambdas'].items():
                share = (value / physical_lambda * 100.0) if physical_lambda else 0.0
                print(f"   - {part:>14}: λ = {value:.4e}  ({share:.2f}% of Λ)")
            res = estimate_sparse_resources(mh, n_b, num_sites)
            bd = res['breakdown']
            print(f"-> LCU summand count (L_eff): {bd['L_eff']} "
                  f"(boson={bd['boson_terms']}, fermion={bd['fermion_terms']}, "
                  f"mixed={bd['mixed_terms']})")
            print(f"-> Walk T (2·PREP + SELECT) = {res['Walk_T_Count']:.4e}")
            print(f"-> Logical qubits           = {res['Logical_Qubits']}")
            walk_T = res['Walk_T_Count']
            walk_Cl = res['Walk_Clifford_Count']
            logical_q = res['Logical_Qubits']
            breakdown = bd

        # Build a `norm_data`-shaped dict the orchestrator expects.
        return {
            'sub_hamiltonians': [(sub[0].name, mh)],
            'sub_lambdas': [(sub[0].name, physical_lambda)],
            'sub_identity_shifts': [(sub[0].name, identity_shift)],
            'identity_shift': identity_shift,
            'physical_lambda': physical_lambda,
            'delta': 0.0,                          # no Δ for sparse path (no PauliLCU normalize)
            'walk_mode': bundle.walk_mode,
            'sparse_oracle_mode': mode,
            'Walk_T_Count': walk_T,
            'Walk_Clifford_Count': walk_Cl,
            'Logical_Qubits': logical_q,
            'Physical_Lambda': physical_lambda,
            'Per_Sub_Walk': [{
                'name': sub[0].name,
                'T': walk_T,
                'Clifford': walk_Cl,
                'LogicalQubits': logical_q,
                'alpha': physical_lambda,
            }],
            'Sparse_Breakdown': breakdown,   # diagnostic for downstream plots
        }
