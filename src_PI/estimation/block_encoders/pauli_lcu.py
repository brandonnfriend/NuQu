"""
PauliLCU strategy: the current pyLIQTR-PauliLCU pipeline.

Wraps `normalize_for_qpe` + `run_qubitization_analysis` behind the
`BlockEncoderStrategy` protocol so the orchestrator (`evaluate_resources`)
can dispatch on `config.block_encoder` without basis-specific knowledge.

For Phase A this is a pure relocation — every print statement and every
return-dict key matches the pre-Phase-A code exactly so the regression
hashes pass.
"""

from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
from src_PI.estimation.estimators import run_qubitization_analysis


class PauliLCUStrategy:
    name = 'pauli_lcu'
    safety_factor = 2.5

    def estimate(self, bundle, num_sites, n_b, config):
        # 1. Normalize.
        print("Normalizing Hamiltonians for QPE...")
        norm_data = normalize_for_qpe(bundle, safety_factor=self.safety_factor)

        id_shift = norm_data['identity_shift']
        id_real = id_shift.real if hasattr(id_shift, 'real') else id_shift
        print(f"-> Extracted classical energy shift: {id_real:.4e}")
        print(f"-> Physical Lambda (total):          {norm_data['physical_lambda']:.4e}")
        print(f"-> Spectral Delta (Scaling factor):  {norm_data['delta']:.4e}")
        for sub_name, lam in norm_data['sub_lambdas']:
            share = (lam / norm_data['physical_lambda'] * 100.0) if norm_data['physical_lambda'] else 0.0
            print(f"   - sub '{sub_name}': Λ = {lam:.4e}  ({share:.2f}% of total)")

        # 2. Combined Pauli stats (diagnostic).
        num_terms = 0
        weights = []
        for _, H_norm in norm_data['sub_hamiltonians']:
            num_terms += len(H_norm.terms)
            weights.extend(len(t) for t in H_norm.terms)
        max_w = max(weights) if weights else 0
        print("\n" + "=" * 45)
        print(f"Total Pauli Strings (Non-Identity): {num_terms}")
        print(f"Maximum Pauli Weight:               {max_w}")
        print("=" * 45)

        # 3. Walk-operator resource estimation via pyLIQTR.
        print("Starting pyLIQTR analysis...")
        liqtr_results = run_qubitization_analysis(norm_data, num_sites, n_b)
        if not isinstance(liqtr_results, dict):
            liqtr_results = {}

        # 4. Merge walk-step fields into the return dict (legacy key names).
        base_t = liqtr_results.get('T', 0)
        base_clifford = liqtr_results.get('Clifford', 0)
        logical_qubits = liqtr_results.get('LogicalQubits', 0)
        per_sub = liqtr_results.get('per_sub', [])

        norm_data['Walk_T_Count'] = base_t
        norm_data['Walk_Clifford_Count'] = base_clifford
        norm_data['Logical_Qubits'] = logical_qubits
        norm_data['Physical_Lambda'] = norm_data['physical_lambda']
        norm_data['Per_Sub_Walk'] = [
            {
                'name': e['name'],
                'T': e['T'],
                'Clifford': e['Clifford'],
                'LogicalQubits': e['LogicalQubits'],
                'alpha': e['alpha'],
            }
            for e in per_sub
        ]
        return norm_data
