import os

from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
from pyLIQTR.qubitization.qubitized_gates import QubitizedWalkOperator
from pyLIQTR.utils.resource_analysis import estimate_resources

from src_PI.estimation.instances import MyCustomHamiltonian


def _ham_to_pyliqtr_instance(qubit_ham):
    """Helper to convert OpenFermion QubitOperator to MyCustomHamiltonian."""
    pauli_dict = {}
    for term, coeff in qubit_ham.terms.items():
        p_string = " ".join([f"{op}{idx}" for idx, op in term]) if term else "I"
        pauli_dict[p_string] = float(coeff.real)
    return MyCustomHamiltonian(pauli_dict)


# Coarse cache keyed on (sub-Hamiltonian name, instance n_qubits). pyLIQTR's
# T/Clifford/LogicalQubits vary by <0.15% across A at fixed n_b (small physics
# from near-cancellation bands; see NormalizeHamiltonians.py for the diagnostic
# story). We collapse the per-A variation to the first-A representative because
# (a) it converts N expensive estimator calls into ~5–7 across L=2..4 sweeps,
# (b) the first-A in an increasing-A sweep carries the *largest* Λ and so the
# conservative-high QPE cost.
#
# Set NUQU_DISABLE_PYLIQTR_CACHE=1 to bypass for verification runs.
_RESOURCE_CACHE = {}


def _estimate_one(name, instance):
    """Estimate resources for a single sub-Hamiltonian.

    Returns (results_dict, alpha, cache_hit).
    """
    cache_disabled = os.environ.get("NUQU_DISABLE_PYLIQTR_CACHE", "") == "1"
    key = None if cache_disabled else (name, instance.n_qubits())

    if key is not None and key in _RESOURCE_CACHE:
        cached = _RESOURCE_CACHE[key]
        alpha = instance.get_alpha()
        return cached["results"], alpha, True

    encoding_type = VALID_ENCODINGS.PauliLCU
    encoding = getEncoding(encoding_type)(instance)
    walk = QubitizedWalkOperator(encoding)
    results = estimate_resources(walk)

    if key is not None:
        _RESOURCE_CACHE[key] = {"results": dict(results)}
    return results, encoding.alpha, False


def run_qubitization_analysis(norm_data, n_sites, n_qubits_per_site):
    """
    Estimate resources for every sub-Hamiltonian in the normalized bundle.

    The walk-mode in norm_data['walk_mode'] decides how per-sub qubit
    counts combine into a peak logical-qubit count:
        'series':   walks reuse the same hardware → peak = max(per-walk qubits)
        'parallel': walks run simultaneously       → peak = sum(per-walk qubits)
    Gate counts (T, Clifford) are summed in both cases.

    Returns a dict with:
        'T', 'Clifford': summed across sub-Hamiltonians.
        'LogicalQubits': peak (max or sum depending on walk_mode).
        'per_sub': list of {'name', 'T', 'Clifford', 'LogicalQubits', 'alpha', 'cache_hit'}.
    """
    sub_hamiltonians = norm_data['sub_hamiltonians']
    walk_mode = norm_data.get('walk_mode', 'series')

    per_sub = []
    total_T = 0
    total_clifford = 0
    other_summed = {}
    qubit_counts = []
    cache_hits = []

    for name, H_norm in sub_hamiltonians:
        instance = _ham_to_pyliqtr_instance(H_norm)
        results, alpha, hit = _estimate_one(name, instance)

        T_count = results.get('T', 0)
        clifford_count = results.get('Clifford', 0)
        lq = results.get('LogicalQubits', 0)

        per_sub.append({
            'name': name,
            'T': T_count,
            'Clifford': clifford_count,
            'LogicalQubits': lq,
            'alpha': alpha,
            'cache_hit': hit,
            'n_qubits': instance.n_qubits(),
        })
        total_T += T_count
        total_clifford += clifford_count
        qubit_counts.append(lq)
        cache_hits.append(hit)

        # Pass through any other keys (e.g. 'Rotations' when profile=True) by summing.
        for k, v in results.items():
            if k in ('T', 'Clifford', 'LogicalQubits'):
                continue
            other_summed[k] = other_summed.get(k, 0) + v

    if walk_mode == 'series':
        peak_qubits = max(qubit_counts) if qubit_counts else 0
    else:  # parallel
        peak_qubits = sum(qubit_counts)

    combined = {
        'T': total_T,
        'Clifford': total_clifford,
        'LogicalQubits': peak_qubits,
        'per_sub': per_sub,
    }
    combined.update(other_summed)

    # Print a summary suited to either the split-oracle or single-oracle case.
    n_walks = len(sub_hamiltonians)
    header = (
        f"      QUBITIZATION RESOURCE ESTIMATION  "
        f"(walks={n_walks}, mode={walk_mode})"
    )
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    for entry in per_sub:
        tag = "  [n_b bin cache HIT]" if entry['cache_hit'] else ""
        print(
            f"  walk={entry['name']:<10} qubits={entry['n_qubits']:<6} "
            f"LogicalQubits={entry['LogicalQubits']:<6} alpha={entry['alpha']:.4f}{tag}"
        )
    print("-" * 60)
    print(f"Logical Qubits (peak, {walk_mode}): {peak_qubits}")
    print(f"Total Lambda: {sum(e['alpha'] for e in per_sub):.4f}")
    print("-" * 60)
    for key, value in combined.items():
        if key == 'per_sub':
            continue
        label = key.replace('_', ' ').title()
        if isinstance(value, (int, float)) and value > 10000:
            print(f"{label:25}: {value:.4e}")
        else:
            print(f"{label:25}: {value}")
    print("=" * 60 + "\n")

    return combined
