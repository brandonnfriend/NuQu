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


# Coarse cache keyed on (pos_n_qubits, mom_n_qubits). Both are determined by
# (L, dim, n_b), so the key is equivalent to the n_b-bucket. Within a bucket,
# pyLIQTR's T/Clifford/LogicalQubits vary by <0.15% across A — that variation
# is real (small) physics from near-cancellation bands, not noise — see
# NormalizeHamiltonians.py for the diagnostic story. We collapse it to the
# first-A representative because:
#   (a) the variation is well below resource-estimator uncertainty,
#   (b) the first-A in an increasing-A sweep is the smallest A in the bin,
#       which carries the *largest* Lambda (LCU norm) and therefore the
#       conservative-high QPE-depth cost (Lambda x T dominates),
#   (c) it converts 25 expensive estimator calls into ~5-7 across the L=2..4
#       sweeps, giving a ~3x sweep speedup.
#
# Set NUQU_DISABLE_PYLIQTR_CACHE=1 to bypass for verification runs.
_RESOURCE_CACHE = {}


def _estimate_with_cache(pos_instance, mom_instance):
    """Run pyLIQTR estimator, or reuse a cached result for the same n_b bin.

    Returns (pos_results, mom_results, pos_alpha, mom_alpha, cache_hit).
    `alpha` is the LCU norm sum |c_i|; on cache hit we read it directly from
    the MyCustomHamiltonian instance instead of rebuilding the (expensive)
    pyLIQTR PauliStringLCU encoding, which at L=3 was ~7s per cache hit just
    to recompute a value we already have.
    """
    cache_disabled = os.environ.get("NUQU_DISABLE_PYLIQTR_CACHE", "") == "1"
    key = None if cache_disabled else (pos_instance.n_qubits(), mom_instance.n_qubits())

    if key is not None and key in _RESOURCE_CACHE:
        cached = _RESOURCE_CACHE[key]
        pos_alpha = pos_instance.get_alpha()
        mom_alpha = mom_instance.get_alpha()
        return cached["pos_results"], cached["mom_results"], pos_alpha, mom_alpha, True

    encoding_type = VALID_ENCODINGS.PauliLCU
    pos_encoding = getEncoding(encoding_type)(pos_instance)
    mom_encoding = getEncoding(encoding_type)(mom_instance)

    pos_walk = QubitizedWalkOperator(pos_encoding)
    mom_walk = QubitizedWalkOperator(mom_encoding)

    pos_results = estimate_resources(pos_walk)
    mom_results = estimate_resources(mom_walk)

    if key is not None:
        _RESOURCE_CACHE[key] = {
            "pos_results": dict(pos_results),
            "mom_results": dict(mom_results),
        }
    return pos_results, mom_results, pos_encoding.alpha, mom_encoding.alpha, False


def run_qubitization_analysis(pos_ham, mom_ham, n_sites, n_qubits_per_site):
    """
    Analyzes resources for Qubitized Phase Estimation by splitting the
    Hamiltonian into position and momentum space walks.
    """
    # 1. Create instances for both Hamiltonians
    pos_instance = _ham_to_pyliqtr_instance(pos_ham)
    mom_instance = _ham_to_pyliqtr_instance(mom_ham)

    # 2–4. Estimate resources (coarse cache keyed on n_qubits ~ n_b bucket)
    pos_results, mom_results, pos_alpha, mom_alpha, cache_hit = (
        _estimate_with_cache(pos_instance, mom_instance)
    )

    # 5. Combine results with the correct per-key semantics for the split-oracle.
    # Gate counts (T, Clifford) are summed: one walk-step runs both encodings sequentially.
    # LogicalQubits is the peak hardware requirement: since the two walks reuse the same
    # hardware (system register + ancillas), the peak is max(pos, mom), not pos + mom.
    pos_lq = pos_results.get('LogicalQubits', 0)
    mom_lq = mom_results.get('LogicalQubits', 0)

    combined_results = {
        'T': pos_results.get('T', 0) + mom_results.get('T', 0),
        'Clifford': pos_results.get('Clifford', 0) + mom_results.get('Clifford', 0),
        'LogicalQubits': max(pos_lq, mom_lq),
        'Pos_LogicalQubits': pos_lq,
        'Mom_LogicalQubits': mom_lq,
    }

    # Pass through any other keys (e.g. 'Rotations' when profile=True) by summing.
    for key in set(pos_results.keys()).union(mom_results.keys()):
        if key not in combined_results:
            combined_results[key] = pos_results.get(key, 0) + mom_results.get(key, 0)

    # 6. Print the summary
    cache_tag = "  [n_b bin cache HIT]" if cache_hit else ""
    print("\n" + "=" * 50)
    print(f"      SPLIT ORACLE RESOURCE ESTIMATION{cache_tag}")
    print("=" * 50)
    print(f"System Qubits (Pos instance): {pos_instance.n_qubits()}")
    print(f"System Qubits (Mom instance): {mom_instance.n_qubits()}")
    print(f"Logical Qubits (Pos Walk):    {pos_lq}")
    print(f"Logical Qubits (Mom Walk):    {mom_lq}")
    print(f"Logical Qubits (peak, max):   {combined_results['LogicalQubits']}")
    print(f"Lambda (Pos Normalization):   {pos_alpha:.4f}")
    print(f"Lambda (Mom Normalization):   {mom_alpha:.4f}")
    print(f"Total Lambda (Alpha_pos + Alpha_mom): {(pos_alpha + mom_alpha):.4f}")
    print("-" * 50)

    for key, value in combined_results.items():
        label = key.replace('_', ' ').title()
        if isinstance(value, (int, float)) and value > 10000:
            print(f"{label:25}: {value:.4e}")
        else:
            print(f"{label:25}: {value}")
    print("=" * 50 + "\n")

    return combined_results
