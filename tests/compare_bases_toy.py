"""
Toy A/B comparison: Fock basis vs. amplitude basis.

Two questions, one tiny lattice (dim=1, L=2, A=2 by default):
  1. How many Pauli strings does the full EFT Hamiltonian carry in each
     basis, as a function of the per-species register width n_b?
  2. How long does the pyLIQTR resource estimation take (wall time)?

The expected story (see PROJECT_CONTEXT.md, "lambda audit" / bosonic
encodings): the amplitude basis keeps π̂, Π̂ *diagonal* in their respective
registers, so the operators are sums of I/Z/ZZ strings — polynomial in n_b.
The Fock basis expands each ladder operator |i⟩⟨j| as a dense product over
n_b qubits (X/Y/Z mix), so the Pauli count grows exponentially in n_b.

Bounded by construction: small n_b ranges only. Pauli counting is cheap and
swept wide; the timed pyLIQTR runs use a short, explicit list of points.
"""

import time

from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    get_physical_parameters,
)
from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
from src_PI.estimation.estimators import run_qubitization_analysis
from src_PI.utils.Config import Config


# --- Toy-lattice knobs (keep tiny) ----------------------------------------
L = 2
DIM = 1
A = 2

# n_b sweeps: the amplitude basis tolerates wide registers cheaply; the Fock
# basis blows up fast, so only a handful of bits.
AMP_NB = [4, 6, 8, 10, 12, 16, 20]
FOCK_NB = [1, 2, 3, 4, 5]

# pyLIQTR wall-time is the expensive part. Time only these (basis, n_b) points.
TIMED_POINTS = [
    ('amplitude', 4),
    ('amplitude', 10),
    ('amplitude', 20),
    ('fock', 2),
    ('fock', 3),
    ('fock', 4),
    ('fock', 5),
]


def _bundle_term_count(basis, n_b, pi_max, params):
    """Build the full EFT bundle in `basis` and return (build_seconds,
    total_terms, per_sub_terms, total_qubits)."""
    config = Config(pion_basis=basis)
    t0 = time.time()
    bundle, q_count, _ = build_eft_hamiltonian(L, DIM, n_b, pi_max, params, config)
    build_s = time.time() - t0
    per_sub = [(name, len(op.terms)) for name, op in bundle]
    total = sum(n for _, n in per_sub)
    return build_s, total, per_sub, q_count


def pauli_count_sweep(params, pi_max):
    print("=" * 72)
    print(f" PAULI-STRING COUNTS   (lattice: {L}^{DIM} = {L**DIM} sites, A={A})")
    print("=" * 72)
    print(f"{'basis':<10} {'n_b':>4} {'qubits':>7} {'#Pauli (raw bundle)':>22} "
          f"{'build s':>9}")
    print("-" * 72)

    results = {'amplitude': [], 'fock': []}
    for basis, nb_list in (('amplitude', AMP_NB), ('fock', FOCK_NB)):
        for n_b in nb_list:
            build_s, total, per_sub, q_count = _bundle_term_count(
                basis, n_b, pi_max, params
            )
            results[basis].append((n_b, total, q_count, build_s))
            sub_str = ", ".join(f"{name}={n}" for name, n in per_sub)
            print(f"{basis:<10} {n_b:>4} {q_count:>7} {total:>22,} {build_s:>9.3f}   "
                  f"[{sub_str}]")
        print("-" * 72)
    return results


def timed_estimation(params, pi_max):
    print("\n" + "=" * 72)
    print(" pyLIQTR RESOURCE-ESTIMATION WALL TIME")
    print("=" * 72)
    print(f"{'basis':<10} {'n_b':>4} {'#Pauli':>10} {'estimate s':>12} "
          f"{'T-count':>14} {'logical q':>10}")
    print("-" * 72)

    rows = []
    for basis, n_b in TIMED_POINTS:
        config = Config(pion_basis=basis)
        bundle, _, num_sites = build_eft_hamiltonian(
            L, DIM, n_b, pi_max, params, config
        )
        norm = normalize_for_qpe(bundle, safety_factor=2.5)
        # Drop any sub-Hamiltonian that normalized to zero terms (a toy-scale
        # tolerance artifact at small n_b; see header). pyLIQTR can't encode
        # an empty operator (log2(0) → -inf).
        norm['sub_hamiltonians'] = [
            (nm, H) for nm, H in norm['sub_hamiltonians'] if len(H.terms) > 0
        ]
        n_pauli = sum(len(H.terms) for _, H in norm['sub_hamiltonians'])

        t0 = time.time()
        liqtr = run_qubitization_analysis(norm, num_sites, n_b)
        est_s = time.time() - t0

        t_count = liqtr.get('T', 0)
        logical_q = liqtr.get('LogicalQubits', 0)
        rows.append((basis, n_b, n_pauli, est_s, t_count, logical_q))
        print(f"{basis:<10} {n_b:>4} {n_pauli:>10,} {est_s:>12.2f} "
              f"{t_count:>14,} {logical_q:>10}")
    print("-" * 72)
    return rows


def main():
    params = get_physical_parameters()
    # pi_max only scales amplitude-basis coefficients (Λ); it does not change
    # the Pauli-string *structure*, and Fock ignores it. Compute one physical
    # value and reuse it everywhere.
    _, pi_max, _ = calculate_dynamic_cutoffs(L, DIM, A, params)
    print(f"[setup] physical pi_max = {pi_max:.4f} (amplitude coeff scale only)\n")

    pauli_count_sweep(params, pi_max)
    timed_estimation(params, pi_max)


if __name__ == '__main__':
    main()
