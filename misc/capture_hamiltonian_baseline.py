"""
Hamiltonian-construction regression baseline.

Capture canonical-hashed Pauli expansions of every sub-Hamiltonian (pre- and
post-normalize) for the three baseline configurations we use as regression
targets during the block-encoder refactor. The Fock case additionally stores
the full sorted Pauli expansion, which doubles as a sparse-matrix reference
for the new-encoder classical-sim sanity checks (pseudocode step 3).

Output: data/<date>/hamiltonian_baseline.json.

Re-runnable; deterministic given the same code state.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# Run as a script from project root or anywhere — make sure the project
# root (parent of this file's directory) is on sys.path so `src_PI` imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import (
    calculate_dynamic_cutoffs,
    calculate_ns_cutoffs,
    estimate_boson_cutoff,
    get_physical_parameters,
)
from src_PI.utils.Config import Config


def _canonical_terms(qubit_op):
    """Sort QubitOperator terms canonically; return list of [term_list, real_hex, imag_hex]."""
    items = []
    for term, coeff in qubit_op.terms.items():
        term_key = tuple(sorted(term))
        c = complex(coeff)
        items.append((term_key, c.real.hex(), c.imag.hex()))
    items.sort(key=lambda x: (len(x[0]), x[0]))
    return [([list(p) for p in t], r, i) for t, r, i in items]


def _op_hash(canonical):
    sha = hashlib.sha256()
    for t, r, i in canonical:
        sha.update(repr(t).encode())
        sha.update(r.encode())
        sha.update(i.encode())
    return sha.hexdigest()


def _op_summary(qubit_op, include_full):
    canonical = _canonical_terms(qubit_op)
    out = {
        'term_count': len(qubit_op.terms),
        'max_weight': max((len(t) for t in qubit_op.terms), default=0),
        'sha256': _op_hash(canonical),
    }
    if include_full:
        out['canonical_terms'] = canonical
    return out


def _compute_cutoffs(L, dim, A, params, pion_basis, cutoff_method, eps, E_bound, n_b_override):
    if pion_basis == 'amplitude':
        if cutoff_method == 'ns':
            n_b, pi_max, Pi_max = calculate_ns_cutoffs(L, dim, A, params, epsilon_cut=eps, E_bound=E_bound)
        else:
            n_b, pi_max, Pi_max = calculate_dynamic_cutoffs(L, dim, A, params, epsilon_cut=eps, E_bound=E_bound)
    else:
        n_b, pi_max, Pi_max = estimate_boson_cutoff(L, dim, A, params, epsilon_cut=eps, E_bound=E_bound)
    if n_b_override is not None:
        n_b = int(n_b_override)
    return n_b, pi_max, Pi_max


def _capture_one(L, dim, A, pion_basis, cutoff_method, n_b_override, include_full):
    params = get_physical_parameters()
    eps = 0.1
    E_bound = 10.0 * A
    n_b, pi_max, _ = _compute_cutoffs(
        L, dim, A, params, pion_basis, cutoff_method, eps, E_bound, n_b_override
    )
    config = Config(pion_basis=pion_basis, cutoff_method=cutoff_method)

    print(f'  Building H: L={L}, dim={dim}, n_b={n_b}, A={A}, basis={pion_basis}, cutoff={cutoff_method}', flush=True)
    t0 = time.time()
    bundle, n_qubits, n_sites = build_eft_hamiltonian(L, dim, n_b, pi_max, params, config)
    t_build = time.time() - t0
    print(f'    construct: {t_build:.1f}s, sub-H: {bundle.names()}', flush=True)

    t0 = time.time()
    pre_subs = []
    for name, op in bundle.sub_hamiltonians:
        s = _op_summary(op, include_full=include_full)
        s['name'] = name
        pre_subs.append(s)
    t_hash_pre = time.time() - t0
    print(f'    pre-normalize hashes ({t_hash_pre:.1f}s):', flush=True)
    for s in pre_subs:
        print(f'      {s["name"]:>10}: terms={s["term_count"]:>7}, w_max={s["max_weight"]:>3}, sha={s["sha256"][:12]}', flush=True)

    t0 = time.time()
    norm = normalize_for_qpe(bundle, safety_factor=2.5)
    t_norm = time.time() - t0

    t0 = time.time()
    post_subs = []
    for name, op in norm['sub_hamiltonians']:
        s = _op_summary(op, include_full=include_full)
        s['name'] = name
        post_subs.append(s)
    t_hash_post = time.time() - t0
    print(f'    normalize: {t_norm:.2f}s; Λ={norm["physical_lambda"]:.3e}; post-hash {t_hash_post:.1f}s', flush=True)
    for s in post_subs:
        print(f'      {s["name"]:>10}: terms={s["term_count"]:>7}, w_max={s["max_weight"]:>3}, sha={s["sha256"][:12]}', flush=True)

    id_shift = norm['identity_shift']
    return {
        'L': L, 'dim': dim, 'A': A, 'n_b': n_b,
        'n_qubits': n_qubits, 'n_sites': n_sites,
        'identity_shift_real': float(id_shift.real if hasattr(id_shift, 'real') else id_shift),
        'identity_shift_imag': float(id_shift.imag if hasattr(id_shift, 'imag') else 0.0),
        'physical_lambda': float(norm['physical_lambda']),
        'delta': float(norm['delta']),
        'sub_lambdas': [(name, float(lam)) for name, lam in norm['sub_lambdas']],
        'sub_identity_shifts': [
            (name, float(s.real if hasattr(s, 'real') else s))
            for name, s in norm['sub_identity_shifts']
        ],
        'pre_normalize': pre_subs,
        'post_normalize': post_subs,
    }


def main():
    parser = argparse.ArgumentParser(description='Capture Hamiltonian-construction baselines.')
    parser.add_argument('--out', default='data/quantum/2026-05-26/hamiltonian_baseline.json',
                        help='Output path for the consolidated baseline JSON.')
    args = parser.parse_args()

    git_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()

    # (tag, basis, cutoff, n_b_override, A_values, include_full_terms)
    configs = [
        ('amplitude_energy_bound', 'amplitude', 'energy_bound', None, [1], False),
        ('amplitude_ns',           'amplitude', 'ns',           None, [1], False),
        ('fock_nb3',               'fock',      'energy_bound', 3,    [1], True),
    ]
    L, dim = 2, 3

    out = {
        'description': (
            'Hamiltonian-construction regression baseline for the block-encoder '
            'refactor. Each entry holds a canonical sha256 hash + (term_count, '
            'max_weight) per sub-Hamiltonian, both pre- and post-normalize. The '
            'Fock case additionally stores the full sorted Pauli expansion, '
            'used as the sparse-matrix reference for new-encoder classical-sim '
            'sanity checks (pseudocode step 3).'
        ),
        'pre_refactor_git_head': git_head,
        'generated_date': '2026-05-26',
        'configs': {},
    }

    overall_t0 = time.time()
    for tag, basis, cutoff, nb_override, A_values, include_full in configs:
        print(f'\n== {tag} ==', flush=True)
        out['configs'][tag] = {
            'pion_basis': basis,
            'cutoff_method': cutoff,
            'n_b_override': nb_override,
            'per_A': {},
        }
        for A in A_values:
            entry = _capture_one(L, dim, A, basis, cutoff, nb_override, include_full)
            out['configs'][tag]['per_A'][f'A_{A}'] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    sz_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f'\nWROTE {args.out} ({sz_mb:.1f} MB)')
    print(f'Pre-refactor HEAD: {git_head}')
    print(f'Total runtime: {time.time() - overall_t0:.1f}s')


if __name__ == '__main__':
    main()
