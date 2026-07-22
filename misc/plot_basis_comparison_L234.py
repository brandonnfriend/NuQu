"""
Produce one basis-comparison plot per L in {2, 3, 4}, each overlaying:
  - amplitude/energy_bound  (legacy, Watson Lemma 5)
  - amplitude/ns            (Path B, Nyquist-Shannon optimal)
  - fock/sparse             (sparse-oracle block encoding)
  - Watson Trotterization baseline

The fock Pauli LCU path is deliberately excluded — runtime is too long for
the full A range at L ≥ 3.

Inputs are sweep JSONs already on disk (legacy energy_bound + freshly run
amp/ns + fock/sparse). Outputs land in data/<today>/basis_comparison_L{L}_3D.{png,json}.

Run from project root:
    source .venv/bin/activate
    python misc/plot_basis_comparison_L234.py
"""

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plot_sweep_data import plot_basis_comparison_total_qpe


# (L, [list of (label, glob)]) — globs let us pick the most-recent run if
# multiple exist; we sort and take the last (lexicographic timestamp order
# matches chronological order for the HHMMSS naming).
PER_L_INPUTS = {
    2: [
        ('amp/energy_bound (legacy)', 'data/2026-04-21/sweep_L2_3D_164215.json'),
        ('amp/ns',                    'data/2026-05-26/sweep_L2_3D_amplitude_ns_112706.json'),
        ('fock/sparse',               'data/2026-05-30/sweep_L2_3D_fock_*.json'),
    ],
    3: [
        ('amp/energy_bound (legacy, synth)', 'data/2026-04-04/sweep_L3_3D_amplitude_energy_bound_synth.json'),
        ('amp/ns',                           'data/2026-05-30/sweep_L3_3D_amplitude_ns_*.json'),
        ('fock/sparse',                      'data/2026-05-30/sweep_L3_3D_fock_*.json'),
    ],
    4: [
        ('amp/energy_bound (legacy)', 'data/2026-04-06/L4_3D.json'),
        ('amp/ns',                    'data/2026-05-30/sweep_L4_3D_amplitude_ns_*.json'),
        ('fock/sparse',               'data/2026-05-30/sweep_L4_3D_fock_*.json'),
    ],
}


def resolve_glob(pattern):
    """Return the lexically-last match (matches chronological order for
    HHMMSS-suffixed sweep files); None if no match."""
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main():
    for L, entries in PER_L_INPUTS.items():
        print(f'\n=== Building L={L} comparison ===')
        resolved = []
        for label, pat in entries:
            fp = resolve_glob(pat)
            if fp is None:
                print(f'  MISSING: {label}  (pattern {pat})')
                continue
            print(f'  {label:36s} -> {fp}')
            resolved.append(fp)

        if len(resolved) < 2:
            print(f'  Not enough sweep files for L={L} comparison; skipping.')
            continue

        # Trotter A-range: cover the same span as the qubitization data.
        # 1..100 inclusive matches the legacy datasets at every L.
        plot_basis_comparison_total_qpe(
            resolved,
            delta_E=1.0,
            e=0.1,
            E_kin=10,
            Cp=1e-3,
            trotter_A_range=(1, 100),
            save_basename=f'basis_comparison_total_qpe_L{L}_3D',
        )


if __name__ == '__main__':
    main()
