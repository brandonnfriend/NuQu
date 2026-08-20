"""Unit test for the total-T budget optimizer (audit issue 1) + pruning accounting (issue 2).

Pure math for the optimizer (no pyLIQTR); a tiny real estimate for the pruning one-norm.
Run: python -m tests.test_total_t_optimizer
"""
import math
import sys

from src_PI.estimation.total_t_optimizer import (
    fit_walk_t_vs_precision, total_t_at_fraction, optimize_qpe_fraction,
)


def main():
    fails = []

    # --- fit is exact for log-linear data ---------------------------------------
    a_true, b_true = 35448.0, 3082.5
    samples = [(cp, a_true + b_true * math.log2(1.0 / cp)) for cp in (1e-4, 1e-8, 1e-12)]
    a, b, resid = fit_walk_t_vs_precision(samples)
    if abs(a - a_true) > 1e-6 or abs(b - b_true) > 1e-6 or resid > 1e-6:
        fails.append(f"fit wrong: a={a}, b={b}, resid={resid}")

    # --- optimum is interior and beats the endpoints + a naive 50/50 ------------
    lam = 2086.6
    opt = optimize_qpe_fraction(a_true, b_true, lam, delta_E=1.0)
    f_star = opt['qpe_fraction']
    if not (0.02 < f_star < 0.98):
        fails.append(f"f* {f_star} not interior")
    # total_T(f*) <= total_T(f) for a grid of f
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        tt, _ = total_t_at_fraction(a_true, b_true, lam, 1.0, f)
        if opt['total_T'] > tt + 1e-3:
            fails.append(f"f* not optimal: total_T({f})={tt:.3e} < opt {opt['total_T']:.3e}")
    # budget adds up and N_walk uses eps_qpe (not full ΔE)
    if abs(opt['eps_qpe'] + opt['eps_be'] - 1.0) > 1e-9:
        fails.append("eps_qpe + eps_be != ΔE")
    if abs(opt['walk_queries'] - math.sqrt(2) * math.pi * lam / opt['eps_qpe']) > 1e-3:
        fails.append("walk_queries not √2π·λ/eps_qpe")
    if abs(opt['total_T'] - opt['walk_T'] * opt['walk_queries']) > 1e-3:
        fails.append("total_T != walk_T · walk_queries")

    # --- tightening ΔE raises N_walk and total_T --------------------------------
    opt_tight = optimize_qpe_fraction(a_true, b_true, lam, delta_E=0.1)
    if not (opt_tight['walk_queries'] > opt['walk_queries']):
        fails.append("tighter ΔE should raise N_walk")

    # --- pruning one-norm accounting (issue 2) on a tiny real bundle -------------
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
    from src_PI.utils.Config import Config
    cfg = Config(pion_basis='fock', block_encoder='pauli_lcu')
    bundle, _q, _ns = build_eft_hamiltonian(1, 3, 2, 0.0, get_physical_parameters(), cfg)
    nd = normalize_for_qpe(bundle, safety_factor=2.5)
    if 'pruned_one_norm_MeV' not in nd:
        fails.append("normalize_for_qpe must report pruned_one_norm_MeV")
    elif nd['pruned_one_norm_MeV'] < 0:
        fails.append("pruned one-norm must be >= 0")
    # n_b=2 is clean: the discarded one-norm is tiny vs a 0.025 MeV budget slice
    if nd.get('pruned_one_norm_MeV', 0.0) > 0.025:
        fails.append(f"n_b=2 should be clean, got {nd['pruned_one_norm_MeV']} MeV discarded")

    print("=" * 56)
    print("   TOTAL-T OPTIMIZER + PRUNING UNIT TEST")
    print("=" * 56)
    print(f"f*={f_star:.3f}  eps_qpe={opt['eps_qpe']:.3f}  total_T*={opt['total_T']:.4e}")
    print(f"pruned one-norm (L=1 n_b=2) = {nd.get('pruned_one_norm_MeV', 0.0):.3e} MeV "
          f"({nd.get('pruned_term_count')} terms)")
    print("=" * 56)
    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: total-T optimum is interior/optimal; ΔE monotonicity holds; pruning tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
