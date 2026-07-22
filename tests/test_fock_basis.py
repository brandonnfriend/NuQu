"""
Regression checks for the BosonOperator-front-end Fock pion-basis builder.

Frozen at (L=2, dim=2, n_b=2) against the pre-refactor reference: term counts
per sector, Hermiticity (no imaginary survivors after _drop_imag_noise), and
the diagonal-collapse identity m_π·(â†â + ½) for the free-local sector with
ω_0 = m_π. If any of these drift, the new front-end is no longer faithful to
the truncated operator.

Run from the project root:
    python -m tests.test_fock_basis
"""

import sys

from src_PI.hamiltonians.core.pion_basis import fock
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters


# Golden term counts captured against the pre-refactor build at this problem
# size; bit-exact equality verified once at refactor time.
_REF_TERM_COUNTS = {
    'H_free': 253,
    'H_av':   192,
    'H_wt':   1024,
}


def _max_imag(op):
    return max((abs(c.imag) if hasattr(c, 'imag') else 0.0
                for c in op.terms.values()), default=0.0)


def main():
    L, dim, n_b = 2, 2, 2
    params = get_physical_parameters()

    H_free = fock.H_pion_free(L, dim, n_b, params)
    H_av = fock.H_axial_vector(L, dim, n_b, params)
    H_wt = fock.H_WT_Logic(L, dim, n_b, params)

    sectors = {'H_free': H_free, 'H_av': H_av, 'H_wt': H_wt}
    print("=" * 50)
    print(f" FOCK-BASIS REGRESSION  (L={L}, dim={dim}, n_b={n_b})")
    print("=" * 50)

    failures = 0

    for name, op in sectors.items():
        n_terms = len(op.terms)
        expected = _REF_TERM_COUNTS[name]
        imag = _max_imag(op)
        ok_count = (n_terms == expected)
        ok_herm = (imag < 1e-9)
        status = "PASS" if (ok_count and ok_herm) else "FAIL"
        print(f"  {name:8s}  terms={n_terms:>5}  (ref={expected})  "
              f"max|Im|={imag:.2e}  {status}")
        if not ok_count:
            print(f"     -> term count drifted from frozen reference")
            failures += 1
        if not ok_herm:
            print(f"     -> imaginary survivors past _drop_imag_noise")
            failures += 1

    # Diagonal-collapse sanity: H_pion_free_local with ω_0 = m_π should equal
    # m_π·(n̂ + ½) summed over sites and species. We can verify the trace
    # cheaply: Tr(H_free_local) over the full register = m_π · Σ_x Σ_I (Σ_n n + ½·N_f)
    H_local = fock.H_pion_free_local(L, dim, n_b, params)
    identity_coeff = H_local.terms.get((), 0.0)
    expected_identity = (
        params['m_pi'] * (L ** dim) * 3 *
        ((sum(range(2 ** n_b)) / 2 ** n_b) + 0.5)
    )
    drift = abs(identity_coeff - expected_identity)
    ok_id = drift < 1e-6 * abs(expected_identity)
    status = "PASS" if ok_id else "FAIL"
    print(f"  H_local identity coeff = {identity_coeff:.4f}  "
          f"(expected {expected_identity:.4f})  {status}")
    if not ok_id:
        failures += 1

    print("=" * 50)
    if failures == 0:
        print(" ALL CHECKS PASSED")
        return 0
    print(f" {failures} CHECK(S) FAILED")
    return 1


if __name__ == '__main__':
    sys.exit(main())
