"""Tests for the analytic bilinear JW statistics (task 34, I2 deeper fix).

The analytic one-body JW functionals must be bit-identical to what openfermion's
transform yields — both on synthetic operators and on the real mixed-term factors,
and end-to-end through `evaluate_resources`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openfermion import FermionOperator

from src_PI.estimation.EstimateResources import evaluate_resources
from src_PI.estimation.sparse_oracle.fermion_jw_stats import (
    bilinear_jw_stats, fermion_jw_stats, verify_against_openfermion,
)
from src_PI.estimation.sparse_oracle.jw_cache import clear_jw_cache
from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import (
    estimate_boson_cutoff, get_physical_parameters,
)
from src_PI.utils.Config import Config


def test_bilinear_stats_bit_exact_battery():
    """Both index orderings, Hermitian/non-Hermitian, adjacent/far, complex,
    collisions, and number operators — all bit-exact vs openfermion."""
    cases = [
        FermionOperator('0^ 5', 1.7),
        FermionOperator('5^ 0', 1.7),
        FermionOperator('3^ 8', -2.3),
        FermionOperator('0^ 5', 2.0) + FermionOperator('5^ 0', 2.0),      # Hermitian pair
        FermionOperator('1^ 2', 1.0) + FermionOperator('2^ 1', 1.0),      # adjacent (w=2)
        FermionOperator('0^ 40', 0.5) + FermionOperator('40^ 0', 0.5),    # long Z string
        FermionOperator('2^ 7', 1.0 + 0.5j) + FermionOperator('7^ 2', 1.0 - 0.5j),
        FermionOperator('3^ 3', 1.5),                                     # number op
        FermionOperator('1^ 1', 2.0) + FermionOperator('0^ 4', 1.0)
        + FermionOperator('4^ 0', 1.0),                                   # number + hopping
        FermionOperator('5^ 5', 1.0) + FermionOperator('5^ 5', 0.5),      # colliding numbers
    ]
    for op in cases:
        ok, analytic, ref = verify_against_openfermion(op)
        assert ok, f"mismatch: analytic={analytic} ref={ref}"


def test_quartic_falls_back():
    """A quartic (contact) term is not one-body -> analytic returns None -> the
    unified stats falls back to openfermion (still correct)."""
    quartic = FermionOperator('0^ 1^ 2 3', 1.0)
    assert bilinear_jw_stats(quartic) is None
    # unified path still returns sane, non-None stats via openfermion
    stats = fermion_jw_stats(quartic)
    assert stats['n_pauli_terms'] > 0


def test_real_mixed_factors_bit_exact():
    params = get_physical_parameters()
    config = Config(pion_basis='fock', block_encoder='sparse',
                    boson_cutoff_method='heuristic')
    for L in (2, 3):
        n_b, pi_max, _ = estimate_boson_cutoff(L, 3, 10, params, epsilon_cut=0.1,
                                               E_bound=100.0, boson_cutoff_method='heuristic')
        bundle, _, _ = build_eft_hamiltonian(L, 3, n_b, pi_max, params, config)
        mh = bundle.sub_hamiltonians[0].operator
        for mt in mh.mixed_terms:
            ok, analytic, ref = verify_against_openfermion(mt.fermion_factor)
            assert analytic is not None, "mixed factor should be one-body"
            assert ok, f"L={L} mismatch: analytic={analytic} ref={ref}"


def test_evaluate_resources_analytic_equals_openfermion():
    """End-to-end: the full resource numbers are identical with the analytic path
    on vs off (openfermion)."""
    params = get_physical_parameters()
    config = Config(pion_basis='fock', block_encoder='sparse',
                    boson_cutoff_method='heuristic')
    L, dim, A = 2, 3, 10
    n_b, pi_max, _ = estimate_boson_cutoff(L, dim, A, params, epsilon_cut=0.1,
                                           E_bound=10.0 * A, boson_cutoff_method='heuristic')
    keys = ('Physical_Lambda', 'Walk_T_Count', 'Walk_Clifford_Count', 'Logical_Qubits')
    prev = os.environ.get('NUQU_ANALYTIC_BILINEAR_JW')
    try:
        os.environ['NUQU_ANALYTIC_BILINEAR_JW'] = '0'
        clear_jw_cache()
        off = evaluate_resources(L, dim, n_b, pi_max, params, config)
        os.environ['NUQU_ANALYTIC_BILINEAR_JW'] = '1'
        clear_jw_cache()
        on = evaluate_resources(L, dim, n_b, pi_max, params, config)
    finally:
        if prev is None:
            os.environ.pop('NUQU_ANALYTIC_BILINEAR_JW', None)
        else:
            os.environ['NUQU_ANALYTIC_BILINEAR_JW'] = prev
    for k in keys:
        assert off[k] == on[k], f"analytic changed {k}: {off[k]} != {on[k]}"


if __name__ == '__main__':
    import io
    from contextlib import redirect_stdout
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    with redirect_stdout(io.StringIO()):
        for fn in fns:
            fn()
    print(f"test_fermion_jw_stats: {len(fns)} passed")
