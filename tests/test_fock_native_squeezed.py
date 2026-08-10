"""Tests for the squeezed NATIVE Fock builder (Architecture B, sparse pipeline).

Fast + no pyLIQTR: (1) at r=0 the squeezed native MixedHamiltonian is identical to
`fock_native`'s (the collapsed local is recovered), and (2) `fock_squeezed` + sparse
dispatches to the native squeezed path. The end-to-end sparse resource estimate is
exercised by `misc/run_frame_AB_shard.py --encoder sparse` (needs pyLIQTR).

Run:  python -m tests.test_fock_native_squeezed
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.hamiltonians.core.MixedHamiltonian import MixedHamiltonian
from src_PI.hamiltonians.core.pion_basis import fock_native, fock_native_squeezed
from src_PI.utils.Config import Config


def _bosonop_max_diff(a, b):
    diff = a - b
    return max((abs(c) for c in diff.terms.values()), default=0.0)


def test_r_zero_identical_to_fock_native():
    """r=0 ⇒ squeezed native builder == bare fock_native (boson_part + mixed_terms)."""
    params = dict(get_physical_parameters())
    params['squeeze_r'] = 0.0
    L, dim, n_b = 2, 3, 3

    mh_sq = fock_native_squeezed.build_native_mixed_hamiltonian(L, dim, n_b, params)
    mh_ba = fock_native.build_native_mixed_hamiltonian(L, dim, n_b, params)

    d_bos = _bosonop_max_diff(mh_sq.boson_part, mh_ba.boson_part)
    assert d_bos < 1e-9, f"boson_part differs at r=0: max|Δ|={d_bos:.2e}"

    assert len(mh_sq.mixed_terms) == len(mh_ba.mixed_terms), "mixed_terms count differs"
    worst = 0.0
    for ts, tb in zip(mh_sq.mixed_terms, mh_ba.mixed_terms):
        assert abs(ts.coeff - tb.coeff) < 1e-9, "mixed_term coeff differs at r=0"
        worst = max(worst, _bosonop_max_diff(ts.boson_factor, tb.boson_factor))
        # fermion factors are reused by reference (unscaled) -> identical
        assert (ts.fermion_factor - tb.fermion_factor).terms == {} or \
            max(abs(c) for c in (ts.fermion_factor - tb.fermion_factor).terms.values()) < 1e-12
    assert worst < 1e-9
    print(f"[1] r=0 native squeezed == fock_native (boson Δ={d_bos:.1e}, "
          f"{len(mh_sq.mixed_terms)} mixed_terms match) OK")


def test_r_nonzero_squeeze_signature():
    """r≠0 ⇒ boson_part changes, and the mixed sector shows the exact squeeze
    signature: H_AV coeffs × e^{r}, H_WT invariant."""
    import numpy as np
    params = dict(get_physical_parameters())
    L, dim, n_b, r = 2, 3, 3, 0.21
    params['squeeze_r'] = 0.0
    mh0 = fock_native_squeezed.build_native_mixed_hamiltonian(L, dim, n_b, params)
    params['squeeze_r'] = r
    mhr = fock_native_squeezed.build_native_mixed_hamiltonian(L, dim, n_b, params)
    # boson_part changes (local weights rescale by cosh(2r), gradient by e^{2r},
    # local off-diagonal o switches on).
    assert _bosonop_max_diff(mhr.boson_part, mh0.boson_part) > 1e-3
    # mixed coeff ratios: H_AV -> e^{r}, H_WT -> 1 (both must be present).
    ratios = [ts.coeff / tb.coeff for ts, tb in zip(mhr.mixed_terms, mh0.mixed_terms)
              if abs(tb.coeff) > 0]
    er = float(np.exp(r))
    assert any(abs(x - er) < 1e-9 for x in ratios), "H_AV terms should scale by e^r"
    assert any(abs(x - 1.0) < 1e-9 for x in ratios), "H_WT terms should be invariant"
    print(f"[2] r≠0 squeeze signature: H_AV ×e^r={er:.3f}, H_WT invariant, boson changed OK")


def test_dispatch_fock_squeezed_sparse():
    """fock_squeezed + sparse -> native squeezed path, well-formed MixedHamiltonian."""
    params = dict(get_physical_parameters())
    params['squeeze_r'] = 0.21
    config = Config(pion_basis='fock_squeezed', block_encoder='sparse')
    bundle, _, _ = build_eft_hamiltonian(2, 3, 3, 0.0, params, config)
    sh = bundle.sub_hamiltonians[0]
    assert sh.algebra == 'fermion_boson'
    assert sh.name == 'fock_squeezed'
    assert isinstance(sh.operator, MixedHamiltonian)
    # static nucleon folded into fermion_part
    assert len(sh.operator.fermion_part.terms) > 0
    # pauli_lcu still routes fock_squeezed through the Pauli path (unchanged)
    from openfermion import QubitOperator
    cfg_p = Config(pion_basis='fock_squeezed', block_encoder='pauli_lcu')
    b2, _, _ = build_eft_hamiltonian(2, 3, 2, 0.0, params, cfg_p)
    assert isinstance(b2.sub_hamiltonians[0].operator, QubitOperator)
    print("[3] dispatch: fock_squeezed+sparse -> native; +pauli_lcu -> Pauli OK")


if __name__ == '__main__':
    test_r_zero_identical_to_fock_native()
    test_r_nonzero_squeeze_signature()
    test_dispatch_fock_squeezed_sparse()
    print("\nall fock_native_squeezed tests passed")
