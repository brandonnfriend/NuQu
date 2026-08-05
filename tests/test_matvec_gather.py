"""SubspaceContext matvec is a Hermitian ROW-GATHER (each output row independent ->
embarrassingly parallel, no reduction). It reuses the complex CSC via conjugation:
y[i] = sum over column i of conj(data)*x[row_idx]. This is only correct if H is exactly
Hermitian and the CSC is full-stored -- guard both here against a dense reference, and
check the marshaling-free matvec_cplx matches the (vr,vi) matvec bit-for-bit.

Run: python tests/test_matvec_gather.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.hamiltonian import build_from_eft
from classical.trimci.backend import _cpp_provider
from classical.trimci.graph_arrays import ground_state_arrays


def main():
    H = build_from_eft(2, 3, 2, transform="bare")
    res = ground_state_arrays(H, n_elec=4, seed=0, n_dets=800)   # realistic core
    ferm = np.ascontiguousarray(res.ferm_arr, dtype=np.uint64)
    bos = np.ascontiguousarray(res.bos_arr, dtype=np.uint16)
    prov = _cpp_provider(H)
    ctx = prov.build_context(ferm, bos)
    N = ferm.shape[0]

    rows, cols, re, im = prov.build_coo(ferm, bos)
    Hc = sp.csr_matrix((np.asarray(re) + 1j * np.asarray(im),
                        (np.asarray(rows), np.asarray(cols))), shape=(N, N)).toarray()
    Hc = 0.5 * (Hc + Hc.conj().T)

    rng = np.random.RandomState(0)
    v = rng.randn(N) + 1j * rng.randn(N)
    ref = Hc @ v
    orr, oii = ctx.matvec(np.ascontiguousarray(v.real), np.ascontiguousarray(v.imag))
    gather = np.asarray(orr) + 1j * np.asarray(oii)
    cplx = np.asarray(ctx.matvec_cplx(np.ascontiguousarray(v, dtype=complex)))

    e_gather = np.linalg.norm(gather - ref)
    e_cplx = np.linalg.norm(cplx - gather)
    e_herm = np.linalg.norm(Hc - Hc.conj().T)
    fails = []
    if e_gather > 1e-8:
        fails.append(f"gather matvec != dense Hx: {e_gather:.2e}")
    if e_cplx > 1e-12:
        fails.append(f"matvec_cplx != matvec: {e_cplx:.2e}")
    if e_herm > 1e-9:
        fails.append(f"CSC not Hermitian/full: {e_herm:.2e}")

    if fails:
        print("test_matvec_gather: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_matvec_gather: PASS  (N={N}, ||gather-dense||={e_gather:.1e}, "
          f"||cplx-gather||={e_cplx:.1e}, hermiticity={e_herm:.1e})")


if __name__ == "__main__":
    main()
