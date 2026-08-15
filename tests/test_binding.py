"""binding.py -- fixed-A box-convergence binding-energy assembly + finite-volume fit.

Checks the BE formula (vacuum cancels), error propagation, box-size mapping, the exact
vacuum constant vs the Hamiltonian's own .constant(), and that a synthetic
exponentially-converging BE(box) is recovered by the finite-volume extrapolator.

Run: python tests/test_binding.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.binding import (binding_energy, binding_energy_sigma,
                                       box_convergence, box_size_fm,
                                       finite_volume_extrapolate, vacuum_constant)


def main():
    fails = []

    # 1. Vacuum cancels: add the SAME constant to every sector -> BE unchanged.
    E0, E1, EA, A = 5000.0, 5100.0, 5150.0, 2
    be = binding_energy(E0, E1, EA, A)                 # 2*5100 - 1*5000 - 5150 = 50
    if abs(be - 50.0) > 1e-9:
        fails.append(f"BE formula wrong: {be} != 50")
    shift = 43740.0
    be_shift = binding_energy(E0 + shift, E1 + shift, EA + shift, A)
    if abs(be_shift - be) > 1e-6:
        fails.append(f"vacuum constant did NOT cancel: {be_shift} != {be}")

    # 2. A bound system (EA below A free nucleons above vacuum) => BE > 0.
    #    E1-E0 = 100 (one nucleon above vacuum); 2 free = 200 above vacuum; bind to 150.
    if binding_energy(5000, 5100, 5150, 2) <= 0:
        fails.append("bound case should give BE>0")

    # 3. Error propagation.
    sig = binding_energy_sigma(1.0, 1.0, 1.0, 2)       # sqrt(4+1+1)=sqrt6
    if abs(sig - np.sqrt(6)) > 1e-9:
        fails.append(f"sigma prop wrong: {sig}")

    # 4. Box size + vacuum constant vs the actual Hamiltonian constant.
    if abs(box_size_fm(6) - 13.2) > 1e-9:
        fails.append("box_size_fm(6) should be 13.2 fm")
    if abs(vacuum_constant(3, dim=3) - 5467.5) > 1e-6:      # 202.5 * 27
        fails.append(f"vacuum_constant(3) {vacuum_constant(3)} != 5467.5")
    try:
        from classical.trimci.hamiltonian import build_from_eft
        c = build_from_eft(2, 3, 2).constant()
        if abs(c - vacuum_constant(2, dim=3)) > 1e-6:       # 202.5*8 = 1620
            fails.append(f"Hamiltonian constant {c} != vacuum_constant {vacuum_constant(2)}")
    except Exception as e:
        print(f"  (skipped Hamiltonian-constant cross-check: {e})")

    # 5. box_convergence assembles rows and skips incomplete boxes.
    se = {2: {0: (1000., .1), 1: (1100., .1), 2: (1150., .1)},
          3: {0: (2000., .1), 1: (2100., .1)},              # missing A=2 -> skipped
          4: {0: (3000., .1), 1: (3100., .1), 2: (3150., .1)}}
    rows = box_convergence(se, A=2)
    if [r["L"] for r in rows] != [2, 4]:
        fails.append(f"box_convergence should skip incomplete L=3: got {[r['L'] for r in rows]}")

    # 6. Finite-volume extrapolator recovers a known BE_inf from exp-converging data.
    BE_INF, C, K = 28.0, 40.0, 0.35
    boxes = [4.4, 6.6, 8.8, 11.0, 13.2]
    synth = [{"box_fm": b, "BE": BE_INF - C * np.exp(-K * b)} for b in boxes]
    fit = finite_volume_extrapolate(synth)
    if fit is None or abs(fit["BE_inf"] - BE_INF) > 0.5:
        fails.append(f"finite-volume extrapolation off: {fit}")

    if fails:
        print("test_binding: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_binding: PASS  (BE formula + vacuum-cancel + fv-extrap BE_inf="
          f"{fit['BE_inf']:.2f} [true {BE_INF}], r2={fit['r2']:.4f})")


if __name__ == "__main__":
    main()
