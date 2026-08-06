"""_adaptive_ladder_solve's pt2_max_core gate. The EN-PT2 external space scales ~223x
core (measured L=3), materializing ~228M determinants / ~150 GB at 1M core -- it OOMs
long before the variational solve does. Past the cap the ladder must record the E_var
energy ONLY (dE_pt2=None), so deep runs reach 1M+ on E_var while PT2 stays on the
shallow rungs for the extrapolation. This guards that gate.

Run: python tests/test_pt2_cap.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.hamiltonian import build_from_eft
from classical.trimci.run_cpp import _adaptive_ladder_solve, _pick_solver


def main():
    H = build_from_eft(2, 3, 2)
    solver, pt2_diag, _ = _pick_solver(arrays=True)
    cap = 1200
    rungs = _adaptive_ladder_solve(H, 4, 500, 4, solver, pt2_diag, n_runs=2, seed=0,
                                   verbose=False, pt2_max_core=cap)
    fails = []
    if len(rungs) < 3:
        fails.append(f"expected several rungs, got {len(rungs)}")
    for r in rungs:
        if r["E_var"] is None:
            fails.append(f"core={r['core']} missing E_var")
        if r["core"] <= cap and r["dE_pt2"] is None:
            fails.append(f"core={r['core']} <= cap but PT2 skipped")
        if r["core"] > cap and r["dE_pt2"] is not None:
            fails.append(f"core={r['core']} > cap but PT2 computed (the OOM path)")

    # backward compat: no cap -> PT2 on every rung
    plain = _adaptive_ladder_solve(H, 4, 500, 2, solver, pt2_diag, n_runs=2, seed=0,
                                   verbose=False)
    if any(r["dE_pt2"] is None for r in plain):
        fails.append("pt2_max_core=None should compute PT2 on every rung")

    if fails:
        print("test_pt2_cap: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    capped = [r["core"] for r in rungs if r["dE_pt2"] is None]
    print(f"test_pt2_cap: PASS  (PT2 skipped above cap={cap} at cores {capped}; "
          f"E_var recorded on all {len(rungs)} rungs)")


if __name__ == "__main__":
    main()
