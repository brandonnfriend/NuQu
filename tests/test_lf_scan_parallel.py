"""frame.optimize_displacement — the LF scale-scan is now a FLAT fork over
(scale x ensemble-seed) tasks instead of a serial loop with a forked ensemble.

The parallelization MUST be a pure wall-clock win: taking the min energy over a
scale's seeds reproduces the old ensemble-best exactly (same seeds -> same random
cores -> same energies), so serial (NUQU_NUM_WORKERS=1) and parallel (>1) must return
BIT-IDENTICAL results. This guards that invariant.

Run: python tests/test_lf_scan_parallel.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci.hamiltonian import build_from_eft
from classical.trimci import frame


def _run(nw, H, A):
    os.environ["NUQU_NUM_WORKERS"] = str(nw)
    return frame.optimize_displacement(H, n_elec=A, core=600, n_runs=3, seed=0)


def main():
    H = build_from_eft(2, 3, 2, transform="bare")   # L=2, d=3, n_b=2: 8 sites
    A = 5

    ser = _run(1, H, A)          # serial scan
    par = _run(4, H, A)          # flat fork over 17*3 = 51 tasks

    fails = []
    for k in ("scale", "energy", "n_dets", "bare_energy"):
        if ser[k] != par[k]:
            fails.append(f"{k}: serial={ser[k]!r} parallel={par[k]!r}")
    if len(ser["scan"]) != len(par["scan"]) or any(
            a != b for a, b in zip(ser["scan"], par["scan"])):
        fails.append("scan lists differ")
    if ser["gen"].keys() != par["gen"].keys():
        fails.append("generator keys differ")

    # sanity: a real scan happened (bare + the non-zero scales), and the winning frame
    # is at least as good as bare (selected-CI is variational at fixed core).
    if len(ser["scan"]) != 17:
        fails.append(f"expected 17 scan points (bare + 16 scales), got {len(ser['scan'])}")
    if ser["energy"] > ser["bare_energy"] + 1e-9:
        fails.append(f"best E {ser['energy']} worse than bare {ser['bare_energy']}")

    if fails:
        print("test_lf_scan_parallel: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_lf_scan_parallel: PASS  "
          f"(serial==parallel; best_scale={ser['scale']:+.2f}, "
          f"E={ser['energy']:.4f} <= bare {ser['bare_energy']:.4f})")


if __name__ == "__main__":
    main()
