"""run_dmrg's on_chi callback (the HPC-shard incremental-save hook) must fire once
per bond dimension, in order, with each rung's (chi, E, S_max_bond). Fast: L=2 1D A=2
(block2's validated ~seconds case). Guards the hook the DMRG HPC workflow relies on to
keep partial results when a large-chi shard times out.

Run: python tests/test_dmrg_callback.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.baselines.dmrg_block2 import run_dmrg


def main():
    bond_dims = (20, 40, 80)
    seen = []
    res, _ = run_dmrg(2, 1, 2, N_f=2, n_b=1, bond_dims=bond_dims,
                      n_sweeps_per=3, on_chi=lambda r: seen.append(dict(r)))

    fails = []
    if [r["chi"] for r in seen] != list(bond_dims):
        fails.append(f"callback chis {[r['chi'] for r in seen]} != {list(bond_dims)}")
    if len(seen) != len(res) or any(seen[i]["E"] != res[i]["E"] for i in range(len(res))):
        fails.append("callback rungs disagree with returned list")
    for r in seen:
        if not {"chi", "E", "S_max_bond"} <= set(r):
            fails.append(f"rung missing keys: {r}")
    # DMRG energy is variational -> non-increasing as chi grows
    Es = [r["E"] for r in seen]
    if any(Es[i + 1] > Es[i] + 1e-6 for i in range(len(Es) - 1)):
        fails.append(f"energy not monotone non-increasing in chi: {Es}")

    # each rung now carries its own wall time
    if any("wall_s" not in r for r in seen):
        fails.append("rung missing wall_s")

    # per-chi TIME CAP: with a ~0 budget, any chi step exceeds it, so it must stop
    # after the FIRST chi (keeping that one) rather than run the whole schedule.
    capped, _ = run_dmrg(2, 1, 2, N_f=2, n_b=1, bond_dims=(20, 40, 80),
                         n_sweeps_per=3, max_chi_seconds=1e-9)
    if len(capped) != 1:
        fails.append(f"max_chi_seconds cap did not stop early: got {len(capped)} rungs")

    if fails:
        print("test_dmrg_callback: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_dmrg_callback: PASS  (callback fired {len(seen)}x, "
          f"E: {Es[0]:.3f} -> {Es[-1]:.3f}; time-cap stopped at {len(capped)} rung)")


if __name__ == "__main__":
    main()
