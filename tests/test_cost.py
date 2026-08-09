"""classical.trimci.cost -- the Goal-3 cost analysis (Tier-1 core*, Tier-2 support).
Pure analysis on synthetic rungs (no solves), so fast + deterministic.

Run: python tests/test_cost.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.trimci import cost


def main():
    fails = []
    E_exact = 100.0
    sites = 8
    # E_var - E_exact = 40 * core^-0.5 : gap = 4 at 100, 2 at 400, 1 at 1600, 0.5 at 6400
    rungs = []
    for c, gap in [(100, 4.0), (400, 2.0), (1600, 1.0), (6400, 0.5)]:
        rungs.append({"core": c, "E_var": E_exact + gap,
                      "support": {"n90": 5, "n99": 20, "n999": 60, "n9999": 180,
                                  "participation_ratio": 3.0}})

    # Tier-1: core* for dE=1 should land ~1600 (exactly on a rung); dE=4 -> ~100
    c1, ok1 = cost.core_star(rungs, E_exact, 1.0)
    if not (ok1 and abs(c1 - 1600) < 50):
        fails.append(f"core*(1.0) = {c1} (expected ~1600)")
    c4, ok4 = cost.core_star(rungs, E_exact, 4.0)
    if not (ok4 and abs(c4 - 100) < 5):
        fails.append(f"core*(4.0) = {c4} (expected ~100)")
    # dE below the deepest gap (0.5) -> not reached -> lower bound flag
    c_lb, ok_lb = cost.core_star(rungs, E_exact, 0.1)
    if ok_lb or c_lb != 6400:
        fails.append(f"core*(0.1) should be a lower bound at 6400, got {c_lb}, reached={ok_lb}")

    # interpolation: dE=1.5 is between core 400 (gap 2) and 1600 (gap 1) -> in (400,1600)
    c15, ok15 = cost.core_star(rungs, E_exact, 1.5)
    if not (ok15 and 400 < c15 < 1600):
        fails.append(f"core*(1.5) = {c15} (expected between 400 and 1600)")

    # per-site: dE/site = 0.5 -> total 4.0 -> ~100
    t1 = cost.tier1_costs(rungs, E_exact, sites, dEs=(0.5,))
    if abs(t1["per_site"][0.5]["core_star"] - 100) > 5:
        fails.append(f"per-site core* wrong: {t1['per_site'][0.5]}")

    # Tier-2: support flat across rungs -> converged; weight exponent from n(1-delta)
    conv, val = cost.support_converged(rungs, "n999")
    if not (conv and val == 60):
        fails.append(f"support_converged wrong: {conv}, {val}")
    we = cost.support_weight_exponent(rungs[-1])
    if we is None or not (0.3 < we["gamma"] < 1.2):
        fails.append(f"weight exponent unreasonable: {we}")
    svc = cost.support_vs_core(rungs)["n999"]
    if svc[-1] != (6400, 60):
        fails.append(f"support_vs_core wrong: {svc}")

    if fails:
        print("test_cost: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_cost: PASS  (Tier-1 core*: 100/1600/LB@6400; interp ok; "
          f"Tier-2 n999 converged=60, gamma={we['gamma']:.2f})")


if __name__ == "__main__":
    main()
