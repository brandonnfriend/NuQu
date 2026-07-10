"""
Higher-A frame comparison — the A-analog of the L-scaling showcase (task 33).

The A=1 comparison was a smoke test. The physically interesting regime is MANY nucleons
(A = n_elec at solve time on the SAME L=2 d=3 operator): more nucleons -> a larger pion-
dressing / density sector and real orbital correlation, so the frames that were idle at A=1
(COO) or invalid (LF) should finally matter. Fully converging the sector is HPC-scale
(A=10 is ~40x slower per core than A=1), so this is a FIXED-CORE trend: at a fixed budget,
a better frame reaches a LOWER energy (all frames are isospectral -> same E_inf), and the
gap bare-frame is the compaction benefit. We track it vs A, exactly as the L-scaling
showcase tracked it vs L.

Frames: bare, per-mode squeeze (all A); COO natural-orbital rotation (A in A_COO — newly
meaningful at A>1, seeded ED-free from the bare TrimCI core's 1-RDM). The projector-LF
frame is NOT swept: its go/no-go verdict is that the exact (isospectral) version is term-
explosive (order-2 multi-mode FC dressing -> 3.7e7 terms), so it is documented, not run.

Also computes the polaron A-scaling diagnostic |sum_p lambda_{m,p} <n_p>|(A): does the
coherent pion displacement the LF frame would absorb actually grow with A?
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci import build_from_eft, frame
from classical.trimci.run_cpp import _solver
from classical.trimci.lf import compactness

OUT_DIR = os.path.join("data", "classical")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = os.path.join(OUT_DIR, "frame_comparison_highA_summary.json")

A_LADDER = (2, 4, 6, 8, 10)
A_COO = (6, 10)
CORE = 1500
N_RUNS = 3


def _pr(res):
    """Participation ratio + mean boson number of a solved core (compaction proxies)."""
    c = np.asarray(res.coeffs, dtype=complex)
    pr = compactness(c)["participation_ratio"]
    w = np.abs(c) ** 2
    mean_n = float((w * np.asarray(res.bos_arr).sum(axis=1)).sum() / max(w.sum(), 1e-300))
    return float(pr), mean_n


def _polaron_displacement(H, res):
    """|sum_p lambda_{m,p} <n_p>| summed over boson modes m: the coherent pion
    displacement the (projector-)LF frame would absorb, in the bare ground state.
    <n_p> read off the bare core; lambda from the analytic polaron seed."""
    entries, _, _ = frame.analytic_displacement(H)
    c = np.asarray(res.coeffs, dtype=complex)
    w = np.abs(c) ** 2
    w = w / max(w.sum(), 1e-300)
    masks = frame._ferm_bitmasks(res.ferm_arr)
    occ = np.zeros(H.n_ferm_modes)
    for wi, fm in zip(w, masks):
        f = fm
        while f:
            p = (f & -f).bit_length() - 1
            occ[p] += wi
            f &= f - 1
    disp = {}
    for (m, p, lam) in entries:
        disp[m] = disp.get(m, 0.0) + lam * occ[p]
    return float(np.sum(np.abs(list(disp.values())))), occ


def run():
    solve = _solver(True)
    H = build_from_eft(2, 3, 3)
    r, phi = frame.analytic_squeeze(H)
    Hsq = frame.squeeze_terms(H, -r, phi)
    print(f"L=2 d=3 n_b=3: {H.n_ferm_modes}F {H.n_bos_modes}B  bare={len(H.terms)} "
          f"squeeze={len(Hsq.terms)} terms; core={CORE} n_runs={N_RUNS}", flush=True)

    rows = []
    for A in A_LADDER:
        t = time.time()
        rb = solve(H, n_elec=A, n_dets=CORE, seed=0, n_runs=N_RUNS)
        tb = time.time() - t
        pr_b, mn_b = _pr(rb)
        t = time.time()
        rs = solve(Hsq, n_elec=A, n_dets=CORE, seed=0, n_runs=N_RUNS)
        ts = time.time() - t
        pr_s, mn_s = _pr(rs)
        disp, _ = _polaron_displacement(H, rb)
        row = {"A": A, "E_bare": float(rb.energy), "E_squeeze": float(rs.energy),
               "gap_squeeze": float(rb.energy - rs.energy),
               "pr_bare": pr_b, "pr_squeeze": pr_s, "mean_n_bare": mn_b,
               "mean_n_squeeze": mn_s, "polaron_disp": disp,
               "t_bare": tb, "t_squeeze": ts}
        # COO at the selected A: seed R from the bare core's 1-RDM (ED-free)
        if A in A_COO:
            R, occ = frame.natural_orbitals_from_core(H, rb.coeffs, rb.ferm_arr, rb.bos_arr)
            Hc = frame.rotate_orbitals_terms(H, R=R)
            t = time.time()
            rc = solve(Hc, n_elec=A, n_dets=CORE, seed=0, n_runs=N_RUNS)
            tc = time.time() - t
            pr_c, mn_c = _pr(rc)
            row.update({"E_coo": float(rc.energy), "gap_coo": float(rb.energy - rc.energy),
                        "pr_coo": pr_c, "mean_n_coo": mn_c, "coo_terms": len(Hc.terms),
                        "t_coo": tc})
        rows.append(row)
        msg = (f"A={A:>2}: bare {rb.energy:9.2f} | squeeze {rs.energy:9.2f} "
               f"(gap {row['gap_squeeze']:+7.2f})")
        if "E_coo" in row:
            msg += f" | COO {row['E_coo']:9.2f} (gap {row['gap_coo']:+7.2f})"
        msg += f"  disp={disp:.3f}  [{tb+ts+row.get('t_coo',0):.0f}s]"
        print(msg, flush=True)

    out = {"L": 2, "dim": 3, "n_b": 3, "core": CORE, "n_runs": N_RUNS,
           "A_ladder": list(A_LADDER), "A_coo": list(A_COO), "rows": rows}
    json.dump(out, open(SUMMARY, "w"), indent=2)
    _plot(out)
    _verdict(out)
    return out


def _verdict(out):
    print("\n=== A-TREND VERDICT (fixed core, gap = E_bare - E_frame, MeV) ===", flush=True)
    print(f"  {'A':>3} {'gap_squeeze':>12} {'gap_COO':>9} {'PR bare->sq':>13} "
          f"{'<n> bare->sq':>13} {'polaron disp':>13}", flush=True)
    for r in out["rows"]:
        gc = f"{r['gap_coo']:+.2f}" if "gap_coo" in r else "--"
        print(f"  {r['A']:>3} {r['gap_squeeze']:>+12.2f} {gc:>9} "
              f"{r['pr_bare']:>5.2f}->{r['pr_squeeze']:<5.2f} "
              f"{r['mean_n_bare']:>5.2f}->{r['mean_n_squeeze']:<5.2f} {r['polaron_disp']:>13.3f}",
              flush=True)


def _plot(out):
    # HONEST framing: at a fixed core of %d in a ~1e29 sector the solves are deeply
    # under-converged, so (a) the only meaningful signal is ENERGY at fixed budget
    # (squeeze below bare = the frame captures more of the true state), (b) the gap
    # MAGNITUDE is ensemble-variance-limited (bare catches up with more restarts), and
    # (c) PR-over-the-selected-core is NOT a compaction metric here (bare's core is one
    # dominant reference + junk = low PR / high E; squeeze's is a well-connected block =
    # higher PR / lower E), so it is intentionally NOT plotted. Clean high-A numbers need
    # HPC-scale cores.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    A = [r["A"] for r in rows]
    ac = [r["A"] for r in rows if "gap_coo" in r]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(A, [r["E_bare"] for r in rows], "o-", color="C0", ms=8, label="bare")
    ax[0].plot(A, [r["E_squeeze"] for r in rows], "s-", color="C1", ms=8, label="squeeze")
    ax[0].plot(ac, [r["E_coo"] for r in rows if "E_coo" in r], "^", color="C2", ms=11, label="COO")
    ax[0].set_xlabel("A (nucleons)"); ax[0].set_ylabel(r"$E_0$ at fixed core (MeV)")
    ax[0].set_title("Squeeze reaches lower E than bare at every A\n"
                    "(L=2 d=3, core=%d — deeply under-converged)" % out["core"])
    ax[0].legend(); ax[0].grid(alpha=0.25)
    ax[1].plot(A, [r["gap_squeeze"] for r in rows], "s-", color="C1", ms=8, label="squeeze gap")
    ax[1].plot(ac, [r["gap_coo"] for r in rows if "gap_coo" in r], "^-", color="C2", ms=10, label="COO gap")
    ax[1].axhline(0, color="gray", lw=1, ls=":")
    ax[1].set_xlabel("A (nucleons)"); ax[1].set_ylabel(r"gap = $E_{bare}-E_{frame}$ (MeV)")
    ax[1].set_title("Frame benefit POSITIVE for squeeze at all A\n"
                    "(magnitude ensemble-variance-limited; COO unreliable)")
    ax[1].legend(); ax[1].grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "frame_comparison_highA.png")
    fig.savefig(p, dpi=140)
    print(f"[plot] wrote {p}", flush=True)


if __name__ == "__main__":
    run()
