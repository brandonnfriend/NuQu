"""
n_b / N_f Fock-cutoff convergence study for the classical TrimCI baseline.

Two deliverables (see `claude/research/bosonic-encodings/02_tong_fock_cutoff.md`):

  GOAL 1 — TrimCI performance vs n_b. As n_b grows the Hilbert space grows
  exponentially, but the term list is n_b-INDEPENDENT (the cutoff only enters at
  apply-time) and the GS is near pion vacuum, so a selected-CI at fixed core should
  (a) barely slow down and (b) barely lower the energy past small n_b. We measure
  runtime(n_b) and E(n_b) to show low n_b captures the important Fock states.

  GOAL 2 — certify the quantum-side n_b bound. The doc derives an engineering cutoff
  (n_b ~ 1-2) and a rigorous spectral bound (n_b ~ 4-5) for eps=1e-3. ED can only
  check this at L=1 (trivial: no gradient -> pion vacuum). TrimCI does the ED-cross-
  check at L>=2 (the real system), where we read off the empirical n_b needed for a
  target accuracy and confirm the rigorous bound safely over-estimates it. We also
  measure <N>/mode (should match the SCS ~0.045 at L=2 d=3) and the leaked-weight
  tail (the physical reason the cutoff is small).

Usage:  python misc/run_nb_convergence.py [A|B|C|all]
  A: L=2 d=1 (exact Lanczos reachable -> validates TrimCI+PT2 against truth)
  B: L=2 d=3 (the real system; ED-impossible; the headline)
  C: L=3, L=4 d=3 (preliminary + runtime->HPC extrapolation)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from classical.trimci.run_cpp import (nb_convergence_sweep, occupation_vs_A_sweep,
                                       exact_occupation_vs_A)

OUT_DIR = os.path.join("data", "classical", "nb_convergence")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(out, name):
    path = os.path.join(OUT_DIR, f"{name}.json")
    # np types -> native for json
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"[save] {path}", flush=True)
    return path


def study_A():
    """L=2 d=1 A=1 — exact Lanczos reachable up to N_f=6; validates TrimCI+PT2."""
    out = nb_convergence_sweep(L=2, dim=1, A=1,
                               N_f_list=(2, 3, 4, 5, 6, 8, 16),
                               core=800, n_runs=3, seed=0)
    _save(out, "studyA_L2d1A1")
    return out


def study_B(A=1, core=2500):
    """L=2 d=3 — the real system, ED-impossible. Headline. Powers of two give the
    n_b axis; the odd N_f (3,6) probe the non-variational wobble."""
    out = nb_convergence_sweep(L=2, dim=3, A=A,
                               N_f_list=(2, 3, 4, 6, 8, 16, 32, 64),
                               core=core, n_runs=3, seed=0)
    _save(out, f"studyB_L2d3A{A}")
    return out


def study_D(core=2500, n_runs=5):
    """L=2 d=3 A=1 — UNIFORM (unbiased) init control for Study B.

    Study B seeds the core near vacuum (truncated-geometric, mean 0.5), which could
    in principle bias the high-n_b runs away from the high-occupation states newly
    available to them — i.e. we might be seeding our expected (low-occupation) answer.
    Study D removes the low-occupation prior entirely: uniform occupation sampling
    over [0, N_f) and NO vacuum anchor, so the search must FIND the near-vacuum GS.
    Same system/cutoffs as B; more runs (the ensemble min-over-runs must escape the
    high-occupation cloud). If D still converges to B's energy and <N>, the seed is
    not biasing the result. Matched core = B's, so only the init differs."""
    out = nb_convergence_sweep(L=2, dim=3, A=1,
                               N_f_list=(2, 4, 8, 16, 32, 64),
                               core=core, n_runs=n_runs, seed=0,
                               boson_init_mean=None)     # <-- uniform, unbiased
    _save(out, "studyD_L2d3A1_uniforminit")
    return out


def study_E(A_list=(1, 2, 3, 4, 6, 8), cores=(2000, 4000, 8000, 16000)):
    """Occupation vs A at L=2 d=3 — how the pion cloud responds to the nucleon
    source. The SCS mean field predicts <N> nearly flat in A (A-independent squeeze
    + negligible A^2 displacement); we measure whether the empirical <N> grows.
    Per-A core LADDER: high A has a bigger fermion sector and only reaches a
    finite-core LOWER BOUND on the laptop (near-vacuum seed approaches <N> from
    below). N_f=16 (n_b=4) is safely past the cutoff for all these A."""
    out = occupation_vs_A_sweep(L=2, dim=3, A_list=A_list, n_b=4,
                                cores=cores, n_runs=2, seed=0)
    _save(out, "studyE_occ_vs_A_L2d3")
    return out


def study_G(A_list=(1, 2, 3, 4, 5, 6, 7, 8)):
    """EXACT (full-ED) occupation vs A at L=2 d=1 — the clean, convergence-free
    anchor for study_E. Full Lanczos over each A-sector (fits for all A at N_f=4),
    so <N>(A) is exact truth. If it is flat, the pion occupation is A-independent
    (vacuum-squeezing dominated), which the selected-CI d=3 study can only show up
    to core-convergence noise."""
    out = exact_occupation_vs_A(L=2, dim=1, A_list=A_list, N_f=4)
    _save(out, "studyG_exact_occ_vs_A_L2d1")
    return out


def study_F(A_list=(6, 10), core=8000):
    """n_b cross-check at LARGER A. As A grows the pion source grows, so re-verify
    that low n_b (n_b=3, N_f=8) still converges the energy and keeps the leaked
    tail negligible. Fixed core across N_f -> the E(N_f=8) vs E(N_f=16,32)
    comparison is clean even if the core is not fully converged in absolute E."""
    outs = {}
    for A in A_list:
        out = nb_convergence_sweep(L=2, dim=3, A=A, N_f_list=(2, 4, 8, 16, 32),
                                   core=core, n_runs=2, seed=0)
        _save(out, f"studyF_nbcheck_L2d3A{A}")
        outs[A] = out
    return outs


def study_C():
    """L=3 and L=4 d=3 — preliminary, smaller n_b range + core, for runtime scaling.

    L=4 runs with pt2=False here mainly because at core 600 the PT2 correction is
    meaningless anyway (the core is far too incomplete). The former pure-Python
    EN-PT2 external-sum bottleneck at 12353 terms is now fixed by the C++ pass-1
    port (backend.cpp_pt2_external, ~50-60× at these term counts; auto-detected),
    so PT2 can be turned back on for larger cores. The RAW E_var plateau
    (N_f=4≈8≈16) is the robust cutoff-convergence signal and needs no PT2."""
    out3 = nb_convergence_sweep(L=3, dim=3, A=1, N_f_list=(2, 4, 8, 16),
                                core=1500, n_runs=2, seed=0)
    _save(out3, "studyC_L3d3A1")
    out4 = nb_convergence_sweep(L=4, dim=3, A=1, N_f_list=(2, 4, 8, 16),
                                core=600, n_runs=2, seed=0, pt2=False)
    _save(out4, "studyC_L4d3A1")
    return out3, out4


def plot_all():
    """Read whatever studies have been saved and produce the summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_studies = {}
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".json"):
            all_studies[f[:-5]] = json.load(open(os.path.join(OUT_DIR, f)))
    if not all_studies:
        print("(no saved studies to plot)")
        return

    def _rows(v):
        return v["rows"] if isinstance(v, dict) and isinstance(v.get("rows"), list) else []

    # split by schema; skip anything that is neither (e.g. the descent-control list)
    studies = {k: v for k, v in all_studies.items()
               if isinstance(v, dict) and "predictions" in v}
    occ_studies = {k: v for k, v in all_studies.items()
                   if isinstance(v, dict) and "predictions" not in v
                   and _rows(v) and "A" in _rows(v)[0]}
    if occ_studies:
        plot_occupation_vs_A(occ_studies, plt)
    if not studies:
        return

    # --- Plot 1: truncation error vs n_b, with Tong predictions ---
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for name, s in studies.items():
        rows = s["rows"]
        nb = [r["n_b"] for r in rows]
        err = [max(abs(r["trunc_err"]), 1e-4) for r in rows]
        ax.semilogy(nb, err, "o-", ms=7, label=name.replace("study", ""))
    p = list(studies.values())[0]["predictions"]
    ax.axvline(p["n_b_eng"], ls=":", color="green", alpha=0.7,
               label=f"SCS engineering n_b={p['n_b_eng']}")
    ax.axvline(p["n_b_spec2"], ls="--", color="orange", alpha=0.7,
               label=f"spectral(2nd) n_b={p['n_b_spec2']}")
    ax.axvline(p["n_b_spec1"], ls="--", color="red", alpha=0.7,
               label=f"spectral(1st) n_b={p['n_b_spec1']}")
    ax.axhline(13.5, ls="-.", color="gray", alpha=0.5, lw=1, label="ΔE_QPE=13.5 MeV")
    ax.set_xlabel("n_b  (N_f = 2^n_b)")
    ax.set_ylabel(r"|E_var+PT2(n_b) − E(n_b,max)|  (MeV)")
    ax.set_title("Fock-cutoff convergence — empirical n_b bound vs Tong prediction")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "truncation_error_vs_nb.png"), dpi=140)
    print(f"[plot] {OUT_DIR}/truncation_error_vs_nb.png", flush=True)

    # --- Plot 2: runtime vs n_b (Goal 1: does TrimCI slow down?) ---
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, s in studies.items():
        rows = s["rows"]
        ax.plot([r["n_b"] for r in rows], [r["runtime_s"] for r in rows],
                "s-", ms=6, label=name.replace("study", ""))
    ax.set_xlabel("n_b  (N_f = 2^n_b)")
    ax.set_ylabel("solve runtime (s)")
    ax.set_title("TrimCI runtime vs n_b (Hilbert space grows 2^n_b per mode)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "runtime_vs_nb.png"), dpi=140)
    print(f"[plot] {OUT_DIR}/runtime_vs_nb.png", flush=True)

    # --- Plot 3: <N>/mode vs n_b, with SCS prediction ---
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, s in studies.items():
        rows = s["rows"]
        ax.plot([r["n_b"] for r in rows], [r["N_per_mode"] for r in rows],
                "o-", ms=6, label=f"{name.replace('study','')} (measured)")
        ax.axhline(s["predictions"]["N_per_mode"], ls=":", alpha=0.6,
                   label=f"{name.replace('study','')} SCS={s['predictions']['N_per_mode']:.3f}")
    ax.set_xlabel("n_b")
    ax.set_ylabel(r"⟨N⟩ per pion mode")
    ax.set_title("Measured GS pion occupation vs SCS prediction (near-vacuum)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "occupation_vs_nb.png"), dpi=140)
    print(f"[plot] {OUT_DIR}/occupation_vs_nb.png", flush=True)


def plot_occupation_vs_A(occ_studies, plt):
    """Occupation vs A: measured <N>/mode (filled=core-converged, open=finite-core
    lower bound) with the SCS mean-field prediction and the max-mode occupation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ci, (name, s) in enumerate(sorted(occ_studies.items())):
        rows = sorted(s["rows"], key=lambda r: r["A"])
        A = np.array([r["A"] for r in rows], dtype=float)
        Nm = np.array([r["N_per_mode"] for r in rows])
        Nmax = np.array([r["N_max_mode"] for r in rows])
        scs = np.array([r["scs_N_per_mode"] for r in rows])
        # exact-ED rows have no "converged" flag -> treat as converged (filled)
        conv = np.array([bool(r.get("converged", True)) for r in rows])
        col = f"C{ci}"
        exact = "exact" in name or "d1" in name
        tag = name.replace("study", "").split("_")[0] + (" (exact d=1)" if exact
                                                          else " (SCI d=3, lower bnd)")
        ax1.plot(A[conv], Nm[conv], "o", ms=8, color=col, label=f"{tag} <N>")
        if (~conv).any():
            ax1.errorbar(A[~conv], Nm[~conv],
                         yerr=[0.25 * Nm[~conv], 0 * Nm[~conv]],
                         fmt="o", ms=8, mfc="white", color=col, uplims=True)
        ax1.plot(A, scs, "--", color=col, alpha=0.5, lw=1,
                 label=f"{tag.split('(')[0]}SCS")
        ax2.plot(A, Nmax, "s-", ms=7, color=col, label=f"{tag} max-mode")
    ax1.set_xlabel("A (nucleon number)")
    ax1.set_ylabel(r"$\langle N\rangle$ per pion mode")
    ax1.set_title("GS pion occupation vs A (L=2)\nexact d=1 + selected-CI d=3 "
                  "(filled=converged, open=lower bound)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax2.set_xlabel("A (nucleon number)")
    ax2.set_ylabel(r"max-mode occupation $\langle \max_m n_m\rangle$")
    ax2.axhline(7, ls=":", color="green", alpha=0.6, label="n_b=3 ceiling (occ 7)")
    ax2.set_title("Hottest-mode occupation vs A\n(stays << the n_b=3 Fock ceiling)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "occupation_vs_A.png")
    fig.savefig(out, dpi=140)
    print(f"[plot] {out}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        study_A()
    if which in ("B", "all"):
        study_B()
    if which in ("C", "all"):
        study_C()
    if which in ("D", "all"):
        study_D()
    if which in ("E", "all"):
        study_E()
    if which in ("F", "all"):
        study_F()
    if which in ("G", "all"):
        study_G()
    if which == "plot":
        plot_all()
        return
    plot_all()


if __name__ == "__main__":
    main()
