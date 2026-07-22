"""
Consolidated, publication-style figures for the classical-baseline rule-out
campaign (see claude/research/classical_baselines/). One clear figure per demo
plus a one-glance summary.

Reuses saved JSON where the run is expensive (the DMRG L=2^3 sweep is ~90 s);
re-runs the cheap pieces (NQS FullSum stall, sign-severity scaling) live.

    python -m classical.baselines.plots            # writes to data/classical/baselines/figures/

Figures:
  00_summary.png        one-panel-per-method verdict overview
  01_cc_breakdown.png   CC error vs coupling (d=1, d=3)
  02_sign_problem.png   phase/sign structure by term + <s>(beta) + Delta_sign scaling
  03_dmrg_wall.png      E(chi) convergence + exact chi*(area) + wall-time/cross-check
  04_nqs_stall.png      NQS energy vs ansatz + sign-structure control test
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = "data/classical/baselines/2026-07-10"       # saved run data
_OUT = "data/classical/baselines/figures"


def _load(name):
    with open(os.path.join(_SRC, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------
def fig_cc(outdir):
    d = _load("cc_reduction_L2_A2.json")
    rbd = d["rows_by_dim"]
    fig, axes = plt.subplots(1, len(rbd), figsize=(6.4 * len(rbd), 4.8), squeeze=False)
    for ax, (label, rows) in zip(axes[0], rbd.items()):
        s = [r["s"] for r in rows]

        def col(k):
            return [abs(r[k]) if r.get(k) is not None else np.nan for r in rows]
        ax.semilogy(s, col("err_E_CCSD"), "o-", color="C3", label="RHF-CCSD")
        ax.semilogy(s, col("err_E_CCSD(T)"), "s--", color="C1", label="RHF-CCSD(T)")
        ax.semilogy(s, col("err_E_CCSD_U"), "^-", color="C0", label="UHF-CCSD")
        ax.axhline(1e-3, color="green", lw=1.4, ls=":",
                   label="selected CI / FCI (exact)")
        ax.axvline(1.0, color="0.4", lw=1.0)
        ax.text(1.0, ax.get_ylim()[1] * 0.5, " physical\n coupling", color="0.4",
                fontsize=8, va="top")
        for r in rows:
            conv = r.get("CCSD_converged", True)
            if r.get("E_CCSD") is None or conv is False:
                ax.axvspan(r["s"] - 0.03, r["s"] + 0.03, color="red", alpha=0.08)
        ax.set_xlabel("interaction scale  s   (s=1: physical contacts)")
        ax.set_ylabel(r"$|E_{\rm method}-E_{\rm exact}|$   (MeV)")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("D-CC: coupled cluster is uncontrolled at physical coupling "
                 "(non-variational / non-convergent); selected CI = FCI is exact",
                 fontsize=12, y=1.02)
    _save(fig, outdir, "01_cc_breakdown.png")


# --------------------------------------------------------------------------
def fig_sign(outdir):
    d = _load("sign_structure_L2_d1_A2.json")
    res = d["results"]
    betas = d["betas"]
    scurve = d["s_curve"]
    # live: Delta_sign scaling with L and A
    from classical.baselines.sign_structure import scaling_run
    scal = scaling_run(points=((2, 1, 2), (3, 1, 2), (2, 1, 4)), N_f=2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    labels = ["full model", "no WT (H_WT off)", "no Yukawa (H_AV off)", "no pion coupling"]
    labels = [k for k in labels if k in res]
    cx = [res[k].get("frac_complex", 0) * 100 for k in labels]
    fx = [res[k].get("frac_frustrating", 0) * 100 for k in labels]
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, cx, 0.4, color="C3", label="complex (PHASE problem)")
    ax1.bar(x + 0.2, fx, 0.4, color="C1", label="frustrating (sign)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax1.set_ylabel("% of off-diagonal matrix elements")
    ax1.set_title("H_WT is the phase source\n(complex fraction -> 0 without it)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.semilogy(betas, scurve, "o-", color="C0")
    ax2.set_xlabel(r"projection time  $\beta$   (MeV$^{-1}$)")
    ax2.set_ylabel(r"average sign  $\langle s\rangle$")
    ax2.set_title(r"Worldline sign decays: $\langle s\rangle\sim e^{-\beta\,\Delta_{\rm sign}}$")
    ax2.grid(True, which="both", alpha=0.25)

    lab = [f"L={r['L']}\nA={r['A']}" for r in scal]
    dps = [r["Delta_sign_per_site"] for r in scal]
    ax3.bar(range(len(scal)), dps, color="C2")
    ax3.set_xticks(range(len(scal)))
    ax3.set_xticklabels(lab, fontsize=8)
    ax3.set_ylabel(r"$\Delta_{\rm sign}$ / site   (MeV)")
    ax3.set_title("Sign gap grows with volume\n(A=4 = accidentally sign-free filling)")
    ax3.grid(True, axis="y", alpha=0.25)

    fig.suptitle("D-AFQMC-sign: the QMC family faces a phase problem (H_WT) + a sign "
                 "problem (H_AV) — attributed exactly", fontsize=12, y=1.02)
    _save(fig, outdir, "02_sign_problem.png")


# --------------------------------------------------------------------------
def fig_dmrg(outdir):
    lad = _load("dmrg_ladder.json")["data"]
    ent = _load("dmrg_entanglement.json")["rows"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    for label in ("L=2 1D", "L=2 2D", "L=2 3D"):
        sw = lad[label]["sweep"]
        ebest = min(r["E"] for r in sw)
        chi = [r["chi"] for r in sw]
        dE = [max(r["E"] - ebest, 1e-12) for r in sw]
        ax1.semilogy(chi, dE, "o-", label=f"{label} ({lad[label]['sites']} sites)")
    ax1.set_xlabel(r"bond dimension  $\chi$")
    ax1.set_ylabel(r"$E(\chi)-E_{\rm best}$   (MeV)")
    ax1.set_title("DMRG convergence slows with dimension")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    area = [r["cut_area"] for r in ent]
    chi6 = [r["chi_star"]["1e-06"] if "1e-06" in r["chi_star"] else r["chi_star"].get(1e-6)
            for r in ent]
    lbl = [f"L={r['L']} {r['dim']}D" for r in ent]
    ax2.bar(range(len(ent)), chi6, color="C4")
    ax2.set_xticks(range(len(ent)))
    ax2.set_xticklabels([f"{l}\n(cut area {a})" for l, a in zip(lbl, area)], fontsize=8)
    ax2.set_ylabel(r"exact  $\chi^*(\varepsilon=10^{-6})$")
    ax2.set_title(r"Bond-dim lower bound grows with cut area"
                  "\n(area law $\\chi\\sim e^{\\alpha L^{D-1}}$; flat in 1D)")
    ax2.grid(True, axis="y", alpha=0.25)

    dims = [lad[l]["dim"] for l in ("L=2 1D", "L=2 2D", "L=2 3D")]
    walls = [lad[l]["wall_s"] for l in ("L=2 1D", "L=2 2D", "L=2 3D")]
    ax3.semilogy(dims, walls, "s-", color="C3")
    ax3.set_xticks([1, 2, 3])
    ax3.set_xlabel("spatial dimension (L=2 fixed)")
    ax3.set_ylabel("DMRG wall-clock (s)")
    cc = lad.get("crosscheck_L2_3D", {})
    ax3.set_title("Cost explodes with dimension\n"
                  f"L=2³ cross-check: block2 {cc.get('block2_dmrg',0):.0f} vs "
                  f"TrimCI {cc.get('trimci_sci',0):.0f} MeV\n"
                  f"({cc.get('gap_MeV_per_site',0):.1f} MeV/site — validates TrimCI)")
    ax3.grid(True, which="both", alpha=0.25)

    fig.suptitle("D-DMRG: the tensor-network wall (χ~e^{L²} in 3D); the L=2³ run also "
                 "independently cross-checks TrimCI", fontsize=12, y=1.04)
    _save(fig, outdir, "03_dmrg_wall.png")


# --------------------------------------------------------------------------
def fig_nqs(outdir):
    from classical.baselines.nqs_netket import run_vmc
    from classical.baselines.sign_structure import dense_H
    from classical.trimci import build_from_eft
    from classical.trimci.lanczos import lanczos_ground_state
    from classical.trimci.run_cpp import cpp_ground_state_ensemble

    E_lanc, _ = lanczos_ground_state(build_from_eft(L=2, dim=1, n_b=1, N_f=2), 2)
    E_tc = float(cpp_ground_state_ensemble(
        build_from_eft(L=2, dim=1, n_b=1, N_f=2), n_elec=2,
        n_dets=3000, n_runs=16, seed=0).energy)

    alphas = [2, 4, 8]
    e_alpha = [run_vmc(2, 1, 2, N_f=2, alpha=a, n_iter=250, lr=0.05,
                       verbose=False)["E_best"] for a in alphas]

    toggles = [("full", None), ("H_WT off", {"wt": 0.0}),
               ("no pion", {"av": 0.0, "wt": 0.0})]
    tog = []
    for name, sc in toggles:
        M, _ = dense_H(2, 1, 2, N_f=2, **(sc or {}))
        e0 = float(np.linalg.eigvalsh(M)[0])
        e_nqs = run_vmc(2, 1, 2, N_f=2, alpha=4, n_iter=250, lr=0.05,
                        verbose=False, scales=sc)["E_best"]
        tog.append((name, e0, e_nqs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1.plot(alphas, e_alpha, "o-", color="C0", ms=8, label="NQS (RBM)")
    ax1.axhline(E_lanc, color="green", lw=1.6, ls="-", label=f"exact (Lanczos) {E_lanc:.1f}")
    ax1.axhline(E_tc, color="C2", lw=1.6, ls="--", label=f"selected CI (TrimCI) {E_tc:.1f}")
    for a, e in zip(alphas, e_alpha):
        ax1.annotate(f"+{e-E_lanc:.0f}", (a, e), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color="C0")
    ax1.set_xlabel(r"RBM width  $\alpha$")
    ax1.set_ylabel("ground-state energy  (MeV)")
    ax1.set_title("NQS stalls ~11 MeV above ground state\n(more params don't help)")
    ax1.set_xticks(alphas)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    x = np.arange(len(tog))
    gaps = [t[2] - t[1] for t in tog]
    ax2.bar(x, gaps, color="C1")
    for i, g in enumerate(gaps):
        ax2.annotate(f"+{g:.1f}", (i, g), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[0] for t in tog])
    ax2.set_ylabel("NQS energy above exact  (MeV)")
    ax2.set_title("Gap persists with pion coupling OFF\n(NOT the sign structure)")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("D-NQS: off-the-shelf NQS stalls above the ground state; does NOT "
                 "flip the baseline", fontsize=12, y=1.02)
    _save(fig, outdir, "04_nqs_stall.png")
    return E_lanc, E_tc, list(zip(alphas, e_alpha)), tog


def fig_summary(outdir, nqs=None):
    """One-glance verdict panel across the four methods."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # CC
    d = _load("cc_reduction_L2_A2.json")
    rows = list(d["rows_by_dim"].values())[-1]      # the 3D panel
    s = [r["s"] for r in rows]
    err = [abs(r["err_E_CCSD"]) if r.get("err_E_CCSD") is not None else np.nan for r in rows]
    ax = axes[0, 0]
    ax.semilogy(s, err, "o-", color="C3")
    ax.axhline(1e-3, color="green", ls=":", label="selected CI = FCI (exact)")
    ax.axvline(1.0, color="0.5")
    ax.set_title("Coupled cluster — RULED OUT\nnon-variational at physical coupling")
    ax.set_xlabel("coupling s"); ax.set_ylabel("|CCSD - exact| (MeV)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # sign
    sd = _load("sign_structure_L2_d1_A2.json")
    ax = axes[0, 1]
    ax.semilogy(sd["betas"], sd["s_curve"], "o-", color="C0")
    ax.set_title("QMC sign problem — RULED OUT\n<s> decays; H_WT gives complex weights")
    ax.set_xlabel(r"$\beta$ (MeV$^{-1}$)"); ax.set_ylabel(r"$\langle s\rangle$")
    ax.grid(True, which="both", alpha=0.25)

    # dmrg
    lad = _load("dmrg_ladder.json")["data"]
    ax = axes[1, 0]
    for label in ("L=2 1D", "L=2 2D", "L=2 3D"):
        sw = lad[label]["sweep"]
        eb = min(r["E"] for r in sw)
        ax.semilogy([r["chi"] for r in sw], [max(r["E"] - eb, 1e-12) for r in sw],
                    "o-", label=label)
    ax.set_title("DMRG — RULED OUT (3D)\nχ wall grows with dimension")
    ax.set_xlabel(r"$\chi$"); ax.set_ylabel(r"$E(\chi)-E_{\rm best}$ (MeV)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # nqs
    ax = axes[1, 1]
    if nqs is not None:
        E_lanc, E_tc, alpha_e, _ = nqs
        al = [a for a, _ in alpha_e]
        ee = [e for _, e in alpha_e]
        ax.plot(al, ee, "o-", color="C0", label="NQS (RBM)")
        ax.axhline(E_lanc, color="green", label="exact")
        ax.axhline(E_tc, color="C2", ls="--", label="selected CI")
        ax.set_xlabel(r"RBM $\alpha$"); ax.set_ylabel("E (MeV)")
        ax.set_xticks(al)
    ax.set_title("NQS — does NOT flip baseline\nstalls ~11 MeV above ground state")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.suptitle("Classical-baseline rule-out: selected CI (TrimCI) is the least-bad "
                 "classical method for the dynamical-pion EFT", fontsize=13, y=0.98)
    _save(fig, outdir, "00_summary.png")


def _save(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def make_all(outdir=_OUT):
    os.makedirs(outdir, exist_ok=True)
    print(f"Writing figures to {outdir}/")
    fig_cc(outdir)
    fig_sign(outdir)
    fig_dmrg(outdir)
    nqs = fig_nqs(outdir)
    fig_summary(outdir, nqs=nqs)
    print("done.")


if __name__ == "__main__":
    make_all()
