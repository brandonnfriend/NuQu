"""Headline quantum-resource figure + table from the regenerated PauliLCU anchor (r3 / 290826).

Reads the manifest-versioned quantum round-3 shards (Fock/PauliLCU, n_b=2 anchor L=1..10 +
n_b sweep L=2,3), VALIDATES them as an accepted dataset, and writes:
  * headline_resources.pdf / .png   — 3-panel figure (qubits vs L, walk-query T vs L, T vs n_b)
  * headline_resources_table.md     — the anchor + n_b-sweep tables (review) + provenance
  * headline_resources_table.tex    — LaTeX booktabs versions (manuscript)
  * accepted_data_manifest.json     — file hashes + dependency versions (reproducibility)

Reproducible: point --data at the pulled shard dir. Colors are the validated dataviz
reference slots (blue slot-1, orange slot-2; critical status for any pruning-flagged point).

The accepted-data validator (codex round-3 audit, reproducibility section) refuses to plot
unless every record is complete, single-commit, single-configuration, pruning-clean, has a
3-sample log-linear walk-T fit with negligible residual, and has coherent total-T arithmetic.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the QPE-cost helpers (repo root on path; script lives in misc/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src_PI.estimation.qpe_cost import (            # noqa: E402
    WALK_QUERY_CONSTANT_BABBUSH_UB,
    WALK_QUERY_CONSTANT_HEISENBERG,
    qpe_phase_register_qubits_from_nwalk,
    total_logical_qubits,
    walk_queries,
)

# Headline adopts the Heisenberg constant π (N_walk = π·λ/ε_qpe); the raw shards
# carry the Babbush upper bound √2·π. Reporting recomputes N_walk / coherent-T /
# QPE-phase-register from the shard's own λ and ε_qpe with π — a documented ×1/√2
# tightening (see results/quantum_pauli_lcu_resources.md §3 and total_costs review).
N_WALK_CONSTANT = WALK_QUERY_CONSTANT_HEISENBERG

# --- dataviz reference palette (light; slots 1,2 pass all-pairs) + ink tokens ------------- #
BLUE, ORANGE, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data_dir):
    """Return (anchor, sweep, allrecs). `allrecs` is EVERY done record (incl. the determinism
    repeat) with the provenance/validation fields; anchor/sweep are the plotted subsets."""
    anchor, sweep, allrecs = [], [], []
    for f in sorted(glob.glob(f"{data_dir}/*fock_pauli*.json")):
        j = json.load(open(f))
        if not j.get("done") or not j["results"]:
            continue
        r = j["results"][0]
        b = r.get("QPE_Budget") or {}
        man = (j.get("metadata") or {}).get("manifest") or {}
        fit = b.get("walk_T_fit") or {}
        lam, walkT = r["Physical_Lambda"], r["Walk_T_Count"]
        eps_qpe = b.get("eps_qpe")
        # Headline reporting: recompute N_walk / coherent-T with the ADOPTED π constant
        # from the shard's own λ, ε_qpe (a documented ×1/√2 tightening of the raw √2·π
        # shard values). Then the QPE phase register m = ⌈log₂(N_walk/2)⌉ and the honest
        # total logical width = walk register + m (state prep a_prep=0 until modeled).
        nwalk_hl = walk_queries(lam, eps_qpe, constant=N_WALK_CONSTANT) if eps_qpe else None
        qpeT_hl = nwalk_hl * walkT if nwalk_hl else None
        m_qpe = qpe_phase_register_qubits_from_nwalk(nwalk_hl) if nwalk_hl else None
        qtot = total_logical_qubits(r["Logical_Qubits"], m_qpe) if m_qpe else None
        rec = dict(L=r["L"], n_b=r["n_b"], lam=lam, q=r["Logical_Qubits"],
                   # raw shard values (√2·π) — kept for the data-integrity validator:
                   nwalk_raw=r["QPE_Walk_Queries"], walkT=walkT,
                   qpeT_raw=r["QPE_Total_T_Count"], eps_qpe=eps_qpe,
                   # adopted-headline values (π):
                   nwalk=nwalk_hl, qpeT=qpeT_hl, m=m_qpe, qtot=qtot,
                   clean=b.get("prune_within_budget", None),
                   terms=r.get("Pauli_Term_Count"), rot=r.get("Rotation_Count"),
                   fit_resid=fit.get("resid"), fit_n=fit.get("n_samples"),
                   delta_E=b.get("delta_E"), commit=man.get("git_commit"),
                   dirty=man.get("git_dirty"), path=f, base=os.path.basename(f),
                   rep="rep2" in os.path.basename(f))
        allrecs.append(rec)
        if rec["n_b"] == 2 and not rec["rep"]:
            anchor.append(rec)
        if rec["L"] in (2, 3) and not rec["rep"]:
            sweep.append(rec)
    anchor.sort(key=lambda x: x["L"])
    sweep.sort(key=lambda x: (x["L"], x["n_b"]))
    return anchor, sweep, allrecs


def _dep_versions():
    import importlib.metadata as md
    v = {"python": sys.version.split()[0]}
    for p in ("pyLIQTR", "openfermion", "cirq-core", "numpy", "matplotlib"):
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = None
    return v


def validate(anchor, sweep, allrecs):
    """Accepted-data assertions (codex audit reproducibility section). Raises on any failure so
    a defective dataset can never be silently plotted as accepted."""
    import math
    assert allrecs, "no done records found"
    # single generating commit, clean tree
    commits = {r["commit"] for r in allrecs}
    assert len(commits) == 1 and None not in commits, f"multiple/absent generating commits: {commits}"
    assert all(r["dirty"] is False for r in allrecs), "a record was generated from a dirty tree"
    # single requested-accuracy budget
    des = {r["delta_E"] for r in allrecs}
    assert des == {1.0}, f"expected every record at delta_E=1 MeV, got {des}"
    # expected anchor + sweep census
    aL = [r["L"] for r in anchor]
    assert aL == list(range(1, 11)), f"anchor must be L=1..10 (n_b=2), got {aL}"
    sweep_keys = sorted((r["L"], r["n_b"]) for r in sweep)
    want = sorted((L, nb) for L in (2, 3) for nb in (1, 2, 3, 4))
    assert sweep_keys == want, f"cutoff sweep must be L={{2,3}} x n_b={{1,2,3,4}}, got {sweep_keys}"
    # per-record integrity
    for r in allrecs:
        tag = f"L={r['L']} n_b={r['n_b']} ({r['base']})"
        assert r["clean"] is True, f"{tag}: pruning-flagged (not within budget)"
        assert (r["fit_n"] or 0) >= 3, f"{tag}: walk-T fit has <3 samples ({r['fit_n']})"
        assert r["fit_resid"] is not None and r["fit_resid"] < 1.0, \
            f"{tag}: walk-T fit residual not negligible ({r['fit_resid']})"
        for k in ("lam", "q", "nwalk_raw", "walkT", "qpeT_raw", "eps_qpe",
                  "nwalk", "qpeT", "m", "qtot", "terms", "rot"):
            v = r[k]
            assert v is not None and math.isfinite(v) and v > 0, f"{tag}: {k} missing/nonfinite ({v})"
        # raw shard arithmetic (integrity check on the stored √2·π values):
        prod_raw = r["nwalk_raw"] * r["walkT"]
        assert abs(prod_raw - r["qpeT_raw"]) <= 1e-9 * r["qpeT_raw"], \
            f"{tag}: raw total-T arithmetic incoherent ({prod_raw:.6e} vs {r['qpeT_raw']:.6e})"
        # adopted-headline arithmetic (π) is self-consistent by construction:
        prod_hl = r["nwalk"] * r["walkT"]
        assert abs(prod_hl - r["qpeT"]) <= 1e-9 * r["qpeT"], \
            f"{tag}: headline total-T arithmetic incoherent ({prod_hl:.6e} vs {r['qpeT']:.6e})"
        # headline is exactly the π/√2·π tightening of the raw value:
        assert abs(r["nwalk"] / r["nwalk_raw"] - math.pi / (math.sqrt(2) * math.pi)) < 1e-9, \
            f"{tag}: headline N_walk is not the π rescale of the raw √2·π value"
        # total logical qubits = one walk register + the QPE phase register m:
        assert r["qtot"] == r["q"] + r["m"], \
            f"{tag}: total qubits {r['qtot']} != walk {r['q']} + QPE phase reg {r['m']}"
    # uniqueness within each plotted set
    for name, recs in (("anchor", anchor), ("sweep", sweep)):
        keys = [(r["L"], r["n_b"]) for r in recs]
        assert len(keys) == len(set(keys)), f"duplicate (L,n_b) in {name}"
    print(f"[validate] PASS — {len(allrecs)} records, commit {list(commits)[0][:10]}, "
          f"anchor L=1..10, sweep {len(sweep)} pts, all pruning-clean, fit resid<1, arithmetic coherent")
    return list(commits)[0]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0)
    ax.set_axisbelow(True)


def make_figure(anchor, sweep, out_base):
    # All r3 points are pruning-clean; assert it so an invalid point can never be plotted
    # silently as accepted data (codex plot-audit requirement).
    bad = [(r["L"], r["n_b"]) for r in anchor + sweep if r["clean"] is not True]
    assert not bad, f"refusing to plot pruning-flagged points: {bad}"
    # L=1 is a degenerate anchor (no ordinary intersite bonds) — shown but SEPARATED from the
    # bulk line and excluded from any scaling read (audit item 8).
    bulk = [r for r in anchor if r["L"] >= 2]
    deg = [r for r in anchor if r["L"] == 1]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 4.1))
    fig.patch.set_facecolor(SURFACE)

    def _deg(ax, y, log=False):                          # L=1 as a hollow degenerate marker
        if deg:
            plot = ax.semilogy if log else ax.plot
            plot([1], [deg[0][y]], "o", color=SURFACE, ms=7, mec=MUTED, mew=1.5, zorder=3)

    # (A) total logical qubits vs L = one walk/block-encoding register ((4+3 n_b)·S + ~16
    #     block-encoding ancilla) PLUS the QPE phase register m=⌈log₂(N_walk/2)⌉. The walk
    #     register is the thin lower line; the total is the marker line — the gap is m.
    axA.plot([r["L"] for r in bulk], [r["q"] for r in bulk], "--", color=MUTED, lw=1.2,
             zorder=2, label="walk register only")
    axA.plot([r["L"] for r in bulk], [r["qtot"] for r in bulk], "-o", color=BLUE, lw=2.0, ms=7,
             mec=SURFACE, mew=1.5, zorder=3, label="total (+ QPE phase reg.)")
    _deg(axA, "qtot")
    axA.annotate(f"{bulk[-1]['qtot']:,}\nqubits\n(+{bulk[-1]['m']} QPE)",
                 (bulk[-1]["L"], bulk[-1]["qtot"]),
                 textcoords="offset points", xytext=(-6, -2), ha="right", va="top",
                 fontsize=8.5, color=INK2)
    axA.set_xlabel("lattice size $L$  (dim=3, $L^3$ sites)", color=INK2, fontsize=9.5)
    axA.set_ylabel("total logical qubits (walk + QPE phase reg.)", color=INK2, fontsize=9.5)
    axA.set_title("a  Compiled logical width", color=INK, fontsize=11, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=7.6, loc="upper left", labelcolor=INK2)
    _style(axA)

    # (B) coherent walk-query T-count vs L (log y) — the compiled anchor, all pruning-clean
    axB.semilogy([r["L"] for r in bulk], [r["qpeT"] for r in bulk], "-o", color=BLUE, lw=2.0,
                 ms=7, mec=SURFACE, mew=1.5, zorder=3, label="compiled, pruning-clean")
    _deg(axB, "qpeT", log=True)
    axB.plot([], [], "o", color=SURFACE, mec=MUTED, mew=1.5, label="$L$=1 (degenerate anchor)")
    axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("coherent walk-query $T$-count", color=INK2, fontsize=9.5)
    axB.set_title("b  Compiled query cost vs volume", color=INK, fontsize=11, loc="left", weight="bold")
    axB.annotate(r"QPE+BE synth. $\leq 1$ MeV", (0.04, 0.93), xycoords="axes fraction",
                 fontsize=8.5, color=MUTED)
    axB.legend(frameon=False, fontsize=8.0, loc="lower right", labelcolor=INK2)
    _style(axB)

    # (C) coherent walk-query T-count vs n_b — resource sensitivity CONDITIONAL on the cutoff
    for Lx, col, mk in ((2, BLUE, "o"), (3, ORANGE, "s")):
        pts = sorted([r for r in sweep if r["L"] == Lx], key=lambda x: x["n_b"])
        nb = [r["n_b"] for r in pts]; qpe = [r["qpeT"] for r in pts]
        axC.semilogy(nb, qpe, "-", color=col, lw=2.0, marker=mk, ms=7, mec=SURFACE, mew=1.5,
                     zorder=3)
        axC.annotate(f"$L$={Lx}", (nb[-1], qpe[-1]), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=9, color=col, weight="bold")
    axC.set_xlabel("boson register $n_b$  (bits/mode)", color=INK2, fontsize=9.5)
    axC.set_ylabel("coherent walk-query $T$-count", color=INK2, fontsize=9.5)
    axC.set_title("c  Cost sensitivity to cutoff $n_b$", color=INK, fontsize=11, loc="left", weight="bold")
    axC.set_xticks([1, 2, 3, 4])
    axC.annotate("conditional on $n_b$\n(physical adequacy set\nby classical study)",
                 (1.05, 0.05), xycoords="axes fraction", fontsize=7.8, color=MUTED, style="italic")
    _style(axC)

    fig.suptitle("Compiled PauliLCU resources for QPE ground-state energy estimation",
                 fontsize=12.5, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def _fmt(x, sci=False):
    return f"{x:.3e}" if sci else f"{x:,.0f}"


# Caption revised per the round-3 finalization audit (six required caption/provenance edits).
CAPTION = (
    "Logical fault-tolerant COHERENT WALK-QUERY resources for qubitized phase estimation of the "
    "corrected dynamical-pion chiral-EFT Hamiltonian (Fock encoding, dim=3, one PauliLCU walk). "
    "The allocated QPE-resolution plus block-encoding-synthesis error is at most 1 MeV for the "
    "finite-n_b Hamiltonian, split by a total-T optimization between phase resolution and "
    "block-encoding synthesis that is optimal WITHIN the implemented fixed internal error model "
    "(an even eps_be share via qpe_error_budget; NOT a proof of global optimality over all "
    "coefficient-loading, synthesis, QPE, or success strategies); every walk-T is at the resulting "
    "circuit precision (log-linear in log2(1/cp), verified with a 3rd interior sample, residual 0). "
    "N_walk uses the adopted Heisenberg constant N_walk=pi*lambda/eps_qpe; the raw shards carry the "
    "Babbush 2018 Eq.26 upper bound sqrt(2)*pi (sqrt(2)x larger, conservative). "
    "Boson-cutoff, EFT, lattice-spacing, and finite-volume errors are OUTSIDE this 1 MeV budget. "
    "All points are pruning-clean: no Pauli terms are pruned from the constructed finite-n_b "
    "Hamiltonian (this is NOT a claim of zero boson-truncation, EFT, lattice-spacing, or "
    "finite-volume error). lambda is an exactly evaluated Pauli one-norm of the constructed "
    "Hamiltonian; N_walk follows from the phase-resolution allocation; the logical-qubit count is a "
    "deterministic pyLIQTR compiler estimate for the selected walk implementation (none is a broad "
    "'exact physical' quantity). 'total qubits' = one walk/block-encoding register PLUS the QPE phase "
    "register m=ceil(log2(N_walk/2)) (Babbush 2018 Eq.24); it still EXCLUDES input-state preparation "
    "(which enters as max(m, a_prep) once modeled), and the physical-layer magic-state factories and "
    "routing. Costs are CONDITIONAL on n_b=2 (boson cutoff; physical adequacy to be set by the "
    "classical cutoff study) and on input-state preparation / success probability (GSEE repetition "
    "~1/p0 sampling, ~1/sqrt(p0) with amplitude-amplified filtering, for input-state success is NOT "
    "included here); they are NOT a complete ground-state-energy algorithm cost. L varies at "
    "fixed lattice spacing (finite-volume, not continuum). L=1 has no ordinary intersite bonds "
    "(degenerate smoke anchor; excluded from bulk scaling).")


def prune_tag(r):
    return "clean" if r["clean"] is True else ("FLAGGED" if r["clean"] is False else "?")


def _provenance_line(commit, versions):
    vs = ", ".join(f"{k} {versions[k]}" for k in ("pyLIQTR", "openfermion", "cirq-core", "numpy")
                   if versions.get(k))
    return (f"Generated from data commit `{commit}` (single clean generating commit across all "
            f"records). Compiler/estimator stack: {vs}, python {versions.get('python')}.")


def make_tables(anchor, sweep, out_base, commit, versions):
    prov = _provenance_line(commit, versions)
    # --- markdown ---
    adopt = (f"**N_walk convention:** headline adopts the Heisenberg constant "
             f"N_walk = π·λ/ε_qpe (π≈{WALK_QUERY_CONSTANT_HEISENBERG:.3f}); the raw shards "
             f"carry the Babbush upper bound √2·π≈{WALK_QUERY_CONSTANT_BABBUSH_UB:.3f} "
             f"(conservative bound, ×√2 larger). 'total qubits' = one walk register + the "
             f"QPE phase register m=⌈log₂(N_walk/2)⌉ (Babbush 2018 Eq. 24); input-state prep "
             f"enters later as max(m, a_prep). Magic-state factories + routing are physical-"
             f"layer and excluded here.")
    md = ["# Headline quantum-resource anchor (compiled Fock/PauliLCU, corrected H)\n",
          "_" + CAPTION + "_\n",
          "**Provenance:** " + prov + "\n",
          adopt + "\n",
          "**Anchor — n_b=2, dim=3:**\n",
          "| L | sites | λ (MeV) | walk qubits | QPE phase reg. m | total qubits | Pauli terms "
          "| rotations | N_walk (π) | walk-T | coherent-query T (π) | pruning |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in anchor:
        note = " *(degen.)*" if r["L"] == 1 else ""
        md.append(f"| {r['L']}{note} | {r['L']**3} | {r['lam']:,.0f} | {r['q']:,} | {r['m']} "
                  f"| {r['qtot']:,} | {r['terms']:,} | {r['rot']:,} | {r['nwalk']:.2e} "
                  f"| {r['walkT']:.2e} | {r['qpeT']:.2e} | {prune_tag(r)} |")
    md.append("\n**Cost sensitivity to the boson cutoff n_b** (conditional; L=2, L=3):\n")
    md.append("| L | n_b | λ (MeV) | walk qubits | QPE phase reg. m | total qubits "
              "| Pauli terms | coherent-query T (π) | pruning |")
    md.append("|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for Lx in (2, 3):
        for r in sorted([x for x in sweep if x["L"] == Lx], key=lambda x: x["n_b"]):
            md.append(f"| {r['L']} | {r['n_b']} | {r['lam']:,.0f} | {r['q']:,} | {r['m']} "
                      f"| {r['qtot']:,} | {r['terms']:,} | {r['qpeT']:.2e} | {prune_tag(r)} |")
    open(f"{out_base}.md", "w").write("\n".join(md) + "\n")

    # --- LaTeX (booktabs) ---
    cap_tex = (CAPTION
               # multi-token phrases first (they consume the underscore-bearing tokens):
               .replace("N_walk=pi*lambda/eps_qpe", r"$N_\mathrm{walk}{=}\pi\lambda/\epsilon_\mathrm{qpe}$")
               .replace("m=ceil(log2(N_walk/2))", r"$m{=}\lceil\log_2(N_\mathrm{walk}/2)\rceil$")
               .replace("max(m, a_prep)", r"$\max(m, a_\mathrm{prep})$")
               .replace("~1/sqrt(p0)", r"$\sim 1/\sqrt{p_0}$")
               .replace("~1/p0", r"$\sim 1/p_0$")
               .replace("sqrt(2)*pi", r"$\sqrt2\,\pi$")
               .replace("eps_be", r"$\epsilon_\mathrm{be}$")
               .replace("qpe_error_budget", r"\texttt{qpe\_error\_budget}")
               .replace("lambda is", r"$\lambda$ is").replace("n_b=2", r"$n_b{=}2$")
               .replace("finite-n_b", r"finite-$n_b$")
               .replace("N_walk", r"$N_\mathrm{walk}$").replace("log2(1/cp)", r"$\log_2(1/\mathrm{cp})$"))
    tex = [r"% Compiled PauliLCU anchor (corrected H). Requires booktabs, siunitx.",
           r"% Provenance: " + prov.replace("`", ""),
           r"\begin{table}[t]\centering\small",
           r"\caption{" + cap_tex + r"}",
           r"\label{tab:pauli_lcu_anchor}",
           r"\begin{tabular}{r r r r r r r r r l}\toprule",
           r"$L$ & sites & $\lambda$ (MeV) & walk qb & $m$ & total qb & Pauli terms & "
           r"$N_\mathrm{walk}$ & coh.\ query $T$ & pruning\\\midrule"]
    for r in anchor:
        note = r"\,$^{\dagger}$" if r["L"] == 1 else ""
        tex.append(f"{r['L']}{note} & {r['L']**3} & {r['lam']:,.0f} & {r['q']:,} & {r['m']} & "
                   f"{r['qtot']:,} & {r['terms']:,} & \\num{{{r['nwalk']:.2e}}} & "
                   f"\\num{{{r['qpeT']:.2e}}} & {prune_tag(r)}\\\\")
    tex += [r"\bottomrule\end{tabular}",
            r"\par\footnotesize$^{\dagger}$ $L{=}1$ degenerate anchor (no intersite bonds); "
            r"excluded from bulk scaling. $m{=}\lceil\log_2(N_\mathrm{walk}/2)\rceil$ is the "
            r"QPE phase register (Babbush 2018 Eq.~24); total qb $=$ walk register $+\,m$. "
            r"$N_\mathrm{walk}{=}\pi\lambda/\epsilon_\mathrm{qpe}$ (Heisenberg; the raw "
            r"$\sqrt2\,\pi$ upper bound is $\sqrt2\times$ larger).",
            r"\end{table}"]
    open(f"{out_base}.tex", "w").write("\n".join(tex) + "\n")
    print(f"[tbl] wrote {out_base}.md / .tex")


def write_accepted_manifest(allrecs, commit, versions, out_path):
    """Machine-readable accepted-data manifest: per-file sha256 + the dependency versions the
    shard manifests lack (audit item 5 + reproducibility section)."""
    files = [{"file": r["base"], "L": r["L"], "n_b": r["n_b"], "rep": r["rep"],
              "sha256": _sha256(r["path"])} for r in sorted(allrecs, key=lambda x: (x["L"], x["n_b"], x["rep"]))]
    man = {"accepted": True, "n_records": len(allrecs), "generating_commit": commit,
           "requested_accuracy_MeV": 1.0, "dependency_versions": versions, "files": files}
    with open(out_path, "w") as f:
        json.dump(man, f, indent=2)
    print(f"[manifest] wrote {out_path}  ({len(files)} files hashed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--out-dir", default="data/quantum/2026-08-21/headline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    anchor, sweep, allrecs = load(args.data)
    print(f"[load] anchor n_b=2: L={[r['L'] for r in anchor]}; sweep points: {len(sweep)}; "
          f"all records: {len(allrecs)}")
    commit = validate(anchor, sweep, allrecs)          # refuses to proceed on a defective dataset
    versions = _dep_versions()
    make_figure(anchor, sweep, f"{args.out_dir}/headline_resources")
    make_tables(anchor, sweep, f"{args.out_dir}/headline_resources_table", commit, versions)
    write_accepted_manifest(allrecs, commit, versions, f"{args.out_dir}/accepted_data_manifest.json")


if __name__ == "__main__":
    main()
