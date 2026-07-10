"""
Lang-Firsov (polaron-frame) compactification — the `transform="LF"` axis vs the
bare Fock basis (`transform="COO"`). See tasks/32 and
`claude/research/trimci/00_literature_review.md` §7.

WHY. Our Hamiltonian is a multi-flavor Hubbard-Holstein/Fröhlich model, so the
ground state is a POLARON: the nucleon dressed by a pion cloud. In the bare Fock
basis (COO) that cloud spreads over many boson Fock levels, costing determinants.
The Lang-Firsov unitary U(λ) = exp(λ S) moves to the polaron frame,

    S = Σ_m ( D_m b†_m - D_m† b_m )         (anti-Hermitian ⇒ U unitary),

where D_m is the fermion operator that couples LINEARLY to boson mode m (extracted
from the axial-vector coupling H_AV). U shifts each boson by its polaron cloud, so
the transformed ground state |ḡ⟩ = U† |g⟩ should be MORE COMPACT in the Fock basis.
Because U is unitary, the transformed Hamiltonian Ū = U† H U is ISOSPECTRAL — the
frame change is exact; only the basis representation of the ground state changes.

WHAT THIS MODULE DOES (Phase 1 — the go/no-go, ED scale). It builds U(λ) as a dense
matrix on a small enumerable system, forms |ḡ⟩ = U†|g⟩, and measures how much more
compact |ḡ⟩ is than |g⟩ (COO) as a function of the displacement amplitude λ. This
answers "does LF help our ground state, and at what λ?" WITHOUT the heavy term-list
Franck-Condon machinery needed to run LF through TrimCI at large L (that is Phase 2;
see tasks/32). The single global λ scanned here is the simplest variational-LF
parameter (the textbook λ=g/ω is only exact for a density vertex; ours is a
transition on a gradient, so λ is optimized — the bosonic analog of orbital opt.).

The generator uses ONLY the linear (H_AV) coupling; the quadratic H_WT/dispersion
sector is a squeezing (Bogoliubov) generalization, deferred to the Gaussian-transform
phase. If a system has no linear coupling, S=0 and LF is a no-op (reported).
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import MixedH, OperatorTerm, build_from_eft
from .hij import build_dense
from .state import enumerate_basis


def _dagger_ferm(ops):
    """Hermitian conjugate of a fermion ladder-op product: reverse order, flip
    creation<->annihilation. (b/f ops: (mode, action), action 1=create, 0=annih.)"""
    return tuple((i, 1 - a) for (i, a) in reversed(ops))


def displacement_generator(H):
    """Anti-Hermitian LF displacement generator S = Σ_m (D_m b†_m - D_m† b_m) as a
    MixedH term list, built from H's LINEAR fermion-boson coupling (mixed terms with
    exactly one boson ladder op). D_m = the fermion factor multiplying b†_m in H_AV.

    Returns a MixedH (may have zero terms if H has no linear coupling)."""
    gen = []
    for t in H.terms:
        # linear coupling term F · b†_m  (skip pure-boson H_pion_free: no ferm_ops)
        if t.ferm_ops and len(t.bos_ops) == 1 and t.bos_ops[0][1] == 1:
            m = t.bos_ops[0][0]
            gen.append(OperatorTerm(t.coeff, t.ferm_ops, ((m, 1),)))          # +D_m b†_m
            gen.append(OperatorTerm(-np.conj(t.coeff),                        # -D_m† b_m
                                    _dagger_ferm(t.ferm_ops), ((m, 0),)))
    return MixedH(gen, H.n_ferm_modes, H.n_bos_modes, H.N_f)


def scale_linear_coupling(H, s):
    """Return a copy of H with its LINEAR fermion-boson coupling (the H_AV terms,
    one boson ladder op) scaled by `s` — a controlled el-ph coupling-strength knob.
    s>1 mimics the stronger-coupling (small-a_L / higher-dim) regime where the
    polaron cloud is large and LF is expected to pay off; the ED-tractable physical
    systems are weak-coupling (d=1), so this lets us probe the interesting regime
    on a small lattice. (Scales only the linear vertex g ⇒ the dimensionless
    coupling λ=g/ω; H_WT and the boson dispersion are left physical.)"""
    if s == 1.0:
        return H
    terms = []
    for t in H.terms:
        if t.ferm_ops and len(t.bos_ops) == 1:
            terms.append(OperatorTerm(t.coeff * s, t.ferm_ops, t.bos_ops))
        else:
            terms.append(t)
    out = MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)
    out.meta = dict(H.meta)
    return out


def compactness(vec, fracs=(0.99, 0.999)):
    """Compactness metrics of a state vector over a basis:
      participation_ratio = 1/Σ p_i²  (effective # basis states),
      n{99,999}            = # basis states holding 99 % / 99.9 % of the weight.
    Lower = more compact."""
    p = np.abs(np.asarray(vec)) ** 2
    s = p.sum()
    if s <= 0:
        return {"participation_ratio": 0.0, "n99": 0, "n999": 0}
    p = p / s
    csum = np.cumsum(np.sort(p)[::-1])
    out = {"participation_ratio": float(1.0 / np.sum(p ** 2))}
    for f in fracs:
        label = "n" + f"{f * 100:g}".replace(".", "")   # 0.99->n99, 0.999->n999
        out[label] = int(np.searchsorted(csum, f) + 1)
    return out


def _ground_vector(H, n_elec):
    """Exact ground state as (basis, vector, E0) over the enumerated A-sector."""
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec)
    M = build_dense(H, basis)
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    return basis, V[:, 0], float(w[0])


def lf_compactness_scan(L, dim, n_b, A=1, lambdas=None, params=None,
                        coupling_scale=1.0, out_png=None, verbose=True):
    """Phase-1 go/no-go: how compact is the LF-frame ground state |ḡ⟩=U†|g⟩ vs the
    bare (COO) |g⟩, as a function of the displacement amplitude λ, on an
    ED-enumerable system. `coupling_scale`>1 strengthens the (linear) el-ph coupling
    to probe the polaron regime where LF should pay off (the ED-tractable physical
    systems are weak-coupling). Returns per-λ records (λ=0 row = COO baseline) and
    writes a compactness-vs-λ plot. Validated automatically: U is unitary (spectrum
    preserved), and λ=0 reproduces COO exactly."""
    from scipy.linalg import expm
    if lambdas is None:
        lambdas = np.linspace(0.05, 1.2, 12)
    H = scale_linear_coupling(build_from_eft(L, dim, n_b, params), coupling_scale)
    basis, g, E0 = _ground_vector(H, A)
    S0 = build_dense(displacement_generator(H), basis)
    anti_err = float(np.max(np.abs(S0 + S0.conj().T))) if S0.size else 0.0
    coo = compactness(g)
    I = np.eye(len(basis))
    recs = [{"lam": 0.0, **coo, "unitarity_err": 0.0}]
    if verbose:
        print("=" * 68)
        print(f"  LF COMPACTNESS SCAN  L={L} dim={dim} n_b={n_b} A={A} "
              f"coupling×{coupling_scale:g}  ({len(basis)} states)")
        print(f"  generator anti-Hermitian err={anti_err:.1e}; E0={E0:.4f} MeV")
        print(f"  {'lambda':>8} {'PR':>10} {'n99':>6} {'n999':>6} {'U†U-I':>9}")
        print("  " + "-" * 46)
        print(f"  {'COO':>8} {coo['participation_ratio']:>10.1f} "
              f"{coo['n99']:>6} {coo['n999']:>6} {0.0:>9.1e}")
    if S0.size == 0 or np.max(np.abs(S0)) == 0:
        if verbose:
            print("  (no linear coupling — LF is a no-op for this system)")
        return recs
    for lam in lambdas:
        U = expm(lam * S0)
        uerr = float(np.max(np.abs(U.conj().T @ U - I)))
        gbar = U.conj().T @ g
        nrm = np.linalg.norm(gbar)
        if nrm > 0:
            gbar = gbar / nrm
        c = compactness(gbar)
        recs.append({"lam": float(lam), **c, "unitarity_err": uerr})
        if verbose:
            print(f"  {lam:>8.3f} {c['participation_ratio']:>10.1f} "
                  f"{c['n99']:>6} {c['n999']:>6} {uerr:>9.1e}")
    best = min(recs, key=lambda r: r["n999"])
    if verbose:
        print("  " + "-" * 46)
        gain = coo["n999"] / max(best["n999"], 1)
        print(f"  best λ={best['lam']:.3f}: n999 {coo['n999']} -> {best['n999']} "
              f"({gain:.2f}× compaction of the 99.9%-weight core)")
        print("=" * 68)
    if out_png:
        _plot_scan(recs, L, dim, n_b, A, out_png)
    return recs


def _plot_scan(recs, L, dim, n_b, A, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = [r["lam"] for r in recs]
    n999 = [r["n999"] for r in recs]
    pr = [r["participation_ratio"] for r in recs]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(lam, n999, "o-", color="C0", label="n(99.9% weight)")
    ax.plot(lam, pr, "s--", color="C1", alpha=0.7, label="participation ratio")
    ax.axhline(recs[0]["n999"], ls=":", color="gray", lw=1,
               label=f"COO baseline ({recs[0]['n999']})")
    ax.set_xlabel("LF displacement amplitude λ")
    ax.set_ylabel("basis states in the compact core")
    ax.set_title(f"LF frame compaction of the GS  (L={L}, dim={dim}, N_f={2**n_b}, A={A})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[plot] wrote {out_png}")
    return fig
