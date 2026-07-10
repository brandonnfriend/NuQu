"""
Frame (basis) transformations for compactness — task 33.

A frame is Hamiltonian PRE-PROCESSING: `Ū = U†HU` produced as a new `MixedH` term
list, which the existing solver (`connections` / `build_*` / TrimCI) consumes
UNCHANGED. Every frame is a variational **Perelomov (group-theoretic) coherent-state**
transform (see `claude/research/frame_optimization/00_literature_review.md` §3.0):

    Heisenberg-Weyl   -> displacement          = Lang-Firsov   (boson; see lf.py, STEP 3)
    symplectic Sp(2N) -> squeezing / Bogoliubov = Gaussian      (boson; THIS module, STEP 1)
    U(n)              -> orbital rotation        = COO           (fermion; STEP 5)

Universal correctness test = **isospectrality**: a frame unitary preserves the
spectrum, so `Ū` and `H` share their low eigenvalues (exactly in the untruncated
limit; on a finite Fock cutoff the match is cutoff-limited — see `isospectral_check`).

STEP 1 here is the Gaussian/Bogoliubov squeeze. It is the cheap, exact first frame:
because our boson operators are degree <= 2, the Bogoliubov substitution keeps the
Hamiltonian a finite POLYNOMIAL in ladder operators — no Franck-Condon expansion
(contrast the displacement/LF frame). It targets the quadratic pion sector: if that
sector has pair terms (b†b† + bb), its exact ground state is a squeezed vacuum, and
the right squeeze makes it a plain (compact) vacuum in the new frame.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .hamiltonian import MixedH, OperatorTerm
from .hij import build_dense, _apply_fermion_ops
from .state import enumerate_basis
from .lf import compactness, _ground_vector, _dagger_ferm


# ---------------------------------------------------------------------------
#  STEP 0 — validation harness
# ---------------------------------------------------------------------------

def _low_spectrum(H, n_elec, k=None):
    """Sorted (lowest-k) eigenvalues of H over the A=n_elec sector (ED)."""
    basis = enumerate_basis(H.n_ferm_modes, H.n_bos_modes, H.N_f, n_elec)
    M = build_dense(H, basis)
    M = 0.5 * (M + M.conj().T)
    w = np.sort(np.linalg.eigvalsh(M))
    return w if k is None else w[:min(k, len(w))]


def isospectral_check(H, H_frame, n_elec, k=None, tol=1e-9):
    """Max |ΔE| between the ED spectra of `H` and `H_frame` (A=n_elec sector).

    A frame unitary preserves the spectrum, so this is ~0. `k` compares only the
    lowest k eigenvalues (the physically relevant ones): a Bogoliubov/coherent
    transform on a FINITE Fock cutoff leaks a little past the boundary, so the
    HIGH eigenvalues (near the cutoff) pick up truncation artifacts while the low
    ones stay clean — use `k` (and/or a larger N_f) for the physical test.
    Returns the max mismatch; asserts `< tol`. Small (enumerable) systems only."""
    e1 = _low_spectrum(H, n_elec, k)
    e2 = _low_spectrum(H_frame, n_elec, k)
    n = min(len(e1), len(e2))
    err = float(np.max(np.abs(e1[:n] - e2[:n]))) if n else 0.0
    assert err < tol, f"frame not isospectral (k={k}): max|dE|={err:.2e} > {tol:.0e}"
    return err


def frame_compactness(H, H_frame, n_elec):
    """Compare the bare vs framed ED ground state: energies (should match) and
    Fock-basis compactness (participation ratio; #states holding 99/99.9% weight).
    A good frame keeps `dE≈0` while shrinking the core."""
    _, g0, e0 = _ground_vector(H, n_elec)
    _, gf, ef = _ground_vector(H_frame, n_elec)
    c0, cf = compactness(g0), compactness(gf)
    return {
        "E_bare": e0, "E_frame": ef, "dE": ef - e0,
        "bare": c0, "frame": cf,
        "n999_gain": c0["n999"] / max(cf["n999"], 1),
        "pr_gain": c0["participation_ratio"] / max(cf["participation_ratio"], 1e-12),
    }


# ---------------------------------------------------------------------------
#  STEP 1 — Gaussian / Bogoliubov squeeze (symplectic Perelomov coherent state)
# ---------------------------------------------------------------------------

def _per_mode(x, n):
    """Broadcast a scalar or length-n array to a length-n float array."""
    return np.full(n, float(x)) if np.isscalar(x) else np.asarray(x, dtype=float)


def squeeze_terms(H, r, phi=0.0, modes=None):
    """Bogoliubov frame `Ū = U†HU`, `U = exp[½ Σ_m (ζ_m b_m†² − ζ_m* b_m²)]`,
    `ζ_m = r_m e^{iφ_m}` — the symplectic Perelomov coherent-state transform.

    Implemented as the canonical substitution in every term's boson ops
        b_m  → cosh r_m · b_m  + e^{ iφ_m} sinh r_m · b_m†
        b_m† → cosh r_m · b_m† + e^{-iφ_m} sinh r_m · b_m
    (which is exactly `U†b_mU`), distributing the products. |μ|²−|ν|²=1 ⇒ the
    substitution is canonical ⇒ `Ū` is unitarily equivalent to `H` (isospectral).
    Our boson terms are degree ≤2, so each term spawns ≤4 children — a finite
    POLYNOMIAL result, no Franck-Condon.

    `r`, `phi`: scalar (all modes) or length-n_bos_modes array. `modes`: iterable
    of boson-mode indices to squeeze (default all; others pass through unchanged)."""
    n = H.n_bos_modes
    rv, pv = _per_mode(r, n), _per_mode(phi, n)
    sq = set(range(n)) if modes is None else set(int(m) for m in modes)
    mu = np.cosh(rv)                        # real
    nu = np.sinh(rv) * np.exp(1j * pv)      # e^{iφ} sinh r

    def subst(m, a):
        """(mode m, action a) -> list of (coeff, (m, a')) Bogoliubov children."""
        if m not in sq:
            return [(1.0 + 0j, (m, a))]
        if a == 0:                          # b_m  -> μ b_m + ν b_m†
            return [(mu[m] + 0j, (m, 0)), (nu[m], (m, 1))]
        return [(mu[m] + 0j, (m, 1)), (np.conj(nu[m]), (m, 0))]   # b_m† -> μ b_m† + ν* b_m

    acc = {}   # (ferm_ops, bos_ops) -> summed coeff
    for t in H.terms:
        choices = [subst(m, a) for (m, a) in t.bos_ops]
        if not choices:                     # constant / fermion-only term: unchanged
            key = (t.ferm_ops, ())
            acc[key] = acc.get(key, 0j) + complex(t.coeff)
            continue
        for combo in product(*choices):
            c = complex(t.coeff)
            ops = []
            for cf, op in combo:
                c *= cf
                ops.append(op)
            key = (t.ferm_ops, tuple(ops))
            acc[key] = acc.get(key, 0j) + c
    terms = [OperatorTerm(c, fo, bo) for (fo, bo), c in acc.items()
             if abs(c) > 1e-14]
    out = MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)
    out.meta = dict(H.meta)
    out.meta["frame"] = {"transform": "gaussian", "r": rv.tolist(),
                         "phi": pv.tolist(), "modes": sorted(sq)}
    return out


def squeeze_generator_terms(H, r, phi=0.0, modes=None):
    """The anti-Hermitian squeeze generator `G = Σ_m ½(ζ_m b_m†² − ζ_m* b_m²)` as a
    MixedH term list, so `U = exp(G)` and `U†|g⟩ = e^{−G}|g⟩` (used by
    `squeeze_state` for the generator-vs-substitution cross-check)."""
    n = H.n_bos_modes
    rv, pv = _per_mode(r, n), _per_mode(phi, n)
    sq = range(n) if modes is None else [int(m) for m in modes]
    zeta = rv * np.exp(1j * pv)
    terms = []
    for m in sq:
        terms.append(OperatorTerm(0.5 * zeta[m], (), ((m, 1), (m, 1))))
        terms.append(OperatorTerm(-0.5 * np.conj(zeta[m]), (), ((m, 0), (m, 0))))
    return MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)


# ---------------------------------------------------------------------------
#  STEP 2 — analytic seed + variational refinement of the squeeze
# ---------------------------------------------------------------------------

def boson_quadratic_form(H):
    """Per-mode `(omega_m, kappa_m)` of the BOSON-ONLY quadratic form:
    `omega_m` = coefficient of `b_m†b_m`, `kappa_m` = coefficient of `b_m†²`
    (same-mode pair). Cross-mode pair terms `b_i†b_j†` (present at d≥1, L≥2) are
    NOT captured — they need the full multi-mode Bogoliubov (STEP 2b); this
    per-mode form is the analytic seed."""
    n = H.n_bos_modes
    omega = np.zeros(n)
    kappa = np.zeros(n, dtype=complex)
    for t in H.terms:
        if t.ferm_ops or len(t.bos_ops) != 2:
            continue
        (m0, a0), (m1, a1) = t.bos_ops
        if m0 != m1:
            continue
        if (a0, a1) in ((1, 0), (0, 1)):     # b†b or bb† -> number
            omega[m0] += t.coeff.real
        elif (a0, a1) == (1, 1):             # b†²  (bb is its h.c.)
            kappa[m0] += t.coeff
    return omega, kappa


def analytic_squeeze(H, max_ratio=0.98):
    """Per-mode Bogoliubov `(r*, phi)` that DIAGONALIZES the same-mode boson
    quadratic form `H_m = omega_m b†b + (kappa_m b†² + h.c.)`:
        `r*_m = ¼ ln((omega+2|kappa|)/(omega−2|kappa|))`, `phi_m = arg(kappa_m)`.
    `2|kappa|/omega` is clamped to `max_ratio` (the form is unbounded-below past 1).
    Sign of the squeeze is resolved by `optimize_squeeze` (scans ±). Zero where the
    mode has no pair coupling (e.g. every mode at L=1)."""
    omega, kappa = boson_quadratic_form(H)
    ratio = np.clip(2.0 * np.abs(kappa) / np.maximum(omega, 1e-12), 0.0, max_ratio)
    r = 0.25 * np.log((1.0 + ratio) / (1.0 - ratio))
    phi = np.angle(kappa)
    return r, phi


def optimize_squeeze(H, n_elec, r0=None, scales=None, verbose=False):
    """Refine the squeeze to MINIMISE the framed ground-state compactness by ED.
    (The energy is frame-invariant, so we minimise the CORE SIZE, not the energy.)
    Scans a global multiplier over the per-mode analytic seed — which fixes the
    sign and the best magnitude — and returns the best `(scale, r_array, phi,
    compactness)`. For systems past ED, swap the objective to the fixed-budget
    TrimCI energy (a better frame reaches lower E at fixed n_dets); this ED version
    is the go/no-go. `r0`: seed (default `analytic_squeeze`)."""
    r_seed, phi = (analytic_squeeze(H) if r0 is None
                   else (np.asarray(r0, float), np.zeros(H.n_bos_modes)))
    if scales is None:
        scales = np.linspace(-1.6, 1.6, 33)
    _, g0, _ = _ground_vector(H, n_elec)
    c0 = compactness(g0)
    best = {"scale": 0.0, "r": np.zeros(H.n_bos_modes), "phi": phi,
            "compactness": c0, "bare": c0}
    for s in scales:
        Hf = squeeze_terms(H, s * r_seed, phi)
        _, gf, _ = _ground_vector(Hf, n_elec)
        c = compactness(gf)
        if verbose:
            print(f"    scale={s:+.2f}  n999={c['n999']:>4}  PR={c['participation_ratio']:.2f}")
        if c["n999"] < best["compactness"]["n999"]:
            best = {"scale": float(s), "r": s * r_seed, "phi": phi,
                    "compactness": c, "bare": c0}
    return best


def squeeze_state(H, n_elec, r, phi=0.0, modes=None):
    """The framed ground state `|ḡ⟩ = U†|g⟩ = e^{−G}|g⟩` built EXPLICITLY from the
    generator (dense `expm`), returned with its basis. Independent of
    `squeeze_terms`; the test checks the two agree (generator ⇔ substitution)."""
    from scipy.linalg import expm
    basis, g, _ = _ground_vector(H, n_elec)
    G = build_dense(squeeze_generator_terms(H, r, phi, modes), basis)
    gbar = expm(-G) @ g                     # U† |g> = e^{-G} |g>
    nrm = np.linalg.norm(gbar)
    if nrm > 0:
        gbar = gbar / nrm
    return basis, gbar


# ---------------------------------------------------------------------------
#  STEP 2b — full multi-mode Bogoliubov (the cross-mode pairs the per-mode
#  seed can't reach). Same symplectic Perelomov group Sp(2N,ℝ), but the general
#  element mixes modes, so it also kills the inter-mode pair terms b_i†b_j†
#  (i≠j) — 72 of them at L=2 d=3, vs 48 same-mode (STEP 2 note).
# ---------------------------------------------------------------------------

def boson_quadratic_matrices(H, modes=None):
    """Full boson-only quadratic form as matrices over `modes` (default all boson
    modes). `A` (Hermitian) = coefficient of the number sector `b_i† b_j`; `B`
    (complex-symmetric) = the PAIR matrix, normalised so `½ Σ_ij B_ij b_i† b_j† +
    h.c.` reproduces the degree-2 pair terms (⇒ `B_ii = 2 κ_i`, and `B_ij` for i≠j
    is the coefficient of `b_i† b_j†`). The CROSS-mode block of `B` (off-diagonal) is
    exactly what per-mode `analytic_squeeze` drops. Reordering constants are dropped
    (they shift energy, not the Bogoliubov angles). Returns `(A, B, mode_list)`."""
    n = H.n_bos_modes
    ms = list(range(n)) if modes is None else [int(m) for m in modes]
    idx = {m: i for i, m in enumerate(ms)}
    k = len(ms)
    A = np.zeros((k, k), dtype=complex)
    Braw = np.zeros((k, k), dtype=complex)   # raw coeff of b_i†b_j† as written
    for t in H.terms:
        if t.ferm_ops or len(t.bos_ops) != 2:
            continue
        (m0, a0), (m1, a1) = t.bos_ops
        if m0 not in idx or m1 not in idx:
            continue
        i, j = idx[m0], idx[m1]
        c = complex(t.coeff)
        if (a0, a1) == (1, 0):               # b_i† b_j
            A[i, j] += c
        elif (a0, a1) == (0, 1):             # b_i b_j† = b_j† b_i + δ_ij  (drop const)
            A[j, i] += c
        elif (a0, a1) == (1, 1):             # b_i† b_j†  -> pair
            Braw[i, j] += c
        # (0,0) bb terms are the h.c. of the pairs; B is fixed by Hermiticity below
    A = 0.5 * (A + A.conj().T)               # clean tiny numerical asymmetry
    B = Braw + Braw.T                        # symmetric: B_ii=2 Braw_ii, B_ij=Braw_ij+Braw_ji
    return A, B, ms


def analytic_bogoliubov(H, modes=None, max_ratio=0.98, thresh=1e-12):
    """Multi-mode Bogoliubov `(α, β)` that DIAGONALISES the FULL boson-only quadratic
    form — including the cross-mode pair terms `b_i†b_j†` (i≠j) the per-mode
    `analytic_squeeze` leaves behind (STEP 2b). Solves the symplectic eigenproblem by
    Colpa's method: with `M = [[A, B],[B*, A*]]` the para-unitary `S` obeying
    `S†MS = diag(Ω,Ω)` (Ω>0) is the transform whose NEW-frame boson vacuum is the
    exact quadratic-sector ground state — maximally compact. Returns `(α, β)`, each
    `n_bos×n_bos`, defining the substitution `b_i -> Σ_j (α_ij b_j + β_ij b_j†)`;
    identity on modes outside `modes`. Reduces to the single-mode `cosh r / e^{iφ}
    sinh r` when `B` is diagonal (checked in the suite). Entries `|·|<thresh` are
    zeroed to keep the framed term list sparse.

    Stability: a pure Bogoliubov can only reach positive Ω, so if the pair coupling
    makes `M` non-positive-definite (form unbounded below) the pair block is scaled
    down (global factor) until `M` is PD — the multi-mode analogue of the per-mode
    `2|κ|/ω` clamp. Returns identity `(I, 0)` if there are no pair terms."""
    A, B, ms = boson_quadratic_matrices(H, modes)
    k = len(ms)
    n = H.n_bos_modes
    alpha = np.eye(n, dtype=complex)
    beta = np.zeros((n, n), dtype=complex)
    if k == 0 or not np.any(np.abs(B) > thresh):
        return alpha, beta                   # nothing to squeeze

    floor = 0.02 * max(float(np.min(np.linalg.eigvalsh(A).real)), 1e-12)
    s = 1.0
    for _ in range(80):                      # shrink pairs until M is safely PD
        M = np.block([[A, s * B], [s * B.conj(), A.conj()]])
        M = 0.5 * (M + M.conj().T)
        if float(np.linalg.eigvalsh(M).min()) > floor or s < 1e-6:
            break
        s *= 0.9
    if s < 1.0:
        print(f"    [analytic_bogoliubov] pair matrix scaled x{s:.3f} for PD stability")

    # Colpa: M = K†K ; W = K J K† (Hermitian, eigenvalues ±Ω) ; S = K^{-1} Q √|Λ|
    J = np.diag(np.concatenate([np.ones(k), -np.ones(k)]))
    K = np.linalg.cholesky(M).conj().T       # M = K†K, K upper-triangular
    W = K @ J @ K.conj().T
    W = 0.5 * (W + W.conj().T)
    lam, Q = np.linalg.eigh(W)
    pos = np.where(lam > 0)[0]
    neg = np.where(lam < 0)[0]
    pos = pos[np.argsort(-lam[pos])]         # Ω1 >= Ω2 >= ...
    neg = neg[np.argsort(lam[neg])]          # -Ω1 <= -Ω2 <= ...  (|Ω| desc, matches pos)
    order = np.concatenate([pos, neg])       # -> Λ = diag(Ω1..Ωk, -Ω1..-Ωk)
    lam, Q = lam[order], Q[:, order]
    S = np.linalg.solve(K, Q * np.sqrt(np.abs(lam)))   # K^{-1} @ Q diag(√|Λ|)

    # The physical bosonic Bogoliubov S_phys = [[α, β],[β*, α*]] is read off the
    # POSITIVE-Ω columns (α; β*): α = top-left, β* = bottom-left. (Colpa's raw S is
    # para-unitary but NOT in this block form — its top-right block is a different
    # eigenvector, so β must come from conj(bottom-left), not the top-right.)
    for a, mi in enumerate(ms):              # embed the k×k blocks into full n×n
        for b, mj in enumerate(ms):
            alpha[mi, mj] = S[a, b]
            beta[mi, mj] = np.conj(S[k + a, b])
    alpha[np.abs(alpha) < thresh] = 0.0
    beta[np.abs(beta) < thresh] = 0.0
    return alpha, beta


def bogoliubov_terms(H, alpha, beta):
    """General multi-mode Bogoliubov frame `Ū = U†HU` via the LINEAR substitution
        b_i  -> Σ_j (α_ij b_j + β_ij b_j†),   b_i† -> Σ_j (α*_ij b_j† + β*_ij b_j)
    with `(α, β)` a canonical (para-unitary) pair (`αα† − ββ† = I`, `αβ^T = βα^T`).
    Generalises `squeeze_terms` (the α=diag(cosh r), β=diag(e^{iφ}sinh r) diagonal
    case) to CROSS-mode mixing — the transform that also removes inter-mode pair
    terms (STEP 2b). Boson degree stays ≤2, so the result is still a finite
    polynomial (no Franck-Condon); each 2-boson term spawns ≤ (2·nnz)² children."""
    alpha = np.asarray(alpha, dtype=complex)
    beta = np.asarray(beta, dtype=complex)
    n = H.n_bos_modes

    def subst(m, a):
        """(mode m, action a) -> list of (coeff, (mode', action')) children."""
        out = []
        if a == 0:                           # b_m -> Σ_j α_mj b_j + β_mj b_j†
            for j in range(n):
                if alpha[m, j]:
                    out.append((alpha[m, j], (j, 0)))
                if beta[m, j]:
                    out.append((beta[m, j], (j, 1)))
        else:                                # b_m† -> Σ_j α*_mj b_j† + β*_mj b_j
            for j in range(n):
                if alpha[m, j]:
                    out.append((np.conj(alpha[m, j]), (j, 1)))
                if beta[m, j]:
                    out.append((np.conj(beta[m, j]), (j, 0)))
        return out or [(0j, (m, a))]

    acc = {}
    for t in H.terms:
        choices = [subst(m, a) for (m, a) in t.bos_ops]
        if not choices:                      # constant / fermion-only: unchanged
            key = (t.ferm_ops, ())
            acc[key] = acc.get(key, 0j) + complex(t.coeff)
            continue
        for combo in product(*choices):
            c = complex(t.coeff)
            ops = []
            for cf, op in combo:
                c *= cf
                ops.append(op)
            key = (t.ferm_ops, tuple(ops))
            acc[key] = acc.get(key, 0j) + c
    terms = [OperatorTerm(c, fo, bo) for (fo, bo), c in acc.items()
             if abs(c) > 1e-14]
    out = MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)
    out.meta = dict(H.meta)
    out.meta["frame"] = {"transform": "gaussian", "bogoliubov": True,
                         "n_terms": len(terms)}
    return out


def multimode_squeeze_terms(H, modes=None, max_ratio=0.98, thresh=1e-12):
    """STEP 2b production entry point: solve the analytic multi-mode Bogoliubov for
    the full boson quadratic form, then apply it as a term list, in one call. Wired
    to `build_from_eft(transform='gaussian', frame_params={'bogoliubov': True})`."""
    alpha, beta = analytic_bogoliubov(H, modes=modes, max_ratio=max_ratio,
                                      thresh=thresh)
    return bogoliubov_terms(H, alpha, beta)


# ---------------------------------------------------------------------------
#  STEP 3 — Lang-Firsov displacement (Heisenberg-Weyl Perelomov coherent state),
#  the term-list transform for TrimCI. U = exp[Σ_m λ_m (D_m b†_m − D_m† b_m)] shifts
#  each boson by its fermion-conditioned polaron cloud D_m. Implemented as the closed
#  operator substitution  b_m → b_m + λ_m D_m ,  b_m† → b_m† + λ_m D_m†  — which is
#  EXACT and finite iff the D_m mutually commute (density/Holstein coupling: D_m a
#  number/projector). Our physical vertex is a TRANSITION (H_AV: D_m = a_p†a_q), for
#  which this substitution is only the LEADING Franck-Condon order (the higher orders
#  come from [D_m, D_n]≠0); the exact finite transition frame is the projector-
#  conditioned generator of STEP 4. The ED go/no-go (dense U, any vertex) is `lf.py`.
# ---------------------------------------------------------------------------

def _displacement_map(H):
    """Per-boson-mode displacement operator `D_m` (the fermion factor coupling
    LINEARLY to `b†_m`) as `{m: [(coeff, ferm_ops), ...]}`, read off H's linear
    coupling (mixed terms with a single boson CREATION op). The `b_m` h.c. terms are
    the conjugate `D_m† b_m` and are reconstructed by daggering — not re-read."""
    gen = {}
    for t in H.terms:
        if t.ferm_ops and len(t.bos_ops) == 1 and t.bos_ops[0][1] == 1:
            m = t.bos_ops[0][0]
            gen.setdefault(m, []).append((complex(t.coeff), t.ferm_ops))
    return gen


def _fc_displacement_ops(m, alpha, order):
    """Normal-ordered, truncated Franck-Condon expansion of the displacement operator
    `X_m(α) = exp[α(b†_m − b_m)] = e^{−α²/2} e^{α b†_m} e^{−α b_m}` as a list of
    `(coeff, bos_ops)`: `e^{−α²/2} Σ_{j+k≤order} α^j(−α)^k/(j!k!) (b†_m)^j (b_m)^k`.
    `order→∞` is exact; a finite `order` (~N_f) is the Franck-Condon truncation."""
    import math
    pref = np.exp(-0.5 * alpha * alpha)
    out = []
    for j in range(order + 1):
        for k in range(order + 1 - j):
            c = pref * (alpha ** j) * ((-alpha) ** k) / (math.factorial(j) * math.factorial(k))
            if abs(c) < 1e-15:
                continue
            out.append((c, ((m, 1),) * j + ((m, 0),) * k))
    return out


def displace_terms(H, lambdas, gen=None, fc_dress=None, order=4, tol=1e-14):
    """Lang-Firsov displacement frame `Ū = U†HU`,
    `U = exp[Σ_m λ_m (D_m b†_m − D_m† b_m)]` — the Heisenberg-Weyl Perelomov coherent
    state (polaron frame). Two mechanisms of `U†·U`, applied operator-by-operator:

    1. **Boson substitution** (always): `b_m → b_m + λ_m D_m`, `b_m† → b_m† + λ_m D_m†`
       (`= U†b_mU` when the `D_m` commute). Each boson ladder op spawns a child that is
       either the kept boson op or the fermion string `D_m` appended to the term's
       fermion factor (bosons commute with fermions ⇒ `D_m` moves right of ferm_ops).
    2. **Franck-Condon fermion dressing** (`fc_dress` given): a fermion operator that
       does NOT commute with the displacement (hopping/transition) picks up a
       displacement factor `U†a_pU = a_p X_{m}(±α)`. `fc_dress = {p: (m, α)}` maps
       fermion mode `p` → its boson mode `m` and amplitude `α`; `a_p` (annihilation)
       gets `X_m(+α)`, `a_p†` gets `X_m(−α)`, each expanded to `order` (see
       `_fc_displacement_ops`) and appended to the term's boson factor.

    EXACT (isospectral) for a DENSITY/Holstein vertex: with a STATIC (number-diagonal)
    fermion sector, mechanism 1 alone is exact and finite; WITH fermion hopping, adding
    mechanism 2 (`fc_dress`) restores exactness as `order→N_f`. For a TRANSITION vertex
    (our H_AV, `D_m = a_p†a_q`) the substitution is only leading order and even the
    boson image doesn't terminate — the exact finite frame is the projector-conditioned
    generator (STEP 4). `lambdas`: scalar or length-n_bos (textbook polaron `λ_m=−1/ω_m`).
    `gen`: explicit `{m:[(coeff,ferm_ops)]}` (default: read from H's linear coupling)."""
    n = H.n_bos_modes
    lam = _per_mode(lambdas, n)
    if gen is None:
        gen = _displacement_map(H)
    gendag = {m: [(np.conj(c), _dagger_ferm(fo)) for (c, fo) in dl]
              for m, dl in gen.items()}
    fc_dress = fc_dress or {}

    def subst_boson(m, a):
        """boson op (m,a) -> children ('b', coeff, bosop) | ('f', coeff, ferm_ops)."""
        out = [("b", 1.0 + 0j, (m, a))]
        src = gen.get(m, []) if a == 0 else gendag.get(m, [])   # b→+λD, b†→+λD†
        for (c, fo) in src:
            out.append(("f", lam[m] * c, fo))
        return out

    def dress_fermion(p, a):
        """fermion op (p,a) -> children (coeff, bos_ops): keep the op, append the FC
        factor. `fc_dress[p]` is `(m, α)` for a single boson mode, or a LIST
        `[(m, α), ...]` when the fermion mode couples to SEVERAL boson modes (our real
        vertex: each fermion mode dresses two pion modes). Different boson modes commute,
        so the multi-mode factor is the product `Π_m X_m(±α_m)` — the Cartesian product
        of their expansions. `a_p → X(+α)`, `a_p† → X(−α)`; amplitude scales with lam[m]
        so the ± global scan drives dressing and displacement together."""
        if p not in fc_dress:
            return [(1.0 + 0j, ())]
        spec = fc_dress[p]
        if not isinstance(spec, (list, tuple)) or (len(spec) == 2 and np.isscalar(spec[1])):
            spec = [spec]                      # normalize single (m, α) -> [(m, α)]
        out = [(1.0 + 0j, ())]
        for (m, alpha) in spec:
            s = alpha if a == 0 else -alpha    # a_p -> X(+α), a_p† -> X(−α)
            xm = _fc_displacement_ops(m, s * lam[m], order)
            out = [(c1 * c2, bo1 + bo2) for (c1, bo1) in out for (c2, bo2) in xm]
        return out

    acc = {}
    for t in H.terms:
        fchoices = [dress_fermion(p, a) for (p, a) in t.ferm_ops]
        bchoices = [subst_boson(m, a) for (m, a) in t.bos_ops]
        for fcombo in (product(*fchoices) if fchoices else [()]):
            for bcombo in (product(*bchoices) if bchoices else [()]):
                c = complex(t.coeff)
                fops = list(t.ferm_ops)        # fermion ops kept as written
                bops = []
                for (cc, xbos) in fcombo:      # FC dressing -> boson factor (ferm order)
                    c *= cc
                    bops.extend(xbos)
                for kind, cc, payload in bcombo:
                    c *= cc
                    if kind == "b":
                        bops.append(payload)
                    else:                      # D_m fermion string appended (boson order)
                        fops.extend(payload)
                key = (tuple(fops), tuple(bops))
                acc[key] = acc.get(key, 0j) + c
    terms = [OperatorTerm(c, fo, bo) for (fo, bo), c in acc.items()
             if abs(c) > tol]
    out = MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)
    out.meta = dict(H.meta)
    out.meta["frame"] = {"transform": "LF", "lambdas": lam.tolist(),
                         "fc_order": order if fc_dress else None}
    return out


# ---------------------------------------------------------------------------
#  STEP 4 — combined (squeeze ∘ displace) + projector-conditioned displacement
#  = the recommended layered frame (affine-symplectic / Jacobi-group Perelomov CS).
#  Displacement (STEP 3) absorbs the linear H_AV coupling; squeezing (STEP 1/2)
#  handles the quadratic + H_WT sector — the two Gaussian layers together.
# ---------------------------------------------------------------------------

def projector_generator(entries):
    """Build a projector-conditioned displacement map `gen = {m: [(λ, P_ops), ...]}`
    for `displace_terms`, from `entries = [(boson_mode m, fermion_mode p, λ), ...]`.
    Each `P = n̂_p = a_p†a_p` is the OCCUPATION PROJECTOR of fermion mode `p` (a
    commuting, number-diagonal matter projector). This is the state-dependent
    (Phuc-2025 projector-conditioned) generator `U=exp[Σ_{m,μ} λ_{m,μ} P̂_μ(b†_m−b_m)]`:
    each boson mode is displaced by an amplitude that depends on WHICH matter
    configuration the system is in — strictly more variational freedom than a single
    global (density) displacement. Because the projectors commute, the boson
    substitution in `displace_terms` stays EXACT and finite (no infinite generator
    series) even though our physical vertex is a transition — this is what lets LF
    "handle H_AV directly" (review §3.2). (The non-commuting transition parts of H are
    still Franck-Condon dressed via `fc_dress`.)"""
    gen = {}
    for (m, p, lam) in entries:
        gen.setdefault(int(m), []).append((complex(lam), ((int(p), 1), (int(p), 0))))
    return gen


def analytic_displacement(H, thresh=1e-12):
    """Polaron (Lang-Firsov) seed for the projector-conditioned displacement frame.
    Reads the DIAGONAL density couplings `g^m_pp` = coefficient of `n̂_p (b_m†+b_m)`
    off H's linear boson coupling (`_displacement_map`), and the per-mode pion
    frequency `ω_m` (`boson_quadratic_form`), and returns the textbook polaron
    amplitudes `λ_{m,p} = − g^m_pp / ω_m` as projector-generator ENTRIES
    `[(m, p, λ), ...]` ready for `projector_generator`. For `H = ω b†b + g n̂(b†+b)`,
    `U=exp[λ n̂(b†−b)]` sends `b→b+λ n̂` and completing the square at `λ=−g/ω` cancels
    the linear term (net shift `−g²/ω · n̂²`) — the exact seed; its global sign and
    magnitude are then refined by `optimize_displacement`'s ± scan (mirroring how
    `optimize_squeeze` fixes the squeeze sign). OFF-diagonal transition couplings
    `a_p†a_q` (p≠q) are NOT seeded here — the commuting `n̂_p` projectors keep the
    displacement exact/finite (STEP 4); the transition legs are the Franck-Condon
    residual (`fc_dress`), not part of the projector seed.

    Returns `(entries, scale, info)`. `info` records the diagonal-source count and how
    many boson modes got a nonzero seed — a null λ means that channel has NO density to
    absorb (a lattice/chiral selection rule), NOT that the frame converged. Density
    coupling is read in the deterministic create-first form `n̂_p = ((p,1),(p,0))`."""
    omega, _ = boson_quadratic_form(H)
    dmap = _displacement_map(H)
    entries = []
    n_diag = 0
    for m, lst in dmap.items():
        w = max(float(omega[m]), 1e-12)
        for (c, fo) in lst:
            if (len(fo) == 2 and fo[0][1] == 1 and fo[1][1] == 0
                    and fo[0][0] == fo[1][0]):          # n̂_p = a_p† a_p (diagonal)
                n_diag += 1
                lam = -float(np.real(c)) / w            # g^m_pp is real (Hermiticity)
                if abs(lam) > thresh:
                    entries.append((int(m), int(fo[0][0]), lam))
    info = {"n_diag_entries": n_diag, "n_seeded": len(entries),
            "modes_with_seed": len({m for (m, _p, _l) in entries}),
            "n_bos_modes": int(H.n_bos_modes),
            "max_abs_lambda": float(max((abs(l) for (_m, _p, l) in entries), default=0.0))}
    return entries, 1.0, info


def fc_dress_from_entries(entries):
    """Build the Franck-Condon `fc_dress` map `{p: [(m, λ_{m,p}), ...]}` from
    `analytic_displacement` entries — each fermion mode `p` dressed by the SIGNED
    displacement amplitudes of the boson modes it couples to (mechanism 2 of
    `displace_terms`). Passing this to `displace_terms(..., fc_dress=...)` restores
    EXACT isospectrality of the projector-conditioned LF frame even with fermion
    hopping / transition legs (validated: order 3 → |ΔE|~5e-4, order 4 → ~1e-6).

    WARNING — this is the "full" LF frame, and it is CATASTROPHICALLY term-explosive
    for our Hamiltonian: every off-diagonal fermion leg spawns `Π_m (order+1)(order+2)/2`
    boson children, and with ~two boson modes per leg + ~200 off-diagonal legs the count
    blows up (measured on L=2 d=3: order-1 → 1.6e5 terms, order-2 → 3.7e7 terms, order-3
    → ~1e9). Since accuracy needs order ≥3 (λ≈0.28), the exact projector-LF frame is
    INFEASIBLE to solve — this is the quantitative reason the Gaussian squeeze frame
    (finite polynomial, no FC expansion) is the practical choice. Kept for small-system
    validation and to document the go/no-go verdict."""
    from collections import defaultdict
    d = defaultdict(list)
    for (m, p, lam) in entries:
        d[int(p)].append((int(m), float(lam)))
    return dict(d)


def optimize_displacement(H, n_elec, entries=None, scales=None, core=2000,
                          n_runs=3, seed=0, fc_dress=None, order=4, verbose=False):
    """Refine the projector-conditioned LF amplitude by MINIMISING the TrimCI
    fixed-core ground energy (the past-ED objective flagged in `optimize_squeeze`'s
    docstring: for an isospectral frame, selected-CI is variational, so a better frame
    reaches a LOWER E at fixed `n_dets`). Scans a global multiplier `s` over the
    analytic polaron seed (`analytic_displacement`); the frame is
    `displace_terms(H, lambdas=s, gen=projector_generator(entries), fc_dress=...)`, so
    the effective amplitude is `s·λ_{m,p}` (mechanism-1 applies `lam[m]*c`) and the ±
    scan fixes the global sign — exactly as `optimize_squeeze` scans `s·r_seed`.

    IMPORTANT (go/no-go): the leading-order projector-LF (`fc_dress=None`) is exact on
    every boson-carrying term but leaves pure-fermion hopping legs undressed, so it may
    be only APPROXIMATELY isospectral. The energy objective is trustworthy ONLY once the
    frame is verified isospectral (compare the converged energy to the shared E∞ from
    the isospectral frames; a value BELOW E∞ is a red flag). Pass `fc_dress`/`order` to
    dress the residual transition/hopping legs (the STEP-4 escalation). Each scan point
    is a full TrimCI solve — keep `core` modest. Returns `{scale, energy, n_dets,
    entries, gen, bare_energy, scan, info}` (`scan` = list of `(s, energy, n_dets)`)."""
    from .run_cpp import _solver
    solve = _solver(True)
    info = None
    if entries is None:
        entries, _, info = analytic_displacement(H)
    gen = projector_generator(entries)
    if scales is None:
        scales = np.linspace(-1.6, 1.6, 17)
    res0 = solve(H, n_elec=n_elec, n_dets=core, seed=seed, n_runs=n_runs)
    best = {"scale": 0.0, "energy": float(res0.energy), "n_dets": int(res0.n_dets),
            "entries": entries, "gen": gen, "bare_energy": float(res0.energy),
            "scan": [(0.0, float(res0.energy), int(res0.n_dets))], "info": info}
    for s in scales:
        if abs(s) < 1e-12:
            continue
        Hf = displace_terms(H, lambdas=float(s), gen=gen, fc_dress=fc_dress, order=order)
        res = solve(Hf, n_elec=n_elec, n_dets=core, seed=seed, n_runs=n_runs)
        best["scan"].append((float(s), float(res.energy), int(res.n_dets)))
        if verbose:
            print(f"    scale={s:+.2f}  E={res.energy:.4f}  n_dets={res.n_dets}", flush=True)
        if res.energy < best["energy"]:
            best.update({"scale": float(s), "energy": float(res.energy),
                         "n_dets": int(res.n_dets)})
    return best


def combined_frame_terms(H, r=0.0, phi=0.0, lambdas=0.0, gen=None, fc_dress=None,
                         order=4, squeeze_first=True):
    """The layered squeeze ∘ displace frame `Ū = U†HU`, `U = U_sq · U_disp` (STEP 4).
    Composes the two Gaussian layers as term-list transforms:
      `squeeze_first=True`  → `displace_terms(squeeze_terms(H, r, φ), λ, ...)`
        (squeeze the quadratic/WT sector first, then displace the — now modified —
         linear coupling; the displacement generator is auto-read from the squeezed H),
      `squeeze_first=False` → `squeeze_terms(displace_terms(H, λ, ...), r, φ)`.
    Both orderings are exact frames (composition of unitaries ⇒ isospectral); the order
    is the variational convention. Exact when each layer is exact (density vertex +
    quadratic boson sector); for the transition vertex the displacement layer is
    leading-order / FC-dressed (see `displace_terms`). Reduces to pure squeeze (λ=0) or
    pure LF (r=0)."""
    if squeeze_first:
        Hs = squeeze_terms(H, r, phi)
        out = displace_terms(Hs, lambdas, gen=gen, fc_dress=fc_dress, order=order)
    else:
        Hd = displace_terms(H, lambdas, gen=gen, fc_dress=fc_dress, order=order)
        out = squeeze_terms(Hd, r, phi)
    out.meta = dict(H.meta)
    out.meta["frame"] = {"transform": "gaussian+LF", "r": _per_mode(r, H.n_bos_modes).tolist(),
                         "lambdas": _per_mode(lambdas, H.n_bos_modes).tolist(),
                         "squeeze_first": bool(squeeze_first)}
    return out


def frame_comparison(H, n_elec, r=0.0, phi=0.0, lambdas=0.0, gen=None, fc_dress=None,
                     order=4, squeeze_first=True):
    """ED frame-comparison (STEP 4 headline): compactness of the ground state in the
    bare / gaussian / LF / gaussian+LF frames on one enumerable system, at the given
    `(r, φ)` squeeze and `λ` displacement. Returns `{frame: compactness-dict}` plus the
    frame ground energies (all equal to bare up to cutoff leak — the isospectral
    check). Use to state "the layered frame gives X× fewer determinants."."""
    frames = {
        "bare": H,
        "gaussian": squeeze_terms(H, r, phi),
        "LF": displace_terms(H, lambdas, gen=gen, fc_dress=fc_dress, order=order),
        "gaussian+LF": combined_frame_terms(H, r, phi, lambdas, gen=gen,
                                            fc_dress=fc_dress, order=order,
                                            squeeze_first=squeeze_first),
    }
    _, g0, e0 = _ground_vector(H, n_elec)
    out = {}
    for name, Hf in frames.items():
        _, gf, ef = _ground_vector(Hf, n_elec)
        c = compactness(gf)
        c["E"] = ef
        c["dE_vs_bare"] = ef - e0
        c["n999_gain"] = compactness(g0)["n999"] / max(c["n999"], 1)
        out[name] = c
    return out


# ---------------------------------------------------------------------------
#  STEP 5 — fermion orbital rotation (U(n) Perelomov CS) = Core-Optimized Orbitals.
#  The FERMION analogue of the boson transforms: rotate the single-particle basis so
#  the ground state's determinant expansion is compact. Idle at A=1 (one particle has
#  no orbital-correlation to compact); first-class at large A (arXiv:2605.22977).
# ---------------------------------------------------------------------------

def rotate_orbitals_terms(H, R=None, kappa=None):
    """Fermion orbital-rotation frame `Ū = U†HU`,
    `U = exp[Σ_{pq} κ_{pq}(a_p†a_q − a_q†a_p)]` — the U(n) Perelomov coherent state
    (Core-Optimized Orbitals). Rotates the single-particle basis; each fermion ladder
    op transforms LINEARLY:
        a_p  → Σ_q R_{pq} a_q ,    a_p† → Σ_q R*_{pq} a_q† ,    R = exp(κ − κ^T).
    Pass either the rotation matrix `R` (n_ferm×n_ferm unitary) or the antisymmetric
    generator `kappa`. Isospectral (U unitary). Rotating to the NATURAL ORBITALS (the
    1-RDM eigenbasis, `natural_orbitals`) compacts the determinant expansion — a no-op
    at A=1, first-class at large A. Boson factors pass through unchanged. Cost: a
    k-fermion term spawns ≤ nnz(R)^k children (dense O(n⁴) for two-body ⇒ HPC-scale at
    n≈256; the sparse/structured regime keeps it tractable)."""
    from scipy.linalg import expm
    n = H.n_ferm_modes
    if R is None:
        if kappa is None:
            raise ValueError("rotate_orbitals_terms: pass R or kappa")
        k = np.asarray(kappa, dtype=complex)
        R = expm(k - k.T)                    # antisymmetrize -> unitary rotation
    R = np.asarray(R, dtype=complex)

    def subst(p, a):
        if a == 0:                           # a_p  -> Σ_q R_{pq} a_q
            return [(R[p, q], (q, 0)) for q in range(n) if abs(R[p, q]) > 1e-14]
        return [(np.conj(R[p, q]), (q, 1)) for q in range(n) if abs(R[p, q]) > 1e-14]

    acc = {}
    for t in H.terms:
        if not t.ferm_ops:                   # pure-boson / constant: unchanged
            key = (t.ferm_ops, t.bos_ops)
            acc[key] = acc.get(key, 0j) + complex(t.coeff)
            continue
        for combo in product(*[subst(p, a) for (p, a) in t.ferm_ops]):
            c = complex(t.coeff)
            fops = []
            for cf, op in combo:
                c *= cf
                fops.append(op)
            key = (tuple(fops), t.bos_ops)
            acc[key] = acc.get(key, 0j) + c
    terms = [OperatorTerm(c, fo, bo) for (fo, bo), c in acc.items() if abs(c) > 1e-14]
    out = MixedH(terms, H.n_ferm_modes, H.n_bos_modes, H.N_f)
    out.meta = dict(H.meta)
    out.meta["frame"] = {"transform": "COO", "n_ferm_modes": n}
    return out


def one_rdm(H, n_elec):
    """Fermion 1-RDM `γ_{pq} = ⟨g| a_p† a_q |g⟩` of the ED ground state (bosons traced
    out automatically — `a_p†a_q` acts only on the fermion factor). The seed for the
    natural-orbital rotation / the MCSCF orbital gradient."""
    basis, g, _ = _ground_vector(H, n_elec)
    n = H.n_ferm_modes
    gamma = np.zeros((n, n), dtype=complex)
    for p in range(n):
        for q in range(n):
            M = build_dense(MixedH([OperatorTerm(1.0, ((p, 1), (q, 0)), ())],
                                   n, H.n_bos_modes, H.N_f), basis)
            gamma[p, q] = np.vdot(g, M @ g)
    return gamma


def natural_orbitals(H, n_elec):
    """Natural-orbital rotation `R` (columns = the 1-RDM eigenvectors) and occupations
    (descending). `rotate_orbitals_terms(H, R=...)` moves into this basis, where the
    1-RDM is diagonal and the determinant expansion is maximally compact for a
    single-reference state (→ 1 determinant for a mean-field/one-body ground state)."""
    gamma = one_rdm(H, n_elec)
    gamma = 0.5 * (gamma + gamma.conj().T)
    occ, V = np.linalg.eigh(gamma)
    order = np.argsort(-occ)
    return V[:, order], occ[order]


def natural_orbital_terms(H, n_elec):
    """COO production entry point (ED scale): rotate H into its 1-RDM natural-orbital
    basis in one call. Wired to `build_from_eft(transform='COO', frame_params={
    'natural': True, 'n_elec': A})`. At scale, replace the ED `one_rdm` with the RDM
    read off the mixed TrimCI core (trace out bosons) and iterate (the COO loop)."""
    R, _ = natural_orbitals(H, n_elec)
    return rotate_orbitals_terms(H, R=R)


def _ferm_bitmasks(ferm_arr):
    """Reassemble the arrays-solver `(N, W)` little-endian uint64 fermion words into
    Python-int bitmasks (mirrors `backend.cpp_expand`'s `f |= word << 64*w`)."""
    a = np.asarray(ferm_arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    out = []
    for row in a:
        f = 0
        for w, word in enumerate(row):
            f |= int(word) << (64 * w)
        out.append(f)
    return out


def one_rdm_from_core(H, coeffs, ferm_arr, bos_arr):
    """Fermion 1-RDM `γ_pq = Σ_ij c_i* c_j ⟨det_i|a_p†a_q|det_j⟩` read off a TrimCI CORE
    (ED-free — the scalable replacement for `one_rdm`). Bosons trace out automatically:
    `a_p†a_q` is diagonal in boson occupation, so only core pairs with the SAME boson
    row and fermion configs related by a single `p←q` excitation contribute. Signs use
    the solver's own OpenFermion-parity primitive (`_apply_fermion_ops`), so this equals
    the ED `one_rdm` up to the selected-CI truncation of the core. `coeffs`/`ferm_arr`/
    `bos_arr` are the row-aligned arrays-solver result fields."""
    n = H.n_ferm_modes
    full = (1 << n) - 1
    c = np.asarray(coeffs, dtype=complex)
    masks = _ferm_bitmasks(ferm_arr)
    bos = np.asarray(bos_arr)
    gamma = np.zeros((n, n), dtype=complex)
    index = {(bos[i].tobytes(), masks[i]): i for i in range(len(masks))}
    for j, fj in enumerate(masks):
        cj = c[j]
        if cj == 0:
            continue
        bkey = bos[j].tobytes()
        wj = (cj.conjugate() * cj).real
        # diagonal γ_pp = Σ_j |c_j|² n_p(det_j)
        occ = fj
        while occ:
            p = (occ & -occ).bit_length() - 1
            gamma[p, p] += wj
            occ &= occ - 1
        # off-diagonal a_p†a_q : q occupied in det_j, p empty
        occ_q = fj
        while occ_q:
            q = (occ_q & -occ_q).bit_length() - 1
            occ_q &= occ_q - 1
            empt = (~fj) & full
            while empt:
                p = (empt & -empt).bit_length() - 1
                empt &= empt - 1
                fi = (fj & ~(1 << q)) | (1 << p)
                i = index.get((bkey, fi))
                if i is None:
                    continue
                sign, _ = _apply_fermion_ops(fj, ((p, 1), (q, 0)))
                gamma[p, q] += c[i].conjugate() * sign * cj
    return gamma


def natural_orbitals_from_core(H, coeffs, ferm_arr, bos_arr):
    """Natural-orbital rotation `R` (+ occupations, descending) from a TrimCI-core 1-RDM
    — the ED-free, scalable COO seed (production path at A=10–20). Same eigen-decomposition
    as `natural_orbitals`, but the 1-RDM comes from `one_rdm_from_core` instead of an
    enumerated ED vector. Apply with `rotate_orbitals_terms(H, R=R)`. Any unitary R is
    isospectral by construction, so a truncated-core seed only means LESS compaction,
    never a broken spectrum."""
    gamma = one_rdm_from_core(H, coeffs, ferm_arr, bos_arr)
    gamma = 0.5 * (gamma + gamma.conj().T)
    occ, V = np.linalg.eigh(gamma)
    order = np.argsort(-occ)
    return V[:, order], occ[order]


def natural_orbitals_smallcutoff(L, dim, n_elec, seed_n_b=1, params=None):
    """Validation COO seed: compute the natural-orbital rotation `R` from a SMALL-cutoff
    (`seed_n_b`, e.g. N_f=2) ED slice — cheap because the boson box is tiny. `n_ferm_modes`
    is independent of `n_b`, so `R` (an n_ferm×n_ferm U(n) rotation) transfers unchanged to
    the full-`n_b` Hamiltonian via `rotate_orbitals_terms(H_full, R=R)`; the natural-orbital
    basis depends only weakly on the boson cutoff. Used to cross-check
    `natural_orbitals_from_core` wherever both are enumerable. Returns `(R, occ)`."""
    from .hamiltonian import build_from_eft
    H_seed = build_from_eft(L, dim, seed_n_b, params=params)
    return natural_orbitals(H_seed, n_elec)


# ---------------------------------------------------------------------------
#  STEP 6 — non-Gaussianity DIAGNOSTIC (the gate). The non-Gaussian layer (Shi-
#  Demler-Cirac / Guaita generalized CS) is heavy and only worth building IF the
#  ground state, in the best Gaussian (squeeze+displace) frame, keeps a strongly
#  non-Gaussian residual. So STEP 6 = measure the residual non-Gaussianity and DECIDE.
#  Measure = relative-entropy non-Gaussianity of the boson-reduced state (Genoni-Paris-
#  Ferraro): δ = S(τ_G) − S(ρ_B), τ_G the Gaussian with ρ_B's 1st+2nd moments. δ=0 iff
#  ρ_B is Gaussian (vacuum, coherent, squeezed); δ>0 flags a non-Gaussian residual.
# ---------------------------------------------------------------------------

def _boson_ladder_matrices(n_bos, N_f):
    """Annihilation operators `a_m` (dense) on the full boson Fock space `[0,N_f)^n_bos`
    (dim N_f**n_bos), mode 0 most significant. Toy/ED sizes only."""
    a1 = np.diag(np.sqrt(np.arange(1, N_f)), 1)      # single-mode a: a|n>=√n|n-1>
    I = lambda d: np.eye(d)
    out = []
    for m in range(n_bos):
        left = N_f ** m
        right = N_f ** (n_bos - 1 - m)
        out.append(np.kron(np.kron(I(left), a1), I(right)))
    return out


def boson_reduced_dm(g, basis, n_bos, N_f):
    """Boson reduced density matrix `ρ_B = Tr_fermion |g⟩⟨g|` (dense, dim N_f**n_bos),
    from the ED ground vector `g` over `basis`. Groups amplitudes by fermion config and
    sums the boson outer products."""
    D = N_f ** n_bos
    groups = {}
    for i, st in enumerate(basis):
        bidx = 0
        for m in range(n_bos):
            bidx = bidx * N_f + st.bos[m]
        groups.setdefault(st.ferm, np.zeros(D, dtype=complex))[bidx] += g[i]
    rho = np.zeros((D, D), dtype=complex)
    for v in groups.values():
        rho += np.outer(v, v.conj())
    return rho


def _covariance(rho, a_list):
    """First moments `d` and covariance `σ` (quadrature convention x=(a+a†)/√2,
    p=(a−a†)/i√2; vacuum σ=I) of state `rho` over boson modes with ladder ops `a_list`."""
    n = len(a_list)
    quads = []
    for a in a_list:
        quads.append((a + a.conj().T) / np.sqrt(2))          # x_m
        quads.append((a - a.conj().T) / (1j * np.sqrt(2)))   # p_m
    d = np.array([np.trace(rho @ R).real for R in quads])
    sigma = np.zeros((2 * n, 2 * n))
    for i in range(2 * n):
        for j in range(2 * n):
            anti = quads[i] @ quads[j] + quads[j] @ quads[i]
            sigma[i, j] = np.trace(rho @ anti).real - 2 * d[i] * d[j]
    return d, sigma


def _symplectic_eigs(sigma):
    """Symplectic eigenvalues ν_k ≥ 1 of covariance `σ` (= |eig(iΩσ)|, positive half)."""
    n = sigma.shape[0] // 2
    omega = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    ev = np.abs(np.linalg.eigvals(1j * omega @ sigma))
    ev = np.sort(ev)[::-1]
    return ev[:n]


def _gaussian_entropy(sigma):
    """von Neumann entropy (bits) of the Gaussian state with covariance `σ`."""
    nu = np.clip(_symplectic_eigs(sigma), 1.0, None)
    def g(x):
        xp, xm = (x + 1) / 2, (x - 1) / 2
        return xp * np.log2(xp) - np.where(xm > 1e-12, xm * np.log2(np.maximum(xm, 1e-300)), 0.0)
    return float(np.sum(g(nu)))


def non_gaussianity(H, n_elec, frame=None):
    """Relative-entropy non-Gaussianity `δ = S(τ_G) − S(ρ_B)` (bits) of the boson-
    reduced ED ground state. VALIDATION ONLY (small enumerable systems): builds the
    full boson reduced DM (dim N_f**n_bos), so it does NOT scale — use
    `core_non_gaussianity` on a TrimCI core for the realistic-size STEP 6 go/no-go.
    `δ≈0`: Gaussian (a squeeze+displace frame captures it); `δ` large: non-Gaussian
    residual. Optional `frame` callable `H->MixedH` measures the residual after a frame.
    Returns `{delta, S_gaussian, S_reduced, purity, mean_n}`."""
    Hf = frame(H) if frame is not None else H
    basis, g, _ = _ground_vector(Hf, n_elec)
    rho = boson_reduced_dm(g, basis, Hf.n_bos_modes, Hf.N_f)
    a_list = _boson_ladder_matrices(Hf.n_bos_modes, Hf.N_f)
    _, sigma = _covariance(rho, a_list)
    S_g = _gaussian_entropy(sigma)
    w = np.clip(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real, 0, None)
    w = w[w > 1e-14]
    S_r = float(-np.sum(w * np.log2(w)))
    mean_n = float(sum(np.trace(rho @ (a.conj().T @ a)).real for a in a_list))
    return {"delta": max(S_g - S_r, 0.0), "S_gaussian": S_g, "S_reduced": S_r,
            "purity": float(np.trace(rho @ rho).real), "mean_n": mean_n}


def core_non_gaussianity(coeffs, ferm_arr, bos_arr, N_f):
    """Per-mode single-mode boson non-Gaussianity `δ_m` (bits) from a TrimCI CORE
    (ground-state amplitudes + boson-occupation array) — the SCALABLE STEP 6 gate: it
    never builds the full boson reduced DM (dim N_f**n_bos), only the small single-mode
    ρ_m (N_f×N_f), so it runs at realistic sizes (L=2/3, 3D). For each boson mode, ρ_m
    is assembled by grouping core determinants that agree on all OTHER degrees of
    freedom (fermion words + the other boson occupations), then `δ_m = S(τ_G) − S(ρ_m)`.
    A transition vertex makes each fermion config displace the boson differently ⇒ the
    boson marginal is a mixture of Gaussians ⇒ `δ_m>0`; `δ_m≈0` ⇒ Gaussian marginals ⇒
    the layered Gaussian frame suffices. (Marginal witness: `δ_m>0` proves
    non-Gaussian; `δ_m≈0` leaves only cross-mode-correlation non-Gaussianity untested.)
    Returns `{max, mean, per_mode, mean_n}`."""
    c = np.asarray(coeffs, dtype=complex)
    nrm = np.linalg.norm(c)
    if nrm > 0:
        c = c / nrm
    bos = np.asarray(bos_arr)
    ferm = np.asarray(ferm_arr)
    N, n_bos = bos.shape
    a1 = [np.diag(np.sqrt(np.arange(1, N_f)), 1)]     # single-mode annihilation
    deltas = np.zeros(n_bos)
    for m in range(n_bos):
        groups = {}
        for i in range(N):
            key = (ferm[i].tobytes(), bos[i, :m].tobytes(), bos[i, m + 1:].tobytes())
            groups.setdefault(key, []).append(i)
        rho = np.zeros((N_f, N_f), dtype=complex)
        for idxs in groups.values():
            v = np.zeros(N_f, dtype=complex)
            for i in idxs:
                v[bos[i, m]] += c[i]                  # occ_m is the only free index in a group
            rho += np.outer(v, v.conj())
        _, sig = _covariance(rho, a1)
        S_g = _gaussian_entropy(sig)
        w = np.clip(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real, 0, None)
        w = w[w > 1e-14]
        deltas[m] = max(S_g + float(np.sum(w * np.log2(w))), 0.0)   # S_g - S_r
    mean_n = float(np.sum(np.abs(c) ** 2 * bos.sum(axis=1)))
    return {"max": float(deltas.max()), "mean": float(deltas.mean()),
            "per_mode": deltas, "mean_n": mean_n}
