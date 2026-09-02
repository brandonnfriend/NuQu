"""Original-H back-evaluation — restore the Ritz upper bound for the approximate LF frame.

See `docs/lf_backevaluation.md`. The truncated (leading-order) Lang-Firsov frame is NOT
isospectral, so its frame-internal energy `E_frame = ⟨ψ̃|H̃_m|ψ̃⟩` has no variational relation
to the physical spectrum. The fix: take the compact frame-solved state `|ψ̃⟩`, map it to the
physical frame with the EXACT unitary `|ψ⟩ = exp(λS)|ψ̃⟩`, and score it against the reference
(bare / squeezed) Hamiltonian:

    E_orig = ⟨ψ|H_ref|ψ⟩ / ⟨ψ|ψ⟩ ≥ E0(H_ref)   — a legitimate variational energy (Ritz bound).

Everything is sparse and reuses the existing determinant machinery (`hij.apply_term`,
`hij.connections_nocache`), so the same code serves both tiers:
  * Tier A (ED validation): feed the dense ED framed ground state as a state dict, and
    cross-check `E_orig` against the exact dense `expm(λS)` (see `back_evaluate_ed`).
  * Tier B (production): feed the selected-CI solved core `(dets, coeffs)`; the exp(λS)
    map-back is a Taylor series in the (anti-Hermitian) generator `S`, evaluated ONCE on the
    converged state — not inside the solve loop, so the Franck-Condon fan-out that makes
    *solving* the exact-dressed frame intractable does not apply. Cost is the fan-out of a
    compact core (mapping it back UN-compacts it); back-evaluate at representative core sizes.

This module handles the LF displacement generator (the frame that needs it); the squeeze is
~isospectral and keeps its `E_frame`. A composed `gaussian+lf` back-map (squeeze ∘ LF) is a
straightforward extension (compose the generators) once needed.
"""

import numpy as np

from .hij import apply_term, connections_nocache
from .lf import displacement_generator, _ground_vector
from .state import enumerate_basis


def _apply_terms(op_terms, state_dict, N_f, prune=0.0):
    """Apply a MixedH term list to {MixedState: coeff} -> {MixedState: coeff} (sparse).
    `prune`=0 keeps all non-zero children (accurate for the Taylor map-back — pruning
    intermediates biases exp(λS) at strong displacement)."""
    out = {}
    for st, c in state_dict.items():
        if c == 0:
            continue
        for term in op_terms:
            res = apply_term(term, st, N_f)
            if res is None:
                continue
            amp, s2 = res
            out[s2] = out.get(s2, 0.0j) + c * amp
    if prune <= 0:
        return {s: v for s, v in out.items() if v != 0}
    return {s: v for s, v in out.items() if abs(v) > prune}


def _norm(sd):
    return float(np.sqrt(sum(abs(v) ** 2 for v in sd.values())))


def _truncate_weight(d, cap):
    """Keep the top-`cap` entries of `{key: coeff}` by |coeff|²; return (kept, dropped_weight)
    where dropped_weight = Σ|dropped|² / Σ|all|² (a renormalization-invariant loss measure)."""
    if len(d) <= cap:
        return d, 0.0
    items = sorted(d.items(), key=lambda kv: -abs(kv[1]) ** 2)
    tot = sum(abs(c) ** 2 for _s, c in items)
    kept = dict(items[:cap])
    dropped = sum(abs(c) ** 2 for _s, c in items[cap:]) / tot if tot > 0 else 0.0
    return kept, dropped


def exp_generator_apply(S, lam, state_dict, N_f, max_order=200, tol=1e-12, support_cap=None):
    """`|ψ⟩ = exp(λS)|ψ̃⟩` via sparse Taylor with ENFORCED convergence. `S` anti-Hermitian,
    `λ` real ⇒ exp(λS) is unitary. Convergence is declared only when the series is in a
    DECREASING regime and a geometric TAIL BOUND `‖v_k‖·ρ/(1−ρ)` (ρ = ‖v_k‖/‖v_{k−1}‖ < 1)
    is below `tol` — a small individual term is not by itself sufficient (audit gap 4).

    `support_cap` (audit tractability fallback): if set, weight-truncate the running Taylor
    term (and final state) to the top-`support_cap` determinants after each order, bounding the
    combinatorial fan-out that makes dense-filling LF map-back intractable. The truncated map
    yields a DIFFERENT trial state whose Rayleigh quotient is STILL variational; `dropped_weight`
    is returned so the caller can convergence-test (E_orig should converge as the cap rises).

    Returns `(psi_dict, info)`. `converged=False` means the order cap was hit without a certified
    tail — treat as a failure, not silently trusted.
    """
    psi = dict(state_dict)
    v = dict(state_dict)
    prev_norm = _norm(v)
    dropped_total = 0.0
    info = {'order': 0, 'converged': False, 'tail_bound': None,
            'max_term_norm': prev_norm, 'max_support': len(v), 'dropped_weight': 0.0}
    for k in range(1, max_order + 1):
        Sv = _apply_terms(S.terms, v, N_f)                 # S · v_{k-1}
        v = {s: (lam / k) * c for s, c in Sv.items()}      # v_k = (λ/k) S v_{k-1} = (λS)^k/k!
        info['order'] = k
        info['max_support'] = max(info['max_support'], len(v))
        if not v:                                          # exact termination (finite generator)
            info['converged'] = True
            info['tail_bound'] = 0.0
            break
        for s, c in v.items():
            psi[s] = psi.get(s, 0.0j) + c
        vk = _norm(v)
        info['max_term_norm'] = max(info['max_term_norm'], vk)
        rho = vk / prev_norm if prev_norm > 0 else float('inf')
        prev_norm = vk
        if support_cap:                                    # bound the fan-out (dense fallback)
            v, drop = _truncate_weight(v, support_cap)
            dropped_total += drop
        if rho < 1.0:                                      # decreasing regime → geometric tail bound
            tail = vk * rho / (1.0 - rho)
            info['tail_bound'] = tail
            if tail < tol:
                info['converged'] = True
                break
    if support_cap and len(psi) > support_cap:
        psi, drop = _truncate_weight(psi, support_cap)
        dropped_total += drop
    info['dropped_weight'] = dropped_total
    return {s: c for s, c in psi.items() if abs(c) > 1e-14}, info


def kato_temple_lower(E, residual, beta):
    """Kato–Temple certified LOWER bound on the true ground energy from a Rayleigh
    quotient `E`, its residual norm `residual = ‖(H−E)ψ‖/‖ψ‖`, and `beta` — ANY rigorous
    lower bound on the FIRST EXCITED level (E_1) with `beta > E`. Then
        E − residual² / (beta − E)  ≤  E_0  ≤  E.
    Together with the Ritz upper bound `E`, this brackets E_0 in a certified interval at
    zero extra cost (the residual is already computed by `rayleigh`). Returns None if the
    hypothesis `beta > E` fails (can't certify) or inputs are missing. See Kato 1949 /
    Temple 1928; the residual is the honest one from the mapped-back (variational) state."""
    if E is None or residual is None or beta is None:
        return None
    if not (beta > E):
        return None
    return float(E - (residual ** 2) / (beta - E))


def rayleigh(H_ref, state_dict):
    """`(E, residual)` = `⟨ψ|H_ref|ψ⟩/⟨ψ|ψ⟩` and `‖(H_ref−E)ψ‖/‖ψ‖`, sparse (matrix-free)."""
    Hpsi = {}
    for st, c in state_dict.items():
        for row, hij in connections_nocache(H_ref, st).items():   # H_ref|st⟩ column
            Hpsi[row] = Hpsi.get(row, 0.0j) + hij * c
    den = sum(abs(v) ** 2 for v in state_dict.values())
    num = sum(np.conj(state_dict.get(s, 0.0j)) * v for s, v in Hpsi.items())
    E = float((num / den).real)
    r2 = sum(abs(Hpsi.get(s, 0.0j) - E * state_dict.get(s, 0.0j)) ** 2
             for s in set(Hpsi) | set(state_dict))
    return E, float(np.sqrt(r2 / den))


def back_evaluate(H_ref, S, lam, state_dict, max_order=200, tol=1e-12, strict=False,
                  support_cap=None):
    """Map the frame-solved `|ψ̃⟩` (state dict) through a SINGLE exp(λS) to the physical frame
    and score it against `H_ref`. Returns the variational `E_orig`, the residual, and
    provenance (support growth, Taylor order/convergence, norm ratio, dropped weight).
    `strict=True` raises if the Taylor map-back did not certify convergence (audit gap 4);
    `support_cap` bounds the fan-out (dense fallback — still variational, convergence-test the cap)."""
    psi, info = exp_generator_apply(S, lam, state_dict, H_ref.N_f, max_order, tol, support_cap)
    if strict and not info['converged']:
        raise RuntimeError(
            f"exp(λS) map-back did not converge (order {info['order']}, "
            f"tail_bound {info['tail_bound']}) — tighten max_order/tol or reduce λ‖S‖")
    E, resid = rayleigh(H_ref, psi)
    nr = _norm(psi) / max(_norm(state_dict), 1e-300)
    return {
        'E_orig': E, 'residual': resid,
        'support_in': len(state_dict), 'support_out': len(psi),
        'max_support': info['max_support'], 'dropped_weight': info['dropped_weight'],
        'taylor_order': info['order'], 'converged': info['converged'],
        'tail_bound': info['tail_bound'], 'max_term_norm': info['max_term_norm'],
        'norm_ratio': nr,
        # ε_leak = 1 − ‖P_{N_f} U|ψ̃⟩‖² / ‖U|ψ̃⟩‖²: the exact state-specific norm the frame
        # unitary pushes past the Fock ceiling (U is unitary ⇒ ‖U|ψ̃⟩‖=‖|ψ̃⟩‖). It quantifies
        # how TIGHT E_orig is (validity never depends on it; ε_leak→0 ⇒ back-eval is exact for
        # this state). Meaningful only with support_cap=None (else it conflates cap pruning).
        'eps_leak': max(0.0, 1.0 - nr * nr),
    }


def state_dict_from_result(result):
    """Adapter: a production `GroundStateResult` -> {MixedState: coeff}. Handles both the
    object-native (`.states`) and array-native (`.ferm_arr`/`.bos_arr`) core formats."""
    from .state import MixedState
    coeffs = result.coeffs
    if getattr(result, 'states', None):
        return {s: complex(c) for s, c in zip(result.states, coeffs)
                if abs(c) > 1e-14}
    fa, ba = result.ferm_arr, result.bos_arr
    if fa is None or ba is None:
        raise ValueError("result has neither .states nor .ferm_arr/.bos_arr")
    out = {}
    for i in range(fa.shape[0]):
        ferm = 0
        for w in range(fa.shape[1] if fa.ndim > 1 else 1):
            word = int(fa[i, w]) if fa.ndim > 1 else int(fa[i])
            ferm |= word << (64 * w)
        c = complex(coeffs[i])
        if abs(c) > 1e-14:
            out[MixedState(ferm, tuple(int(x) for x in ba[i]))] = c
    return out


def back_evaluate_result(H_ref, S, lam, result, max_order=60, tol=1e-13):
    """Tier-B convenience: back-evaluate a production selected-CI `GroundStateResult`
    (frame-solved) against `H_ref`. Returns the `back_evaluate` dict plus `E_frame`
    (the result's frame-internal energy) and `frame_shift = E_frame − E_orig`."""
    sd = state_dict_from_result(result)
    out = back_evaluate(H_ref, S, lam, sd, max_order=max_order, tol=tol)
    out['E_frame'] = float(getattr(result, 'energy', float('nan')))
    out['frame_shift_vs_orig'] = out['E_frame'] - out['E_orig']
    return out


def generator_from_disp_gen(disp_gen, n_ferm, n_bos, N_f):
    """Anti-Hermitian LF state-map generator `S = Σ_m Σ_(λ,P) [λ P b†_m − conj(λ) P† b_m]`
    from a PRODUCTION projector-conditioned `disp_gen = {m: [(λ, ferm_ops), ...]}`
    (frame.projector_generator: `P = n̂_p`). This is the exact generator matching the
    production `frame.displace_terms(gen=disp_gen)` — NOT `lf.displacement_generator(H)`,
    which reads the physical transition coupling instead (audit gap 2). Because the `n̂_p`
    projectors commute, this generator's boson substitution is exact/finite."""
    from .hamiltonian import MixedH, OperatorTerm
    from .lf import _dagger_ferm
    terms = []
    for m, dl in disp_gen.items():
        for (lam, ferm_ops) in dl:
            fo = tuple(ferm_ops)
            terms.append(OperatorTerm(complex(lam), fo, ((int(m), 1),)))          # +λ P b†_m
            terms.append(OperatorTerm(-np.conj(complex(lam)), _dagger_ferm(fo),
                                      ((int(m), 0),)))                             # −conj(λ) P† b_m
    return MixedH(terms, n_ferm, n_bos, N_f)


def back_evaluate_frame(H_bare, state, result, strict=False, max_order=200, tol=1e-12,
                        support_cap=None):
    """Back-evaluate a PRODUCTION composed frame against the original H (audit gap 3).

    `state` is the `frame_workflow` frame state (squeeze `r`/`phi`, projector-LF
    `disp_gen`/`disp_scale`, optional COO `R`). The physical trial state is
    `|ψ⟩ = U_sq U_lf |ψ̃⟩ = exp(G_sq) · exp(disp_scale·S_lf) · |ψ̃⟩` (apply LF then squeeze,
    matching `apply_frame`'s squeeze→LF frame build so `U = U_sq U_lf`). Scores
    `E_orig = ⟨ψ|H_bare|ψ⟩` (variational). COO orbital-rotation map-back is not yet
    implemented — raises if `R` is set. `result` is a `GroundStateResult` or a state dict.
    `support_cap` bounds the map-back fan-out for the dense-filling case (audit tractability
    fallback): still variational; `dropped_weight` is returned to convergence-test the cap.
    """
    from . import frame as _frame
    if state.get("R") is not None:
        raise NotImplementedError(
            "COO orbital-rotation map-back not implemented; use bare/gaussian/lf/gaussian+lf")
    sd = dict(result) if isinstance(result, dict) else state_dict_from_result(result)
    N_f = H_bare.N_f
    psi = sd
    steps, converged, max_supp, dropped = [], True, len(sd), 0.0

    # inner: LF (projector-conditioned), exp(disp_scale · S_lf)
    if state.get("disp_gen") is not None and abs(state.get("disp_scale", 0.0)) > 1e-12:
        S_lf = generator_from_disp_gen(state["disp_gen"], H_bare.n_ferm_modes,
                                       H_bare.n_bos_modes, N_f)
        psi, info = exp_generator_apply(S_lf, float(state["disp_scale"]), psi, N_f,
                                        max_order, tol, support_cap)
        converged &= info['converged']; max_supp = max(max_supp, info['max_support'])
        dropped += info['dropped_weight']; steps.append(('lf', info['order'], info['converged']))

    # outer: squeeze (Gaussian), exp(G_sq)
    r = state.get("r")
    if r is not None and np.any(np.abs(np.asarray(r, dtype=float)) > 1e-12):
        G_sq = _frame.squeeze_generator_terms(H_bare, r, state.get("phi", 0.0))
        psi, info = exp_generator_apply(G_sq, 1.0, psi, N_f, max_order, tol, support_cap)
        converged &= info['converged']; max_supp = max(max_supp, info['max_support'])
        dropped += info['dropped_weight']; steps.append(('sq', info['order'], info['converged']))

    if strict and not converged:
        raise RuntimeError(f"composed frame map-back did not converge: steps={steps}")
    E, resid = rayleigh(H_bare, psi)
    nr = _norm(psi) / max(_norm(sd), 1e-300)
    out = {'E_orig': E, 'residual': resid, 'support_in': len(sd), 'support_out': len(psi),
           'max_support': max_supp, 'dropped_weight': dropped, 'map_steps': steps,
           'converged': converged, 'norm_ratio': nr,
           # ε_leak = 1 − ‖P_{N_f} U|ψ̃⟩‖²: norm the composed frame unitary leaks past the Fock
           # ceiling for THIS state (the silent hij.py ceiling drop, now surfaced). Sets how
           # TIGHT E_orig is, not its validity. Raise the reference N_f (H_bare) if it's large.
           'eps_leak': max(0.0, 1.0 - nr * nr)}
    ef = None if isinstance(result, dict) else getattr(result, 'energy', None)
    if ef is not None:
        out['E_frame'] = float(ef)
        out['frame_shift_vs_orig'] = float(ef) - E
    return out


def back_evaluate_ed(L, dim, n_b, lam, A=1, params=None, coupling_scale=1.0,
                     dense_check=True):
    """Tier-A ED validation (TINY, memory-bounded systems only). Solve the leading-order LF
    frame, back-evaluate on the bare H, and (optionally) cross-check the sparse map-back
    against the exact dense `expm(λS)`. Returns the full E_bare / E_frame / E_orig comparison.

    E_frame is the non-variational frame-internal energy; E_orig is the variational number
    (≥ E_bare). `gap_orig = E_orig − E_bare ≥ 0` is the honest LF penalty; `E_frame − E_bare`
    (which can be NEGATIVE) is the non-isospectral spectrum shift the fix removes from claims.

    CAVEAT: back-eval is well-defined only when the framed ground state is non-degenerate and
    away from the Fock-cutoff boundary. Artificially strong coupling on a tiny `N_f` drives the
    leading-order frame's ground state into a DEGENERATE boundary artifact (E_frame above
    E_bare); `eigh` then returns an arbitrary degenerate combination and the back-map is
    ill-defined — use adequate `N_f`. The physical regime (`coupling_scale=1`, converged `N_f`)
    is clean.
    """
    from .hamiltonian import build_from_eft
    from .frame import displace_terms
    from .lf import scale_linear_coupling

    H = scale_linear_coupling(build_from_eft(L, dim, n_b, params), coupling_scale)
    S = displacement_generator(H)
    Hf = displace_terms(H, lam)                             # leading-order LF frame (fc_dress=None)
    basis, gf, ef = _ground_vector(Hf, A)                  # framed ground state
    _, g0, e0 = _ground_vector(H, A)                       # bare exact ground

    state_dict = {basis[i]: complex(gf[i]) for i in range(len(basis))
                  if abs(gf[i]) > 1e-14}
    res = back_evaluate(H, S, lam, state_dict)
    out = {
        'E_bare': e0, 'E_frame': ef, 'E_orig': res['E_orig'],
        'gap_orig': res['E_orig'] - e0,                    # ≥ 0 (variational)
        'frame_shift': ef - e0,                            # can be < 0 (non-isospectral)
        'residual': res['residual'], 'norm_ratio': res['norm_ratio'],
        'support_in': res['support_in'], 'support_out': res['support_out'],
        'taylor_order': res['taylor_order'], 'n_states': len(basis),
    }
    if dense_check:
        from scipy.linalg import expm
        from .hij import build_dense
        U = expm(lam * build_dense(S, basis))
        psi_d = U @ gf
        Hd = build_dense(H, basis)
        E_dense = float((psi_d.conj() @ Hd @ psi_d).real / (psi_d.conj() @ psi_d).real)
        out['E_orig_dense'] = E_dense
        out['sparse_vs_dense'] = abs(res['E_orig'] - E_dense)
    return out
