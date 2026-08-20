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


def exp_generator_apply(S, lam, state_dict, N_f, max_order=60, tol=1e-13):
    """`|ψ⟩ = exp(λS)|ψ̃⟩` via sparse Taylor. `S` anti-Hermitian, `λ` real ⇒ exp(λS) unitary
    (norm-preserving — a convergence check). Returns `(psi_dict, taylor_order)`."""
    psi = dict(state_dict)
    v = dict(state_dict)
    order = 0
    for k in range(1, max_order + 1):
        Sv = _apply_terms(S.terms, v, N_f)                 # S · v_{k-1}
        v = {s: (lam / k) * c for s, c in Sv.items()}      # v_k = (λ/k) S v_{k-1} = (λS)^k/k!
        if not v:
            order = k
            break
        for s, c in v.items():
            psi[s] = psi.get(s, 0.0j) + c
        order = k
        if _norm(v) < tol:
            break
    return {s: c for s, c in psi.items() if abs(c) > 1e-14}, order


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


def back_evaluate(H_ref, S, lam, state_dict, max_order=60, tol=1e-13):
    """Map the frame-solved `|ψ̃⟩` (state dict) to the physical frame and score it against
    `H_ref`. Returns the variational `E_orig`, the residual, and provenance (support growth,
    Taylor order, norm ratio — a unitarity check on the map-back)."""
    psi, order = exp_generator_apply(S, lam, state_dict, H_ref.N_f, max_order, tol)
    E, resid = rayleigh(H_ref, psi)
    return {
        'E_orig': E, 'residual': resid,
        'support_in': len(state_dict), 'support_out': len(psi),
        'taylor_order': order,
        'norm_ratio': _norm(psi) / max(_norm(state_dict), 1e-300),
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
