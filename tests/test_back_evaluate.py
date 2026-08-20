"""Original-H back-evaluation for the approximate LF frame (docs/lf_backevaluation.md).

TINY ED systems only (memory-bounded): L=2 d=1 n_b=1 A=1 is ~512 states. Verifies:
  * the non-isospectrality trap is real — E_frame can fall BELOW E_bare (non-variational);
  * back-evaluation restores the Ritz bound — E_orig >= E_bare for every case;
  * the sparse exp(λS) map-back (the production/Tier-B path) equals the exact dense expm;
  * the map-back is unitary (norm preserved).

Run: python -m tests.test_back_evaluate
"""
import sys

from classical.trimci.back_evaluate import back_evaluate_ed, back_evaluate
from classical.trimci.hamiltonian import build_from_eft
from classical.trimci.lf import displacement_generator, _ground_vector
from classical.trimci.frame import displace_terms


def main():
    fails = []
    L, dim, n_b, A = 2, 1, 1, 1                      # ~512 states — tiny

    # Physical coupling only (coupling_scale=1.0). NOTE: artificially strong coupling on a
    # tiny N_f=2 cutoff drives the leading-order framed ground state into a DEGENERATE
    # Fock-boundary artifact (E_frame above E_bare); back-eval on a degenerate boundary
    # state is ill-defined (arbitrary eigh combination) and outside the diagnostic's
    # validity — use adequate N_f. See the module docstring caveat.
    saw_negative_shift = False
    for lam in (0.1, 0.2, 0.3):
        r = back_evaluate_ed(L, dim, n_b, lam=lam, A=A, coupling_scale=1.0)
        tag = f"lam={lam}"
        # (1) Ritz bound restored: E_orig >= E_bare (any normalized state).
        if r['gap_orig'] < -1e-6:
            fails.append(f"{tag}: E_orig below E_bare (gap {r['gap_orig']:.2e}) — Ritz violated")
        # (2) sparse map-back == exact dense expm(λS).
        if r['sparse_vs_dense'] > 1e-6:
            fails.append(f"{tag}: sparse vs dense {r['sparse_vs_dense']:.2e} > 1e-6")
        # (3) unitary map-back.
        if abs(r['norm_ratio'] - 1.0) > 1e-9:
            fails.append(f"{tag}: norm_ratio {r['norm_ratio']} != 1 (map-back not unitary)")
        if r['frame_shift'] < -1e-4:
            saw_negative_shift = True

    # (4) The motivating pathology must actually occur: for some frame the internal energy
    #     drops BELOW the true ground energy (non-variational), which back-eval removes.
    if not saw_negative_shift:
        fails.append("expected some E_frame < E_bare (the non-isospectral trap) — none seen")

    # (5) Tier-B interface: feed a solved state dict straight to back_evaluate (no ED wrapper).
    H = build_from_eft(L, dim, n_b)
    S = displacement_generator(H)
    Hf = displace_terms(H, 0.25)
    basis, gf, ef = _ground_vector(Hf, A)
    sd = {basis[i]: complex(gf[i]) for i in range(len(basis)) if abs(gf[i]) > 1e-14}
    res = back_evaluate(H, S, 0.25, sd)
    _, _, e0 = _ground_vector(H, A)
    if res['E_orig'] < e0 - 1e-6:
        fails.append(f"Tier-B back_evaluate: E_orig {res['E_orig']:.4f} < E_bare {e0:.4f}")
    if abs(res['norm_ratio'] - 1.0) > 1e-9:
        fails.append(f"Tier-B back_evaluate: non-unitary map-back {res['norm_ratio']}")

    # report
    demo = back_evaluate_ed(L, dim, n_b, lam=0.2, A=A, coupling_scale=1.0)
    print("=" * 60)
    print("   LF ORIGINAL-H BACK-EVALUATION")
    print("=" * 60)
    print(f"demo (cs×1, lam=0.2, {demo['n_states']} states):")
    print(f"  E_bare  = {demo['E_bare']:.5f}")
    print(f"  E_frame = {demo['E_frame']:.5f}  (shift {demo['frame_shift']:+.4f}  <- NON-variational)")
    print(f"  E_orig  = {demo['E_orig']:.5f}  (gap  {demo['gap_orig']:+.5f}  >= 0, Ritz-valid)")
    print(f"  sparse-vs-dense map-back = {demo['sparse_vs_dense']:.1e}   norm_ratio = {demo['norm_ratio']:.9f}")
    print(f"  support {demo['support_in']} -> {demo['support_out']} (Taylor order {demo['taylor_order']})")
    print("=" * 60)
    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: E_frame is non-variational (can drop below E_bare); back-eval restores "
          "E_orig >= E_bare; sparse map-back == exact dense expm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
