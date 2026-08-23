"""Initial-state-preparation cost for the classically-informed QPE warm start (total_costs §3–4).

The warm start `|ψ_init⟩ = U|ψ̃⟩` is built in two pieces (`docs/frame_warmstart_p0_fidelity.md`):

  1. **Fermionic core load** — a superposition of `D` Slater determinants (the compact classical
     ground state) loaded by a QROAM / SelectSwap state-prep (Low–Kliuchnikov–Schaeffer 2018,
     arXiv:1812.00954). Cost `O(√(D·b))` Toffoli at the space–time sweet spot; this is the
     dominant, D-dependent piece.
  2. **Gaussian frame circuit U** — per-mode single-mode squeeze (⊗ optional displacement), an
     analytic Gaussian unitary: a handful of rotations per pion mode, no QROM. Cheap.

Both are ONE-TIME per QPE shot and, per Berry 2024 §VII.C, **sub-dominant to the qubitized walk**
— this module quantifies that. It returns the Toffoli/T count `T_prep` and the ancilla width
`a_prep`; `a_prep` enters the logical total as `walk + max(m_QPE, a_prep)` (NOT the sum — state
prep resets before the phase register is populated; see `qpe_cost.total_logical_qubits`).

⚠ **Name collision:** Low 2018's space–time knob is written `λ` (the QROAM copy count). That is
NOT NuQu's `λ` (the block-encoding / Pauli one-norm). Here it is `lambda_qr` and only ever the
copy count.

References (local): `claude/research/qpe_gsee_algorithms/Low_2018_QROAM_dirty_qubits_state_prep.pdf`,
`.../Berry_2024_rapid_initial_state_prep_filtering.pdf`.
"""

import math


# One Toffoli decomposed to T gates. Modern catalyzed/measurement-assisted Toffoli = 4 T
# (Gidney 2018); the older Toffoli-via-T-ladder is 7 T. We report Toffoli AND T=4·Toffoli
# and expose the factor so the assembler can switch conventions.
T_PER_TOFFOLI = 4

# Amplitude-precision bits for the loaded coefficients. b≈17 ↔ ε≈1e-3 (Low 2018 Fig. 2 /
# Berry 2024); the coefficient error folds into the QPE overlap, not the ΔE budget.
DEFAULT_AMPLITUDE_BITS = 17

# Berry 2024 interferometric synthesis reduces the Low/LKS Toffoli by ~7× (Table I). Optional.
BERRY_SYNTHESIS_SPEEDUP = 7.0


def _nearest_pow2(x):
    if x <= 1:
        return 1
    return 2 ** int(round(math.log2(x)))


def qroam_state_prep_cost(D, b=DEFAULT_AMPLITUDE_BITS, lambda_qr=None, berry_synthesis=False):
    """QROAM / SelectSwap cost to load a `D`-determinant, `b`-bit amplitude state (Low 2018).

    SelectSwap with `k = lambda_qr` copies costs `Toffoli ≈ ⌈D/k⌉ + b·(k−1)` — the QROM readout
    `D/k` plus the swap-up `b·(k−1)` — minimized at `k ≈ √(D/b)` giving `≈ 2√(D·b)`. Ancilla:
    `b + ⌈log₂D⌉` clean (output + address) and `(k−1)·b` dirty (the SelectSwap copies, borrowable).

    `lambda_qr=None` picks the T-optimal power-of-two `k`; `lambda_qr=1` is the serial min-width
    QROM (`Toffoli ≈ D`, ancilla `b + 2⌈log₂D⌉`) — the width↔depth knob. `berry_synthesis=True`
    applies the ~7× Berry-2024 reduction. Returns `{toffoli, t, ancilla_clean, ancilla_dirty,
    ancilla_total, lambda_qr, depth_toffoli}`.
    """
    D = max(1, int(D))
    b = int(b)
    addr = max(1, math.ceil(math.log2(max(D, 2))))
    k = _nearest_pow2(math.sqrt(D / b)) if lambda_qr is None else max(1, int(lambda_qr))
    toffoli = math.ceil(D / k) + b * (k - 1)
    if berry_synthesis:
        toffoli = math.ceil(toffoli / BERRY_SYNTHESIS_SPEEDUP)
    ancilla_clean = b + addr
    ancilla_dirty = (k - 1) * b
    return {
        "toffoli": int(toffoli), "t": int(toffoli * T_PER_TOFFOLI),
        "ancilla_clean": int(ancilla_clean), "ancilla_dirty": int(ancilla_dirty),
        "ancilla_total": int(ancilla_clean + ancilla_dirty),
        "lambda_qr": int(k), "depth_toffoli": int(toffoli),
    }


def _t_rot(eps_synth=1e-10):
    """Ross–Selinger T-count for one arbitrary single-qubit rotation to precision ε."""
    return 3.067 * math.log2(1.0 / eps_synth) + 9.678


def gaussian_prep_cost(n_bos_modes, N_f, *, displace=False, eps_synth=1e-10):
    """Per-mode Gaussian frame circuit U cost (analytic; no QROM).

    Each pion mode carries a single-mode squeeze = an `N_f×N_f` unitary on its `⌈log₂N_f⌉`
    qubits, synthesized with ~`N_f²/2` Givens rotations (a conservative general-small-unitary
    count; the squeeze is banded so the true count is lower). Optional per-mode displacement adds
    ~`⌈log₂N_f⌉` rotations. In-place on the mode register → `a_prep ≈ 0` extra ancilla (a handful
    for rotation synthesis, absorbed). Returns `{rotations, t, ancilla}`."""
    n = int(n_bos_modes)
    rot_sq = 0.5 * N_f * N_f
    rot_dp = math.ceil(math.log2(max(N_f, 2))) if displace else 0
    rotations = n * (rot_sq + rot_dp)
    return {"rotations": rotations, "t": rotations * _t_rot(eps_synth), "ancilla": 0}


def state_prep_cost(D, n_bos_modes, N_f, *, b=DEFAULT_AMPLITUDE_BITS, lambda_qr=None,
                    displace=False, berry_synthesis=False, eps_synth=1e-10):
    """Total warm-start prep = QROAM fermionic-core load + Gaussian frame circuit U.

    Returns `{T_prep, a_prep, T_qroam, T_gaussian, qroam, gaussian}`:
      * `T_prep`  — total T-count (QROAM Toffoli→T + Gaussian rotations→T);
      * `a_prep`  — peak state-prep ancilla = the QROAM ancilla (Gaussian U is in-place). This is
        the value that enters `total_logical_qubits(walk, m, a_prep)` as `walk + max(m, a_prep)`.
    """
    qr = qroam_state_prep_cost(D, b=b, lambda_qr=lambda_qr, berry_synthesis=berry_synthesis)
    gp = gaussian_prep_cost(n_bos_modes, N_f, displace=displace, eps_synth=eps_synth)
    return {
        "T_prep": qr["t"] + gp["t"], "a_prep": qr["ancilla_total"],
        "T_qroam": qr["t"], "T_gaussian": gp["t"], "qroam": qr, "gaussian": gp,
    }
