"""
Precision / error budget for the compiled sparse QPE estimate (C1 sparse P0-4).

Codex's audit flagged the hard-coded `probability_epsilon=1e-3`, `kappa=8`:
"A single unexplained setting is not publication-grade. Derive `kappa` and
alias/QROM precision from the desired energy accuracy and query count."

This module derives every approximation parameter from one physical input — the
target QPE energy accuracy `ΔE` (MeV) — and the block-encoding subnormalisation
`λ`, with a transparent, worst-case error allocation.

The error model (standard qubitized QPE)
----------------------------------------
Two independent error sources add to the total energy error:

  ΔE_total  ≤  ε_QPE  +  ε_BE  ≤  ΔE.

* **ε_QPE** — the phase-estimation *resolution*. Qubitization maps eigenvalue
  `E` to eigenphase `arccos(E/λ)`, resolved to `ε_QPE` with
  `N_walk = √2·π·λ / ε_QPE` walk queries (Babbush 2018 Eq. 9; the same constant
  `qpe_cost.walk_queries` uses).

* **ε_BE** — the *block-encoding* error. The compiled walk qubitizes `H + δH`,
  not `H`, where `δH` collects the approximations in PREP (finite-precision LCU
  coefficients) and in the amplitude-loading **rotations** (finite-precision
  synthesis). This is a **systematic** Hamiltonian perturbation: it shifts every
  eigenvalue by `≤ ‖δH‖` *independent of N_walk* (the walk qubitizes `H+δH`
  exactly, so QPE returns its eigenvalues). Hence the budget is simply
  `‖δH‖ ≤ ε_BE`, not a per-query accumulation.

`‖δH‖` splits (triangle inequality, worst case) into the PREP and rotation
contributions, each an energy `= λ · (dimensionless unitary error)`:

  ‖δH‖  ≤  λ·ε_prep  +  λ·ε_rot   ≤  ε_BE,

so with an even split `ε_prep = ε_rot = ε_BE / (2λ)`:

  * `probability_epsilon` (alias-sampling coeff precision)  = ε_prep,
  * `circuit_precision` (total rotation-synthesis error fed to
    `estimate_resources`) = ε_rot   (pyLIQTR divides it across the rotations),
  * `kappa` / QROM angle bits (only used by the QROM cost form) = ⌈log₂(1/ε_rot)⌉.

`ΔE` is split between resolution and block encoding by `qpe_fraction`
(default ½ each). Tightening `ΔE` raises `N_walk` (∝ 1/ΔE) and the precision
requirements (bits ∝ log(1/ΔE)); `sensitivity_table` reports this.
"""

import math

from src_PI.estimation.qpe_cost import walk_queries

DEFAULT_DELTA_E_MEV = 1.0
DEFAULT_QPE_FRACTION = 0.5


def qpe_error_budget(physical_lambda, delta_E=DEFAULT_DELTA_E_MEV,
                     qpe_fraction=DEFAULT_QPE_FRACTION):
    """Derive the full precision budget from `ΔE` (MeV) and `λ`.

    Returns a dict with the energy allocation (`eps_qpe`, `eps_be`), the derived
    approximation parameters (`probability_epsilon`, `circuit_precision`,
    `kappa`, `alias_mu_bits`), and `walk_queries`. All are functions of `ΔE`, so
    there are no unexplained constants.
    """
    if not (0.0 < qpe_fraction < 1.0):
        raise ValueError("qpe_fraction must be in (0, 1)")
    if physical_lambda <= 0:
        raise ValueError("physical_lambda must be positive")

    eps_qpe = qpe_fraction * delta_E
    eps_be = (1.0 - qpe_fraction) * delta_E

    n_walk = walk_queries(physical_lambda, eps_qpe)

    # ε_BE = λ·(ε_prep + ε_rot), split evenly (both are dimensionless unitary errors).
    eps_prep = eps_be / (2.0 * physical_lambda)
    eps_rot = eps_be / (2.0 * physical_lambda)

    # Integer precisions: bits so that 2^-bits ≤ ε.
    alias_mu_bits = max(1, math.ceil(math.log2(1.0 / eps_prep)))
    kappa = max(1, math.ceil(math.log2(1.0 / eps_rot)))

    return {
        'delta_E': delta_E,
        'qpe_fraction': qpe_fraction,
        'physical_lambda': physical_lambda,
        'eps_qpe': eps_qpe,
        'eps_be': eps_be,
        'walk_queries': n_walk,
        # approximation parameters wired into the compiled cost:
        'probability_epsilon': eps_prep,        # alias-sampling PREP precision
        'circuit_precision': eps_rot,           # total rotation-synthesis error
        'kappa': kappa,                         # QROM angle bits (QROM cost form)
        'alias_mu_bits': alias_mu_bits,         # alias keep-register bits
    }


def sensitivity_table(physical_lambda, delta_Es=(10.0, 1.0, 0.1, 0.01),
                      qpe_fraction=DEFAULT_QPE_FRACTION):
    """Budget as a function of `ΔE` — the P0-4 sensitivity deliverable.

    Returns a list of budget dicts (one per `ΔE`). Demonstrates the expected
    scaling: `N_walk ∝ 1/ΔE`, precision bits `∝ log(1/ΔE)`.
    """
    return [qpe_error_budget(physical_lambda, dE, qpe_fraction) for dE in delta_Es]
