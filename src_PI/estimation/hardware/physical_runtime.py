"""Logical -> physical resource translation (task 30).

Turns a compiled logical fingerprint (algorithm qubits Q_alg, magic-state/T count
M_T, optional Toffoli count and Toffoli-depth) into physical deliverables: code
distance d, total physical qubits, and wall-clock runtime -- following the
Beverland 2211.07629 closed-form model with the SOTA choices from
`claude/research/ft_runtime_estimation/00_literature_review.md`.

Runtime is reported (or will be) as a PAIR:
  * throughput-limited  t = C * d * t_cycle   (serial magic-state consumption) -- built here.
  * reaction-limited    t = Toffoli_depth * tau_react  (the architecture floor) -- NEXT STEP,
    stubbed: returned only when a Toffoli-depth is supplied.

Key facts encoded (see lit review §4):
  * Cliffords are compiled away for free (Pauli-frame / lattice surgery), so runtime
    is set by the T/Toffoli sub-circuit; logical-qubit count sets machine SIZE.
  * The code distance d is fixed by the LOGICAL error budget (Q*C*P(d) <= eps/3) and
    is independent of the magic-state method; the cultivation-vs-distillation choice
    only moves the factory (space) cost, not d or the throughput runtime.
"""

import math
from dataclasses import dataclass

from src_PI.estimation.hardware.assumptions import DEFAULT_PROFILE


@dataclass
class PhysicalCost:
    d: int                          # code distance
    physical_qubits: float
    data_qubits: float
    factory_qubits: float
    runtime_throughput_s: float     # C * d * t_cycle
    runtime_reaction_s: float       # Toffoli_depth * tau_react, or nan if no depth given
    logical_cycles: float           # C
    magic_states: float             # M
    n_factories: int
    cultivation_self_sufficient: bool
    per_T_error_needed: float
    profile_name: str

    @property
    def runtime_throughput_years(self):
        return self.runtime_throughput_s / (3600 * 24 * 365.25)

    @property
    def runtime_reaction_years(self):
        return self.runtime_reaction_s / (3600 * 24 * 365.25)


def _distance_for(Q, C, profile):
    """Smallest ODD d with per-cycle logical error P(d)=a*(p/p_th)^((d+1)/2) small
    enough that the whole computation's logical error Q*C*P(d) <= eps/3."""
    target = profile.epsilon / 3.0 / (Q * C)          # allowed per-cycle logical error
    a, ratio = profile.a, profile.p / profile.p_th    # ratio < 1
    d = 3
    while d < 10001:
        if a * ratio ** ((d + 1) / 2.0) <= target:
            return d
        d += 2
    return d


def translate_to_physical(Q_alg, M_T, profile=None, M_Tof=0.0, toffoli_depth=None):
    """Logical fingerprint -> PhysicalCost.

    Args:
        Q_alg: algorithm logical qubits.
        M_T:   T-gate count (our QPE_Total_T_Count). For a qubitized walk this is the
               magic-state count (Toffolis, if separated, go in M_Tof).
        profile: HardwareProfile (default = the SOTA superconducting profile).
        M_Tof: Toffoli count, if separated from M_T (Toffoli = 4 T states, 3 cycles).
        toffoli_depth: adaptive Toffoli-depth for the reaction-limited floor (optional).
    """
    profile = profile or DEFAULT_PROFILE
    M = M_T + 4.0 * M_Tof                              # magic states
    C = M_T + 3.0 * M_Tof                              # logical cycles (throughput, serial)
    Q = 2.0 * Q_alg + math.ceil(math.sqrt(8.0 * Q_alg)) + 1  # tiles incl. routing (PSSPC)

    d = _distance_for(Q, C, profile)
    n_d = profile.tile_qubits_coeff * d * d           # physical qubits / tile
    data_qubits = Q * n_d

    # magic-state factory (space cost). d and runtime are independent of this.
    per_T_needed = profile.epsilon / 3.0 / M          # required per-T error
    pt_cult = profile.cultivation_pt()
    self_suff = pt_cult <= per_T_needed
    # cultivation fits in a patch and keeps pace with serial consumption -> ~1 factory;
    # if not self-sufficient, a distillation top-up round is folded in (still ~one
    # factory footprint of extra magic-state hardware; runtime unchanged).
    n_fac = 1 if self_suff else 2
    factory_qubits = n_fac * profile.factory_qubits

    runtime_throughput_s = C * d * profile.t_cycle_s
    runtime_reaction_s = (toffoli_depth * profile.tau_react_s
                          if toffoli_depth is not None else float('nan'))

    return PhysicalCost(
        d=d, physical_qubits=data_qubits + factory_qubits,
        data_qubits=data_qubits, factory_qubits=factory_qubits,
        runtime_throughput_s=runtime_throughput_s, runtime_reaction_s=runtime_reaction_s,
        logical_cycles=C, magic_states=M, n_factories=n_fac,
        cultivation_self_sufficient=self_suff, per_T_error_needed=per_T_needed,
        profile_name=profile.name)
