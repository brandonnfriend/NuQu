"""Hardware assumption profiles for the logical -> physical translation (task 30).

The full derivation, formulas, and SOTA-vs-range justification are in
`claude/research/ft_runtime_estimation/00_literature_review.md`. This file fixes the
DEFAULT profile the NuQu headline uses and the toggles/alternatives.

Standing decisions (2026-08, user):
  * Match the Gratsea-Otten 2025 full-stack SOTA (arXiv:2510.26547) as closely as we can.
  * Analytic surface-code error model (a=0.03, p_th=1e-2), NOT the empirical Willow
    Λ-form — the application is a decade+ out; assume analytic-scale hardware.
  * p = 1e-4 (optimistic superconducting): cultivation is then self-sufficient
    (per-T error ~6e-15), so no distillation top-up is needed at our Toffoli counts.
  * Superconducting, t_cycle = 1 us, reaction time tau_react = 1 us.
  * Magic states: cultivation (Gidney 2024, ~9200 physical qubits / factory).
"""

from dataclasses import dataclass, field


# Cultivation output per-T error vs physical error rate (Gidney 2409.17595; the
# points quoted in the lit review). Interpolated log-linearly between anchors.
_CULTIVATION_PT = {1e-3: 2e-9, 5e-4: 4e-11, 1e-4: 6e-15}


@dataclass(frozen=True)
class HardwareProfile:
    """A fault-tolerant hardware + architecture assumption set."""
    name: str
    p: float = 1e-4              # physical error rate
    p_th: float = 1e-2          # surface-code threshold
    a: float = 0.03             # analytic per-cycle-error prefactor (surface, gate-based)
    t_cycle_s: float = 1e-6     # one syndrome-extraction round (s); logical cycle = d * t_cycle
    tau_react_s: float = 1e-6   # decode->feedback latency for the reaction-limited floor
    epsilon: float = 1e-2       # total target failure (split /3: logical/distill/synth)
    tile_qubits_coeff: float = 2.0   # n(d) = coeff * d^2 (rotated surface patch = 2 d^2)
    magic_state: str = 'cultivation'
    factory_qubits: float = 9200.0   # cultivation factory footprint (Gratsea-Otten)
    comment: str = ''

    def cultivation_pt(self):
        """Per-T logical error the cultivation factory delivers at this p (log-interp)."""
        import math
        ps = sorted(_CULTIVATION_PT)
        if self.p <= ps[0]:
            return _CULTIVATION_PT[ps[0]]
        if self.p >= ps[-1]:
            return _CULTIVATION_PT[ps[-1]]
        lo = max(x for x in ps if x <= self.p)
        hi = min(x for x in ps if x >= self.p)
        if lo == hi:
            return _CULTIVATION_PT[lo]
        f = (math.log(self.p) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return math.exp(math.log(_CULTIVATION_PT[lo]) * (1 - f) + math.log(_CULTIVATION_PT[hi]) * f)


# The headline NuQu profile.
SUPERCONDUCTING_SOTA = HardwareProfile(
    name='superconducting_sota',
    p=1e-4, p_th=1e-2, a=0.03, t_cycle_s=1e-6, tau_react_s=1e-6, epsilon=1e-2,
    magic_state='cultivation', factory_qubits=9200.0,
    comment='Gratsea-Otten-matched analytic superconducting; cultivation self-sufficient at p=1e-4.')

# Conservative sensitivity toggle: today's realistic node (p=1e-3). Cultivation is
# NOT self-sufficient here (PT~2e-9) -> flagged by the translator.
SUPERCONDUCTING_P1E3 = HardwareProfile(
    name='superconducting_p1e-3', p=1e-3,
    comment='realistic-today node; cultivation not self-sufficient (needs top-up/lower p).')

PROFILES = {p.name: p for p in (SUPERCONDUCTING_SOTA, SUPERCONDUCTING_P1E3)}
DEFAULT_PROFILE = SUPERCONDUCTING_SOTA
