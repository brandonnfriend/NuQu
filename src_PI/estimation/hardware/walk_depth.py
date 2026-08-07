"""Walk-step Toffoli-depth model -> the reaction-limited runtime floor (task 30 / 34).

The throughput runtime (`physical_runtime.translate_to_physical`) serializes every
magic state: t = C * d * t_cycle. The REACTION-LIMITED floor instead serializes only
the algorithm's *critical path* of magic-state consumptions:

    t_react = adaptive_Toffoli_depth * tau_react = (N_walk * D_walk) * tau_react

where the N_walk qubitized walk applications W^(2^j) are strictly sequential (QPE's
controlled-power chain -- unavoidable), and D_walk is the Toffoli critical-path depth
of ONE walk step. Unlike N_walk, D_walk is a FREE KNOB: the QROAM / SelectSwap ancilla
parameter (Low-Kliuchnikov-Schaeffer 2018; Berry et al. 2019) trades ancilla qubits
for depth in the PREPARE data-lookup and the SELECT control tree, at *fixed T-count*.
So the reaction-limited runtime is honestly a BAND, set by the qubit budget.

APPROACH A (analytical composition, production-consistent to L=10):
  1. MEASURE the single-mode (a + a_dag) BCK atom's Toffoli-depth by lowering the real
     pyLIQTR/qualtran primitive (`SparseSingleLadderBlockEncoding`, ~n_b+10 qubits) to
     a cirq circuit and reading its critical path. Small -> laptop-safe; no L>2 build.
  2. COMPOSE to D_walk mirroring `sparse_oracle/resources.py`'s LCU aggregation
     (walk = 2*PREPARE + SELECT over L_eff terms), with the QROAM knob giving the band:
       * serial : plain QROM, no parallelism      -> D_walk = walk Toffoli COUNT
                  (a rigorous upper bound: fully serialized depth = count).
       * qroam  : SelectSwap PREPARE ~2*sqrt(L_eff) + log2(L_eff) SELECT control tree
                  + p_max * d_atom  (only the selected term acts) -- the realistic headline.
       * log    : max-ancilla ~log2(L_eff) everywhere + p_max * d_atom -- aggressive floor.

Depth is counted in TOFFOLI layers (each Toffoli / And = one magic-state consumption =
one reaction step under an active-volume architecture). The rotation-synthesis and raw
T critical path are carried as informational fields only.

NOTE (honest scope): PREPARE QROAM depth-reduction is textbook; the SELECT one-hot
depth-reduction (only the selected term on the critical path) is the more
architecture-dependent assumption -- that is exactly why `serial` (no parallelism at
all) is reported alongside as the conservative bound. The band brackets the truth.
"""

import math
from dataclasses import dataclass, asdict

import cirq
import qualtran.bloqs.mcmt  # noqa: F401  -- pyLIQTR reflection reaches qt.bloqs.mcmt by
#                                            dotted access; preload so decompose works.

from src_PI.estimation.sparse_oracle.block_encoding import (
    SingleLadderProblemInstance,
    SparseSingleLadderBlockEncoding,
)


# --------------------------------------------------------------------------------------
# 1. Measured atomic primitive depth
# --------------------------------------------------------------------------------------

@dataclass
class AtomicDepth:
    """Measured depth/count of the single-mode (a + a_dag) BCK block encoding."""
    n_b: int                    # requested n_b
    toffoli_depth: int          # critical path of 3-qubit (And/Toffoli) gates
    toffoli_count: int
    t_depth: int                # critical path of exact T gates (excl. rotation synth)
    t_count: int
    n_qubits: int
    measured_at_n_b: int        # n_b actually built (== n_b unless clamped)
    clamped: bool               # True iff a conservative proxy was used (see below)


_ATOM_CACHE = {}

# qualtran 0.4.0's Addition.decompose_from_registers indexes ancillas[0] with an empty
# carry list at bitsize=1 (IndexError). The n_b=1 atom is strictly cheaper than n_b=2
# (fewer Fock levels, trivial 1-bit shift, smaller QROM), so measuring at n_b=2 is a
# CONSERVATIVE upper-bound proxy for the rare n_b=1 point (e.g. the L=1 anchor). Flagged
# via AtomicDepth.clamped so downstream can note it.
_MIN_MEASURABLE_N_B = 2


def _gate_type_depth(ops, is_counted):
    """Critical-path depth over a topologically-ordered op list, counting only ops for
    which is_counted(op) is True. Per-qubit running-max respects data dependencies
    regardless of how the linear order packs (any valid topo order gives the same
    critical-path length)."""
    depth = {}
    maxd = 0
    for op in ops:
        base = max((depth.get(q, 0) for q in op.qubits), default=0)
        nd = base + (1 if is_counted(op) else 0)
        for q in op.qubits:
            depth[q] = nd
        if nd > maxd:
            maxd = nd
    return maxd


def _is_t(op):
    g = op.gate
    return isinstance(g, cirq.ZPowGate) and abs(abs(g.exponent) - 0.25) < 1e-9


def _is_toffoli(op):
    return cirq.num_qubits(op) >= 3


def atomic_depths(n_b):
    """Measure the single-mode BCK atom's Toffoli/T depth & count by lowering the real
    pyLIQTR/qualtran primitive to a cirq circuit. Cached by n_b (laptop-safe: the atom
    is a single mode, ~n_b + 10 qubits, thousands of gates)."""
    n_b = int(n_b)
    if n_b in _ATOM_CACHE:
        return _ATOM_CACHE[n_b]

    meas_n_b = max(_MIN_MEASURABLE_N_B, n_b)   # conservative clamp for n_b=1 (see above)
    enc = SparseSingleLadderBlockEncoding(SingleLadderProblemInstance(meas_n_b))
    q = 0
    quregs = {}
    for reg in enc.signature:
        quregs[reg.name] = [cirq.LineQubit(q + k) for k in range(reg.bitsize)]
        q += reg.bitsize
    op = enc.on_registers(**quregs)

    # Toffoli level: keep 3-qubit (And/Toffoli) gates as leaves.
    tof_ops = list(cirq.decompose(op, keep=lambda o: cirq.num_qubits(o) <= 3,
                                  on_stuck_raise=None))
    # Clifford+T level: full decompose (exact T only; rotations stay as rotations).
    ct_ops = list(cirq.decompose(op, on_stuck_raise=None))

    ad = AtomicDepth(
        n_b=n_b,
        toffoli_depth=_gate_type_depth(tof_ops, _is_toffoli),
        toffoli_count=sum(1 for o in tof_ops if _is_toffoli(o)),
        t_depth=_gate_type_depth(ct_ops, _is_t),
        t_count=sum(1 for o in ct_ops if _is_t(o)),
        n_qubits=q,
        measured_at_n_b=meas_n_b,
        clamped=(meas_n_b != n_b),
    )
    _ATOM_CACHE[n_b] = ad
    return ad


# --------------------------------------------------------------------------------------
# 2. Analytical composition -> per-walk-step depth band
# --------------------------------------------------------------------------------------

@dataclass
class WalkDepthBand:
    """Toffoli critical-path depth of ONE qubitized walk step, as a band over the QROAM
    ancilla knob. serial >= qroam >= log always."""
    serial: int         # no parallelism (rigorous upper bound = walk Toffoli count)
    qroam: int          # SelectSwap PREPARE (~2*sqrt L_eff) + log SELECT control + atom
    log: int            # max-ancilla (~log2 L_eff) everywhere + atom
    l_eff: int          # LCU term count
    p_max: int          # deepest monomial degree (atoms in series on the selected term)
    d_atom_toffoli: int
    walk_toffoli_count: int

    def as_dict(self):
        return asdict(self)


def walk_toffoli_count(l_eff, total_atom_applications, atom_toffoli_count):
    """Rigorous serial-depth upper bound = total Toffoli count of the walk, mirroring
    resources.py in Toffoli units:  walk = 2*PREPARE + SELECT.
      * PREPARE (alias sampling over L_eff terms): ~1 Toffoli/term -> 2 * L_eff.
      * SELECT: sum over terms of (P single-mode atoms) each atom_toffoli_count Toffolis
                -> total_atom_applications * atom_toffoli_count, where
                total_atom_applications = sum_terms P = select_T / single_mode_walk_T.
    (Fermion-only Pauli strings add a sub-1% Toffoli count; folded into the atom sum via
    the caller's total_atom_applications when present, else negligible.)"""
    return int(2 * max(1, l_eff) + total_atom_applications * atom_toffoli_count)


def walk_depth_band(l_eff, d_atom_toffoli, wtc, p_max=2):
    """Compose the per-walk-step Toffoli-depth band.

    Args:
        l_eff: LCU term count (`breakdown['L_eff']`).
        d_atom_toffoli: measured single-mode atom Toffoli-depth (`atomic_depths().toffoli_depth`).
        wtc: walk Toffoli count (serial upper bound; from `walk_toffoli_count`).
        p_max: deepest monomial degree = atoms applied in series on the selected term.
               Our H is degree <=2 (H_WT boson-quadratic); default 2.
    """
    l = max(2, int(l_eff))
    log_le = math.ceil(math.log2(l))
    sqrt_le = math.ceil(math.sqrt(l))
    atom_path = p_max * d_atom_toffoli

    # qroam: 2*PREPARE(SelectSwap ~2 sqrt) + SELECT control tree (log) + selected-term atom path.
    qroam = 2 * (2 * sqrt_le) + log_le + atom_path
    # log: max-ancilla PREPARE+SELECT control all ~log2 + selected-term atom path.
    log_ = 3 * log_le + atom_path

    serial = int(wtc)
    qroam = int(min(serial, qroam))     # band can't exceed the serial bound
    log_ = int(min(qroam, log_))
    return WalkDepthBand(serial=serial, qroam=qroam, log=log_,
                         l_eff=int(l_eff), p_max=int(p_max),
                         d_atom_toffoli=int(d_atom_toffoli), walk_toffoli_count=serial)


# --------------------------------------------------------------------------------------
# 3. Reaction-limited runtime
# --------------------------------------------------------------------------------------

def reaction_runtime_s(n_walk, walk_depth_toffoli, tau_react_s):
    """Reaction-limited wall-clock for one scalar D_walk = N_walk * D_walk * tau_react."""
    return float(n_walk) * float(walk_depth_toffoli) * float(tau_react_s)


def reaction_band_s(n_walk, band, tau_react_s):
    """Reaction-limited runtime band (seconds) for a WalkDepthBand."""
    return {
        'serial': reaction_runtime_s(n_walk, band.serial, tau_react_s),
        'qroam': reaction_runtime_s(n_walk, band.qroam, tau_react_s),
        'log': reaction_runtime_s(n_walk, band.log, tau_react_s),
    }


# --------------------------------------------------------------------------------------
# 4. Convenience: sparse `breakdown` sub-dict -> depth band (minimal JSON coupling)
# --------------------------------------------------------------------------------------

def walk_depth_from_breakdown(breakdown, n_b, p_max=2):
    """Build the WalkDepthBand from a sparse-estimate `breakdown` sub-dict + n_b.

    Uses only the stable keys `L_eff`, `select_T`, `single_mode_walk_T`; measures the
    atom for `n_b`. `total_atom_applications = select_T / single_mode_walk_T` recovers
    sum_terms P faithfully (resources.py: select_T = sum_terms P * single_mode_walk_T)."""
    l_eff = int(breakdown['L_eff'])
    single_T = float(breakdown['single_mode_walk_T'])
    select_T = float(breakdown.get('select_T', 0.0))
    total_atom_apps = round(select_T / single_T) if single_T > 0 else l_eff * p_max

    atom = atomic_depths(n_b)
    wtc = walk_toffoli_count(l_eff, total_atom_apps, atom.toffoli_count)
    band = walk_depth_band(l_eff, atom.toffoli_depth, wtc, p_max=p_max)
    return band, atom
