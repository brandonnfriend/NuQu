"""Architecture study — where a frame enters the quantum computation (task 34).

Companion to `docs/frame_on_quantum_side.md`. That doc frames three architectures
for how a classical frame `U` can meet the fault-tolerant QPE pipeline:

  * **A — warm-start only** (SAFE, no proof): qubitize the BARE H, use `U` only to
    prepare the QPE initial state `U|ref>`. QPE returns spec(bare H) exactly; the
    frame's only effect is the warm-start overlap `p0` (=> ~1/p0 repetitions) and,
    on a Fock/occupation register, a smaller boson cutoff `n_b` from the reduced
    mean occupation. A wrong/truncated `U` cannot shift the energy — it can only
    lower `p0`. THIS MODULE implements Architecture A.
  * **B — walk the squeeze frame** (certified n_b win): qubitize `squeeze_terms(H)`.
    Requires a new `src_PI` squeezed-basis walk (Stage 2, not here).
  * **C — walk squeeze+LF** (needs an error bound): qubitize the truncated-LF frame.
    Gated by the `‖R_trans‖` admissibility test (a separate ED order-sweep).

Architecture A is ~90% pre-built: `frame_qpe.qpe_payoff` already computes the QPE
impact of a `(p0_bare, p0_frame, <n>)` triple against a `src_PI` (Λ, T_step) point,
and the classical HPC frame shards (`misc/run_frame_shard.py`) already record `p0`
and `mean_occ` per rung. What was missing is the GLUE that (1) reads those shards,
(2) aggregates the seed ensemble honestly (the runs are seed-fragile — see
[[project_frame_value_quantified]] / the D-DMRG n_runs>=16 finding), and (3) folds
them through the doc-faithful total-T so the A-vs-bare verdict is a real number.

**Doc-faithful total-T** (`docs/frame_on_quantum_side.md` §4):

    total_T = (T_prep + N_walk · T_step) · QPE_reps,   QPE_reps = 1/p0,
    N_walk  = π·Λ/ΔE   (the qubitized-walk query count; `src_PI.qpe_cost`).

The repo's saved `QPE_Total_T_Count` is only the inner `N_walk·T_step` (QPE_reps=1,
T_prep=0) — Architecture A is exactly what folds in the `1/p0` and `T_prep` factors.
`T_prep` is charged PER REPETITION (each QPE run re-prepares `U|ref>`); it is
negligible vs `N_walk·T_step` by construction (that is the point), but we carry it
explicitly so the doc's "does the repetition win beat T_prep?" question gets a
number rather than an assertion.

**Honesty note surfaced by the data (2026-08-05):** the warm-start `p0` win is
FILLING-DEPENDENT and often NEGATIVE. At L=3 mid-filling the framed core is *less*
concentrated on its dominant determinant than the bare core (bare p0≈0.21 vs
gaussian+lf≈0.03), so the frame HURTS the warm start there; it helps only at the
dilute and high-filling extremes (e.g. `tail` A=108: gaussian p0=0.75 vs bare 0.26).
So `repetition_factor > 1` is a real, reportable outcome — this module never assumes
the frame wins. The frame's robust A-payoff is the boson-register reduction, which is
a Fock/occupation-register quantity: it is only apples-to-apples against a Fock/tong
`n_b` baseline, NOT the large field-amplitude `n_b`.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

from classical.trimci.frame_qpe import qpe_payoff


# Frame name -> which state-prep layers `U|ref>` charges (frame_qpe.stateprep_tcount).
FRAME_LAYERS = {
    'bare': (),
    'gaussian': ('squeeze',),
    'squeeze': ('squeeze',),
    'lf': ('displace',),
    'gaussian+lf': ('squeeze', 'displace'),
    'gaussian+coo': ('squeeze', 'orbital'),
    'gaussian+coo+lf': ('squeeze', 'orbital', 'displace'),
}


@dataclass
class FrameRecord:
    """One (L, dim, A, frame) point, aggregated over the seed ensemble.

    `p0_best` / `mean_occ_best` are taken from the MOST-CONVERGED seed (lowest
    `E_pt2`) — the best variational estimate of the true state, hence the most
    trustworthy warm-start metrics. `p0_lo`/`p0_hi` carry the full seed band as an
    honest error bar (these runs are seed-fragile). `mean_occ` is the per-mode boson
    occupation `<n>` (drives the Fock cutoff)."""
    L: int
    dim: int
    A: int
    frame: str
    sites: int
    fill: float | None
    n_b_classical: int
    N_f: int
    p0_best: float
    p0_lo: float
    p0_hi: float
    mean_occ_best: float | None
    E_pt2_best: float | None
    core_best: int
    n_seeds: int
    source_dir: str
    seeds: list = field(default_factory=list)

    @property
    def n_bos_modes(self) -> int:
        return 3 * self.sites

    @property
    def n_ferm_modes(self) -> int:
        return 4 * self.sites


def _deepest_rung(shard: dict):
    """The largest-core rung of a classical frame shard (most converged)."""
    rungs = [r for r in shard.get('rungs', []) if isinstance(r, dict) and 'p0' in r]
    if not rungs:
        return None
    return max(rungs, key=lambda r: r.get('core', 0))


def load_frame_shard(path: str):
    """Parse one classical frame-core shard JSON into `(key, seed_row)`.

    `key = (L, dim, A, frame)`; `seed_row` is the deepest rung's metrics plus the
    top-level geometry. Returns `None` for a file that is not a frame shard (no
    top-level `frame`/`rungs`) or has no usable rung.
    """
    try:
        with open(path) as f:
            shard = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(shard, dict) or 'frame' not in shard or 'rungs' not in shard:
        return None
    rung = _deepest_rung(shard)
    if rung is None:
        return None
    L, dim, A = shard.get('L'), shard.get('dim'), shard.get('A')
    frame = shard.get('frame')
    if L is None or A is None or frame is None:
        return None
    sites = shard.get('sites') or (int(L) ** int(dim) if dim else None)
    key = (int(L), int(dim), int(A), str(frame))
    seed_row = {
        'seed': shard.get('seed'),
        'p0': float(rung['p0']),
        'mean_occ': (float(rung['mean_occ']) if rung.get('mean_occ') is not None else None),
        'E_pt2': (float(rung['E_pt2']) if rung.get('E_pt2') is not None else None),
        'E_var': (float(rung['E_var']) if rung.get('E_var') is not None else None),
        'core': int(rung.get('core', 0)),
        'sites': int(sites) if sites is not None else None,
        'fill': shard.get('filling'),
        'n_b_classical': shard.get('n_b'),
        'N_f': shard.get('N_f'),
        'source_dir': os.path.basename(os.path.dirname(path)),
    }
    return key, seed_row


def collect_frame_records(dirs, mean_occ_max=0.5):
    """Scan classical HPC dirs -> `{(L, dim, A, frame): FrameRecord}`.

    Aggregates the seed ensemble per key: the central `p0`/`mean_occ` come from the
    lowest-`E_pt2` seed (best converged); the `p0` band spans all seeds. `mean_occ`
    rows above `mean_occ_max` are dropped as non-physical outliers (e.g. a broken run
    with <n>~0.77 — the framed near-vacuum state has <n>~0.01-0.09); if that leaves a
    key with no usable occupation, `mean_occ_best` is None.
    """
    buckets: dict = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.json'))):
            parsed = load_frame_shard(path)
            if parsed is None:
                continue
            key, row = parsed
            buckets.setdefault(key, []).append(row)

    records: dict = {}
    for key, rows in buckets.items():
        L, dim, A, frame = key
        # central = most-converged seed (lowest E_pt2; fall back to first if no PT2)
        with_e = [r for r in rows if r['E_pt2'] is not None]
        central = min(with_e, key=lambda r: r['E_pt2']) if with_e else rows[0]
        p0s = [r['p0'] for r in rows]
        occs = [r['mean_occ'] for r in rows
                if r['mean_occ'] is not None and r['mean_occ'] <= mean_occ_max]
        mean_occ_best = central['mean_occ']
        if mean_occ_best is not None and mean_occ_best > mean_occ_max:
            mean_occ_best = (sum(occs) / len(occs)) if occs else None
        records[key] = FrameRecord(
            L=L, dim=dim, A=A, frame=frame,
            sites=central['sites'], fill=central['fill'],
            n_b_classical=central['n_b_classical'], N_f=central['N_f'],
            p0_best=central['p0'], p0_lo=min(p0s), p0_hi=max(p0s),
            mean_occ_best=mean_occ_best,
            E_pt2_best=central['E_pt2'], core_best=central['core'],
            n_seeds=len(rows), source_dir=central['source_dir'],
            seeds=rows,
        )
    return records


def recommended_n_b(mean_occ, margin_sigmas=5.0):
    """Fock-register bits/mode justified by a per-mode occupation `<n>` — the single
    source of truth shared with the sweep (`src_PI.estimation.qpe_cost`)."""
    from src_PI.estimation.qpe_cost import recommended_n_b_from_occupation
    return recommended_n_b_from_occupation(mean_occ, margin_sigmas=margin_sigmas)


def architecture_A_point(bare: FrameRecord, frame: FrameRecord, quantum: dict,
                         *, delta_E=1.0, n_b_fock_bare=None):
    """One Architecture-A comparison row: bare cold-start vs frame warm-start.

    Folds the measured `(p0_bare, p0_frame, <n>_frame)` through `frame_qpe.qpe_payoff`
    against a bare-H resource point, then assembles the doc-faithful total-T:

        total_T_bare = N_walk · T_step / p0_bare               (cold; T_prep=0)
        total_T_A    = (T_prep + N_walk · T_step) / p0_frame   (warm; T_prep/rep)

    Args:
        bare, frame: `FrameRecord`s for the same `(L, dim, A)`; `frame.frame` picks
            the state-prep layers.
        quantum: bare-H resource point `{physical_lambda, total_t_count, n_b}`
            (from a `src_PI` shard at this `(L, A)` or a re-estimate). `n_b` is the
            walk register used for `qpe_payoff`'s state-prep sizing.
        n_b_fock_bare: OPTIONAL Fock/tong bare boson cutoff to use as the
            apples-to-apples baseline for the boson-qubit saving (the amplitude `n_b`
            is a field-register size, not comparable to the occupation-derived
            frame `n_b`). Defaults to `quantum['n_b']`.

    Returns a dict row: p0s (+band), N_walk, T_prep, total-T (bare/A, +band from the
    p0 band), the repetition/qpe ratios, boson-qubit saving, and a `verdict`.
    """
    p0_bare = bare.p0_best
    p0_frame = frame.p0_best
    layers = FRAME_LAYERS.get(frame.frame, ('squeeze',))
    lam = quantum['physical_lambda']
    t_step = quantum['total_t_count']
    n_b_walk = quantum['n_b']

    pay = qpe_payoff(
        p0_bare=p0_bare, p0_frame=p0_frame,
        mean_n_bare=(bare.mean_occ_best or 0.0),
        mean_n_frame=(frame.mean_occ_best or 0.0),
        t_step=t_step, physical_lambda=lam,
        n_bos=frame.n_bos_modes, n_ferm=frame.n_ferm_modes, n_b=n_b_walk,
        delta_E=delta_E, frame_layers=layers,
    )
    n_walk = pay['N_walk']
    t_prep = pay['T_prep']

    def _total_bare(p0):
        return n_walk * t_step / max(p0, 1e-15)

    def _total_warm(p0):
        return (t_prep + n_walk * t_step) / max(p0, 1e-15)

    total_bare = _total_bare(p0_bare)
    total_A = _total_warm(p0_frame)
    ratio = total_A / total_bare if total_bare else float('inf')

    # Fock/occupation-register qubit saving (apples-to-apples): a Fock/tong bare n_b
    # vs the frame-recommended n_b from <n>. Distinct from the walk register above.
    n_b_base = n_b_fock_bare if n_b_fock_bare is not None else n_b_walk
    n_b_frame = (recommended_n_b(frame.mean_occ_best)
                 if frame.mean_occ_best is not None else None)
    qubit_saving_per_mode = (n_b_base - n_b_frame) if n_b_frame is not None else None

    return {
        'L': bare.L, 'dim': bare.dim, 'A': bare.A, 'frame': frame.frame,
        'fill': frame.fill, 'sites': frame.sites, 'layers': list(layers),
        'p0_bare': p0_bare, 'p0_frame': p0_frame,
        'p0_bare_band': [bare.p0_lo, bare.p0_hi],
        'p0_frame_band': [frame.p0_lo, frame.p0_hi],
        'p0_gain': pay['p0_gain'],
        'repetition_factor': pay['repetition_factor'],   # <1 => fewer runs (win)
        'N_walk': n_walk, 'T_step': t_step, 'physical_lambda': lam,
        'T_prep': t_prep, 'prep_vs_walk': pay['prep_vs_walk'],
        'total_T_bare': total_bare, 'total_T_A': total_A, 'total_T_ratio': ratio,
        # total-T band from the p0 seed bands (best/worst warm vs central bare):
        'total_T_A_band': [_total_warm(frame.p0_hi), _total_warm(frame.p0_lo)],
        'mean_occ_bare': bare.mean_occ_best, 'mean_occ_frame': frame.mean_occ_best,
        'n_b_walk': n_b_walk, 'n_b_fock_bare': n_b_base, 'n_b_frame': n_b_frame,
        'boson_qubit_saving_per_mode': qubit_saving_per_mode,
        'n_seeds_bare': bare.n_seeds, 'n_seeds_frame': frame.n_seeds,
        'core_bare': bare.core_best, 'core_frame': frame.core_best,
        'verdict': ('warm-start WIN' if ratio < 1.0 else
                    'warm-start LOSS (frame lowers p0)'),
    }


def build_architecture_A(records, quantum_lookup, *, frames=None, delta_E=1.0,
                         n_b_fock_lookup=None):
    """Assemble Architecture-A rows for every `(L, dim, A)` that has a bare record,
    a matching frame record, and a bare-H resource point.

    Args:
        records: `{(L,dim,A,frame): FrameRecord}` from `collect_frame_records`.
        quantum_lookup: callable `(L, A) -> {physical_lambda, total_t_count, n_b}`
            or a dict keyed by `(L, A)`. Rows whose `(L, A)` is missing are skipped
            (with a note in the returned `skipped` list).
        frames: which frames to compare against bare (default: all present except
            'bare').
        n_b_fock_lookup: optional callable/dict `(L, A) -> Fock/tong bare n_b` for
            the boson-qubit-saving baseline.

    Returns `{rows: [...], skipped: [...]}`.
    """
    def _lookup(fn, L, A):
        if fn is None:
            return None
        if callable(fn):
            return fn(L, A)
        return fn.get((L, A))

    bare_keys = {(L, dim, A): rec for (L, dim, A, fr), rec in records.items()
                 if fr == 'bare'}
    rows, skipped = [], []
    for (L, dim, A, fr), frec in sorted(records.items()):
        if fr == 'bare':
            continue
        if frames is not None and fr not in frames:
            continue
        brec = bare_keys.get((L, dim, A))
        if brec is None:
            skipped.append({'L': L, 'dim': dim, 'A': A, 'frame': fr,
                            'reason': 'no bare record at this (L,dim,A)'})
            continue
        qp = _lookup(quantum_lookup, L, A)
        if qp is None:
            skipped.append({'L': L, 'dim': dim, 'A': A, 'frame': fr,
                            'reason': f'no bare-H resource point at (L={L}, A={A})'})
            continue
        nbf = _lookup(n_b_fock_lookup, L, A)
        rows.append(architecture_A_point(brec, frec, qp, delta_E=delta_E,
                                         n_b_fock_bare=nbf))
    return {'rows': rows, 'skipped': skipped}


def format_table(rows):
    """Compact fixed-width Architecture-A table for a terminal summary."""
    if not rows:
        return '(no Architecture-A rows)'
    hdr = (f"{'L':>2} {'A':>4} {'frame':16} {'p0_bare':>8} {'p0_frame':>9} "
           f"{'rep×':>7} {'totT_bare':>11} {'totT_A':>11} {'A/bare':>7} "
           f"{'Δn_b':>5} {'verdict':<28}")
    lines = [hdr, '-' * len(hdr)]
    for r in sorted(rows, key=lambda r: (r['L'], r['A'], r['frame'])):
        dnb = r['boson_qubit_saving_per_mode']
        lines.append(
            f"{r['L']:>2} {r['A']:>4} {r['frame']:16} "
            f"{r['p0_bare']:>8.4f} {r['p0_frame']:>9.4f} "
            f"{r['repetition_factor']:>7.2f} "
            f"{r['total_T_bare']:>11.3e} {r['total_T_A']:>11.3e} "
            f"{r['total_T_ratio']:>7.2f} {('' if dnb is None else f'{dnb:>5d}')} "
            f"{r['verdict']:<28}")
    return '\n'.join(lines)
