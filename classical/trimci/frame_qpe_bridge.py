"""The frame -> QPE bridge (task 34, I1).

`frame_qpe.py` (task 33 STEP 7) built the *pieces* — `warmstart_overlap`,
`mean_boson_number`, `stateprep_tcount`, `qpe_payoff` — but they had never been
run against a real classical-frame core and a real `src_PI` sweep Λ/T together:
`qpe_payoff` was only ever exercised in a unit test with placeholder numbers.
This module is the missing glue. It:

  1. reads a TrimCI ground-state **core** (`coeffs`, boson-occupation `bos`) from a
     saved run and computes its warm-start metrics (`p0`, `⟨n⟩`);
  2. reads the matching `(L, A)` point out of a `src_PI` resource **sweep JSON**
     (`Physical_Lambda`, `Total_T_Count`, `n_b`, `Logical_Qubits`);
  3. folds them through `frame_qpe.qpe_payoff` to get the frame's QPE-cost impact
     (fewer repetitions from `p0`, smaller boson register from `⟨n⟩`, minus the
     one-time state-prep);
  4. applies the **isospectrality gate** — the leading-order projector-LF frame is
     only approximately isospectral for our transition vertex, so a `gaussian+lf`
     *converged* E∞ that dips *below* the shared E∞ of the exactly-isospectral
     frames (bare / squeeze) is a spectrum shift, not compaction, and must NOT be
     certified as a resource reduction (see task 34 and [[project_lf_leading_order]]).

Direction of imports stays one-way (`classical` may read `src_PI`; not vice-versa):
this lives on the classical side and imports `src_PI.estimation.qpe_cost` via
`frame_qpe`. It is a post-hoc analysis/plumbing layer, not something `run_sweep`
calls inline.
"""

from __future__ import annotations

import json
import os

from classical.trimci.frame_qpe import (
    mean_boson_number, qpe_payoff, warmstart_overlap,
)


# ---------------------------------------------------------------------------
#  1. Core -> warm-start metrics
# ---------------------------------------------------------------------------

def core_metrics(coeffs, bos_arr=None, n_bos_modes=None):
    """Warm-start metrics of one (framed or bare) TrimCI core.

    Returns `{p0, mean_n_total, mean_n_per_mode}`:
      * `p0` = `|c_dominant|² / Σ|c|²` — QPE warm-start overlap (→ ~1/p0 reps).
      * `mean_n_total` = `Σ_i |c_i|² Σ_m n_{i,m}` — total mean boson number.
      * `mean_n_per_mode` = `mean_n_total / n_bos_modes` (None if not given) — the
        per-mode occupation that sets the Fock cutoff `n_b`.
    `bos_arr` may be omitted (p0-only) when the core carries no boson array.
    """
    p0 = warmstart_overlap(coeffs)
    mean_n_total = None
    mean_n_per_mode = None
    if bos_arr is not None:
        mean_n_total = mean_boson_number(coeffs, bos_arr)
        if n_bos_modes:
            mean_n_per_mode = mean_n_total / n_bos_modes
    return {'p0': p0, 'mean_n_total': mean_n_total,
            'mean_n_per_mode': mean_n_per_mode}


def metrics_from_run_dir(run_dir):
    """Load a saved classical run (`classical.io.load_classical_run` layout:
    `groundstate.npz` + `metadata.json`) and return its `core_metrics` plus the
    frame tag and energy. `n_bos_modes` is read from metadata when present."""
    from classical.io import load_classical_run
    run = load_classical_run(run_dir)
    meta = run.get('metadata', {})
    n_bos_modes = meta.get('n_bos_modes') or meta.get('n_bos') \
        or _infer_n_bos_modes(run['bos'])
    m = core_metrics(run['coeffs'], run['bos'], n_bos_modes)
    m['energy'] = run['energy']
    m['transform'] = meta.get('transform', meta.get('frame', 'bare'))
    m['run_dir'] = run_dir
    return m


def _infer_n_bos_modes(bos_arr):
    """Number of boson modes = width of the (N, n_bos) occupation array."""
    try:
        return int(bos_arr.shape[1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  2. Sweep JSON -> the matching (L, A) resource point
# ---------------------------------------------------------------------------

def load_sweep_point(sweep_json_path, A=None, L=None):
    """Pull one result entry (Λ, T, n_b, qubits) out of a `src_PI` sweep JSON.

    The sweep schema is `{metadata, results: [{A, L, dim, n_b, Physical_Lambda,
    Total_T_Count, Logical_Qubits, ...}]}` (`DataIO.save_sweep_data`). Selects the
    result with matching `A` (and `L` if given); if neither is given and there is
    exactly one result, returns it. Raises if the match is ambiguous or missing.
    """
    with open(sweep_json_path) as f:
        data = json.load(f)
    results = data.get('results', [])
    if not results:
        raise ValueError(f"no results in sweep JSON {sweep_json_path}")

    def _match(r):
        return (A is None or r.get('A') == A) and (L is None or r.get('L') == L)

    hits = [r for r in results if _match(r)]
    if not hits:
        raise ValueError(f"no result with A={A}, L={L} in {sweep_json_path}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous: {len(hits)} results match A={A}, L={L} "
                         f"in {sweep_json_path}; disambiguate with both A and L")
    r = hits[0]
    return {
        'A': r.get('A'), 'L': r.get('L'), 'dim': r.get('dim'), 'n_b': r.get('n_b'),
        'physical_lambda': r['Physical_Lambda'],
        'total_t_count': r['Total_T_Count'],
        'logical_qubits': r.get('Logical_Qubits'),
    }


# ---------------------------------------------------------------------------
#  3. The isospectrality gate (LF certification)
# ---------------------------------------------------------------------------

# Frames whose term-list transform is EXACTLY isospectral (unitary, no truncation):
# bare (identity) and the Gaussian squeeze (polynomial, degree-preserving). Their
# converged E∞ is the trustworthy ground-state reference the gate compares against.
EXACT_ISOSPECTRAL_FRAMES = ('bare', 'gaussian', 'squeeze')


def isospectrality_gate(e_inf_by_frame, sites, test_frame,
                        tol_mev_per_site=0.5,
                        isospectral_frames=EXACT_ISOSPECTRAL_FRAMES):
    """Decide whether a frame's CONVERGED E∞ is consistent with the true spectrum.

    A true (unitary) frame is isospectral, so its converged ground energy must
    equal that of the exactly-isospectral frames — it may sit *above* the
    reference (under-converged, still a valid upper bound) but must not sit
    meaningfully *below* it (that would be an energy under the true ground state,
    which only an approximate/spectrum-shifting frame can fake). The leading-order
    projector-LF frame is exactly such an approximate frame, so this gate is what
    keeps a `gaussian+lf` "win" honest.

    Args:
        e_inf_by_frame: `{frame_name: converged E∞ in MeV}`. Must include at least
            one `isospectral_frames` entry (the reference) and `test_frame`.
        sites: lattice site count (to turn the per-site tolerance into MeV).
        test_frame: the frame to certify (e.g. 'gaussian+lf', 'lf').
        tol_mev_per_site: how far below the reference counts as noise vs a real
            shift. Default 0.5 MeV/site.

    Returns dict:
        `certified` (bool), `reason`, `e_inf_ref` (tightest isospectral E∞),
        `e_inf_test`, `delta_per_site` (`(test − ref)/sites`; negative = below the
        reference), `ref_frames_used`, `ref_spread_per_site` (bare-vs-squeeze
        disagreement, a convergence-health signal).
    """
    refs = {k: v for k, v in e_inf_by_frame.items()
            if k in isospectral_frames and v is not None}
    if not refs:
        raise ValueError("isospectrality_gate needs at least one exactly-isospectral "
                         f"reference frame ({isospectral_frames}); got "
                         f"{list(e_inf_by_frame)}")
    if test_frame not in e_inf_by_frame or e_inf_by_frame[test_frame] is None:
        raise ValueError(f"test_frame {test_frame!r} not in e_inf_by_frame")

    # Tightest (lowest, most-converged) exact reference; spread is a health check.
    e_ref = min(refs.values())
    ref_spread = (max(refs.values()) - min(refs.values())) / sites if len(refs) > 1 else 0.0
    e_test = e_inf_by_frame[test_frame]
    delta_per_site = (e_test - e_ref) / sites
    tol = tol_mev_per_site

    if delta_per_site < -tol:
        certified = False
        reason = (f"{test_frame} E∞ is {-delta_per_site:.3f} MeV/site BELOW the "
                  f"isospectral reference (> {tol} tol): spectrum shift, not "
                  f"compaction — NOT certified as a resource reduction")
    else:
        certified = True
        reason = (f"{test_frame} E∞ is {delta_per_site:+.3f} MeV/site vs reference "
                  f"(within {tol} tol): consistent with the true spectrum — "
                  f"compaction certified")
    return {
        'certified': certified, 'reason': reason,
        'e_inf_ref': e_ref, 'e_inf_test': e_test,
        'delta_per_site': delta_per_site,
        'ref_frames_used': sorted(refs),
        'ref_spread_per_site': ref_spread,
    }


# ---------------------------------------------------------------------------
#  4. The payoff: fold real metrics through frame_qpe.qpe_payoff (+ gate)
# ---------------------------------------------------------------------------

def frame_qpe_reduction(bare_metrics, frame_metrics, sweep_point, *,
                        n_bos, n_ferm, frame_layers=('squeeze',),
                        delta_E=1.0, test_frame=None, e_inf_by_frame=None,
                        sites=None, tol_mev_per_site=0.5):
    """End-to-end frame -> QPE resource reduction for one `(L, A)` point.

    Combines a bare core, a frame core, and a resource sweep point into the
    QPE-cost impact via `frame_qpe.qpe_payoff`, and (when E∞ data is supplied)
    runs the isospectrality gate and stamps a `certified` flag.

    Args:
        bare_metrics / frame_metrics: dicts from `core_metrics` (need `p0`, and
            `mean_n_per_mode` for the boson-register saving).
        sweep_point: dict from `load_sweep_point` (Λ, T, n_b).
        n_bos, n_ferm: mode counts for the state-prep cost.
        frame_layers: which state-prep layers to charge ('squeeze', 'displace',
            'orbital'). For `gaussian+lf` use `('squeeze', 'displace')`.
        test_frame / e_inf_by_frame / sites: if all given, run the gate.

    Returns the `qpe_payoff` dict plus `p0_bare/p0_frame`, `mean_n_*`, the sweep
    point, and (if gated) a `gate` sub-dict and top-level `certified`.
    """
    p0_bare = bare_metrics['p0']
    p0_frame = frame_metrics['p0']
    mean_n_bare = bare_metrics.get('mean_n_per_mode')
    mean_n_frame = frame_metrics.get('mean_n_per_mode')
    payoff = qpe_payoff(
        p0_bare=p0_bare, p0_frame=p0_frame,
        mean_n_bare=(mean_n_bare if mean_n_bare is not None else 0.0),
        mean_n_frame=(mean_n_frame if mean_n_frame is not None else 0.0),
        t_step=sweep_point['total_t_count'],
        physical_lambda=sweep_point['physical_lambda'],
        n_bos=n_bos, n_ferm=n_ferm, n_b=sweep_point['n_b'],
        delta_E=delta_E, frame_layers=frame_layers,
    )
    out = dict(payoff)
    out.update({
        'p0_bare': p0_bare, 'p0_frame': p0_frame,
        'mean_n_bare_per_mode': mean_n_bare, 'mean_n_frame_per_mode': mean_n_frame,
        'sweep_point': sweep_point, 'frame_layers': list(frame_layers),
    })
    if e_inf_by_frame is not None and test_frame is not None and sites is not None:
        gate = isospectrality_gate(e_inf_by_frame, sites, test_frame,
                                   tol_mev_per_site=tol_mev_per_site)
        out['gate'] = gate
        out['certified'] = gate['certified']
    return out


# ---------------------------------------------------------------------------
#  5. Seam (b) as a post-process: write warm-start columns back into a sweep
# ---------------------------------------------------------------------------

def fold_warmstart_into_sweep(sweep_json_path, bare_metrics, frame_metrics, *,
                              n_bos, n_ferm, frame_name, frame_layers=('squeeze',),
                              A=None, L=None, delta_E=1.0, write=True):
    """Post-process (mirrors `qpe_cost.compute_total_qpe_cost`): fold a frame's
    warm-start reduction into the matching sweep result entry as a new
    `warmstart[frame_name]` column. Keeps `run_sweep` untouched — the raw sweep is
    the reviewable gate; this augments it after the fact. Returns the updated dict.
    """
    with open(sweep_json_path) as f:
        data = json.load(f)
    point = load_sweep_point(sweep_json_path, A=A, L=L)
    reduction = frame_qpe_reduction(
        bare_metrics, frame_metrics, point,
        n_bos=n_bos, n_ferm=n_ferm, frame_layers=frame_layers, delta_E=delta_E)
    for r in data.get('results', []):
        if (A is None or r.get('A') == A) and (L is None or r.get('L') == L):
            r.setdefault('warmstart', {})[frame_name] = {
                'p0_gain': reduction['p0_gain'],
                'repetition_factor': reduction['repetition_factor'],
                'qpe_T_ratio': reduction['qpe_T_ratio'],
                'boson_qubit_saving_per_mode': reduction['boson_qubit_saving_per_mode'],
                'T_prep': reduction['T_prep'],
                'prep_vs_walk': reduction['prep_vs_walk'],
            }
    if write:
        with open(sweep_json_path, 'w') as f:
            json.dump(data, f, indent=4)
    return data
