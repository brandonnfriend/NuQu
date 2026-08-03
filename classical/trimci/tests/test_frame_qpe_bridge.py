"""Tests for the frame -> QPE bridge (task 34, I1).

Exercises the bridge logic on synthetic cores + a synthetic sweep JSON (no
classical solve or pyLIQTR needed), including the isospectrality gate that keeps
a leading-order-LF "win" from being certified when it is actually a spectrum shift.
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from classical.trimci.frame_qpe_bridge import (
    core_metrics, fold_warmstart_into_sweep, frame_qpe_reduction,
    isospectrality_gate, load_sweep_point,
)


# --- 1. core metrics ----------------------------------------------------

def test_core_metrics_p0_and_mean_n():
    coeffs = [0.8, 0.6]                       # p0 = 0.64 / 1.00 = 0.64
    bos = np.array([[0], [2]])                # one mode; ⟨n⟩ = 0.36*2 = 0.72
    m = core_metrics(coeffs, bos, n_bos_modes=1)
    assert abs(m['p0'] - 0.64) < 1e-12
    assert abs(m['mean_n_total'] - 0.72) < 1e-12
    assert abs(m['mean_n_per_mode'] - 0.72) < 1e-12


def test_core_metrics_p0_only_without_bos():
    m = core_metrics([1.0, 0.0, 0.0])
    assert m['p0'] == 1.0                     # single-determinant state
    assert m['mean_n_total'] is None


# --- 2. sweep point loading --------------------------------------------

def _write_sweep(tmpdir, results):
    path = os.path.join(tmpdir, 'sweep.json')
    with open(path, 'w') as f:
        json.dump({'metadata': {'L': 2, 'dim': 3}, 'results': results}, f)
    return path


def test_load_sweep_point_matches_A():
    with tempfile.TemporaryDirectory() as d:
        path = _write_sweep(d, [
            {'A': 1, 'L': 2, 'dim': 3, 'n_b': 8, 'Physical_Lambda': 1e6,
             'Total_T_Count': 2e6, 'Logical_Qubits': 200},
            {'A': 4, 'L': 2, 'dim': 3, 'n_b': 8, 'Physical_Lambda': 4e6,
             'Total_T_Count': 5e6, 'Logical_Qubits': 240},
        ])
        p = load_sweep_point(path, A=4)
        assert p['physical_lambda'] == 4e6
        assert p['total_t_count'] == 5e6
        assert p['n_b'] == 8


# --- 3. isospectrality gate --------------------------------------------

def test_gate_certifies_consistent_frame():
    # gaussian+lf E∞ sits just above the isospectral reference -> real compaction.
    e_inf = {'bare': 7229.0, 'gaussian': 7220.0, 'gaussian+lf': 7225.0}
    g = isospectrality_gate(e_inf, sites=27, test_frame='gaussian+lf')
    assert g['certified'] is True
    assert g['e_inf_ref'] == 7220.0
    assert g['delta_per_site'] > 0


def test_gate_flags_spectrum_shift():
    # gaussian+lf dips well below the isospectral reference -> shift, not compaction.
    e_inf = {'bare': 7229.0, 'gaussian': 7220.0, 'gaussian+lf': 7180.0}
    g = isospectrality_gate(e_inf, sites=27, test_frame='gaussian+lf')
    assert g['certified'] is False
    assert g['delta_per_site'] < 0
    assert 'spectrum shift' in g['reason']


def test_gate_requires_a_reference():
    try:
        isospectrality_gate({'gaussian+lf': 100.0}, sites=8, test_frame='gaussian+lf')
    except ValueError:
        return
    raise AssertionError("gate should require an exactly-isospectral reference frame")


# --- 4. end-to-end reduction -------------------------------------------

def test_frame_qpe_reduction_end_to_end():
    bare = {'p0': 0.50, 'mean_n_per_mode': 1.0}
    frame = {'p0': 0.65, 'mean_n_per_mode': 0.5}          # better warm start, lower ⟨n⟩
    point = {'A': 4, 'L': 2, 'dim': 3, 'n_b': 8,
             'physical_lambda': 4.5e6, 'total_t_count': 5e6}
    red = frame_qpe_reduction(
        bare, frame, point, n_bos=24, n_ferm=32,
        frame_layers=('squeeze',),
        test_frame='gaussian', sites=8,
        e_inf_by_frame={'bare': 2421.0, 'gaussian': 2418.0})
    assert abs(red['p0_gain'] - 1.3) < 1e-9                # 0.65 / 0.50
    assert red['repetition_factor'] < 1.0                  # fewer QPE runs
    assert red['qpe_T_ratio'] < 1.0                        # net cost down
    assert red['boson_qubit_saving_per_mode'] >= 1         # ⟨n⟩ 1.0 -> 0.5 shrinks n_b
    assert red['prep_vs_walk'] < 1e-3                      # state-prep is negligible
    assert red['certified'] is True


def test_fold_warmstart_writes_column():
    with tempfile.TemporaryDirectory() as d:
        path = _write_sweep(d, [
            {'A': 4, 'L': 2, 'dim': 3, 'n_b': 8, 'Physical_Lambda': 4.5e6,
             'Total_T_Count': 5e6, 'Logical_Qubits': 240},
        ])
        bare = {'p0': 0.50, 'mean_n_per_mode': 1.0}
        frame = {'p0': 0.65, 'mean_n_per_mode': 0.5}
        data = fold_warmstart_into_sweep(
            path, bare, frame, n_bos=24, n_ferm=32,
            frame_name='gaussian+lf', frame_layers=('squeeze', 'displace'), A=4)
        col = data['results'][0]['warmstart']['gaussian+lf']
        assert abs(col['p0_gain'] - 1.3) < 1e-9
        assert col['repetition_factor'] < 1.0
        # persisted to disk
        with open(path) as f:
            assert 'gaussian+lf' in json.load(f)['results'][0]['warmstart']


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f"test_frame_qpe_bridge: {len(fns)} passed")
