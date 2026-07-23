"""
Data I/O for the classical ground-state pipeline.

Method-agnostic: every run records the **method** (TrimCI for now; we will
compare DMRG / AFQMC / NQS / ... later — just change the field) and the
**transform / frame** axis — the change-of-basis in which the ground state is
expressed, chosen for compactness (see tasks/32, tasks/33):
    "bare"     — the native fermion-occupation ⊗ boson-Fock basis (what every run
                 uses TODAY; no frame transform).
    "COO"      — Core-Optimized Orbitals: fermion single-particle basis co-optimized
                 with the sparse CI core (Zhang & Otten 2026, arXiv:2605.22977). NOT
                 YET IMPLEMENTED.
    "LF"       — Lang-Firsov polaron frame: coherent displacement of the boson sector
                 (variational for our transition/gradient vertex). NOT YET IMPLEMENTED.
    "gaussian" — Gaussian boson transform = displacement ⊕ squeezing (Bogoliubov).
                 NOT YET IMPLEMENTED.
NOTE: runs saved before 2026-07-01 were tagged "COO" by an earlier mislabel — they
are actually "bare". The frame axis is being reworked in tasks/33.

Each run is saved as a self-describing folder under `data/classical/<date>/<run-id>/`:

    metadata.json    — method, transform, system + solver params, wall-clock
                       runtime, energy, convergence history, exact reference
    hamiltonian.mixedfci  — the generalized Hamiltonian dump (dump.write_dump)
    groundstate.npz  — the compact wavefunction: fermion dets + boson occ + coeffs

The wall-clock runtime is saved explicitly so we can see where the classical
solver stops running quickly enough (the whole point of the cost-confront study).
`data/` is gitignored, so runs stay local.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from math import comb

import numpy as np

DATA_ROOT = os.path.join("data", "classical")

# the transform / frame axis values (change-of-basis for compactness; see tasks/33)
TRANSFORM_BARE = "bare"          # native fermion-Fock ⊗ boson-Fock basis (current)
TRANSFORM_COO = "COO"            # Core-Optimized Orbitals (fermion; arXiv:2605.22977) — planned
TRANSFORM_LF = "LF"              # Lang-Firsov polaron frame (boson displacement) — STEP 3
TRANSFORM_GAUSSIAN = "gaussian"  # Gaussian boson transform (squeeze/Bogoliubov) — STEP 1/2/2b
TRANSFORM_GAUSSIAN_LF = "gaussian+LF"  # layered squeeze ∘ displace (recommended) — STEP 4


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sector_size(H, A):
    return comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)


_MASK64 = (1 << 64) - 1


def _states_arrays(result):
    """Compact wavefunction arrays from a GroundStateResult.

    Fermion masks are stored as (N, W) little-endian uint64 words so the bitmask
    survives any mode count (e.g. L=4 in 3d => 256 fermion modes => W=4); numpy
    has no native >64-bit integer. `load_classical_run` reassembles the Python-int
    mask. W = ceil((max occupied bit + 1)/64).
    """
    # Tier-2 array-native result: the compact arrays are already in exactly this
    # layout (ferm (N,W) uint64, bos (N,n_bos) uint16). Use them directly instead
    # of iterating a MixedState per state — the whole point of the array path is
    # never to materialize those objects.
    if getattr(result, "ferm_arr", None) is not None:
        ferm = np.ascontiguousarray(result.ferm_arr, dtype=np.uint64)
        bos = np.ascontiguousarray(result.bos_arr, dtype=np.uint16)
        coeffs = np.asarray(result.coeffs, dtype=complex)
        return ferm, bos, coeffs

    states = result.states
    n = len(states)
    max_bits = max((s.ferm.bit_length() for s in states), default=0)
    n_words = max(1, (max_bits + 63) // 64)
    ferm = np.empty((n, n_words), dtype=np.uint64)
    for w in range(n_words):
        shift = 64 * w
        ferm[:, w] = np.fromiter(((s.ferm >> shift) & _MASK64 for s in states),
                                 dtype=np.uint64, count=n)
    n_bos = len(states[0].bos) if states else 0
    bos = np.array([s.bos for s in states], dtype=np.uint16) if states \
        else np.zeros((0, n_bos), dtype=np.uint16)
    coeffs = np.array(result.coeffs, dtype=complex)
    return ferm, bos, coeffs


def save_classical_run(result, H, A, *, runtime_s, solver_params=None,
                       method="TrimCI", transform=TRANSFORM_BARE,
                       exact_reference=None, params=None, data_root=DATA_ROOT,
                       label=None, save_hamiltonian=True, convergence=None,
                       hpc=False):
    """Save a classical ground-state run as a self-describing folder.

    Args:
        result: a GroundStateResult (energy, states, coeffs, n_dets, history).
        H: the MixedH that was solved.
        A: nucleon number (the conserved fermion sector).
        runtime_s: wall-clock seconds for the classical solve (REQUIRED).
        solver_params: dict of solver knobs (n_dets, n_runs, seed, backend, ...).
        method: solver method name ("TrimCI" now; swappable for future methods).
        transform: boson-frame axis — "COO" (bare) or "LF" (Lang-Firsov).
        exact_reference: optional {"method","energy"} for a cross-checked E0.
        params: physical-parameter dict (a_L, m_pi, ...); from H.meta if None.
        label: optional short tag added to the run-id.

    Returns:
        The run directory path.
    """
    ts = datetime.now()
    date = ts.strftime("%Y-%m-%d")
    stamp = ts.strftime("%H%M%S")
    meta_in = H.meta or {}
    L = meta_in.get("L", "?")
    dim = meta_in.get("dim", "?")
    tag = f"_{label}" if label else ""
    run_id = f"{method}_{transform}_L{L}d{dim}_A{A}_nb{meta_in.get('n_b','?')}_{stamp}{tag}"
    run_dir = os.path.join(data_root, date, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # --- compact ground state ---------------------------------------------
    ferm, bos, coeffs = _states_arrays(result)
    np.savez_compressed(os.path.join(run_dir, "groundstate.npz"),
                        ferm=ferm, bos=bos, coeffs=coeffs,
                        energy=np.array([result.energy]))

    # --- Hamiltonian dump --------------------------------------------------
    ham_file = None
    if save_hamiltonian:
        from classical.trimci.dump import write_dump
        ham_file = "hamiltonian.mixedfci"
        write_dump(H, os.path.join(run_dir, ham_file), n_elec=A)

    # --- metadata ----------------------------------------------------------
    # convergence: explicit (n_dets, energy) sweep if given, else the solver's
    # internal round history (filtered to integer det-count entries).
    if convergence is not None:
        history = [[int(n), float(e)] for (n, e) in convergence]
    else:
        history = [[int(n), float(e)] for (n, e) in result.history
                   if isinstance(n, int)]
    metadata = {
        "method": method,
        "transform": transform,
        "label": label,
        "timestamp": ts.isoformat(timespec="seconds"),
        "runtime_s": float(runtime_s),
        "system": {
            "L": L, "dim": dim, "A": int(A),
            "n_b": meta_in.get("n_b"), "N_f": H.N_f,
            "n_ferm_modes": H.n_ferm_modes, "n_bos_modes": H.n_bos_modes,
            "n_terms": len(H.terms),
            "sector_size": float(_sector_size(H, A)),
            "a_L": (params or {}).get("a_L"),
        },
        "solver": dict(solver_params or {}),
        "result": {
            "energy": float(result.energy),
            "n_dets": int(result.n_dets),
            "frac_of_sector": float(result.n_dets / max(_sector_size(H, A), 1)),
        },
        "convergence": history,
        "exact_reference": exact_reference,
        "files": {"groundstate": "groundstate.npz", "hamiltonian": ham_file},
        # provenance: git commit + where it ran (hpc flag distinguishes cluster
        # runs from laptop runs when JSONs are pooled).
        "code": {"git_commit": _git_commit(), "hpc": bool(hpc),
                 "host": __import__("platform").node()},
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def load_classical_run(run_dir):
    """Load a saved run: returns dict with `metadata`, `ferm`, `bos`, `coeffs`.

    `ferm` is reassembled into a Python-int object array (full bitmask, any
    width). Backward-compatible with the legacy 1-D uint64 `ferm` format.
    """
    with open(os.path.join(run_dir, "metadata.json")) as f:
        metadata = json.load(f)
    gs = np.load(os.path.join(run_dir, "groundstate.npz"))
    ferm_raw = gs["ferm"]
    if ferm_raw.ndim == 2:   # (N, W) little-endian words -> Python-int bitmask
        ferm = np.array([sum(int(word) << (64 * w) for w, word in enumerate(row))
                         for row in ferm_raw], dtype=object)
    else:                    # legacy 1-D uint64
        ferm = ferm_raw
    return {
        "metadata": metadata,
        "ferm": ferm, "bos": gs["bos"], "coeffs": gs["coeffs"],
        "energy": float(gs["energy"][0]),
        "run_dir": run_dir,
    }
