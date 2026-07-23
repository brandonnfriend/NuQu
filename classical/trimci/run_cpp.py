"""
Larger-L TrimCI driver — full C++ hot path (connections + matrix build +
expansion in C++ via mixed_ci, diagonalization via the official C++ sparse
Davidson). Solves systems far past exact reach in seconds.

Build prerequisites once:
    .venv/bin/python -m ensurepip --upgrade   # if needed
    VIRTUAL_ENV=.venv uv pip install pybind11
    VIRTUAL_ENV=.venv uv pip install /path/to/TrimCI      # Tier-1 sparse fork
    bash classical/trimci/backend_fork/build_mixed_ci.sh  # Tier-2 connections port

Run:
    python -m classical.trimci.run_cpp --L 2 --dim 2 --A 1 --n_b 2 --n_dets 2000
    python -m classical.trimci.run_cpp --L 3 --dim 1 --A 1 --n_b 2
"""

from __future__ import annotations

import argparse
import os
import time
from math import comb

import numpy as np

from .hamiltonian import build_from_eft
from .backend import (cpp_available, cpp_ground_state_ensemble,
                      cpp_ground_state_ensemble_arrays)
from .lanczos import lanczos_ground_state, DEFAULT_MAX_STATES


def _solver(arrays):
    """Pick the ensemble solver: Tier-2 array-native (compact arrays end-to-end,
    the scaling path toward 1e6 states) or the object path (MixedState sets)."""
    return cpp_ground_state_ensemble_arrays if arrays else cpp_ground_state_ensemble


def run(L=2, dim=2, A=1, n_b=2, n_dets=2000, n_runs=4, seed=0, save=False,
        arrays=True, verbose=True):
    if not cpp_available():
        raise RuntimeError(
            "Full C++ path unavailable. Build the sparse-Davidson fork "
            "(uv pip install /path/to/TrimCI) and the connections port "
            "(bash classical/trimci/backend_fork/build_mixed_ci.sh)."
        )
    solver = _solver(arrays)
    H = build_from_eft(L, dim, n_b)
    sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
    if verbose:
        print("=" * 70)
        print(f"  TrimCI (full C++ path)  L={L} dim={dim} A={A} N_f={H.N_f}")
        print(f"  {H.n_ferm_modes}F + {H.n_bos_modes}B modes, {len(H.terms)} terms; "
              f"sector ~ {sector:.2e} states")
        print("=" * 70)

    # Exact reference only if the full sector is enumerable.
    E_ref = None
    if sector <= DEFAULT_MAX_STATES:
        E_ref, _ = lanczos_ground_state(H, n_elec=A)
        if verbose:
            print(f"  exact (Lanczos): E0 = {E_ref:.8f} MeV  ({sector:,} states)")
    elif verbose:
        print(f"  (no exact reference: sector {sector:.1e} >> Lanczos reach)")

    # det-count convergence sweep.
    targets = sorted({int(x) for x in
                      np.unique(np.geomspace(max(50, n_dets // 16), n_dets, 5))})
    if verbose:
        head = f"  {'n_dets':>8} {'E (MeV)':>16} {'dE vs prev':>13}"
        if E_ref is not None:
            head += f" {'dE vs ED':>12}"
        print(head)
        print("  " + "-" * (52 if E_ref is None else 64))

    rows = []
    prev = None
    final_res = final_dt = None
    for nd in targets:
        t = time.time()
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=nd, seed=seed)
        dt = time.time() - t
        dprev = (res.energy - prev) if prev is not None else None
        line = f"  {res.n_dets:>8} {res.energy:>16.8f} " \
               f"{(f'{dprev:.2e}' if dprev is not None else '--'):>13}"
        if E_ref is not None:
            line += f" {res.energy - E_ref:>12.2e}"
        line += f"   [{dt:.1f}s]"
        if verbose:
            print(line)
        rows.append((res.n_dets, res.energy, dt))
        prev = res.energy
        final_res, final_dt = res, dt   # the production (largest-n_dets) solve

    variational_ok = (E_ref is None) or (min(e for _, e, _ in rows) >= E_ref - 1e-6)
    if verbose:
        print("  " + "-" * (52 if E_ref is None else 64))
        if E_ref is not None:
            print(f"  variational (E >= E_ED): {variational_ok}")

    if save and final_res is not None:
        from classical.io import save_classical_run, TRANSFORM_BARE
        from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
        run_dir = save_classical_run(
            final_res, H, A, runtime_s=final_dt,
            method="TrimCI", transform=TRANSFORM_BARE,
            solver_params={"n_dets": n_dets, "n_runs": n_runs, "seed": seed,
                           "backend": "cpp", "boson_init_mean": 0.5},
            exact_reference=({"method": "Lanczos", "energy": E_ref}
                             if E_ref is not None else None),
            params=get_physical_parameters(),
            convergence=[(n, e) for (n, e, _) in rows],
        )
        if verbose:
            print(f"  saved -> {run_dir}")
    if verbose:
        print("=" * 70)
    return rows


def runtime_sweep(systems=None, n_dets=2000, n_runs=4, seed=0,
                  max_seconds=480, out_png=None, arrays=True, verbose=True):
    """Solve a spread of systems once each (fixed n_dets) and save them, to map
    classical wall-clock vs Hilbert-sector size. Stops launching new systems
    once `max_seconds` of cumulative solve time is reached. Writes the
    runtime-vs-size plot. Each `system` is (L, dim, A, n_b)."""
    from classical.io import save_classical_run, TRANSFORM_BARE
    from classical.plotting import plot_runtime_vs_size
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    solver = _solver(arrays)

    if systems is None:
        # span sector sizes ~10^2 .. 10^17 via N_f (runtime-flat) and #sites
        # (runtime grows). Capped at 16 sites (<=64 fermion modes for the C++ mask).
        systems = [
            (1, 1, 1, 2), (2, 1, 1, 2), (2, 1, 1, 3), (2, 1, 1, 4), (2, 1, 1, 5),
            (3, 1, 1, 2), (4, 1, 1, 2), (2, 2, 1, 2), (2, 2, 1, 3),
            (2, 3, 1, 2), (3, 2, 1, 2),
        ]
    params = get_physical_parameters()
    if verbose:
        print("=" * 74)
        print(f"  RUNTIME-vs-SIZE SWEEP  (n_dets={n_dets}, n_runs={n_runs}, "
              f"budget {max_seconds}s)")
        print(f"  {'system':>16} {'sites':>5} {'modes':>9} {'sector':>10} "
              f"{'E (MeV)':>14} {'t(s)':>7}")
        print("  " + "-" * 70)
    spent = 0.0
    done = []
    for (L, dim, A, n_b) in systems:
        H = build_from_eft(L, dim, n_b)
        # No mode-count cap: the C++ fermion mask is arbitrary-width. Scale is
        # bounded by RAM/time (the cache byte-budget guards RAM); see l_scaling_sweep.
        sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
        t = time.time()
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=n_dets, seed=seed)
        dt = time.time() - t
        spent += dt
        save_classical_run(
            res, H, A, runtime_s=dt, method="TrimCI", transform=TRANSFORM_BARE,
            solver_params={"n_dets": n_dets, "n_runs": n_runs, "seed": seed,
                           "backend": "cpp", "boson_init_mean": 0.5},
            params=params, convergence=[(n_dets, res.energy)], label="sweep")
        done.append((L, dim, A, n_b, sector, res.energy, dt))
        if verbose:
            print(f"  L={L} d={dim} A={A} N_f={H.N_f:<4} "
                  f"{L**dim:>5} {H.n_ferm_modes:>3}F+{H.n_bos_modes:>2}B "
                  f"{sector:>10.1e} {res.energy:>14.4f} {dt:>7.1f}")
        if spent > max_seconds:
            if verbose:
                print(f"  [budget] {spent:.0f}s used > {max_seconds}s — stopping sweep")
            break
    if out_png is None:
        out_png = os.path.join("data", "classical", "runtime_vs_size.png")
    plot_runtime_vs_size(out_path=out_png,
                         title=f"TrimCI (bare) runtime vs sector size  (n_dets={n_dets})")
    if verbose:
        print("  " + "-" * 70)
        print(f"  {len(done)} systems, {spent:.0f}s total. plot -> {out_png}")
        print("=" * 74)
    return done


def _predict_runtime(records, next_sites):
    """Power-law extrapolation runtime ~ a * sites^b from solved (sites, dt)
    records. Sub-second points are dropped first — the smallest lattices are
    non-asymptotic (L=1 has no hopping) and a single such point badly skews the
    log-log slope. Needs >=2 asymptotic points; else returns (None, None) so the
    caller does NOT gate on a thin/biased fit."""
    import numpy as np
    good = [r for r in records if r["runtime_s"] >= 1.0]
    if len(good) < 2:
        return None, None
    sites = np.array([r["sites"] for r in good], dtype=float)
    dt = np.array([r["runtime_s"] for r in good], dtype=float)
    b, loga = np.polyfit(np.log(sites), np.log(dt), 1)
    return float(np.exp(loga) * next_sites ** b), float(b)


def _history_points(res):
    """(core_size, energy) round-history of a solve, filtered to numeric entries
    (drops the ('ensemble', per_run) tail) and sorted by core size. This is the
    convergence trajectory — the core ramps from ~n_init up to the final size."""
    pts = [(int(n), float(e)) for (n, e) in res.history if isinstance(n, int)]
    return sorted(set(pts))


# NOTE: a 1/N extrapolation to the "true GS" was tried and dropped — for this
# strongly-bosonic system the selected-CI tail is too noisy for it to be reliable
# (it disagreed with direct energy comparisons). The robust convergence metric is
# the last-core-doubling drop (graph.halving_drop): "does adding ~2x more
# determinants still move the energy?".


def l_scaling_sweep(dim=3, L_values=(1, 2, 3, 4), A=1, n_b=2,
                    max_n_dets=4000, ladder_start=1000, target_gs_rel=1e-2,
                    gate="relative", eps_persite=0.1,
                    n_runs=3, seed=0, label="Lscan3d", next_budget_s=3600,
                    cache_bytes=128 << 20, hpc=False, hpc_max_n_dets=50000,
                    arrays=True, out_dir=None, verbose=True):
    """Solve the A=A ground state for each L in L_values at fixed `dim`, saving
    every run, to map classical cost / compression / energy / convergence vs
    lattice size L.

    DYNAMICAL core via an INDEPENDENT LADDER: each L is solved from scratch at
    core sizes r = `ladder_start`, 2r, 4r, ... (each a full ensemble solve),
    growing until the energy drop between successive INDEPENDENT solves,
    |E(2N)-E(N)|/|E|, falls below `target_gs_rel` — or the core reaches the ceiling
    `max_n_dets`. Independent solves are the ONLY reliable convergence signal for
    this strongly-bosonic system: a single growing run's ramp plateaus deceptively
    (marginal-dE and 1/N extrapolation both mislead), so we compare full solves at
    well-separated sizes instead. Bigger systems need more rungs to converge, so
    the final core grows with L. The ladder (E vs N) is saved as the convergence
    curve, and `runtime_s` is the total ladder wall-clock.

    CONVERGENCE GATE (`gate`) — kept side by side as a comparison switch; BOTH
    drops are always recorded, so one run shows what either gate would decide:
      * "relative" (default, legacy): stop when |E(2N)-E(N)|/|E| < `target_gs_rel`.
        Size-EXTENSIVE-normalized: E ~ sites, so a fixed relative target silently
        LOOSENS the per-site tolerance as L grows (the extensivity trap).
      * "per_site": stop when |E(2N)-E(N)|/sites < `eps_persite` (MeV/site). The
        size-INTENSIVE gate — a genuinely fixed per-site accuracy across L, which
        is what the dets-vs-L exponent must hold constant.

    HPC SWITCH: `hpc=True` raises the ceiling to `hpc_max_n_dets` (default 50000)
    — flip it for big-memory runs; the laptop default `max_n_dets=4000` keeps each
    L to a few minutes (runtime scales ~core^2). `cache_bytes` = C++ cache budget.

    Runtime gate: before each L the next runtime is extrapolated as a power law in
    #sites (dropping sub-second non-asymptotic points); an L whose predicted time
    exceeds `next_budget_s` is skipped and the sweep stops (scales out on HPC).
    Writes the five vs-L plots and returns the per-L records + the un-run prediction."""
    from classical.io import save_classical_run, TRANSFORM_BARE
    from classical.plotting import (plot_runtime_vs_L, plot_compression_vs_L,
                                    plot_energy_vs_L, plot_convergence_vs_L,
                                    plot_coresize_vs_L)
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from .backend import _ferm_words

    if gate not in ("relative", "per_site"):
        raise ValueError(f"gate must be 'relative' or 'per_site', got {gate!r}")
    solver = _solver(arrays)
    ceiling = int(hpc_max_n_dets if hpc else max_n_dets)
    params = get_physical_parameters()
    gate_str = (f"|dE(2N)|/sites<{eps_persite:g} MeV/site" if gate == "per_site"
                else f"|dE(2N)|/|E|<{target_gs_rel:g}")
    if verbose:
        print("=" * 90)
        print(f"  L-SCALING SWEEP (independent ladder)  dim={dim} A={A} n_b={n_b} "
              f"N_f={2**n_b} n_runs={n_runs} (label={label})")
        print(f"  ladder {ladder_start}x2... until {gate_str} [gate={gate}] "
              f"or ceiling={ceiling} [{'HPC' if hpc else 'laptop'}]; next-L budget "
              f"{next_budget_s}s; cache {cache_bytes/1e9:.1f} GB")
        print(f"  {'L':>2} {'sites':>5} {'modes':>11} {'terms':>6} {'sector':>9} "
              f"{'core':>6} {'stop':>9} {'E0 (MeV)':>13} {'2x-drop':>8} {'t(s)':>7}")
        print("  " + "-" * 88)
    records = []
    prediction = None
    for L in L_values:
        H = build_from_eft(L, dim, n_b)
        H._cpp_cache_bytes = int(cache_bytes)
        sites = L ** dim
        sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
        pred, b = _predict_runtime(records, sites)
        if pred is not None and pred > next_budget_s:
            prediction = {"L": L, "sites": sites, "pred_s": pred, "exponent": b}
            if verbose:
                print(f"  L={L}: predicted ~{pred/60:.1f} min "
                      f"(~sites^{b:.2f}) > budget {next_budget_s/60:.0f} min "
                      f"— STOP (raise the budget / run on HPC).")
            break

        # DYNAMICAL core via an INDEPENDENT LADDER: solve from scratch at
        # r = ladder_start, 2r, 4r, ... (each a full ensemble solve), growing
        # until the drop between successive INDEPENDENT solves passes the active
        # gate — the only reliable convergence signal here, since a single growing
        # run's ramp plateaus deceptively. Both the relative drop |E(2N)-E(N)|/|E|
        # and the per-site drop |E(2N)-E(N)|/sites are recorded each rung; the
        # active gate decides the stop. Capped at `ceiling`.
        rungs = []                    # (core_size, energy) — reliable convergence curve
        total_dt = 0.0
        r = min(ladder_start, ceiling)
        final_res = None
        stop_reason = "capped"
        drop = drop_ps = None
        while True:
            # enough expansion rounds to actually grow the core to r (the default
            # 12 caps the x1.5 ramp at ~3.5k dets, silently short of large rungs).
            mr = max(12, int(np.ceil(np.log(max(r, 2) / 20.0) / np.log(1.5))) + 4)
            t = time.time()
            res = solver(H, n_elec=A, n_runs=n_runs,
                         n_dets=r, seed=seed, max_rounds=mr)
            total_dt += time.time() - t
            final_res = res
            rungs.append((int(res.n_dets), float(res.energy)))
            if len(rungs) >= 2:
                dE = abs(rungs[-1][1] - rungs[-2][1])
                drop = dE / max(abs(rungs[-1][1]), 1e-12)   # size-extensive-normalized
                drop_ps = dE / sites                        # size-intensive (MeV/site)
                converged = (drop_ps < eps_persite if gate == "per_site"
                             else drop < target_gs_rel)
                if converged:
                    stop_reason = "converged"
                    break
            if r >= min(ceiling, sector):
                stop_reason = "capped"
                break
            r = min(r * 2, ceiling)

        conv = sorted(set(rungs))     # independent-solve E-vs-N (reliable, monotone)
        core = final_res.n_dets
        e_per_site = float(final_res.energy) / sites
        save_classical_run(
            final_res, H, A, runtime_s=total_dt, method="TrimCI",
            transform=TRANSFORM_BARE,
            solver_params={"max_n_dets": ceiling, "ladder_start": ladder_start,
                           "gate": gate, "target_gs_rel": target_gs_rel,
                           "eps_persite": eps_persite, "n_runs": n_runs,
                           "seed": seed, "backend": "cpp", "ladder": "independent",
                           "cache_bytes": int(cache_bytes),
                           "stop_reason": stop_reason,
                           "last_doubling_drop_rel": drop,
                           "last_doubling_drop_persite": drop_ps},
            params=params, convergence=conv, label=label)
        records.append({"L": L, "sites": sites, "n_ferm_modes": H.n_ferm_modes,
                        "n_words": _ferm_words(H), "n_bos_modes": H.n_bos_modes,
                        "n_terms": len(H.terms), "sector_size": sector,
                        "energy": final_res.energy, "energy_per_site": e_per_site,
                        "n_dets": core, "runtime_s": total_dt,
                        "stop_reason": stop_reason, "n_rungs": len(rungs),
                        "halving_drop_rel": drop,
                        "halving_drop_persite": drop_ps})
        if verbose:
            pstr = f" (pred {pred:.0f}s)" if pred is not None else ""
            active = drop_ps if gate == "per_site" else drop
            hd = f"{active:.1e}" if active is not None else "   -"
            print(f"  {L:>2} {sites:>5} {H.n_ferm_modes:>3}F/{_ferm_words(H)}w"
                  f"+{H.n_bos_modes:>3}B {len(H.terms):>6} {sector:>9.1e} "
                  f"{core:>6} {stop_reason:>9} {final_res.energy:>13.3f} "
                  f"{hd:>8} {total_dt:>7.1f}{pstr}")

    # predict the first un-run L if we stopped naturally at the end of L_values
    if prediction is None and records:
        nxtL = records[-1]["L"] + 1
        pred, b = _predict_runtime(records, nxtL ** dim)
        if pred is not None:
            prediction = {"L": nxtL, "sites": nxtL ** dim, "pred_s": pred,
                          "exponent": b}

    if out_dir is None:
        out_dir = os.path.join("data", "classical")
    suffix = f"_{label}" if label else ""
    plot_runtime_vs_L(dim=dim, label=label, A=A,
                      out_path=os.path.join(out_dir, f"runtime_vs_L{suffix}.png"))
    plot_compression_vs_L(dim=dim, label=label, A=A,
                          out_path=os.path.join(out_dir, f"compression_vs_L{suffix}.png"))
    plot_energy_vs_L(dim=dim, label=label, A=A,
                     out_path=os.path.join(out_dir, f"energy_vs_L{suffix}.png"))
    plot_convergence_vs_L(dim=dim, label=label, A=A,
                          out_path=os.path.join(out_dir, f"convergence_vs_L{suffix}.png"))
    plot_coresize_vs_L(dim=dim, label=label, A=A,
                       out_path=os.path.join(out_dir, f"coresize_vs_L{suffix}.png"))
    if verbose:
        print("  " + "-" * 88)
        print(f"  {len(records)} L-points solved (L={[r['L'] for r in records]}).")
        print(f"  per-L convergence (independent-ladder rungs, last-doubling drop, "
              f"gate={gate}):")
        for r in records:
            active = (r['halving_drop_persite'] if gate == "per_site"
                      else r['halving_drop_rel'])
            unit = "MeV/site" if gate == "per_site" else "of |E0|"
            thresh = eps_persite if gate == "per_site" else target_gs_rel
            hd = f"{active:.2e}" if active is not None else "n/a"
            verdict = ('converged' if (active if active is not None else 1) < thresh
                       else 'still descending — raise ceiling / run on HPC')
            print(f"    L={r['L']}: core={r['n_dets']:>5} ({r['n_rungs']} rungs, "
                  f"{r['stop_reason']}); last independent doubling moved E0 by "
                  f"{hd} {unit} — {verdict}")
        if prediction:
            print(f"  next: L={prediction['L']} (sites={prediction['sites']}) "
                  f"predicted ~{prediction['pred_s']/60:.1f} min "
                  f"(runtime ~ sites^{prediction['exponent']:.2f}).")
        print("=" * 90)
    return records, prediction


def core_convergence_sweep(L=2, dim=3, A=1, n_b=2,
                           cores=(500, 1000, 2000, 4000, 8000, 16000),
                           n_runs=3, seed=0, label=None, cache_bytes=128 << 20,
                           transform="bare", frame_params=None,
                           arrays=True, out_dir=None, verbose=True):
    """Single-system energy-vs-core-size convergence study: solve one (L, dim, A)
    INDEPENDENTLY at each core size in `cores` and record (core, E0, runtime).
    Independent full solves are the reliable convergence signal (a single growing
    run's ramp plateaus deceptively). Each solve is saved to the data pipeline
    (label `{label}_c{core}`); an E0-vs-core plot is written. Returns per-core recs.

    `max_rounds` is sized per core so the x1.5 ramp actually reaches it (the default
    12 silently caps the core near ~3.5k)."""
    from classical.io import save_classical_run, TRANSFORM_BARE
    from classical.plotting import plot_core_convergence
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters

    solver = _solver(arrays)
    if label is None:
        label = f"L{L}core{dim}d" + ("" if transform == "bare" else f"_{transform}")
    H = build_from_eft(L, dim, n_b, transform=transform, frame_params=frame_params)
    H._cpp_cache_bytes = int(cache_bytes)
    sites = L ** dim
    sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
    params = get_physical_parameters()
    if verbose:
        print("=" * 74)
        print(f"  CORE-CONVERGENCE  L={L} dim={dim} A={A} N_f={H.N_f}  {sites} sites, "
              f"{len(H.terms)} terms, sector {sector:.1e}  (label={label}, n_runs={n_runs})")
        print(f"  {'req':>6} {'core':>6} {'E0 (MeV)':>14} {'dE prev':>10} "
              f"{'rel':>9} {'t(s)':>8}")
        print("  " + "-" * 60)
    recs = []
    prevE = None
    for c in cores:
        mr = max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)
        t = time.time()
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=c,
                     seed=seed, max_rounds=mr)
        dt = time.time() - t
        dE = (res.energy - prevE) if prevE is not None else None
        rel = (abs(dE) / max(abs(res.energy), 1e-12)) if dE is not None else None
        save_classical_run(
            res, H, A, runtime_s=dt, method="TrimCI", transform=transform,
            solver_params={"n_dets": c, "n_runs": n_runs, "seed": seed,
                           "backend": "cpp", "cache_bytes": int(cache_bytes),
                           "frame_params": frame_params},
            params=params, convergence=[(int(res.n_dets), float(res.energy))],
            label=f"{label}_c{c}")
        recs.append({"core_req": c, "core": int(res.n_dets),
                     "energy": float(res.energy), "dE": dE, "rel": rel,
                     "runtime_s": dt})
        if verbose:
            ds = f"{dE:.3f}" if dE is not None else "--"
            rs = f"{rel:.1e}" if rel is not None else "--"
            print(f"  {c:>6} {res.n_dets:>6} {res.energy:>14.4f} {ds:>10} "
                  f"{rs:>9} {dt:>8.1f}")
        prevE = res.energy
    if out_dir is None:
        out_dir = os.path.join("data", "classical")
    out_png = os.path.join(out_dir, f"core_convergence_{label}.png")
    plot_core_convergence(recs, L=L, dim=dim, A=A, out_path=out_png)
    if verbose:
        print("  " + "-" * 60)
        print(f"  {len(recs)} cores solved; plot -> {out_png}")
        print("=" * 74)
    return recs


def _pick_solver(arrays):
    """Pick the ensemble solver + a scalable PT2 re-diagonalizer, falling back to
    the pure-Python path when the C++ hot path is not built. Returns
    (solver, pt2_diag, use_cpp)."""
    if cpp_available():
        from .backend import cpp_diagonalize_smart
        return _solver(arrays), cpp_diagonalize_smart, True
    from .graph import ground_state_ensemble as _py_solver
    return _py_solver, None, False        # dense re-diag (fine for small V)


def _solve_ladder(H, A, cores, solver, pt2_diag, n_runs=4, seed=0, pt2=True,
                  verbose=True):
    """Solve H at each core size in `cores` INDEPENDENTLY (each a full ensemble
    solve — the only reliable convergence signal here) and return the rung list
    [{core, E_var, dE_pt2, E_pt2, n_ext}].

    With `pt2`, each rung's E_var is the SELF-CONSISTENT variational energy from the
    PT2 re-diagonalization of the saved core (>= the solver's pre-trim pool energy),
    dE_pt2 the EN-PT2 correction, and E_pt2 their sum (the tighter estimate)."""
    from .pt2 import pt2_from_result
    rungs = []
    for c in cores:
        mr = max(12, int(np.ceil(np.log(max(c, 2) / 20.0) / np.log(1.5))) + 4)
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=c, seed=seed, max_rounds=mr)
        rung = {"core": int(res.n_dets), "E_var": float(res.energy)}
        if pt2:
            pr = pt2_from_result(H, res, diag_fn=pt2_diag)
            rung["E_var"] = float(pr["E_var"])
            rung["dE_pt2"] = float(pr["dE_pt2"])
            rung["E_pt2"] = rung["E_var"] + rung["dE_pt2"]
            rung["n_ext"] = pr["n_ext"]
        if verbose:
            extra = (f"  dE_PT2={rung['dE_pt2']:+.4f} (n_ext={rung['n_ext']})"
                     if pt2 else "")
            print(f"  core={rung['core']:>6}  E_var={rung['E_var']:12.5f} MeV{extra}")
        rungs.append(rung)
    return rungs


def _adaptive_ladder_solve(H, A, ladder_start, n_rungs, solver, pt2_diag,
                           max_core=None, max_rung_seconds=None,
                           n_runs=4, seed=0, verbose=True):
    """Geometric core ladder (ladder_start x2 each rung) with LAPTOP GUARDS: stop
    early once a rung's wall-clock exceeds `max_rung_seconds` (so the next, ~4x
    costlier rung isn't attempted) or the next core would exceed `max_core`. Returns
    the same rung dicts as `_solve_ladder` (PT2 always on), each with an added
    `wall_s`. With both guards None it reproduces `_solve_ladder` over the fixed
    n_rungs geometric ladder (the backward-compatible default)."""
    from .pt2 import pt2_from_result
    rungs = []
    r = int(ladder_start)
    for _ in range(n_rungs):
        if max_core is not None and r > max_core:
            break
        mr = max(12, int(np.ceil(np.log(max(r, 2) / 20.0) / np.log(1.5))) + 4)
        t = time.time()
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=r, seed=seed, max_rounds=mr)
        wall = time.time() - t
        pr = pt2_from_result(H, res, diag_fn=pt2_diag)
        rung = {"core": int(res.n_dets), "E_var": float(pr["E_var"]),
                "dE_pt2": float(pr["dE_pt2"]),
                "E_pt2": float(pr["E_var"]) + float(pr["dE_pt2"]),
                "n_ext": pr["n_ext"], "wall_s": wall}
        rungs.append(rung)
        if verbose:
            print(f"  core={rung['core']:>6}  E_var={rung['E_var']:12.5f} MeV  "
                  f"dE_PT2={rung['dE_pt2']:+.4f} (n_ext={rung['n_ext']})  [{wall:.1f}s]")
        if max_rung_seconds is not None and wall > max_rung_seconds:
            if verbose:
                print(f"  (rung wall {wall:.0f}s > budget {max_rung_seconds:.0f}s "
                      f"— stop growing the ladder for this L)")
            break
        r *= 2
    return rungs


def solve_and_report(L=2, dim=1, A=1, n_b=2, N_f=None,
                     cores=(250, 500, 1000, 2000), n_runs=4, seed=0,
                     transform="bare", frame_params=None, pt2=True,
                     cache_bytes=128 << 20, arrays=True, verbose=True):
    """Solve an INDEPENDENT core ladder and print the honest 3-number energy
    report: variational (best), variational + Epstein-Nesbet PT2, and the
    extrapolated E_infinity (+/- fit uncertainty), plus the exact Lanczos
    reference when the sector is enumerable. Now also reports the SIZE-INTENSIVE
    (per-site) energies (`report_energies(sites=L**dim)`).

    This is the reporting front-end for the classical baseline. Each core size in
    `cores` is a SEPARATE ensemble solve (independent solves are the only reliable
    convergence signal here — see run_cpp / TODO.md), so the extrapolation is fit
    over a trustworthy ladder rather than one growing run's bursty ramp.

    `N_f` overrides the per-mode Fock cutoff (need not be a power of two — a purely
    classical freedom; see hamiltonian.build_from_eft). `pt2=True` adds the
    Epstein-Nesbet correction at each rung (enables the SHCI/PT2 extrapolation).
    Falls back to the pure-Python solver if the C++ hot path is not built.
    """
    from .extrapolation import report_energies

    H = build_from_eft(L, dim, n_b, transform=transform, frame_params=frame_params,
                       N_f=N_f)
    H._cpp_cache_bytes = int(cache_bytes)
    sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
    solver, pt2_diag, use_cpp = _pick_solver(arrays)
    if verbose:
        print("=" * 68)
        print(f"  SOLVE + REPORT  L={L} d={dim} A={A} N_f={H.N_f}"
              f"{'' if transform == 'bare' else ' ' + transform}  "
              f"({H.n_ferm_modes}F+{H.n_bos_modes}B, {len(H.terms)} terms, "
              f"sector {sector:.1e}; {'C++' if use_cpp else 'python'} solver)")
        print("=" * 68)

    E_ref = None
    if sector <= DEFAULT_MAX_STATES:
        E_ref, _ = lanczos_ground_state(H, n_elec=A)
        if verbose:
            print(f"  exact reference (Lanczos): {E_ref:.6f} MeV\n")

    rungs = _solve_ladder(H, A, cores, solver, pt2_diag, n_runs=n_runs, seed=seed,
                          pt2=pt2, verbose=verbose)
    if verbose:
        print()
    return report_energies(rungs, exact=E_ref, sites=L ** dim,
                           label=f"L={L} d={dim} A={A} N_f={H.N_f}"
                           f"{'' if transform == 'bare' else ' ' + transform}",
                           verbose=verbose)


def converged_reference(L=2, dim=3, A=1, n_b=2, N_f=None,
                        cores=None, ladder_start=250, n_rungs=6,
                        max_core=None, max_rung_seconds=None,
                        eps_persite_targets=(1.0, 0.1),
                        sigma_frac=0.3, cross_frac=1.0,
                        n_runs=4, seed=0, transform="bare", frame_params=None,
                        cache_bytes=128 << 20, arrays=True, verbose=True):
    """Per-L converged REFERENCE energy E_inf(L) +/- sigma, for the dets-vs-L study.

    THE CRUX this handles honestly. The dets-vs-L exponent measures N*(L) = the core
    at which a solve reaches a FIXED per-site accuracy eps of the L-converged ground
    energy. But at L>=2 in 3D there is NO exact reference: E_inf(L) is itself the
    N_dets->infinity limit, ESTIMATED by extrapolation with an uncertainty. So this
    routine (a) solves an INDEPENDENT geometric core ladder, (b) EN-PT2 at each rung,
    (c) SHCI/PT2 + power-law extrapolation to E_inf +/- sigma (via `report_energies`),
    and (d) decides, PER accuracy target eps, whether E_inf is pinned tightly enough
    to DEFINE a fixed-eps N* — a "point" — or only bounds it — a "bound". A reference
    with sigma comparable to eps cannot define an eps-accurate N*; saying so is the
    honest alternative to a false-precision exponent.

    PINNING RULE for a target eps (MeV/site): E_inf is a "point" iff it is either
      * VALIDATED vs exact: |extrap - exact|/site < eps (only where the sector is
        enumerable — the pipeline-vs-truth check), OR
      * the fit uncertainty is a fraction of the target AND the two independent
        extrapolators agree: sigma/site < `sigma_frac`*eps  AND
        |E_power - E_pt2|/site < `cross_frac`*eps.
    Otherwise it is a "bound": report E_inf as the best estimate but flag that a
    deeper ladder / HPC is needed to pin an eps-accurate N*.

    Returns a dict carrying the per-rung ladder (for Phase C's N* extraction),
    E_inf/sigma (total + per-site), both extrapolators, the exact ref + per-site gap
    when available, and the per-target pinning verdicts."""
    from .extrapolation import report_energies

    H = build_from_eft(L, dim, n_b, transform=transform, frame_params=frame_params,
                       N_f=N_f)
    H._cpp_cache_bytes = int(cache_bytes)
    sites = L ** dim
    sector = comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
    solver, pt2_diag, use_cpp = _pick_solver(arrays)
    if cores is not None:
        ladder_desc = str(list(cores))
    else:
        ladder_desc = (f"{ladder_start}x2^k, <={n_rungs} rungs"
                       + (f", max_core={max_core}" if max_core else "")
                       + (f", stop rung>{max_rung_seconds:g}s" if max_rung_seconds else ""))

    if verbose:
        print("=" * 74)
        print(f"  CONVERGED REFERENCE  L={L} d={dim} A={A} N_f={H.N_f}  {sites} sites, "
              f"{len(H.terms)} terms, sector {sector:.1e}  "
              f"({'C++' if use_cpp else 'python'} solver)")
        print(f"  independent ladder {ladder_desc}; targets "
              f"{list(eps_persite_targets)} MeV/site "
              f"(pin: sigma/site < {sigma_frac}*eps AND extrapolators agree < "
              f"{cross_frac}*eps)")
        print("=" * 74)

    E_ref = None
    if sector <= DEFAULT_MAX_STATES:
        E_ref, _ = lanczos_ground_state(H, n_elec=A)
        if verbose:
            print(f"  exact reference (Lanczos): {E_ref:.6f} MeV\n")

    if cores is not None:
        rungs = _solve_ladder(H, A, cores, solver, pt2_diag, n_runs=n_runs, seed=seed,
                              pt2=True, verbose=verbose)
    else:
        rungs = _adaptive_ladder_solve(H, A, ladder_start, n_rungs, solver, pt2_diag,
                                       max_core=max_core,
                                       max_rung_seconds=max_rung_seconds,
                                       n_runs=n_runs, seed=seed, verbose=verbose)
    if verbose:
        print()
    rep = report_energies(rungs, exact=E_ref, sites=sites,
                          label=f"L={L} d={dim} A={A} N_f={H.N_f} reference",
                          verbose=verbose)

    # reference E_inf and its per-site uncertainty; cross-check the two extrapolators
    E_inf = rep.get("E_extrap_best")
    sigma_ps = rep.get("E_extrap_best_sigma_per_site")
    power, pt2ex = rep["extrap_power"], rep["extrap_pt2"]
    cross_ps = None
    if power.get("E_inf") is not None and pt2ex.get("E_inf") is not None:
        cross_ps = abs(power["E_inf"] - pt2ex["E_inf"]) / sites
    exact_gap_ps = rep.get("extrap_minus_exact_per_site")

    targets = {}
    for eps in eps_persite_targets:
        validated = (exact_gap_ps is not None and abs(exact_gap_ps) < eps)
        sigma_ok = (sigma_ps is not None and sigma_ps < sigma_frac * eps)
        cross_ok = (cross_ps is not None and cross_ps < cross_frac * eps)
        pinned = bool(validated or (sigma_ok and cross_ok))
        bits = []
        if validated:
            bits.append(f"validated vs exact (|Δ|/site={abs(exact_gap_ps):.2g}<{eps:g})")
        if sigma_ps is not None:
            bits.append(f"σ/site={sigma_ps:.2g}{'<' if sigma_ok else '≥'}{sigma_frac:g}·eps")
        if cross_ps is not None:
            bits.append(f"extrap-gap/site={cross_ps:.2g}{'<' if cross_ok else '≥'}{cross_frac:g}·eps")
        targets[eps] = {"eps_persite": eps, "pinned": pinned,
                        "kind": "point" if pinned else "bound",
                        "reason": "; ".join(bits) if bits else "no extrapolation"}

    # Phase-D robustness diagnostics on the ladder (pure functions of the rungs)
    from .robustness import ladder_monotonicity, pt2_memory_report
    mono = ladder_monotonicity(rungs, energy_key="E_var")
    pt2mem = pt2_memory_report(rungs)

    out = {
        "L": L, "dim": dim, "A": A, "sites": sites, "N_f": H.N_f,
        "n_terms": len(H.terms), "sector": float(sector),
        "cores": [r["core"] for r in rungs], "rungs": rungs,
        "E_inf": E_inf, "sigma": rep.get("E_extrap_best_sigma"),
        "method": rep.get("E_extrap_best_method"),
        "E_inf_per_site": rep.get("E_extrap_best_per_site"),
        "sigma_per_site": sigma_ps,
        "extrap_power": power, "extrap_pt2": pt2ex,
        "cross_check_per_site": cross_ps,
        "exact": E_ref, "extrap_minus_exact_per_site": exact_gap_ps,
        "monotonicity": mono, "pt2_memory": pt2mem,
        "targets": targets, "report": rep,
    }
    if verbose:
        print("  target pinning (can E_inf DEFINE a fixed-eps N*?):")
        for eps, t in targets.items():
            print(f"    eps={eps:>5g} MeV/site -> {t['kind'].upper():5} "
                  f"[{'pinned' if t['pinned'] else 'NOT pinned'}]: {t['reason']}")
        print("  robustness: "
              + ("ladder monotone" if mono["monotone"]
                 else f"NON-MONOTONE — {len(mono['offenders'])} rise(s), "
                      f"max +{mono['max_rise']:.2f} MeV (bursty/under-converged rung)")
              + f"; {pt2mem['trigger']}")
        print("=" * 74)
    return out


def _extract_nstar(rungs, E_inf, sites, eps, energy_key="E_pt2"):
    """Smallest core N* whose energy is STABLY within a fixed per-site accuracy eps
    of the reference E_inf: |E(N) - E_inf|/sites < eps for that rung AND every larger
    rung. "Stably" (not merely the first crossing) is deliberate — selected-CI
    convergence is bursty and PT2 is non-variational, so a lone early rung can dip
    under eps and pop back out; requiring all larger rungs to hold guards against
    that fluke. Scans the ladder from the top for the last FAILING rung; N* brackets
    between it and the next (passing) rung, reported as their geometric mean.

    Returns {status, n_star, n_lo, n_hi, gap_at_max, note}:
      * "bracketed"   — N* lies in (n_lo, n_hi]; n_star = sqrt(n_lo*n_hi) (a fit point).
      * "upper_bound" — even the smallest core is within eps (N* <= n_hi = that core).
      * "lower_bound" — the largest core still fails (N* > n_lo = that core; not reached).
      * "no_reference"— E_inf is None (the ladder was too shallow to extrapolate a
        reference), so N* is undefined. Reported as a bound; NEVER crashes — an
        expensive large-L ladder that self-limits early must not lose the whole run.
    """
    if E_inf is None:
        return {"status": "no_reference", "n_star": None, "n_lo": None, "n_hi": None,
                "gap_at_max": None, "note": "no extrapolated E_inf (ladder too shallow)"}
    rs = sorted(rungs, key=lambda r: r["core"])
    cores = [r["core"] for r in rs]
    gaps = [abs(r[energy_key] - E_inf) / sites for r in rs]
    last_fail = -1
    for i, g in enumerate(gaps):
        if g >= eps:
            last_fail = i
    if last_fail == len(rs) - 1:
        return {"status": "lower_bound", "n_star": None, "n_lo": cores[-1],
                "n_hi": None, "gap_at_max": gaps[-1],
                "note": f"gap/site={gaps[-1]:.3g} >= eps={eps:g} at largest core {cores[-1]}"}
    if last_fail == -1:
        return {"status": "upper_bound", "n_star": cores[0], "n_lo": None,
                "n_hi": cores[0], "gap_at_max": gaps[-1],
                "note": f"smallest core {cores[0]} already within eps (N* <= {cores[0]})"}
    n_lo, n_hi = cores[last_fail], cores[last_fail + 1]
    return {"status": "bracketed", "n_star": int(round((n_lo * n_hi) ** 0.5)),
            "n_lo": n_lo, "n_hi": n_hi, "gap_at_max": gaps[-1],
            "note": "bracketed between failing/passing rungs"}


def _nstar_repr(res):
    """Representative core for a `_extract_nstar` result (for plotting/fitting):
    n_star when bracketed, the passing core for an upper bound, the failing core for
    a lower bound."""
    if res["status"] == "bracketed":
        return res["n_star"]
    if res["status"] == "upper_bound":
        return res["n_hi"]
    return res["n_lo"]


def _fit_line(x, y):
    """Least-squares line y = slope*x + intercept with R^2 and (>=4 points) a slope
    uncertainty. Returns {ok, slope, intercept, r2, sigma, rms_resid}."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return {"ok": False, "reason": f"need >= 2 points, got {len(x)}"}
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    sigma = None
    if len(x) >= 4:
        try:
            _, cov = np.polyfit(x, y, 1, cov=True)
            if np.all(np.isfinite(cov)):
                sigma = float(np.sqrt(cov[0, 0]))
        except Exception:
            sigma = None
    return {"ok": True, "slope": float(coef[0]), "intercept": float(coef[1]),
            "r2": r2, "sigma": sigma, "rms_resid": float((ss_res / len(x)) ** 0.5)}


def _fit_exponent(sites, nstar):
    """Fit N*(V) two ways over the volume V = #sites, to CONTRAST the two hypotheses
    for how classical cost scales with system size:
      * exponential in V: log N* = a + gamma*V      (N* ~ e^{gamma V}; the extensivity
        trap — selected-CI cost blows up exponentially in volume)
      * polynomial in V:  log N* = a + gamma*log V  (N* ~ V^gamma; benign scaling)
    gamma is the slope of each; the better-fitting model (higher R^2) says which
    regime the data are in. Stretched-exponential log N* = a + gamma*V^p (Lee/Chan)
    needs a wider L-range than the laptop L=2-4 to fit p robustly, so it is deferred
    to the HPC push. Returns {n_points, exponential_in_V, polynomial_in_V}."""
    V = [float(s) for s in sites]
    logN = list(np.log(np.asarray(nstar, float)))
    return {"n_points": len(V),
            "exponential_in_V": _fit_line(V, logN),
            "polynomial_in_V": _fit_line([float(np.log(v)) for v in V], logN)}


def dets_vs_L_at_fixed_accuracy(dim=3, L_values=(2, 3, 4), A=1, n_b=2, N_f=None,
                                eps_persite_targets=(1.0, 0.1), filling=None,
                                ladder_start=500, n_rungs=6, max_core=8000,
                                max_rung_seconds=120, sigma_frac=0.3, cross_frac=1.0,
                                n_runs=4, seed=0, transform="bare",
                                cache_bytes=128 << 20, arrays=True, out_dir=None,
                                label=None, hpc=False, verbose=True):
    """PHASE C — the headline dets-vs-L measurement.

    For each L (and each per-site accuracy target eps), find N*(L; eps) = the smallest
    core that reaches eps of the L-converged ground energy E_inf(L), then fit N* vs
    system volume to read the scaling exponent gamma — the paper's central number
    ("does the classical selected-CI cost grow polynomially or exponentially in
    volume?"). The whole chain rests on the Phase-B reference, so it is done HONESTLY:

      1. `converged_reference(L)` builds E_inf(L) +/- sigma and, per eps, decides
         whether E_inf is PINNED tightly enough to define an eps-accurate N*.
      2. `_extract_nstar` brackets N* on the same ladder (using E_var+dE_PT2, the
         production energy), and the reference sigma is propagated by re-extracting at
         E_inf +/- sigma (a horizontal error bar on N*).
      3. A point is FIT-WORTHY only when the reference is pinned AND N* is bracketed
         (not itself a bound). Non-fit-worthy L's are reported as bounds and LOGGED —
         never silently dropped — so an under-resolved laptop run reads as "bounds,
         HPC needed", not a fake exponent.

    Laptop guards: a geometric ladder from `ladder_start` (x2/rung, <= `n_rungs`),
    capped at `max_core`, that stops growing once a rung exceeds `max_rung_seconds`
    (so large L self-limit). `filling=None` => DILUTE (A fixed as given); a float =>
    FIXED FILLING A = round(filling * sites). Writes a JSON summary + the N*-vs-L plot.
    """
    import json
    import platform
    from classical.plotting import plot_dets_vs_L

    if label is None:
        fill_tag = "dilute" if filling is None else f"fill{filling:g}"
        label = f"detsvsL_{dim}d_{fill_tag}"
    if verbose:
        print("#" * 78)
        print(f"  PHASE C — dets vs L at fixed per-site accuracy   dim={dim} "
              f"{'dilute A=%d' % A if filling is None else 'filling=%g' % filling}")
        print(f"  L={list(L_values)}, eps={list(eps_persite_targets)} MeV/site, "
              f"ladder {ladder_start}x2 (<= {n_rungs} rungs, max_core {max_core}, "
              f"stop rung>{max_rung_seconds}s)")
        print("#" * 78)

    per_L = []
    for L in L_values:
        sites = L ** dim
        A_L = A if filling is None else max(1, int(round(filling * sites)))
        ref = converged_reference(
            L, dim=dim, A=A_L, n_b=n_b, N_f=N_f, ladder_start=ladder_start,
            n_rungs=n_rungs, max_core=max_core, max_rung_seconds=max_rung_seconds,
            eps_persite_targets=eps_persite_targets, sigma_frac=sigma_frac,
            cross_frac=cross_frac, n_runs=n_runs, seed=seed, transform=transform,
            cache_bytes=cache_bytes, arrays=arrays, verbose=verbose)
        E_inf, sig = ref["E_inf"], (ref["sigma"] or 0.0)
        eps_out = {}
        for eps in eps_persite_targets:
            pinned = ref["targets"][eps]["pinned"]
            base = _extract_nstar(ref["rungs"], E_inf, sites, eps, "E_pt2")
            base_var = _extract_nstar(ref["rungs"], E_inf, sites, eps, "E_var")
            # propagate the reference sigma into a horizontal bracket on N* — but only
            # when E_inf exists; a bound reference (E_inf is None) has no E_inf +/- sigma
            # to shift, so don't do None arithmetic (it would crash the whole sweep).
            if E_inf is None:
                lo = hi = base
            else:
                lo = _extract_nstar(ref["rungs"], E_inf - sig, sites, eps, "E_pt2")
                hi = _extract_nstar(ref["rungs"], E_inf + sig, sites, eps, "E_pt2")
            reprs = [r for r in (_nstar_repr(base), _nstar_repr(lo), _nstar_repr(hi))
                     if r is not None]
            eps_out[str(eps)] = {
                "eps_persite": eps, "reference_pinned": bool(pinned),
                "nstar_pt2": base, "nstar_var": base_var,
                "nstar_repr": _nstar_repr(base),
                "nstar_sigma_bracket": [min(reprs), max(reprs)] if reprs else None,
                "fit_worthy": bool(pinned and base["status"] == "bracketed"),
            }
        per_L.append({
            "L": L, "sites": sites, "A": A_L, "N_f": ref["N_f"],
            "n_terms": ref["n_terms"], "sector": ref["sector"],
            "E_inf": E_inf, "sigma": ref["sigma"],
            "E_inf_per_site": ref["E_inf_per_site"],
            "sigma_per_site": ref["sigma_per_site"], "method": ref["method"],
            "exact": ref["exact"],
            "monotonicity": ref["monotonicity"], "pt2_memory": ref["pt2_memory"],
            "rungs": [{k: r[k] for k in ("core", "E_var", "dE_pt2", "E_pt2", "n_ext")
                       if k in r} for r in ref["rungs"]],
            "eps": eps_out,
        })

    # fit the exponent per eps over the FIT-WORTHY points only; log everything else
    fits = {}
    for eps in eps_persite_targets:
        pts = [(p["sites"], p["eps"][str(eps)]["nstar_repr"]) for p in per_L
               if p["eps"][str(eps)]["fit_worthy"]]
        dropped = [(p["L"], p["eps"][str(eps)]["nstar_pt2"]["status"],
                    "reference not pinned" if not p["eps"][str(eps)]["reference_pinned"]
                    else p["eps"][str(eps)]["nstar_pt2"]["note"])
                   for p in per_L if not p["eps"][str(eps)]["fit_worthy"]]
        if len(pts) >= 2:
            fit = _fit_exponent([s for s, _ in pts], [n for _, n in pts])
            fit["ok"] = True
            fit["points"] = [{"sites": s, "nstar": n} for s, n in pts]
        else:
            fit = {"ok": False, "n_points": len(pts),
                   "reason": f"only {len(pts)} fit-worthy point(s) — need >= 2; "
                             f"deeper ladders / HPC required",
                   "points": [{"sites": s, "nstar": n} for s, n in pts]}
        fit["dropped"] = [{"L": L, "status": st, "why": why} for L, st, why in dropped]
        fits[str(eps)] = fit

    # Phase-D extensivity signal: at the DEEPEST core reached by EVERY L, the per-site
    # EN-PT2 correction — a size-intensive proxy for how incomplete that fixed budget
    # is — vs L. Growing with volume is the extensivity trap made visible, and is
    # extractable even when N*(eps) itself is only bounded.
    extensivity = None
    if per_L:
        common = set.intersection(*[{r["core"] for r in p["rungs"]} for p in per_L])
        if common:
            c = max(common)
            extensivity = {"fixed_core": c, "rows": [
                {"L": p["L"], "sites": p["sites"],
                 "dE_pt2_per_site": next(r["dE_pt2"] for r in p["rungs"]
                                         if r["core"] == c) / p["sites"]}
                for p in per_L]}
    # top-level robustness roll-up (Phase D)
    robustness = {
        "any_non_monotone": any(not p["monotonicity"]["monotone"] for p in per_L),
        "any_pt2_over_budget": any(p["pt2_memory"]["over_budget"] for p in per_L),
        "max_n_ext": max((p["pt2_memory"]["max_n_ext"] or 0 for p in per_L), default=0),
        "extensivity_signal": extensivity,
    }

    result = {
        "kind": "dets_vs_L", "dim": dim, "A": A, "filling": filling,
        "L_values": list(L_values), "eps_persite_targets": list(eps_persite_targets),
        "label": label, "per_L": per_L, "fits": fits, "robustness": robustness,
        # provenance: where this ran. `hpc` distinguishes cluster runs (deep cores /
        # big ensembles) from laptop smoke runs when the JSONs are pooled later.
        "hpc": bool(hpc), "host": platform.node(),
    }

    if out_dir is None:
        out_dir = os.path.join("data", "classical")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{label}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    plot_path = os.path.join(out_dir, f"{label}.png")
    try:
        plot_dets_vs_L(result, out_path=plot_path)
    except Exception as e:                    # plotting must never lose the data
        print(f"[plot] skipped ({e})")

    if verbose:
        print("#" * 78)
        print(f"  PHASE C SUMMARY  ({label})")
        for eps in eps_persite_targets:
            f = fits[str(eps)]
            print(f"  -- eps = {eps:g} MeV/site --")
            for p in per_L:
                e = p["eps"][str(eps)]
                st = e["nstar_pt2"]["status"]
                tag = ("POINT N*=%d" % e["nstar_repr"] if e["fit_worthy"]
                       else f"{st.upper()} (repr {e['nstar_repr']})")
                pin = "pinned" if e["reference_pinned"] else "NOT pinned"
                print(f"     L={p['L']:>2} ({p['sites']:>3} sites): {tag:<26} "
                      f"[ref {pin}]")
            if f["ok"]:
                ex, po = f["exponential_in_V"], f["polynomial_in_V"]
                better = ("EXPONENTIAL" if (ex["r2"] or -9) > (po["r2"] or -9)
                          else "POLYNOMIAL")
                print(f"     fit ({f['n_points']} pts): "
                      f"exp gamma={ex['slope']:.4g}/site (R^2={ex['r2']:.3f}); "
                      f"poly gamma={po['slope']:.3g} (R^2={po['r2']:.3f}) "
                      f"-> {better} fits better")
                print(f"     NOTE: {f['n_points']} points is PRELIMINARY — the "
                      f"exponent needs the HPC L-range for a trustworthy gamma.")
            else:
                print(f"     fit: {f['reason']}")
            if f["dropped"]:
                for d in f["dropped"]:
                    print(f"       dropped L={d['L']}: {d['status']} ({d['why']})")
        # Phase-D robustness roll-up
        print("  -- robustness (Phase D) --")
        for p in per_L:
            m, pm = p["monotonicity"], p["pt2_memory"]
            mtag = ("monotone" if m["monotone"]
                    else f"NON-MONOTONE (max +{m['max_rise']:.2f} MeV)")
            print(f"     L={p['L']:>2}: ladder {mtag}; PT2 max n_ext="
                  f"{pm['max_n_ext']:,}"
                  + ("  [SEMISTOCHASTIC TRIGGER]" if pm["over_budget"] else ""))
        if extensivity:
            cells = ", ".join(f"L={r['L']}:{r['dE_pt2_per_site']:.2f}"
                              for r in extensivity["rows"])
            print(f"     extensivity signal (dE_PT2/site @ fixed core "
                  f"{extensivity['fixed_core']}): {cells} MeV/site "
                  f"— growing = the trap, made visible")
        print(f"  wrote {json_path}")
        print("#" * 78)
    return result


def seed_robustness(L=2, dim=3, A=1, n_b=2, N_f=None, core=2000,
                    seeds=(0, 1000, 2000, 3000), n_runs=4, eps_persite=1.0,
                    transform="bare", frame_params=None, cache_bytes=128 << 20,
                    arrays=True, verbose=True):
    """PHASE D — is the (load-bearing, no-warm-start) random-init ensemble robust?

    The solver escapes local basins by taking the best of `n_runs` RANDOM inits; a
    base `seed` selects an independent stream (run k uses seed+k). This re-solves the
    SAME (L, core) with several DIFFERENT base seeds — each its own n_runs ensemble —
    and measures the scatter in E_var and E_var+PT2. Small per-site scatter (relative
    to the accuracy target eps) means the energy at that core is NOT an artifact of
    one random stream; scatter comparable to eps means the point is seed-fragile and
    needs more ensemble runs before it can anchor an N* or a reference. No warm starts
    (each seed is independent) — the ensemble randomness is deliberate and preserved.

    Returns {per_seed, E_var_scatter, E_pt2_scatter (total + per-site), robust, eps}.
    """
    from .robustness import scatter_stats
    H = build_from_eft(L, dim, n_b, transform=transform, frame_params=frame_params,
                       N_f=N_f)
    H._cpp_cache_bytes = int(cache_bytes)
    sites = L ** dim
    solver, pt2_diag, use_cpp = _pick_solver(arrays)
    mr = max(12, int(np.ceil(np.log(max(core, 2) / 20.0) / np.log(1.5))) + 4)
    if verbose:
        print("=" * 74)
        print(f"  SEED ROBUSTNESS  L={L} d={dim} A={A} N_f={H.N_f}  core={core}, "
              f"n_runs={n_runs}, seeds={list(seeds)}  ({'C++' if use_cpp else 'python'})")
        print("=" * 74)

    from .pt2 import pt2_from_result
    per_seed = []
    for s in seeds:
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=core, seed=int(s),
                     max_rounds=mr)
        pr = pt2_from_result(H, res, diag_fn=pt2_diag)
        row = {"seed": int(s), "core": int(res.n_dets),
               "E_var": float(pr["E_var"]),
               "E_pt2": float(pr["E_var"] + pr["dE_pt2"])}
        per_seed.append(row)
        if verbose:
            print(f"  seed={s:>5}: E_var={row['E_var']:12.5f}  "
                  f"E_var+PT2={row['E_pt2']:12.5f} MeV  (core {row['core']})")

    var_sc = scatter_stats([r["E_var"] for r in per_seed])
    pt2_sc = scatter_stats([r["E_pt2"] for r in per_seed])
    pt2_ptp_ps = (pt2_sc["ptp"] / sites) if pt2_sc["ptp"] is not None else None
    robust = (pt2_ptp_ps is not None and pt2_ptp_ps < eps_persite)
    out = {"L": L, "dim": dim, "A": A, "sites": sites, "core": core,
           "seeds": list(seeds), "n_runs": n_runs, "per_seed": per_seed,
           "E_var_scatter": var_sc, "E_pt2_scatter": pt2_sc,
           "E_pt2_ptp_per_site": pt2_ptp_ps, "eps_persite": eps_persite,
           "robust": bool(robust)}
    if verbose:
        print("  " + "-" * 70)
        print(f"  E_var    spread: std={var_sc['std']:.4f}  ptp={var_sc['ptp']:.4f} MeV")
        print(f"  E_var+PT2 spread: std={pt2_sc['std']:.4f}  ptp={pt2_sc['ptp']:.4f} MeV "
              f"= {pt2_ptp_ps:.4f} MeV/site")
        print(f"  >> {'ROBUST' if robust else 'SEED-FRAGILE'}: per-site E+PT2 spread "
              f"{pt2_ptp_ps:.4f} {'<' if robust else '>='} eps={eps_persite:g} MeV/site"
              + ("" if robust else "  — add ensemble runs before trusting this core"))
        print("=" * 74)
    return out


def nb_convergence_sweep(L=2, dim=3, A=1, N_f_list=(2, 4, 8, 16, 32),
                         core=2000, n_runs=3, seed=0, pt2=True, cache_bytes=128 << 20,
                         arrays=True, boson_init_mean=0.5, verbose=True):
    """Fock-cutoff (n_b / N_f) convergence study at fixed L.

    For each cutoff N_f in `N_f_list` (powers of two -> the n_b = log2(N_f) axis the
    quantum side controls; non-powers-of-two -> finer N_f resolution), solve the
    A-nucleon ground state at a FIXED core size and record:
      * E_var (self-consistent, re-diagonalized) and E_var + Epstein-Nesbet PT2,
      * runtime (the "does TrimCI slow down as n_b grows?" curve),
      * <N>/mode and the leaked-weight tail (the near-vacuum physics),
      * exact Lanczos energy when the sector is enumerable.
    A FIXED core across N_f makes the runtime comparison clean; PT2 removes most of
    the residual core-incompleteness so E(N_f) reflects the CUTOFF, not the core.

    The truncation error |E(N_f) - E(N_f_max)| vs N_f is the empirical curve that
    certifies the n_b bound (compare against `tong_bound.cutoff_predictions`). Returns
    the per-N_f rows; the caller (misc/run_nb_convergence.py) plots + does the Tong
    comparison. Non-variational-in-N_f wobble (odd N_f can dip low — see TODO.md
    2026-07-07) is why we report both E_var and E_var+PT2 and key the axis on n_b.
    """
    from math import log2, ceil, comb as _comb
    from .observables import mean_occupation, occupation_tail
    from .pt2 import pt2_from_result
    from .tong_bound import cutoff_predictions

    if cpp_available():
        from .backend import cpp_diagonalize_smart
        solver = _solver(arrays)
        pt2_diag = cpp_diagonalize_smart
    else:
        from .graph import ground_state_ensemble as solver
        pt2_diag = None

    pred = cutoff_predictions(L, dim, A)
    init_tag = "uniform(unbiased)" if boson_init_mean is None else f"near-vacuum(mean={boson_init_mean})"
    if verbose:
        print("=" * 84)
        print(f"  N_b / N_f CONVERGENCE  L={L} d={dim} A={A}  core={core} n_runs={n_runs}"
              f"  init={init_tag}")
        print(f"  Tong/SCS: <N>/mode={pred['N_per_mode']:.4f}  N_eng={pred['N_eng']}"
              f"(n_b={pred['n_b_eng']})  N_spec2={pred['N_spec2']}(n_b={pred['n_b_spec2']})"
              f"  N_spec1={pred['N_spec1']}(n_b={pred['n_b_spec1']})")
        print("=" * 84)
        hdr = (f"  {'n_b':>3} {'N_f':>4} {'sector':>9} {'E_var':>13} "
               f"{'E_var+PT2':>13} {'dE_PT2':>9} {'<N>/mode':>9} {'tailN_f':>9} {'t(s)':>7}")
        if verbose:
            print(hdr + f" {'exact':>13}")
            print("  " + "-" * 96)

    rows = []
    for N_f in N_f_list:
        n_b = max(1, ceil(log2(N_f)))
        H = build_from_eft(L, dim, n_b, N_f=N_f)
        H._cpp_cache_bytes = int(cache_bytes)
        sector = _comb(H.n_ferm_modes, A) * (H.N_f ** H.n_bos_modes)
        # exact reference only if the full sector is enumerable
        E_ref = None
        if sector <= DEFAULT_MAX_STATES:
            E_ref, _ = lanczos_ground_state(H, n_elec=A)
        mr = max(12, int(np.ceil(np.log(max(core, 2) / 20.0) / np.log(1.5))) + 4)
        t = time.time()
        res = solver(H, n_elec=A, n_runs=n_runs, n_dets=core, seed=seed,
                     max_rounds=mr, boson_init_mean=boson_init_mean)
        dt = time.time() - t
        E_var = float(res.energy)
        dE_pt2 = None
        if pt2:
            pr = pt2_from_result(H, res, diag_fn=pt2_diag)
            E_var = float(pr["E_var"])            # self-consistent (re-diagonalized)
            dE_pt2 = float(pr["dE_pt2"])
        mo = mean_occupation(res)
        # leaked weight the NEXT-smaller box would drop, measured on this solve
        tail = occupation_tail(res, N_f)
        row = {"n_b": n_b, "N_f": N_f, "sector": sector, "core": int(res.n_dets),
               "E_var": E_var, "dE_pt2": dE_pt2,
               "E_pt2": (E_var + dE_pt2) if dE_pt2 is not None else E_var,
               "N_per_mode": mo["N_per_mode"], "N_max_mode": mo["N_max_mode"],
               "tail_Nf": tail, "runtime_s": dt, "exact": E_ref}
        rows.append(row)
        if verbose:
            es = f"{E_ref:13.5f}" if E_ref is not None else f"{'—':>13}"
            dps = f"{dE_pt2:+9.3f}" if dE_pt2 is not None else f"{'—':>9}"
            print(f"  {n_b:>3} {N_f:>4} {sector:>9.1e} {E_var:>13.5f} "
                  f"{row['E_pt2']:>13.5f} {dps} {mo['N_per_mode']:>9.4f} "
                  f"{tail:>9.1e} {dt:>7.1f} {es}")

    # truncation error vs the largest-cutoff energy (best-converged reference)
    ref_E = rows[-1]["E_pt2"]
    for r in rows:
        r["trunc_err"] = r["E_pt2"] - ref_E
    if verbose:
        print("  " + "-" * 96)
        print(f"  truncation error |E_var+PT2(N_f) - E(N_f={rows[-1]['N_f']})| vs n_b:")
        for r in rows:
            print(f"    n_b={r['n_b']} (N_f={r['N_f']:>2}): "
                  f"{abs(r['trunc_err']):>10.4f} MeV")
        # smallest n_b meeting a few accuracy targets
        for eps_mev in (13.5, 1.0, 0.1):
            hit = next((r["n_b"] for r in rows if abs(r["trunc_err"]) < eps_mev), None)
            print(f"    |trunc err| < {eps_mev:>5} MeV first at n_b={hit}")
        print("=" * 84)

    return {"rows": rows, "predictions": pred, "L": L, "dim": dim, "A": A,
            "core": core, "n_runs": n_runs}


def occupation_vs_A_sweep(L=2, dim=3, A_list=(1, 2, 3, 4), n_b=4, N_f=None,
                          cores=(2000, 4000, 8000, 16000), n_runs=2, seed=0,
                          pt2=False, cache_bytes=128 << 20, arrays=True,
                          boson_init_mean=0.5, verbose=True):
    """GS pion occupation <N> vs nucleon number A, at fixed L/dim/n_b.

    Tests how the pion cloud responds to the nucleon source. The SCS mean field
    (`tong_bound.mean_occupation_scs`) predicts <N> = N_sq (A-independent vacuum
    squeeze) + N_disp*A^2 (AV displacement, ~1e-8*A^2 — negligible), i.e. NEARLY
    FLAT in A. This measures the empirical <N>(A) to test that and to key the
    cutoff-robustness argument: if <N> stays ~vacuum as A grows, the n_b bound is
    A-robust.

    CONVERGENCE CAVEAT. Larger A has a comb(n_ferm, A) larger fermion sector, so a
    FIXED core UNDER-resolves high A — the near-vacuum seed approaches <N> from
    BELOW, so a finite-core <N> is a LOWER BOUND (the bigger the sector, the more
    it is suppressed). That is why this sweep runs a per-A `cores` LADDER: read off
    whether <N> has plateaued (converged) or is still climbing (lower bound). Small
    A (sector <= a few 1e4) converges on a laptop; A >= ~6 needs HPC-scale cores.

    Returns {rows, ...} where each row is the LARGEST-core solve for that A, plus a
    `ladder` list (core, <N>, E_var) and a `converged` flag (last <N> step small)."""
    from math import log2, ceil, comb as _comb
    from .observables import mean_occupation, occupation_tail
    from .pt2 import pt2_from_result
    from .tong_bound import mean_occupation_scs

    if N_f is None:
        N_f = 2 ** n_b
    n_b = max(1, ceil(log2(N_f)))

    if cpp_available():
        from .backend import cpp_diagonalize_smart
        solver = _solver(arrays)
        pt2_diag = cpp_diagonalize_smart
    else:
        from .graph import ground_state_ensemble as solver
        pt2_diag = None

    if verbose:
        print("=" * 92)
        init_tag = "uniform" if boson_init_mean is None else f"near-vac({boson_init_mean})"
        print(f"  OCCUPATION vs A  L={L} d={dim} N_f={N_f} (n_b={n_b})  "
              f"cores={cores} n_runs={n_runs} init={init_tag}")
        print(f"  {'A':>3} {'fill':>6} {'core':>6} {'<N>/mode':>9} {'N_tot':>7} "
              f"{'N_max':>7} {'E_var':>11} {'tail':>8} {'SCS<N>':>8} {'conv?':>6}")
        print("  " + "-" * 86)

    rows = []
    for A in A_list:
        scs = mean_occupation_scs(L, dim, A)["N_per_mode"]
        ladder = []
        last_res = None
        for core in cores:
            H = build_from_eft(L, dim, n_b, N_f=N_f)
            H._cpp_cache_bytes = int(cache_bytes)
            mr = max(12, int(np.ceil(np.log(max(core, 2) / 20.0) / np.log(1.5))) + 4)
            t = time.time()
            res = solver(H, n_elec=A, n_runs=n_runs, n_dets=core, seed=seed,
                         max_rounds=mr, boson_init_mean=boson_init_mean)
            dt = time.time() - t
            mo = mean_occupation(res)
            ladder.append({"core": int(res.n_dets), "N_per_mode": mo["N_per_mode"],
                           "E_var": float(res.energy), "runtime_s": dt})
            last_res = res
        H = build_from_eft(L, dim, n_b, N_f=N_f)          # for fill/mode counts
        mo = mean_occupation(last_res)
        tail = occupation_tail(last_res, N_f)
        E_var = float(last_res.energy)
        dE_pt2 = None
        if pt2:
            pr = pt2_from_result(H, last_res, diag_fn=pt2_diag)
            E_var = float(pr["E_var"])
            dE_pt2 = float(pr["dE_pt2"])
        # converged if the last core-doubling moved <N> by < 3% of its value
        if len(ladder) >= 2 and ladder[-1]["N_per_mode"] > 0:
            step = abs(ladder[-1]["N_per_mode"] - ladder[-2]["N_per_mode"])
            converged = step < 0.03 * ladder[-1]["N_per_mode"]
        else:
            converged = None
        fill = A / H.n_ferm_modes
        row = {"A": A, "fill": fill, "n_ferm_modes": H.n_ferm_modes,
               "core": ladder[-1]["core"], "N_per_mode": mo["N_per_mode"],
               "N_total": mo["N_total"], "N_max_mode": mo["N_max_mode"],
               "E_var": E_var, "dE_pt2": dE_pt2, "tail_Nf": tail,
               "scs_N_per_mode": scs, "converged": converged, "ladder": ladder}
        rows.append(row)
        if verbose:
            cflag = ("yes" if converged else "LOWER") if converged is not None else "-"
            print(f"  {A:>3} {fill:>6.3f} {ladder[-1]['core']:>6} "
                  f"{mo['N_per_mode']:>9.4f} {mo['N_total']:>7.3f} "
                  f"{mo['N_max_mode']:>7.3f} {E_var:>11.2f} {tail:>8.1e} "
                  f"{scs:>8.4f} {cflag:>6}")

    if verbose:
        print("  " + "-" * 86)
        print("  (conv=LOWER: <N> still climbing at largest core -> finite-core "
              "LOWER BOUND on the true <N>; the true value is higher.)")
        print("=" * 92)

    return {"rows": rows, "L": L, "dim": dim, "N_f": N_f, "n_b": n_b,
            "cores": list(cores), "n_runs": n_runs}


def exact_occupation_vs_A(L=2, dim=1, A_list=(1, 2, 3, 4, 5, 6, 7, 8), N_f=4,
                          max_states=500000, verbose=True):
    """EXACT (full-ED) GS pion occupation <N> vs A, at an ED-reachable size.

    The clean truth for the occupation-vs-A question: full Lanczos over the whole
    A-sector, so <N> carries NO core-convergence caveat (unlike the selected-CI
    `occupation_vs_A_sweep`, whose finite-core <N> is only a lower bound). At
    L=2 d=1 the sector fits for all A and N_f<=6, letting us read the exact <N>(A)
    law. The SCS mean field predicts <N> = N_sq (A-independent squeeze) + N_disp*A^2
    (~1e-8*A^2, negligible), i.e. FLAT; this measures whether the exact GS agrees.

    Skips any A whose sector exceeds `max_states`. Returns per-A rows with the
    exact <N>/mode, max-mode occupation, E0, filling, and the SCS prediction."""
    from math import comb as _comb, ceil, log2
    from .observables import occupation_from_coeffs
    from .tong_bound import mean_occupation_scs, squeeze_r_star

    n_b = max(1, ceil(log2(N_f)))

    if verbose:
        print("=" * 84)
        print(f"  EXACT OCCUPATION vs A  L={L} d={dim} N_f={N_f}  "
              f"(r*={squeeze_r_star(L, dim):.4f})")
        print(f"  {'A':>3} {'fill':>6} {'<N>/mode':>9} {'N_max':>8} {'N_tot':>8} "
              f"{'E0':>11} {'SCS<N>':>9} {'states':>9}")
        print("  " + "-" * 74)

    rows = []
    for A in A_list:
        H = build_from_eft(L, dim, n_b, N_f=N_f)
        size = _comb(H.n_ferm_modes, A) * (N_f ** H.n_bos_modes)
        if size > max_states:
            if verbose:
                print(f"  {A:>3}  skip (sector {size:.1e} > max_states {max_states:.0e})")
            continue
        E0, cmap, info = lanczos_ground_state(H, n_elec=A, return_vec=True,
                                              max_states=max_states)
        mo = occupation_from_coeffs(cmap)
        scs = mean_occupation_scs(L, dim, A)["N_per_mode"]
        fill = A / H.n_ferm_modes
        row = {"A": A, "fill": fill, "N_per_mode": mo["N_per_mode"],
               "N_max_mode": mo["N_max_mode"], "N_total": mo["N_total"],
               "E0": float(E0), "scs_N_per_mode": scs,
               "n_states": info["n_states"]}
        rows.append(row)
        if verbose:
            print(f"  {A:>3} {fill:>6.3f} {mo['N_per_mode']:>9.5f} "
                  f"{mo['N_max_mode']:>8.4f} {mo['N_total']:>8.4f} {E0:>11.3f} "
                  f"{scs:>9.5f} {info['n_states']:>9}")

    if verbose and rows:
        span = max(r["N_per_mode"] for r in rows) - min(r["N_per_mode"] for r in rows)
        print("  " + "-" * 74)
        print(f"  <N>/mode spread over A={[r['A'] for r in rows]}: {span:.2e} "
              f"(flat -> occupation is A-INDEPENDENT, set by vacuum squeezing).")
        print("=" * 84)
    return {"rows": rows, "L": L, "dim": dim, "N_f": N_f}


def main():
    ap = argparse.ArgumentParser(description="Larger-L TrimCI (full C++ path)")
    ap.add_argument("--mode",
                    choices=("run", "report", "reference", "seed_robustness"),
                    default="run",
                    help="run: det-count sweep; report: honest 3-number energy report; "
                         "reference: per-L converged E_inf(L)+/-sigma with per-target "
                         "pinning; seed_robustness: ensemble-seed scatter at a fixed core")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--n_b", type=int, default=2)
    ap.add_argument("--N_f", type=int, default=None,
                    help="override the per-mode Fock cutoff (report/reference modes)")
    ap.add_argument("--n_dets", type=int, default=2000)
    ap.add_argument("--n_runs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder-start", type=int, default=250,
                    help="reference mode: first rung of the geometric core ladder")
    ap.add_argument("--n-rungs", type=int, default=6,
                    help="reference mode: number of geometric ladder rungs (x2 each)")
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.1],
                    help="reference mode: per-site accuracy targets (MeV/site)")
    ap.add_argument("--save", action="store_true",
                    help="save the run (metadata + Hamiltonian dump + ground state) to data/classical/")
    ap.add_argument("--no-arrays", dest="arrays", action="store_false",
                    help="use the object (MixedState) path instead of the Tier-2 "
                         "array-native path (default: array-native)")
    ap.set_defaults(arrays=True)
    args = ap.parse_args()
    if args.mode == "reference":
        converged_reference(L=args.L, dim=args.dim, A=args.A, n_b=args.n_b,
                            N_f=args.N_f, ladder_start=args.ladder_start,
                            n_rungs=args.n_rungs, eps_persite_targets=tuple(args.eps),
                            n_runs=args.n_runs, seed=args.seed, arrays=args.arrays)
    elif args.mode == "seed_robustness":
        seed_robustness(L=args.L, dim=args.dim, A=args.A, n_b=args.n_b, N_f=args.N_f,
                        core=args.n_dets, n_runs=args.n_runs,
                        eps_persite=(args.eps[0] if args.eps else 1.0),
                        arrays=args.arrays)
    elif args.mode == "report":
        solve_and_report(L=args.L, dim=args.dim, A=args.A, n_b=args.n_b, N_f=args.N_f,
                         n_runs=args.n_runs, seed=args.seed, arrays=args.arrays)
    else:
        run(L=args.L, dim=args.dim, A=args.A, n_b=args.n_b, n_dets=args.n_dets,
            n_runs=args.n_runs, seed=args.seed, save=args.save, arrays=args.arrays)


if __name__ == "__main__":
    main()
