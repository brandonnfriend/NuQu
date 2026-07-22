"""
Frame-optimization WORKFLOW aligned with TrimCI's recommended three-phase method.

Source of truth: `py/trimci/TrimCI_skill.py` in the TrimCI repo (README "🤖 For AI agents");
Zhang & Otten 2025/2026 (arXiv:2511.14734, 2605.22977). TrimCI's "frame" is the FERMION
orbital rotation (COO); NuQu adds BOSON Gaussian frames (squeeze/Bogoliubov/LF). The workflow
principle is shared:

  Phase 0 — FIND the frame cheaply.  Multi-run STOCHASTIC sampling on a SMALL det space
            (~100–10k), keeping the best; for COO, multi-CYCLE orbital optimization. Dets are
            transient (thrown away); the FRAME is what's kept.
  Phase 1 — GROW the det space while CO-EVOLVING the frame (orbitals) — COO only.
  Phase 2 — FREEZE the frame, grow dets to convergence (+PT2 upstream).

Why this module exists: our earlier frame comparison used EXPENSIVE fixed-large-core solves
(variance-limited at high A) and a ONE-SHOT COO (unreliable). This aligns both — cheap Phase-0
probes for selection, an iterative COO loop, and a 3-phase production runner. `frame.py` keeps
the pure Hamiltonian transforms; this module DRIVES the solver.
"""
import numpy as np

from . import frame
from .run_cpp import _solver


def _per_run_energies(res):
    """Per-run energies from an ensemble result's history (best-of-N came from these)."""
    for tag, payload in reversed(res.history or []):
        if tag == "ensemble":
            return [float(e) for (_s, e, _n) in payload]
    return [float(res.energy)]


def probe_frame(H_frame, n_elec, n_probe=100, num_runs=32, seed=0, solve=None):
    """Phase-0 STOCHASTIC probe of a candidate frame (the cheap frame-quality metric).
    Run `num_runs` independent random-init TrimCI solves at a SMALL core (`n_probe` dets),
    keep the BEST energy, discard the dets. A more compact frame reaches a LOWER best
    energy at the same tiny budget; best-of-N beats down the single-run ensemble variance
    that made fixed-large-core comparisons unreliable at high A. This mirrors TrimCI
    Phase-0's "multi-run stochastic sampling, keep the best run" (TrimCI_skill.py
    `num_runs`, `max_final_dets`). Returns `{best, mean, std, spread, n_probe, num_runs}`."""
    if solve is None:
        solve = _solver(True)
    res = solve(H_frame, n_elec=n_elec, n_dets=n_probe, seed=seed, n_runs=num_runs)
    es = np.asarray(_per_run_energies(res), float)
    return {"best": float(es.min()), "mean": float(es.mean()), "std": float(es.std()),
            "spread": float(es.max() - es.min()), "n_probe": int(res.n_dets),
            "num_runs": int(num_runs)}


def select_frame(candidates, n_elec, n_probe=100, num_runs=32, seed=0, verbose=False):
    """Rank candidate frames by their Phase-0 probe (lowest best-energy wins). This is the
    RECOMMENDED way to SELECT/compare frames — cheap (small core) and robust (best-of-N) —
    replacing an expensive fixed-large-core solve. `candidates` = {name: MixedH}. Returns
    `(ranked, probes)`: ranked = list of (name, probe) sorted by best energy ascending."""
    solve = _solver(True)
    probes = {}
    for name, H in candidates.items():
        p = probe_frame(H, n_elec, n_probe, num_runs, seed, solve)
        probes[name] = p
        if verbose:
            print(f"    {name:14} best={p['best']:.4f}  mean={p['mean']:.4f}  "
                  f"spread={p['spread']:.4f}  (n={p['n_probe']}x{p['num_runs']} runs)", flush=True)
    ranked = sorted(probes.items(), key=lambda kv: kv[1]["best"])
    return ranked, probes


def coo_orbopt(H, n_elec, core=500, num_runs=32, cycles=10, seed=0, margin=0.0,
               conv_tol=1e-6, verbose=False, solve=None):
    """Iterative Core-Optimized-Orbitals loop = TrimCI Phase-0 COO (natural-orbital proxy).
    Each cycle: solve the CURRENT-basis H at a small core (best-of-`num_runs`) → read the
    fermion 1-RDM off that core → rotate to its natural orbitals → ACCEPT the rotation only
    if it lowers the best-of-N energy at fixed core (greedy, guaranteed non-increasing),
    else stop. Accumulates the net rotation `R`. This RETIRES the one-shot
    `natural_orbitals_from_core` COO (which helped at A=10 but HURT at A=6 — an unrefined
    frame). Uses best-of-`num_runs` so the accept/reject survives the small-core ensemble
    variance. NOTE: this is the 1-RDM natural-orbital PROXY for TrimCI's true 2-RDM
    energy-GRADIENT BFGS orbopt (the HPC-scale STEP-5 target) — an intermediate improvement,
    not the full method. Returns `{R, H_frame, energy, cycles_run, occ, history}`."""
    if solve is None:
        solve = _solver(True)
    n = H.n_ferm_modes
    R_total = np.eye(n, dtype=complex)
    H_cur = H
    res = solve(H_cur, n_elec=n_elec, n_dets=core, seed=seed, n_runs=num_runs)
    E_prev = float(res.energy)
    coeffs, farr, barr = res.coeffs, res.ferm_arr, res.bos_arr
    occ = None
    history = [{"cycle": 0, "energy": E_prev, "accepted": True}]
    cycles_run = 0
    for c in range(1, cycles + 1):
        R_cyc, occ = frame.natural_orbitals_from_core(H_cur, coeffs, farr, barr)
        offdiag = float(np.max(np.abs(R_cyc - np.eye(n))))
        if offdiag < conv_tol:                       # 1-RDM already diagonal → converged
            if verbose:
                print(f"    cycle {c}: orbitals converged (offdiag={offdiag:.1e})", flush=True)
            break
        H_new = frame.rotate_orbitals_terms(H_cur, R=R_cyc)
        res = solve(H_new, n_elec=n_elec, n_dets=core, seed=seed, n_runs=num_runs)
        E_new = float(res.energy)
        accepted = E_new < E_prev - margin
        history.append({"cycle": c, "energy": E_new, "offdiag": offdiag,
                        "accepted": bool(accepted)})
        if verbose:
            print(f"    cycle {c}: E {E_prev:.4f} -> {E_new:.4f}  offdiag={offdiag:.2f}  "
                  f"{'accept' if accepted else 'STOP (no gain)'}", flush=True)
        if not accepted:
            break
        H_cur, E_prev = H_new, E_new
        coeffs, farr, barr = res.coeffs, res.ferm_arr, res.bos_arr
        R_total = R_total @ R_cyc
        cycles_run = c
    return {"R": R_total, "H_frame": H_cur, "energy": E_prev, "cycles_run": cycles_run,
            "occ": occ, "history": history}


def _build_frame(H, n_elec, frame_spec, phase0_core, phase0_runs, orbopt_cycles,
                 seed, verbose, solve):
    """Phase 0 — FIND the frame. Boson Gaussian frames are analytic (closed form, no
    search); COO uses the iterative orbopt loop. Returns (H_frame, phase0_info)."""
    spec = frame_spec.lower()
    info = {"frame_spec": frame_spec}
    if spec == "bare":
        return H, info
    if spec in ("squeeze", "gaussian"):
        r, phi = frame.analytic_squeeze(H)
        info["method"] = "analytic squeeze (closed-form r*)"
        return frame.squeeze_terms(H, -r, phi), info
    if spec == "bogoliubov":
        al, be = frame.analytic_bogoliubov(H)
        info["method"] = "analytic multi-mode Bogoliubov"
        return frame.bogoliubov_terms(H, al, be), info
    if spec == "coo":
        oo = coo_orbopt(H, n_elec, core=phase0_core, num_runs=phase0_runs,
                        cycles=orbopt_cycles, seed=seed, verbose=verbose, solve=solve)
        info.update({"method": "iterative COO orbopt", "orbopt_cycles": oo["cycles_run"],
                     "orbopt_energy": oo["energy"]})
        return oo["H_frame"], info
    if spec in ("squeeze+coo", "gaussian+coo"):
        r, phi = frame.analytic_squeeze(H)
        Hsq = frame.squeeze_terms(H, -r, phi)
        oo = coo_orbopt(Hsq, n_elec, core=phase0_core, num_runs=phase0_runs,
                        cycles=orbopt_cycles, seed=seed, verbose=verbose, solve=solve)
        info.update({"method": "analytic squeeze then iterative COO orbopt",
                     "orbopt_cycles": oo["cycles_run"], "orbopt_energy": oo["energy"]})
        return oo["H_frame"], info
    raise ValueError(f"unknown frame_spec {frame_spec!r} "
                     "(bare|squeeze|bogoliubov|COO|squeeze+COO)")


def three_phase_run(H, n_elec, frame_spec="squeeze", *, phase0_core=500, phase0_runs=32,
                    orbopt_cycles=8, phase2_cores=(1000, 2000, 4000), phase2_runs=3,
                    conv_tol_rel=1e-3, seed=0, verbose=False, solve=None):
    """A production frame-optimized TrimCI run in TrimCI's THREE-PHASE structure
    (TrimCI_skill.py):
      Phase 0 — FIND the frame (analytic for boson Gaussian; iterative orbopt for COO).
      Phase 1 — CO-EVOLVE orbitals with the growing det space. For boson Gaussian frames
                there is nothing to co-evolve (the frame is fixed once, analytically);
                for COO the FULL co-evolution is the HPC-scale STEP-5 target, so here we
                fold a single larger-core orbopt refinement into Phase 0 via `phase0_core`
                and note the deferral.
      Phase 2 — FREEZE the frame, grow dets to convergence, judged by the RELIABLE
                independent-solve drop |E(2N)−E(N)|/|E| (NOT the deceptive in-run ramp).
    `frame_spec`: bare | squeeze | bogoliubov | COO | squeeze+COO.
    Returns `{frame_spec, phase0, phase2, energy, converged}`."""
    if solve is None:
        solve = _solver(True)
    if verbose:
        print(f"[Phase 0] find frame: {frame_spec}", flush=True)
    H_frame, phase0 = _build_frame(H, n_elec, frame_spec, phase0_core, phase0_runs,
                                   orbopt_cycles, seed, verbose, solve)
    # Phase 2 — freeze the frame, grow dets to convergence (independent solves).
    if verbose:
        print(f"[Phase 2] freeze frame, grow dets {list(phase2_cores)}", flush=True)
    curve = []
    prevE = None
    converged = False
    for c in phase2_cores:
        res = solve(H_frame, n_elec=n_elec, n_dets=c, seed=seed, n_runs=phase2_runs)
        E = float(res.energy)
        rel = (abs(E - prevE) / max(abs(E), 1e-12)) if prevE is not None else None
        curve.append({"core": int(res.n_dets), "energy": E, "rel_drop": rel})
        if verbose:
            ds = f"{rel:.2e}" if rel is not None else "--"
            print(f"    core={res.n_dets:>6} E={E:.4f}  |ΔE|/E={ds}", flush=True)
        if rel is not None and rel < conv_tol_rel:
            converged = True
            break
        prevE = E
    return {"frame_spec": frame_spec, "phase0": phase0, "phase2": curve,
            "energy": curve[-1]["energy"], "converged": converged, "terms": len(H_frame.terms)}
