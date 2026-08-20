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
    accepted_res = res                 # the solved core in the CURRENT (accepted) basis
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
        accepted_res = res
        R_total = R_total @ R_cyc
        cycles_run = c
    return {"R": R_total, "H_frame": H_cur, "energy": E_prev, "cycles_run": cycles_run,
            "occ": occ, "history": history, "res": accepted_res}


def optimize_frame(H_bare, n_elec, core, has_gaussian=False, has_lf=False, has_coo=False,
                   num_runs=8, cycles=3, seed=0, solve=None, verbose=False):
    """ONE frame-optimization step — fit the WHOLE frame (any mix of squeeze / Lang-Firsov
    polaron / COO orbital rotation) against a determinant space of size `core`. Uniform for
    every frame; absent components are skipped:
      squeeze : analytic r* from the boson quadratic form (core-independent);
      LF      : polaron displacement amplitude, optimized at THIS core;
      COO     : fermion orbitals, optimized at THIS core (coo_orbopt).
    Applied identically at Phase 0 (small core) and repeated at each Phase-1 core, so the
    frame CO-EVOLVES with the growing det space. Returns (H_frame, res) — res is the solved
    core in the framed basis (for PT2 + warm-starting the next phase)."""
    if solve is None:
        solve = _solver(True)
    H = H_bare
    if has_gaussian:
        r, phi = frame.analytic_squeeze(H_bare)
        H = frame.squeeze_terms(H_bare, -r, phi)
    if has_lf:
        best = frame.optimize_displacement(H, n_elec, core=core, seed=seed)
        H = frame.displace_terms(H, lambdas=best["scale"], gen=best["gen"])
    if has_coo:
        oo = coo_orbopt(H, n_elec, core=core, num_runs=num_runs, cycles=cycles,
                        seed=seed, solve=solve, verbose=verbose)
        return oo["H_frame"], oo["res"]
    res = solve(H, n_elec=n_elec, n_dets=core, n_runs=num_runs, seed=seed)
    return H, res


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


# ===========================================================================
#  Faithful Phase-1 CO-EVOLUTION (TrimCI's slow-growth, warm-started refinement)
#  ---------------------------------------------------------------------------
#  TrimCI grows the determinant space SLOWLY in Phase 1 (growth_factor γ≈1.1)
#  and re-optimizes the frame (their κ) at EVERY round, as a SINGLE warm-started
#  deterministic trajectory — NOT the doubling + fresh-random-restart-per-rung
#  our earlier `three_phase_growing_run` Phase 1 used. Stochastic restarts
#  (`num_runs`) live ONLY in Phase 0. The reference confirms: each round screens
#  the pool FROM the current core (superset by construction), maps the previous
#  CI vector forward as the warm guess, then rotates orbitals on the just-grown
#  wavefunction, carrying the rotation + integrals forward.
#
#  Our frame generalizes their κ (fermion orbital rotation, COO) to the full
#  boson+fermion mixed frame: Gaussian squeeze (r*), Lang-Firsov displacement,
#  AND COO. The frame is carried as a STATE dict; every core-DEPENDENT piece is
#  refined once per γ-round. The analytic squeeze r* is core-INDEPENDENT so it is
#  a genuine no-op here UNLESS `squeeze_opt='numerical'` — which re-optimizes r by
#  the fixed-core energy each round and logs ΔE vs the analytic r* (the "verify /
#  beat the closed-form r*" investigation).
# ===========================================================================

def _warm_solve(H, n_elec, n_dets, initial_core=None, seed=0):
    """One warm-started SINGLE-run array-native solve — the Phase-1 continuation
    primitive. Phase 1 is a deterministic warm-started trajectory (num_runs lives
    only in Phase 0), so we GROW the carried core (`initial_core`) rather than
    re-seed a fresh random ensemble at each rung."""
    from .graph_arrays import ground_state_arrays
    return ground_state_arrays(H, n_elec=n_elec, n_dets=n_dets,
                               initial_core=initial_core, seed=seed)


def _around(center, span, n):
    """`n` scan points multiplicatively bracketing `center` by ±`span` (a light
    local scan around the current frame scale). `center=0` falls back to unit
    magnitude so a not-yet-active piece can still be probed."""
    c = float(center) if abs(float(center)) > 1e-12 else 1.0
    return list(np.linspace(c * (1.0 - span), c * (1.0 + span), int(n)))


def new_frame_state(n_bos_modes):
    """Empty frame STATE: no squeeze / displacement / rotation. Populated by
    `initial_frame_state` (Phase 0) and mutated in place by `refine_frame_state`
    (each Phase-1 round). Keys — squeeze: r (signed, applied), phi, r_seed
    (analytic magnitude), sq_scale (1.0≡analytic frame is s=-1); LF: disp_scale,
    disp_gen, disp_entries, disp_fc_dress, disp_order; COO: R (accumulated
    rotation)."""
    return {"r": None, "phi": np.zeros(int(n_bos_modes)), "r_seed": None,
            "sq_scale": 0.0, "disp_scale": 0.0, "disp_gen": None,
            "disp_entries": None, "disp_fc_dress": None, "disp_order": 4,
            "R": None}


def apply_frame(H_bare, state):
    """Build the framed Hamiltonian `Ū = U†HU` from `H_bare` and a frame STATE,
    layers applied outermost-first (squeeze ∘ displace ∘ rotate) — the same order
    `optimize_frame`/`_build_frame` use, so a state built by `initial_frame_state`
    reproduces their H exactly. Every piece is optional; an empty state returns
    `H_bare` unchanged. This is the single source of truth for the carried frame."""
    H = H_bare
    r = state.get("r")
    if r is not None and np.any(np.abs(np.asarray(r)) > 1e-12):
        H = frame.squeeze_terms(H, r, state.get("phi", 0.0))
    if state.get("disp_gen") is not None and abs(state.get("disp_scale", 0.0)) > 1e-12:
        H = frame.displace_terms(H, lambdas=float(state["disp_scale"]),
                                 gen=state["disp_gen"],
                                 fc_dress=state.get("disp_fc_dress"),
                                 order=int(state.get("disp_order", 4)))
    R = state.get("R")
    if R is not None:
        H = frame.rotate_orbitals_terms(H, R=R)
    return H


def optimize_squeeze_energy(H_base, n_elec, r_seed, phi, *, core, initial_core=None,
                            seed=0, scales=None, span=0.25, n_scan=5):
    """NUMERICAL squeeze: choose the global scale `s` over the analytic seed
    `r_seed` that MINIMISES the warm fixed-`core` TrimCI energy — the past-ED
    objective flagged in `frame.optimize_squeeze`'s docstring (an isospectral frame
    is variational, so a better r reaches a LOWER E at fixed N_det). The ANALYTIC
    frame is `s = -1` (`squeeze_terms(H, -r*)`, the diagonalising-sign convention),
    so the scan always includes `s=-1` and reports `E(s=-1)` as the analytic
    baseline + `dE_vs_analytic = E_best - E_analytic` (negative ⇒ optimized r beats
    the closed form). `H_base` is the layer the squeeze sits on (`H_bare` when
    squeeze is outermost). Returns `{sq_scale, r, phi, energy, E_analytic,
    dE_vs_analytic, scan}`."""
    r_seed = np.asarray(r_seed, float)
    if scales is None:
        scales = np.linspace(-1.0 - span, -1.0 + span, int(n_scan))
    grid = sorted({-1.0, *(float(s) for s in scales)})   # always keep the analytic point
    scan, best, E_analytic = [], None, None
    for s in grid:
        Hf = frame.squeeze_terms(H_base, s * r_seed, phi)
        res = _warm_solve(Hf, n_elec, core, initial_core=initial_core, seed=seed)
        E = float(res.energy)
        scan.append((float(s), E, int(res.n_dets)))
        if abs(s + 1.0) < 1e-12:
            E_analytic = E
        if best is None or E < best[1]:
            best = (float(s), E)
    return {"sq_scale": best[0], "r": best[0] * r_seed, "phi": phi,
            "energy": best[1], "E_analytic": E_analytic,
            "dE_vs_analytic": (None if E_analytic is None else best[1] - E_analytic),
            "scan": scan}


def frame_coevolves(has_gaussian, has_lf, has_coo, squeeze_opt):
    """Does this frame have a CORE-DEPENDENT piece to co-evolve in Phase 1? COO and
    LF always do; the analytic squeeze does NOT (closed-form r*, core-independent),
    so it co-evolves only under `squeeze_opt='numerical'`. When False, Phase-1 slow
    growth buys nothing and the runner goes straight to Phase-2 fast expansion —
    exactly the paper's logic (slow growth exists to let the frame track the core)."""
    return bool(has_coo or has_lf or (has_gaussian and squeeze_opt == "numerical"))


def initial_frame_state(H_bare, n_elec, *, has_gaussian=False, has_lf=False,
                        has_coo=False, squeeze_opt="analytic", core=1000,
                        num_runs=64, cycles=10, seed=0, verbose=False, solve=None):
    """Phase-0 DISCOVERY: fit the WHOLE frame heavily at a small `core` and return
    `(state, res, H_frame, info)`. This is the one place the stochastic search
    budget goes (`num_runs` restarts, `cycles` orbopt). Squeeze uses the closed-form
    r* (analytic) unless `squeeze_opt='numerical'`, which also picks the energy-best
    scale here. `state`/`res` seed the Phase-1 co-evolution; `H_frame == apply_frame(
    H_bare, state)`."""
    if solve is None:
        solve = _solver(True)
    state = new_frame_state(H_bare.n_bos_modes)
    info = {"squeeze_opt": squeeze_opt}
    H = H_bare
    if has_gaussian:                                     # --- squeeze (outermost) ---
        r_seed, phi = frame.analytic_squeeze(H_bare)
        # ALWAYS seed with the closed-form r* (s=-1) — even for squeeze_opt='numerical'.
        # The numerical refinement is done in the WARM Phase-1 loop (fair, low-variance),
        # not by a noisy cold single-run scan here; r_seed is kept for that scan.
        state["r_seed"], state["phi"] = r_seed, phi
        state["r"], state["sq_scale"] = -r_seed, -1.0        # analytic frame ≡ s=-1
        info["squeeze"] = {"seed": "analytic r* (s=-1)",
                           "numerical_refine": squeeze_opt == "numerical"}
        H = frame.squeeze_terms(H_bare, state["r"], phi)
    if has_lf:                                           # --- LF displacement ---
        best = frame.optimize_displacement(H, n_elec, core=core, seed=seed)
        state["disp_scale"], state["disp_gen"] = best["scale"], best["gen"]
        state["disp_entries"] = best.get("entries")
        H = frame.displace_terms(H, lambdas=state["disp_scale"], gen=state["disp_gen"])
        info["lf"] = {"scale": best["scale"], "energy": best["energy"]}
    if has_coo:                                          # --- COO orbital rotation ---
        oo = coo_orbopt(H, n_elec, core=core, num_runs=num_runs, cycles=cycles,
                        seed=seed, solve=solve, verbose=verbose)
        state["R"], H, res = oo["R"], oo["H_frame"], oo["res"]
        info["coo"] = {"cycles_run": oo["cycles_run"], "energy": oo["energy"]}
    else:
        res = solve(H, n_elec=n_elec, n_dets=core, n_runs=num_runs, seed=seed)
    return state, res, H, info


def refine_frame_state(H_bare, n_elec, state, *, core, initial_core, seed=0,
                       has_gaussian=False, has_lf=False, has_coo=False,
                       squeeze_opt="analytic", refine_span=0.25, refine_points=5,
                       accept_tol=1e-9):
    """ONE Phase-1 co-evolution round: (1) warm-start-solve the CURRENT frame on the
    grown det space (`core`) from `initial_core`; (2) refine each CORE-DEPENDENT
    piece by a SINGLE light step seeded from `state`, accept-if-lower — numerical
    squeeze scale, LF displacement scale, one COO natural-orbital rotation. Mirrors
    the paper's per-round κ re-rotation on the just-grown, warm-started wavefunction.
    Mutates + returns `state`; returns `(state, res, H_frame, log)` that carry to the
    next (larger) round. The analytic squeeze is skipped (core-independent)."""
    log = {"core": int(core)}
    H = apply_frame(H_bare, state)
    res = _warm_solve(H, n_elec, core, initial_core=initial_core, seed=seed)
    warm = (res.ferm_arr, res.bos_arr)
    log["E_grow"] = float(res.energy)

    if has_gaussian and squeeze_opt == "numerical" and state.get("r_seed") is not None:
        sq = optimize_squeeze_energy(
            H_bare, n_elec, state["r_seed"], state["phi"], core=core, initial_core=warm,
            seed=seed, scales=_around(state["sq_scale"] or -1.0, refine_span, refine_points))
        log["squeeze"] = {"sq_scale": sq["sq_scale"], "dE_vs_analytic": sq["dE_vs_analytic"]}
        if sq["energy"] < float(res.energy) - accept_tol:
            state["r"], state["sq_scale"] = sq["r"], sq["sq_scale"]
            H = apply_frame(H_bare, state)
            res = _warm_solve(H, n_elec, core, initial_core=warm, seed=seed)
            warm = (res.ferm_arr, res.bos_arr)

    if has_lf and state.get("disp_gen") is not None:
        Hpre = H_bare
        if state.get("r") is not None and np.any(np.abs(state["r"]) > 1e-12):
            Hpre = frame.squeeze_terms(H_bare, state["r"], state["phi"])
        bestlf = frame.optimize_displacement(
            Hpre, n_elec, entries=state.get("disp_entries"),
            scales=_around(state["disp_scale"], refine_span, refine_points),
            core=core, n_runs=1, seed=seed)
        log["lf"] = {"disp_scale": bestlf["scale"]}
        if bestlf["energy"] < float(res.energy) - accept_tol:
            state["disp_scale"] = bestlf["scale"]
            H = apply_frame(H_bare, state)
            res = _warm_solve(H, n_elec, core, initial_core=warm, seed=seed)
            warm = (res.ferm_arr, res.bos_arr)

    if has_coo:
        R_step, _occ = frame.natural_orbitals_from_core(H, res.coeffs, res.ferm_arr,
                                                        res.bos_arr)
        offdiag = float(np.max(np.abs(R_step - np.eye(R_step.shape[0]))))
        log["coo"] = {"offdiag": offdiag, "accepted": False}
        if offdiag > 1e-6:
            R_prev = (state["R"] if state.get("R") is not None
                      else np.eye(H_bare.n_ferm_modes, dtype=complex))
            trial = dict(state); trial["R"] = R_prev @ R_step
            Htry = apply_frame(H_bare, trial)
            res_try = _warm_solve(Htry, n_elec, core, initial_core=warm, seed=seed)
            if float(res_try.energy) < float(res.energy) - accept_tol:
                state["R"], H, res = trial["R"], Htry, res_try
                log["coo"]["accepted"] = True

    log["E"] = float(res.energy)
    return state, res, H, log
