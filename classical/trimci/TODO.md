# TrimCI ⟷ NuQu — to-do

Working notes for getting a TrimCI ground-state solve running on the dynamical-pion
EFT. Sketch, not a contract — reorder freely.

## 2026-07-23 — first HPC job package (dets-vs-L) + None-reference crash fix
The Phase-C dets-vs-L exponent is the top classical HPC priority (needs cores past the
laptop's ~16k, where every point is still only a bound). Packaged the first real cluster
run under `hpc/detsvsL/` (`setup_env.sh` + `run_detsvsL.sh` + `submit_detsvsL.sh` +
pinned `requirements-hpc.txt` + `README.md`): L=2,3 dilute 3D, cores→50k, n_runs=4,
qis-pinned, reads code+venv from the shared `/nfs_scratch` checkout, **scipy-eigsh path**
(no official trimci / jax / netket / pyLIQTR — verified minimal by blocking them and
re-running). Provenance: runs now carry an `hpc` tag in the saved JSON/metadata
(`dets_vs_L_at_fixed_accuracy(hpc=)`, `--hpc` CLI, `save_classical_run(hpc=)`).
- **Bug fixed:** `dets_vs_L_at_fixed_accuracy` crashed (`float - NoneType`) when a per-L
  ladder was too shallow to extrapolate E_inf (E_inf=None) — exactly the case a wall-capped
  large-L HPC ladder hits. Now `_extract_nstar` returns a `"no_reference"` bound and the
  caller skips the E_inf±σ arithmetic: honest bound, never loses a multi-hour run.
- **Solver decoupled from official TrimCI (the load-bearing HPC fix):** `cpp_available()`
  now means "mixed_ci C++ built", NOT "official TrimCI Davidson present". Without the
  official package the selected-CI subspace is diagonalized by **scipy eigsh over the
  mixed_ci C++ CSC** (`_diagonalize_arrays_scipy`; dense numpy.eigh only for N<32) instead
  of silently dropping to the pure-Python DENSE solver (capped at 6000 states, ~55× slower
  — which killed the first HPC run at 6188 dets). Exact-equal to the official Davidson
  (E=2349.578936 both at 10k dets; 9e-13 vs Lanczos) and same wall-clock; the official path
  is still used when present (comparison switch, `has_sparse_davidson()`). ⇒ the classical
  baseline scales to large cores with just openfermion + scipy + mixed_ci — no jax/netket.
- HPC division of labor: the cluster runs the LARGE-core rungs (8k→64k) the laptop can't
  reach; small cores stay local and are combined post-hoc (matching dim/A/N_f/n_runs).
- `build_mixed_ci.sh` made OS-portable (Linux drops the Darwin-only `-undefined
  dynamic_lookup`; venv python found relative to the script; still no `-march=native`, so
  the `.so` runs on heterogeneous execute nodes).

## 2026-07-10 (Phase D) — honesty & robustness guards
The layer that makes the Phase-C chain read honestly: surface (and where possible veto)
the ways each estimate can lie. New module `classical/trimci/robustness.py` (pure
functions) + drivers in `run_cpp.py`:
- **`ladder_monotonicity(rungs)`** — flags a RISE in E_var as the core grows (bursty /
  under-converged rung); wired into `converged_reference` output + verbose.
- **`pt2_memory_report(rungs)`** — tracks EN-PT2 external-space size (`n_ext`) and fires a
  **SEMISTOCHASTIC-PT2 trigger** (arXiv:1808.02049) once it exceeds a budget (default 50M).
  Deterministic PT2 sums the FULL connected space (exact, more accurate than sampled) but
  its work/memory grow ~core×#terms; the trigger is the signal to build the sampled-tail
  variant before pushing deeper. On the laptop L=2–4 run max n_ext≈2.4M ⇒ NO trigger
  (deterministic PT2 fine); wired for HPC.
- **`seed_robustness(...)`** (+ CLI `--mode seed_robustness`) — re-solve the SAME (L, core)
  with several independent base seeds (each its own n_runs ensemble, NO warm starts) and
  measure the E_var / E_var+PT2 scatter. **KEY FINDING (real system): the n_runs=4 ensemble
  is SEED-FRAGILE at laptop cores.** L=2 d=3, core=2000, seeds {0,1000,2000,3000}: E_var+PT2
  spans 2352.8–2407.1 MeV = **6.79 MeV/site (54 MeV) peak-to-peak** ⇒ SEED-FRAGILE vs
  eps=1. The single-seed Phase-C ladders all LOOK monotone yet hide this — so those rungs
  carry ~7 MeV/site seed uncertainty ON TOP OF the extrapolation uncertainty. ⇒ n_runs=4 is
  under-powered at these cores; needs many more ensemble runs and/or the HPC-scale cores
  where selection is less basin-dependent. (This compounds — does not change — the Phase-C
  "needs HPC" verdict, and explains the extrapolator instability.)
- **Extensivity signal auto-computed** in the dets-vs-L output: at the deepest common core
  (4000), dE_PT2/site = −2.09 / −6.50 / −9.51 for L=2/3/4 — the trap made visible.
- **Circularity + validation** consolidated in the `robustness.py` docstring: the reference
  is extrapolated (not truth) at L≥2 d=3; the standing proof the chain is trustworthy where
  truth exists is the L=2 d=1 slice (extrap = exact Lanczos to <1e-3 MeV/site, test [36]).
- Test [38]. **STRETCH (not built, flagged for a referee):** an independent-method
  cross-check at one 3D point — DMRG via `block2` (fermion-boson MPO) — would harden the
  reference beyond ED/Lanczos reach.

## 2026-07-09 (Phase C) — dets-vs-L exponent at fixed per-site accuracy (the headline)
The paper's central number: N*(L; eps) = determinants to reach a FIXED per-site accuracy
eps of the L-converged ground energy, fit vs system volume to read the scaling exponent γ
(does classical selected-CI cost grow polynomially or exponentially in volume?). Built on
the Phase-B reference; done honestly (bounds, not fake points, where the reference isn't
pinned). New in `run_cpp.py`:
- `dets_vs_L_at_fixed_accuracy(...)` — per L: `converged_reference(L)` → E_inf±σ + pinning;
  `_extract_nstar` brackets N* on the STABLE criterion (|E_var+dE_PT2 − E_inf|/sites < eps
  for that rung AND every larger rung — guards against a bursty fluke early pass); σ_{E_inf}
  propagated into N*'s bracket by re-extracting at E_inf±σ. A point is FIT-WORTHY only if the
  reference is pinned AND N* is bracketed; non-fit-worthy L's are reported as bounds and
  LOGGED (▲ lower bound = not reached; ▼ upper bound = below the smallest core) — never
  silently dropped. `_fit_exponent` contrasts exponential-in-V (`log N* = a+γV`) vs
  polynomial-in-V (`log N* = a+γ·logV`); the higher-R² model says which regime. Stretched-exp
  (Lee/Chan `a+γV^p`) deferred to the HPC L-range (p needs > 3 L's).
- Laptop guards: geometric ladder from `ladder_start` (×2/rung, ≤ `n_rungs`), capped at
  `max_core`, that stops growing once a rung's wall exceeds `max_rung_seconds` (large L
  self-limit). `filling=None` → dilute A fixed; float → fixed-filling A=round(filling·sites).
  Adaptive ladder added to `converged_reference` (`max_core`/`max_rung_seconds`, backward
  compatible — explicit `cores=` still uses the fixed `_solve_ladder`). Plot
  `plotting.plot_dets_vs_L` (log-lin + log-log panels straighten exp/poly respectively).
  Driver `misc/run_dets_vs_L.py`. Test [37] (synthetic bracketing/bounds/stability +
  exp/poly slope recovery + end-to-end).
- **RESULTS (laptop L=2–4, dilute A=1, dim=3, n_b=2, n_runs=4; 59.5 min, cores→16k for
  L=2,3 and →4k for L=4 where the 240s/rung guard stopped it — L=4 core-4000 = 856s ≈
  L=2 core-4000 × 64, the sites² cost, genuine not sleep).**
  - **No fit-worthy points: every (L, eps) is a LOWER BOUND — nothing pins**, so the
    machinery (correctly) reports bounds, not a fabricated exponent. Two independent
    reasons, both caught by the pinning guards: (i) at the largest core reached, E_var+PT2
    is still **3.4–4.7 MeV/site above** the extrapolated E_inf (L=2 4.44 @16k, L=3 4.65
    @16k, L=4 3.36 @4k) — nowhere near eps=1, let alone 0.1; (ii) the reference itself
    isn't pinned — power-law vs SHCI/PT2 extrapolators disagree by ~1–12 MeV/site (worst at
    L=4, cross-gap/site=12) and σ/site is 2.4/0.86/0.12 for L=2/3/4. **⇒ the exponent needs
    HPC**: cores well past 16k (the gap is only ~4/site there and falling slowly) AND enough
    depth for the two extrapolators to converge. The pipeline + this quantified HPC
    requirement is the deliverable. NB the L=4 lower bound (N*>4k) is shallower than L=2,3
    (N*>16k) only because its ladder was cost-limited — the bounds are NOT apples-to-apples
    across L (the plot shows them as ▲, i.e. "at least this").
  - **Salvageable size-intensive signal (the extensivity trap, made visible):** at a FIXED
    core = 4000 (all three L reached it) the per-site EN-PT2 correction — a proxy for how
    INCOMPLETE that fixed budget is — grows monotonically with volume: **−2.09 (L=2) →
    −6.50 (L=3) → −9.51 (L=4) MeV/site**. A fixed determinant budget is progressively less
    adequate as the system grows — exactly the non-extensivity the whole study is about,
    quantified even though the absolute N*(eps) isn't yet resolvable on a laptop.
  - Output: `data/classical/detsvsL_3d_dilute.{json,png,log}` (gitignored).

## 2026-07-09 — extensivity mitigation, Phase A + B (size-intensive accuracy + per-L reference)
Addressing the "extensivity trap" (selected-CI is not size-extensive; det count at
fixed accuracy grows with volume — Lee/Chan arXiv:2208.02199). Mitigations 1–2 (EN-PT2
`pt2.py`, SHCI/PT2 extrapolation `extrapolation.py`) were already done. This pass adds
the two that were missing — the size-intensive accuracy metric and a trustworthy per-L
reference — as the groundwork for the headline dets-vs-L exponent (Phase C, still to do).

- **Phase A — per-site (size-intensive) accuracy is now first-class.** Total E is
  size-EXTENSIVE (~sites), so a total or RELATIVE gap silently loosens the per-site
  tolerance as L grows — the trap. `extrapolation.report_energies(sites=...)` now emits
  the `_per_site` mirror of every headline number (E_var/site, E_var+PT2/site,
  E_inf/site ± σ/site, and (extrap−exact)/site). `run_cpp.l_scaling_sweep` gained a
  `gate` switch: `"relative"` (legacy `|dE(2N)|/|E| < target_gs_rel`) vs `"per_site"`
  (`|dE(2N)|/sites < eps_persite` MeV/site) — BOTH drops are always recorded and saved,
  so one run shows what either gate would decide (comparison switch, CLAUDE.md). Test [35].
- **Phase B — per-L converged reference `E_inf(L) ± σ` with honest point-vs-bound.**
  New `run_cpp.converged_reference(L)`: independent geometric core ladder → EN-PT2 per
  rung → SHCI/PT2 + power-law extrapolation → E_inf ± σ (total + per-site). THE CRUX it
  handles: at L≥2 in 3D there is no exact reference, so E_inf is extrapolated with a real
  uncertainty. Per accuracy target eps it decides whether E_inf is pinned tightly enough
  to DEFINE a fixed-eps N* — a "point" (VALIDATED vs exact `|Δ|/site<eps`, OR
  `σ/site < sigma_frac·eps` AND the two extrapolators agree `|E_power−E_pt2|/site <
  cross_frac·eps`) — or only a "bound". Refactored `solve_and_report` onto shared
  `_pick_solver`/`_solve_ladder` helpers; `converged_reference` returns the per-rung
  ladder (with per-rung E_var/dE_pt2/E_pt2) that Phase C's N* extraction will consume.
  Wired into the CLI (`--mode reference`). Test [36] (point + bound branches).
  - **Validation:** L=2 d=1 A=2 N_f=2 pins as a POINT — SHCI/PT2 E_inf/site=198.3158 ±
    0.0009, validated vs exact ((extrap−exact)/site = −0.0007 MeV/site). **L=2 d=3 A=1
    from small cores (200–1600) correctly reports BOUND [NOT pinned]** for both eps=1 and
    0.1 MeV/site: the two extrapolators disagree by 1.36 MeV/site (σ/site=0.5) — exactly
    the "3D from small cores is optimistic" caveat (2026-07-07 Task 3), now enforced
    automatically instead of trusted. Pinning the tight targets at L≥2 d=3 needs deeper
    ladders / HPC (consistent with E0 still falling at 16k dets, 2026-07-01).
- **Next — Phase C (the headline):** `dets_vs_L_at_fixed_accuracy`: for each L (dilute
  A=1 first, then fixed-filling A∝sites) and each eps ∈ {1, 0.1} MeV/site, find the
  smallest N*(L) with `|E_var+dE_pt2 − E_inf(L)|/sites < eps`, propagate σ_{E_inf} into
  N*'s error bar, and fit N* vs volume (soft-exp + pure-exp) — γ is the paper's central
  number. Laptop-scale L=2–4 to prove the pipeline, then the HPC push (per the plan).

## 2026-07-08 (later) — occupation vs A: is the Fock cutoff A-robust?
New: `run_cpp.exact_occupation_vs_A` (exact-ED ⟨N⟩(A), no core caveat),
`run_cpp.occupation_vs_A_sweep` (selected-CI, per-A core ladder + lower-bound flag),
`observables.occupation_from_coeffs` (⟨N⟩ off an ED eigenvector dict), driver studies
E/F/G in `misc/run_nb_convergence.py`, plot `occupation_vs_A.png`, synthesis
`bosonic-encodings/04_occupation_vs_A.md`. Test [34] (asserts A-independence).

**Finding: GS pion occupation is A-INDEPENDENT** — set by vacuum squeezing (r*, a
geometry/dispersion property), NOT by nucleon count. Exact ED at L=2 d=1: ⟨N⟩/mode =
0.0126 flat to 5 sig figs across A=1→8 (filling 0.125→1.0) while E₀ sweeps 461→−282 MeV.
The H_AV displacement (the only A-channel) is ~1e-8·A²/mode — dead. d=3 selected-CI
agrees within core-convergence noise (⟨N⟩ is a finite-core LOWER BOUND at high A: the
boson space is huge and ⟨N⟩ converges slowly from below — even A=1 still climbs at core
16k). **n_b bound is A-robust:** at L=2 d=3, n_b=3 (N_f=8) converges E to 1e-4 MeV at A=1,
A=6, A=10 alike; tail past N_f stays exactly 0. **Predict A=100** (needs L≥3 to fit 100
nucleons): ⟨N⟩ ≈ 0.066/mode on L=3 d=3 (filling 0.93) = the A=1 value — near-vacuum, n_b=3–4
safe. Occupation is a function of L (via r*: 0.045→0.066→0.077 for L=2→3→4), not A.
→ implication: `estimate_boson_cutoff`'s `heuristic` A-growth `ceil(4+log₂(1+A))` is not
physically motivated; the A-flat `tong` switch is the better model (direct evidence here).

## 2026-07-08 — n_b / N_f Fock-cutoff convergence study (certifies the quantum cutoff)
New: `classical/trimci/observables.py` (⟨N⟩ + leaked-weight tail off the wavefunction),
`tong_bound.py` (SCS/spectral predictions from `bosonic-encodings/02_tong_fock_cutoff.md`,
validated against the doc), `run_cpp.nb_convergence_sweep`, driver `misc/run_nb_convergence.py`,
synthesis `claude/research/bosonic-encodings/03_nb_convergence_study.md`. Tests [31][32][33].

Two goals, both about the per-mode Fock cutoff N_f=2^n_b (n_b = qubits/mode on the quantum side).

- **GOAL 1 — TrimCI cost is FLAT in n_b.** Cost is set by #Hamiltonian-terms × core, both
  n_b-INDEPENDENT (the cutoff only enters at ladder-apply time); with a near-vacuum GS the
  fixed-core selected-CI never explores high occupations. MEASURED (L=2 d=3 A=1, core 2500):
  runtime ≈4s FLAT from n_b=1→6 while the sector grows **5.4e8 → 7.1e44 (37 orders of
  magnitude)**; raw E_var at N_f=8,16,32,64 is BIT-IDENTICAL (2421.16495) — the GS never
  populates occ≥8, shown directly. So pushing n_b past the plateau is free classically and
  useless physically.

- **GOAL 2 — certify the quantum n_b bound where ED can't.** `02_*.md`'s own ED cross-check is
  stuck at L=1 (trivial: no gradient ⇒ vacuum ⇒ converged at n_b=1). TrimCI does it at L≥2.
  Method: per N_f, fixed-core solve + EN-PT2 (removes ~constant core-incompleteness so the
  truncation DIFFERENCE is clean) + ⟨N⟩/tail; powers-of-two = the n_b axis; odd N_f only to show
  the non-variational wobble. Reference = largest-N_f E_var+PT2.
  - **Study A (L=2 d=1, exact-validated):** TrimCI+PT2 = exact Lanczos to <1e-4 MeV at every N_f
    where exact exists (2..6), incl. reproducing the N_f=3 non-variational dip; extends past the
    exact wall (N_f=8=2.1M, N_f=16=1.3e8). Converged by n_b=3; empirical n_b(ε~0.1)=2, between
    SCS-eng (n_b=1) and spectral (n_b=3) — matches the doc's §6.4 prediction exactly.
  - **★ Study B (L=2 d=3 A=1, THE real system, ED-impossible):** E_var+PT2 = 2426.4(n_b=1) →
    2406.6(n_b=2) → **2402.7(n_b=3) → 2402.7(n_b=4)**. **Converged at n_b=3 (N_f=8) to <0.01 MeV**
    (sector 1.5e23 — no exact method reaches it). ⟨N⟩/mode=0.040 vs SCS 0.045 (near-vacuum
    confirmed). Non-variational wobble bigger here (N_f=3 undershoots ~44 MeV) ⇒ use powers of 2.
    **Bound bracket:** SCS-eng n_b=1 (optimistic — an occupation quantile, not energy), EMPIRICAL
    n_b=3, rigorous spectral n_b=4(2nd)/5(1st) [safe over-estimate ✓], heuristic n_b=6 [2×
    over-pad, IDENTICAL energy to n_b=3]. ⇒ the heuristic `estimate_boson_cutoff` (n_b=6) is 2×
    too large; the spectral `tong_bound` value (n_b≈4-5) is certified safe.
  - **Study C (L=3, L=4 d=3, preliminary):** running/appended in the synthesis doc; runtime grows
    with #terms×core (L=3 ~18s/pt at core 1500) but the n_b axis stays flat, so the n_b BOUND is
    cheap at any L — only the ABSOLUTE energy (core→∞) needs HPC-scale cores (the cutoff DIFFERENCE
    is robust to core-incompleteness).
  - CAVEAT: fixed-core PT2 residual (~constant dE_PT2, cancels in the cutoff difference; absolute
    energy is the separate core axis) and non-variational-in-N_f wobble (powers of two only).
  - **Follow-up:** wire `boson_cutoff_method='tong'` into `estimate_boson_cutoff` (CLAUDE.md open
    homework) — `tong_bound.cutoff_predictions` already computes it; this study certifies it safe.

## 2026-07-07 — accuracy shore-up: basis check, PT2, honest E_infinity
Three tasks to lower/qualify the reported ground energies and raise confidence.

- **Task 1 — bosonic basis is ALREADY the efficient one (verified), + N_f decoupled
  from 2^n_b.** The pion register is stored as an OCCUPATION-NUMBER integer per mode
  on BOTH paths — Python `MixedState.bos` (tuple[int]) and the C++ hot path
  (`std::vector<uint16_t> bos`) — with 1-sparse ladder maps `b|n>=√n|n-1>`
  (`hij._apply_boson_ops`, `mixed_ci.hpp::apply_boson_ops`). This is the "n basis",
  NOT the dense "bit basis" (`|101>`-style) whose ladder ops would be dense. No
  change to the representation. What DID change: `build_from_eft(..., N_f=<int>)` and
  `from_mixed_hamiltonian(mh, n_b, N_f=<int>)` now take an explicit cutoff — the
  power-of-two box is a QUANTUM-hardware constraint (whole qubits), not a classical
  one, and the term list is cutoff-independent (N_f only enters at apply-time), so
  overriding it is exact (no rebuild). Default unchanged (`N_f=2**n_b`). Test [26].
  - **Finding the finer grid exposes: Fock truncation is NON-VARIATIONAL in N_f.**
    L=2 d=1 A=1 exact (Lanczos==denseED to 2e-12): E0(N_f=2,3,4,5) =
    465.55, **460.63**, 461.31, 461.27 — N_f=3 dips BELOW N_f=4. Hermitian but not a
    nested projection (the `a a†`/`a†a†` pieces from π² truncate intermediate
    excursions), so a single odd-N_f point can undershoot. ⇒ converge IN N_f (settles
    by N_f≈4–5 here); don't trust one low odd cutoff. Selected-CI is still variational
    WITHIN a fixed N_f.

- **Task 2 — Epstein-Nesbet PT2 implemented (`pt2.py`); a consistent free win.** The
  released `pt2_correction` flag is fermion-only (Slater-Condon), so PT2 is
  reimplemented over our mixed H_ij: `dE_PT2 = Σ_{a∉V} |<a|H|ψ>|² / (E_var − H_aa)`,
  one `connections` pass over the core + a cheap diagonal-only H_aa. **Bug found &
  fixed:** the solver reports the PRE-trim pool energy but SAVES the POST-trim top-k
  core, so the saved (states, coeffs) are NOT a self-consistent eigenpair — feeding
  them straight in overshot PT2 by ~19×. Fix: RE-DIAGONALIZE the saved core first
  (`epstein_nesbet_pt2` does this by default; Rayleigh gap → ~1e-13). Validated vs
  exact (L=2 d=1 A=2 N_f=2): recovers **88–98% of the correlation at every core, no
  overshoot** (N=30: err 0.274→0.009 MeV). On by default in the report. Test [27].
  **PT2 pass-1 C++ port DONE (2026-07-08).** The `connections` sweep + diagonal
  H_aa evaluation now run in C++ (`MixedProvider.pt2_accumulate` in
  `backend_fork/mixed_ci{.hpp,_pybind.cpp}`, wired via `backend.cpp_pt2_external`);
  `epstein_nesbet_pt2(use_cpp=None)` auto-detects it (the pure-Python path stays as
  fallback + the validated reference). Isolated pass-1 benchmark (cold caches,
  trust_coeffs to remove re-diag): **17× at 229 terms → 47× at 1265 → 57× at 4897
  terms** (L=3 d=3; the speedup GROWS with #terms — precisely the L≥4 bottleneck),
  bit-identical dE_PT2 (≤1e-13). This unblocks PT2 at L≥4 / larger cores. Tests:
  `backend_fork/test_pt2_cpp.py` (C++-vs-Python equivalence over toy sectors).

- **Task 3 — honest E_infinity (`extrapolation.py`) + 3-number report.** PROVENANCE:
  "E∞" was a bare `curve_fit` of `E(N)=E∞+a·N^(-b)` in `misc/run_frame_comparison.py`
  / `misc/plot_L2_loglog.py` — the N_dets→∞ (Full CI at FIXED L, N_f, frame) limit,
  ESTIMATED by extrapolation, NOT exact truth and NOT experiment (still carries L /
  N_f / EFT-truncation / lattice error). Centralized here WITH a fit uncertainty (was
  a bare number) and a robustness guard (needs ≥4 DISTINCT independent-ladder rungs +
  a decreasing ramp; refuses otherwise — the 2026-07-01 finding that single-run ramps
  lie). Added a more principled **SHCI/PT2 extrapolation**: fit `E_var+dE_PT2 =
  E_FCI + c·dE_PT2`, read the intercept at dE_PT2=0. Driver `run_cpp.solve_and_report`
  emits: variational (best) / variational+PT2 / extrapolated ±σ / exact (Lanczos, when
  the sector is enumerable) with the extrapolation-vs-exact gap. Validated: L=2 d=1
  A=2 N_f=2 SHCI extrap **396.6317±0.0019 vs exact 396.6332 (Δ=−0.0015 MeV)**; the
  power-law fit is looser (Δ=−0.011) so the report prefers SHCI/PT2. Test [28].
  - **Caveat (3D):** from small cores the extrapolation is OPTIMISTIC — L=2 d=3 A=1
    from cores 400/800/1600 gives 2390±3.5 MeV, but prior independent solves show E0
    still descending past 2353 at 16k dets. The ±σ is FIT scatter, not model error; 3D
    needs a longer ladder (HPC) for a trustworthy E∞. `report_energies` shows both
    extrapolators + σ so the reader sees the spread.

## 2026-07-02 — optimization push (Tier 0 / 1 / 2): break the ~10^4-state ceiling
Diagnosed why our solver capped near ~10^4 states while stock (fermionic) TrimCI does
10^6 on a laptop. Three compounding issues + fixes:

- **Tier 0 — `pool_factor` 10 → 3.** With pf=10, `global_trim` diagonalizes ~5×N states
  per round and `local_trim` ~2×N per group (not N). Dropping to 3 gave **~2× speedup AND
  5–7 MeV LOWER energy** (a tighter pool focuses the diagonalization on the states that
  matter, less noise). Default changed in `graph.ground_state`. (L=2 3d, cores 1k→5k.)
- **Tier 1 — matrix-free eigsh (`SubspaceContext`).** Every subspace diagonalization used
  to build a fresh 2N real-embedded sparse matrix (scipy COO→CSR ~30% of the solve) and
  throw it away. New C++ `build_context(ferm,bos)` builds the complex CSC ONCE (no 2N
  embedding → 2× fewer entries) and `ctx.matvec` feeds `scipy eigsh` (complex-Hermitian
  native, no embedding). `backend.cpp_diagonalize_matfree` + `_smart` (auto-picks matfree
  for N≥2000). **1.6–3.2× faster than sparse Davidson** at N=2k–20k; end-to-end with pf=3
  **1.93× faster + 5.9 MeV lower** at n_dets=5000. Test [16].
- **Tier 2 — array-native hot loop (`graph_arrays.py`) — the RAM wall.** The real blocker
  to 10^6 was Python `MixedState` objects: the boson TUPLE alone is ~n_bos*8+56 B/state
  (≈3 KB at L=5 3d, and it GROWS with lattice size), and the per-round pool holds
  ~pool_factor× more. Rewrote the loop to carry the core/pool/survivors as **compact numpy
  arrays** (ferm (N,W) uint64, bos (N,n_bos) uint16, coeffs complex) end-to-end — never a
  MixedState. New C++ `expand_topk` returns candidates unique + disjoint from core, so
  pool = concat(core, candidates) and every subsequent keep/union is an **integer row-index**
  op (`argpartition`/`union1d`) — no hash/dedup over heavy state keys. Diagonalization
  array-native (`cpp_diagonalize_matfree_arrays`: fast sparse Davidson small-N, CSC+eigsh
  large-N). `GroundStateResult.ferm_arr/bos_arr` + `io` read them directly → wavefunction
  saved with zero MixedState materialization. Object path kept as the reference /
  comparison switch (`run_cpp --no-arrays`). **Array MATCHES object** on energy (ED to
  1.6e-7, variational) and speed, with LESS RAM (gap widens with L/core). Scales on laptop:
  20k=54s, 50k=145s / 3.2 GB. Test [17]. **17 tests green.**

Remaining for 10^6 (HPC): int32 CSC row-indices (halve RAM when P<2^31); optional full
C++ session object (registry + loop in C++, Python passes only index arrays); the
`global_trim` CSC over ~4×target survivors is the RAM peak — chunk / int32 it.

## 2026-07-01 — dynamical (convergence-driven) core + honest convergence study
Reran L=1→4 in 3d with a **higher, convergence-driven core budget** to see how close
to the true GS we get. `run_cpp.l_scaling_sweep` is now an **independent ladder**: for
each L it solves from scratch at core sizes 500→1000→2000→4000 (each a full ensemble
solve), growing until the drop between *independent* solves |E(2N)−E(N)|/|E| < target
or the ceiling `max_n_dets`. **HPC switch** `hpc=True` raises the ceiling to
`hpc_max_n_dets` (default 50000); laptop default 4000. Five vs-L plots incl.
`convergence_vs_L` (reliable E-vs-N per L) and `coresize_vs_L`. Label `Lladder3d`.

**Two methodology bugs found + fixed (both would have given false "converged"):**
1. **Single-run convergence signals lie here.** A single growing TrimCI run's ramp
   *plateaus deceptively* (the selected-CI finds new important high-boson determinants
   in bursts), so per-round marginal-dE AND a 1/N tail extrapolation both falsely
   declared convergence (e.g. "0.2% from GS" at a core still 1.6% high). Only comparing
   **independent full solves at doubling core sizes** is reliable. Removed the
   extrapolation; `graph.halving_drop` kept as a helper but the driver uses the ladder.
2. **`ground_state(max_rounds=12)` silently capped the core** at ~3482 dets (the x1.5
   ramp from n_init=20 tops out at round 12), so a requested n_dets=4000 rung only
   reached 3482 — badly understating both the energy and the convergence. The driver
   now passes `max_rounds` sized to reach the requested rung. Reaching a real 4000 vs
   3482 dropped L=2's energy 2415→2383 MeV (**1.5%!**), showing 2000-det results were
   meaningfully unconverged.

**Findings (dim=3, A=1, N_f=4, cores to 4000, n_runs=3):**

  | L | sites | core | E0 (MeV) | last-2x drop | runtime |
  |---|-------|------|----------|--------------|---------|
  | 1 | 1  | 94 (exact) | 204.86   | —      | 0.1 s  |
  | 2 | 8  | 4000       | 2382.84  | 1.5%   | 36 s   |
  | 3 | 27 | 4000       | 9378.92  | 0.73%  | 352 s  |
  | 4 | 64 | 4000       | 24280.57 | 0.36%  | 2084 s |

  - **Convergence is SLOW and NON-SMOOTH.** At the laptop ceiling (4000) NONE are
    tightly converged — the last core-doubling still moves E0 by 0.4–1.5%. Tight
    convergence needs much larger cores (10k–100k) ⇒ HPC (the switch).
  - **"Core grows with L" is NOT supported at this scale** (hypothesis corrected): all
    L≥2 saturate the ceiling because none converge below it, so `coresize_vs_L` is flat
    at 4000.
  - ~~relative drop decreases with L ⇒ converges better~~ **RETRACTED 2026-07-01** (see the
    L=2 core sweep below): the cross-L last-doubling drops (1.5%/0.73%/0.36% at core 4000)
    are a **bursty-convergence sampling artifact**, not a real trend — L=2's 1.5% at 4000 is a
    burst; at 8000 its drop is 0.30%. Do NOT claim "convergence improves with L" without a
    full core sweep per L.
  - **Cost-confront implication:** the classical selected-CI baseline for these
    strongly-bosonic systems needs many determinants; the earlier fixed-n_dets=2000
    energies (`Lconv3d`) are ~0.4–1.5% high. Runtime ~ sites² still holds; L=4 to 4000
    took 35 min, L=5 predicted ~2 h (HPC). Possible next lever: broaden the expansion
    (pool_factor, more diverse boson init) so the core reaches new determinants faster.

### 2026-07-01 (later) — L=2 core-size convergence sweep (`core_convergence_sweep`)
Single-system study: L=2 (dim=3, 8 sites, sector 9e15) solved INDEPENDENTLY at cores
500→1000→2000→4000→8000→16000 (n_runs=3; `run_cpp.core_convergence_sweep` +
`plotting.plot_core_convergence` → `data/classical/core_convergence_L2core3d.png`, ~5 min).
**KEY FINDING: E0 is STILL falling at 16 000 dets — no plateau.** E0 (MeV): 2433.9 → 2425.9
→ 2419.0 → 2382.8 → 2375.7 → **2353.6**; total drop 2000→16000 = **65 MeV ≈ 2.7%**, steepest
at the end. Per-doubling rel. drop **bursty/non-monotone**: 0.33, 0.29, **1.5**, 0.30,
**0.94** % — selected-CI reaches important new determinant classes in jumps. ⇒ (1) even the
smallest 3-D system needs ≫16k dets for tight convergence (the compact core is NOT compact for
accuracy — a real cost-confront result); (2) a fixed-budget energy is an *uncontrolled* upper
bound; (3) classical cost lever = determinants-for-accuracy, measured per system, and it's
large. This is the reliable convergence evidence (independent solves; single-run signals lie).

## 2026-06-30 — arbitrary mode count + L-scaling in 3d
- **Fermion mask is now arbitrary-width.** The C++ `MixedDet.ferm` went `uint64`
  → `__uint128_t` → **`std::vector<uint64_t>` of `W = ceil(n_ferm_modes/64)` words**, so
  there is **no cap on lattice size** (was 64 modes = 16 sites). All bit ops route through
  the `Ferm` typedef + `fbit`/`fflip`/`fpopcount_below`; masks cross the Python boundary as
  `(N, W)` uint64 arrays (`backend._states_to_arrays(states, n_words)`, `io` save/load).
  Validated bit-for-bit vs the Python reference at W=1,2,3,5 incl. masks entirely in high
  words (suite test `[14]`, `max|ΔH|=0`). The old 64-mode `runtime_sweep` guard is gone.
- **RAM safeguard** (the user runs on HPC next): the C++ connection cache is bounded by a
  **byte budget** (`max_cache_bytes`, default 1 GiB) as well as a state count, cleared
  wholesale on overflow. At large L each cached state holds a Hamiltonian-sized `ConnMap`
  (n_terms grows with sites), so the byte budget — not the count — is the real guard. Set
  `H._cpp_cache_bytes` to raise it. (L=3 d=3 hit ~639 MB, bounded fine.)
- **L-scaling sweep in 3d** (`run_cpp.l_scaling_sweep`, A=1, n_b=2, n_dets=2000, n_runs=4):
  L=1→5 solved in one pass with a power-law runtime gate (run the next L only if its
  extrapolated time < budget) and a **per-L n_dets convergence check** (label `Lconv3d`):

  | L | sites | modes | terms | sector | E₀ (MeV) | runtime | conv dE (rel) |
  |---|-------|-------|-------|--------|----------|---------|---------------|
  | 1 | 1   | 4F/1w + 3B    | 80    | 2.6e2   | 204.86   | 0.1 s   | 0 (exact)     |
  | 2 | 8   | 32F/1w + 24B  | 1265  | 9.0e15  | 2419.34  | 5.9 s   | 0.56 (2e-4)   |
  | 3 | 27  | 108F/2w + 81B | 4897  | 6.3e50  | 9447.47  | 43.8 s  | 6.2 (6e-4)    |
  | 4 | 64  | 256F/4w +192B | 12353 | 1.0e118 | 24370.92 | 311.5 s | 9.5 (4e-4)    |
  | 5 | 125 | 500F/8w +375B | 25001 | 3.0e228 | 50065.50 | 1109.5 s| 3.3 (7e-5)    |

  **Headline: runtime ~ sites² (fit sites^2.02, holds across L=1→5); a 3×10²²⁸-state sector
  (L=5) solves in 18.5 min with a 2000-det core → ~10²²⁵× compression.** L=4/L=5 predictions
  (299 s/1192 s) matched measured (311 s/1110 s). Next L=6 (216 sites) predicted **~58 min** —
  the local→HPC handoff. Plots `data/classical/{runtime,compression,energy,convergence}_vs_L_
  Lconv3d.png`. E₀/site rises 205→401 MeV toward the thermodynamic value.
- **Convergence-check semantics (caveat).** The default check (`n_conv=1`) saves the
  production solve's free core-ramp history (core ~20→n_dets) as the convergence curve; the
  reported `conv dE` is the *marginal* last-step drop, tiny relative to E₀ (7e-5…6e-4). But
  the convergence plot shows the curves still descending at core≈2000 (residual ~50–130 MeV
  ≈ 0.2–0.5% of E₀ at L=4,5) — so n_dets=2000 is a variational upper bound still a fraction
  of a percent above the true GS at large L. For a rigorous claim use `n_conv>1` (independent
  ensemble re-solves at a geometric n_dets ladder — the conservative check; ~1.5–1.8× the
  wall-clock) on HPC, and/or push n_dets up. `plot_convergence_vs_L` reads either curve.

## 2026-06-29 — performance + scaling + data pipeline (today's pass)
- **C++ connection cache** added to `MixedHijProvider` (bounded, cleared-on-overflow,
  `cache_cap` param). Correct (bit-for-bit preserved) but ~no speedup on the C++ path:
  `connections` is no longer the bottleneck (it was in pure Python). Kept — it's free
  insurance and scales with #terms (helps at larger L); and it enables the 2-pass
  embedding builds. **Profile finding:** the costs are Davidson (~30%), scipy matrix
  assembly (~30%), Python orchestration (~30%). Moved the 2N complex→real embedding into
  C++ (`build_real_embedded_coo`) to drop the complex CSR + Hermitization + bmat → 3.8→3.6 s.
  (A direct-CSC build was tried but is slower — unsorted indices hurt the Davidson matvec;
  scipy's COO→CSR sort is worth it.)
- **A is NOT size-invariant** (correcting the "same-size blocks" intuition): fermion number
  IS conserved and built in (the solver only generates A-fermion dets and H preserves
  number, so it stays in the A-block), but C(n_ferm,A) differs (8/28/56/70 for L=2) AND
  more nucleons activate more terms → denser connectivity → time grows mildly (0.7→1.6 s,
  A=1→4). Same order, not constant.
- **N_f is ~free** (selected-CI): N_f=4→512 all ~1 s, energy converged by N_f≈16.
  Opened N_f>256 via uint16 bosons. **Safe limit: n_b ≤ 16 (N_f ≤ 65536) on the C++ path**;
  physically n_b~5–11 suffices and convergence is by ~16–64, so any real run is fine. n_b>16
  → pure-Python path (you'd never need it; the Tong cutoff is far smaller).
- **Data pipeline** (`classical/io.py`, `analysis.py`, `plotting.py`; data in
  `data/classical/<date>/<run-id>/`): metadata.json (method, **transform axis COO|LF**,
  params, **wall-clock runtime**, energy, convergence, exact ref), Hamiltonian dump, and the
  compact ground state (npz). Method/transform are swappable for future solvers (DMRG/AFQMC)
  and the Lang–Firsov frame ([32](../../tasks/32-lang-firsov-compactification.md)).
  `run_cpp.py --save`. 13 tests green.
- **First runtime-vs-size sweep** (`run_cpp.runtime_sweep`, 11 systems, 26 s total): sector
  sizes **2.6e2 → 6.5e17** (15+ orders) solve in **0.05–5.4 s**. Plots
  `data/classical/runtime_vs_{size,sites}.png`. Headline: **runtime is ~linear in #sites and
  ~flat in N_f** — at 2 sites the sector grows 10^6× (N_f 4→32) with runtime constant ~1.1 s.
  So TrimCI runtime is poly-log in Hilbert dimension — the "where classical stays fast" curve
  for [14]. (`runtime_sweep` has a `max_seconds` budget guard; C++ fermion mask caps at 16
  sites = 64 modes.)

## What we learned from reading the TrimCI source (cloned at `../../../TrimCI`)

The released package (`pip install trimci`, repo `hao-zhang-quantum/TrimCI`) is a
**purely fermionic** selected-CI solver:

- Entry: `trimci.ground_state("FCIDUMP", n_dets=...)` — consumes a standard
  **FCIDUMP** (1-body `h1[n_orb²]` + 2-body `eri[n_orb⁴]`) or a PySCF mean-field.
- The C++ backend (`cpp/trimci_core/common/hamiltonian.hpp::compute_H_ij_t`)
  computes `H_ij` itself from those integrals via **Slater–Condon** on `(alpha,beta)`
  spin-orbital determinants. Determinants are bitstrings; orbital optimization,
  2-RDM, PT2 all assume a fermionic single-particle basis.
- **There is no bosonic mode and no custom-`H_ij` hook.** So we cannot just "hand
  TrimCI our H and its matrix elements" against the stock package — that input
  contract doesn't exist in the release.

⇒ Two viable routes (we're doing both; A first because it's already validated):

## Route A — our own TrimCI over the mixed `H_ij`  ✅ scaffolded + ED-validated

Implemented in this folder, driving the expansion–trim cycle over our generalized
mixed-state `H_ij` (`hij.connections`). Done so far:

- [x] `state.py` mixed Fock state; `hamiltonian.py` builds the full EFT `H` as a
      ladder-term list (reusing the quantum pipeline's `build_eft_hamiltonian`,
      fock+sparse path — so it's the *same* Hamiltonian).
- [x] `hij.py` matrix-free `connections` / dense builder; fermion signs match
      OpenFermion to 1e-14, boson ladder+cutoff exact (`tests/test_hij.py`).
- [x] `graph.py` expansion / local-trim / global-trim / `ground_state`; reproduces
      ED at `(L=1,d=1,A=1,N_f=4)` and the trimmed run hit the exact GS with 6/256 dets.
- [x] `dump.py` generalized-FCIDUMP writer/reader (round-trips exactly).

- [x] **Toy integration run** (`run_toy.py`): full pipeline
      build → write dump → **read dump back** → `ground_state` → compare to ED.
      Results (all variational, reaching ED to machine precision):
        - L=1,d=1,A=1,N_f=4 (256 states): ED at **2 dets** (0.8%)
        - L=1,d=1,A=2,N_f=4 (384 states): ED at **2 dets** (0.5%)
        - L=2,d=1,A=1,N_f=2 (512 states): ED at **18 dets** (3.5%)
        - L=2,d=1,A=2,N_f=2 (1792 states): ED at **64 dets** (3.6%) *(needs ensemble)*
- [x] **Ensemble strategy** (`ground_state_ensemble`): single-run TrimCI is
      seed-dependent and can trap in a local basin (found A=2/L=2: 2/6 seeds nailed
      ED, others stuck at +6.8 MeV). Min-over-N-random-inits escapes it, per the
      paper. Regression-tested (`test_hij.py` [5]).

- [x] **Memory-safety guards + perf pass** (code-reviewed). A prior run OOM-crashed
      the machine via an unguarded full-sector enumeration. Fixed:
        - `enumerate_basis` / `build_dense` now compute size up front and raise
          `MemoryError` past explicit caps (MAX_BASIS=1e5, MAX_DENSE=6000) instead
          of allocating. Verified the guards fire (L=2,N_f=32 would have been 8.6e9
          states / ~1.7 TB).
        - perf: memoized `connections` (per-H bounded cache), `math.sqrt`/`bit_count`
          in the ladder hot loops, `heapq.nlargest` for top-k selection, hoisted the
          bigint `_sector_size` out of the round loop.
- [x] **Sparse full-space Lanczos** (`lanczos.py`, `lanczos_ground_state`). The exact
      reference, scaled past dense ED: builds a sparse Hermitian CSR via
      `connections_nocache` and runs `scipy ... eigsh`. Memory-guarded at every
      allocation (basis size, projected nnz, projected GB). Matches dense ED to 2.8e-13
      (test [6]); solves a **32,768-state** system dense ED refuses, in ~4 s / 0.01 GB,
      where TrimCI(200 dets, 0.6%) lands within 4e-4 MeV from above. `run_toy.py` now
      uses it as the reference.

- [x] **Pushed Lanczos to real N_f** (`nf_convergence.py`). Build made memory-lean
      (direct CSC into a growable buffer, bounded by `max_nnz`; raised caps to
      max_states=4e5 / max_nnz=4e7 / 3 GB). Findings:
        - **Single site (L=1), real N_f=32 = 131,072 states solved in ~14 s / <1 GB.**
          Energy converges at N_f=2 (the only on-site coupling, H_WT ~1e-3 MeV, is far
          below the m_π gap → pion stays in vacuum). Heuristic N_f=32 hugely over-padded.
        - **L=2 (gradient coupling H_AV active): N_f matters** — E0 drops 4.24 MeV from
          N_f=2→4. Exact reach ceiling is N_f=4 (32,768 states); N_f=8 needs 2.1M
          (`N_f**6` blow-up) — beyond exact methods. *This is the TrimCI regime.*
        - **TrimCI ✓ vs Lanczos** at L=2,N_f=4: matches the exact 461.31446 to 3e-6 MeV
          with 500 dets (1.5% of the sector). TrimCI then *extends the curve past the
          exact wall*: L=2 E0(N_f) = 465.553 (N_f=2, exact) → 461.314 (N_f=4, exact) →
          461.277 (N_f=8, TrimCI, 800/2.1M dets, ~0.5 GB) — converging geometrically.
          TrimCI never enumerates the full sector, so its memory stays bounded even at
          the 2.1M-state N_f=8. (this N_f-convergence is the empirical check for the
          Tong-cutoff bound, task 25.) Caveat: pure-Python TrimCI at N_f=8 took ~16 min
          — the boson-aware init + matrix-free items below are the fix.

Next on Route A:
- [ ] Bigger L/A: L=2 caps at N_f=4 exactly; push the TrimCI-vs-Lanczos crossover into
      the regime that is the classical-baseline story for [14]. Pin TrimCI accuracy at
      N_f=8/16/32 (L=2) where only TrimCI reaches.
- [x] **Boson-aware (near-vacuum) random init** — `random_core` now samples each pion
      mode from a truncated-geometric P(n) ∝ p^n (mean `boson_init_mean`, default 0.5),
      so higher occupations are exponentially less likely; vacuum anchor kept. At N_f=8
      the init mean occupation is 0.88 vs 3.61 uniform. Payoff: same L=2,N_f=8 energy
      (461.277, marginally better) in **373 s vs 979 s — 2.6× faster** (starts near the
      weakly-dressed GS instead of the high-occupation cloud). Tunable via `boson_init_mean`.
- [ ] **(now the top perf wall)** pure-Python TrimCI is still slow (N_f=8 ~6 min after the
      2.6× init speedup). Levers: a selected-CI Davidson matvec over `connections`
      (O(N)-vector memory), vectorizing the `apply_term` hot loop, and — past ~1e5 states
      — a matrix-free `LinearOperator` Lanczos (no stored CSR).
- [ ] Boson-aware **expansion** tuning — boson number isn't conserved; check the
      random core / pool spread over boson sectors rather than collapsing to vacuum.
- [ ] **Independent cross-checks** beyond ED/Lanczos (the user's ask): DMRG via `block2`
      (general fermion-boson MPO, near-exact 1D), and a stochastic reference
      (CP-AFQMC for coupled e-ph, or NQS) for the 3D target where DMRG fades.
- [x] **PT2 correction + extrapolation (à la SHCI)** for a tighter energy at fixed
      dets — `pt2.py` (Epstein-Nesbet) + `extrapolation.py` (power-law + SHCI/PT2),
      driver `run_cpp.solve_and_report`. See the 2026-07-07 entry.
- [x] **C++ port of the PT2 `connections` pass** for large-core production PT2 —
      `MixedProvider.pt2_accumulate` + `backend.cpp_pt2_external`, auto-detected by
      `epstein_nesbet_pt2`. 17–47× on pass-1 (grows with #terms); see the 2026-07-08
      note under Task 2 above. Unblocks PT2 at L≥4 / larger cores.
- [x] **Per-site (size-intensive) accuracy + per-L converged reference** — the
      extensivity-trap groundwork. `report_energies(sites=...)` per-site mirror;
      `l_scaling_sweep(gate=...)` relative-vs-per-site convergence switch;
      `run_cpp.converged_reference(L)` → E_inf(L)±σ with honest point-vs-bound pinning.
      See the 2026-07-09 entry. Tests [35][36].
- [x] **Phase C — dets-vs-L exponent at fixed per-site accuracy (the paper's central
      number).** `dets_vs_L_at_fixed_accuracy` (N*(L; eps) from the per-L references +
      exp-vs-poly volume fit, honest point/bound), plot `plot_dets_vs_L`, driver
      `misc/run_dets_vs_L.py`. See the 2026-07-09 (Phase C) entry. Test [37]. Laptop
      L=2–4 dilute run done (all bounds — needs HPC); fixed-filling companion next.
- [x] **Phase D — honesty & robustness guards.** `robustness.py` (ladder monotonicity,
      PT2 semistochastic trigger, scatter), `seed_robustness` driver (found the n_runs=4
      ensemble SEED-FRAGILE at laptop cores: 6.8 MeV/site spread), diagnostics wired into
      the reference / dets-vs-L outputs. See the 2026-07-10 (Phase D) entry. Test [38].

## Route C — HYBRID: our selection + official C++ Davidson  ✅ built + ED-validated (2026-06-29)

Built the official TrimCI (`pip install /path/to/TrimCI`; cmake comes via pip build
isolation, Eigen + robin_hood vendored, OpenMP optional). `classical/trimci/backend.py`
offloads the **diagonalization** (our `O(N_dets^3)` dense-`eigh` bottleneck) to the
official C++ Davidson, keeping our Python selection. `ground_state(..., diag_fn=backend_diagonalize)`.

**What plugs in vs what doesn't (the issue to discuss):**
- ✅ **`davidson_solve_dense(H, diag, ...)` takes an ARBITRARY matrix** — the one real hook.
  We feed it the projected H of any selected det set. **4×–15× faster than numpy `eigh`**
  for the lowest eigenpair at N=512–5000 (grows with N: eigh does all eigvals, Davidson
  only the lowest). Validated: reaches ED to ~1e-12 on the toy.
- ⚠️ **Real-only.** Our H is complex Hermitian (the WT term carries an `i`). We use the
  2N real-symmetric embedding `M=[[ReH,-ImH],[ImH,ReH]]` (spectrum preserved; complex
  eigvec reconstructed). Works, but doubles the dense dimension — a complex Davidson in
  the backend would remove that 2× (worth asking the team).
- ❌ **Selection is NOT injectable.** `pool_build`, `run_trim`, `run_expansion`,
  `generate_excitations`, and even the *matrix-free* `davidson_solve_matfree` are all
  hardwired to fermionic Slater-Condon over `(Determinant, h1, eri)` — **no custom-H_ij
  / provider hook in the exposed API.** So our mixed-state selection can't be offloaded
  to the C++ engine without a fork. The "nodes-have-H_ii, edges-have-H_ij, plug in"
  vision works for the *eigensolver* but not the *selection kernel* as the API stands.

**Net:** the hybrid gives a real, modular speedup on the heaviest single step
(diagonalization), today. Full offload (the matrix-free selection + Davidson over our
`connections`) needs either the Otten group's native mixed-mode generalization (which
presumably exposes exactly the provider interface we want) or a contained fork that
swaps the Slater-Condon-from-FCIDUMP matvec/excitation kernel for our callback.

**Minimal fork — Tier 1 BUILT + tested (`backend_fork/`).** Added one ~35-line pybind
entry `davidson_solve_sparse` (CSR matvec over the same C++ `davidson_solve`; no CMake
change) so the official Davidson runs at **O(nnz) memory** on our sparse mixed H (built
via `connections`, never densified — removes the dense `2N`-embedding wall). Validated to
ED (5e-13); at N=8000 ~0.01s where the dense path would need 2 GB.
`ground_state(..., diag_fn=backend_diagonalize_sparse)`. Patch in `backend_fork/`.

**Tier 2 — connections C++ port + FULL C++ HOT PATH WIRED + WORKING (2026-06-29).**
`backend_fork/mixed_ci.hpp` ports `connections` (validated bit-for-bit, max|ΔH_ij|=0, ~19×);
extended with `build_coo` (matrix build) and `expand` (neighbor scoring), so all three hot
kernels run in C++ and Python only orchestrates (ranking/set-ops). Diagonalization = official
C++ sparse Davidson (Tier 1). `cpp_ground_state[_ensemble]` via `graph`'s diag_fn/expand_fn
hooks; driver `run_cpp.py`. **55× faster than pure Python** (0.9 s vs 48 s, L=2 N_f=4 toy);
**larger systems in seconds**: L=2 N_f=8 (2.1e6) 1.5 s, L=3 d=1 (3.1e6) 2.5 s, **L=2 d=2
(2.7e8 states) ~5 s**. Validated to ED to 1e-12 (`run_cpp`; suite [12]). The deep run_trim
fork is now unnecessary for our goal.

**Code review applied (2026-06-29):** Davidson `residual_tol=tol` + `verbose=0` (tighter +
quiet hot loop); `N_f ≤ 256` guard before uint8 boson packing; int64 COO indices (scaling
ceiling); splitmix64 `MixedDetHash`; vectorized `cpp_expand` rebuild via `.tolist()`.
**Top deferred follow-up: a bounded C++ connection cache in `MixedHijProvider`** (the Python
path caches; the C++ hot path recomputes per column) — highest-leverage remaining perf win
for the iterate-to-convergence workflow. Also follow-up: degeneracy makes the *selected set*
solver-dependent (energy unaffected); document.

- [ ] Push the hybrid to large `N_dets` (where the 4–15× diagonalization win compounds
      over many trim rounds) — needs the compiled `apply_term`/`connections` (else the
      Python selection becomes the new bottleneck).
- [ ] Ask the team: (a) a complex-Hermitian Davidson (drop the 2× embedding); (b) a
      custom-matvec / provider hook for the selection kernel.

## Route B — fermionize the bosons into a standard FCIDUMP, use stock TrimCI

Gets us the C++ speed, orbital optimization, and PT2 for free — but needs an
encoding that turns each pion mode (Fock levels `0..N_f-1`) into fermionic
orbitals so the whole `H` becomes a 1-/2-body fermionic operator.

- [ ] Decide the boson→fermion encoding. Leading candidate: **unary / one-hot per
      mode** — `N_f` orbitals per pion mode, exactly one occupied = the Fock level;
      `a, a†` become nearest-level hoppings (with `√n` weights as one-body integrals),
      `H_WT`/gradient become two-orbital (two-body) integrals across mode-blocks.
      Nucleon number `A` + one fermion per pion-mode block ⇒ total `NELEC` and the
      per-block constraint is auto-preserved by the hoppings.
- [ ] Work out the **spin-orbital ↔ (alpha,beta) spatial** convention TrimCI expects
      (Hubbard FCIDUMP uses spatial orbitals × spin; our 4 nucleon spin-isospin modes
      + boson orbitals likely need a generalized/GHF-style FCIDUMP). Check what
      `py/trimci/api.py` header parsing + the C++ det layout actually assume.
- [ ] Extend `dump.py` with `write_fermion_fcidump(...)`: emit `h1`,`eri` (chemist
      notation) for the fermionized `H`. Validate spectrum vs our Route-A `H_ij`.
- [ ] Run stock `trimci.ground_state(FCIDUMP, n_dets=...)`; compare energy + core
      to Route A on the same toy. (Cost of the unary blow-up: `N_f` orbitals/mode —
      watch `n_orb`.)
- [ ] *Maybe* instead: fork the C++ core to add native bosonic excitation
      generation. More work; only if the unary FCIDUMP route proves too costly.

## Open questions (some for the user / Otten — in-group)

- Stock TrimCI has no custom-`H_ij` callback. Is one planned, or is a bosonic /
  mixed-mode extension on their roadmap? That would make Route B unnecessary.
- For the unary fermionization: does TrimCI tolerate a generalized (non-spin-restricted)
  FCIDUMP, and large `n_orb` with very sparse integrals (it advertises lattice/Hubbard
  sparsity, so probably yes)?
- Which route is the one we actually publish the classical baseline from? Route A is
  cleaner physics (no encoding artifact); Route B borrows their tuned, fast, PT2-capable
  engine. Likely: Route A for correctness truth, Route B for reach.

## Pointers
- Cloned source: `/Users/brandonfriend/Desktop/Projects/TrimCI` (outside NuQu).
- Hamiltonian form + coefficient tensors: `claude/research/trimci/01_hamiltonian_form.md`.
- Lit review: `claude/research/trimci/00_literature_review.md`.
- Tracker: `tasks/31-trimci-classical-solver.md`.
