# NuQu Remediation & Regeneration Plan

Response to the Codex publication-readiness audit (`codex_audit/`, 2026-08-18).
Owner: user (reviewer). Branch: **`remediation/vertex-fix`** (off the campaign branch, so it
carries the binding/HPC infra). All data generated before the vertex fix is **invalid** and
must be regenerated — see Workstream A.

## Scope decisions (locked 2026-08-18)

| Fork | Decision | Consequence |
|---|---|---|
| Sparse oracle (P0-2) | **Build the compiled sparse RE** | Implement `SparseFullBundleBlockEncoding`, cost via pyLIQTR `estimate_resources`. Real circuit-level number, valid sparse-vs-PauliLCU. (Task 26) |
| Amplitude basis (P0-4) | **Fix the composed encoding** | Build a real `H_pos+H_mom` sum block encoding (or costed product formula + error budget), incl. the H_WT species-selective basis-change circuit. Keeps Fock-vs-field-amplitude comparison valid. |
| Tong cutoff (P0-3) | **Complete the rigorous bound** | Full Theorem-6-style derivation resolving the H_WT obstruction, constants retained. (Task 25) |

All three are max-rigor. To keep the paper's spine unblocked by the slow theory/encoder
work, the workstreams are sequenced so the **critical path (Workstream A) unlocks the
core results on its own**; B, C1, C2 extend the claim set.

---

## Current status & handoff (2026-08-18)

Commits on `remediation/vertex-fix` (newest first):
`9bc2e7a` C1 design done · `644d94f` C scoping + C3 anchor · `11beb50` B `tong_rigorous` ·
`d532857` A3 + cross-builder tests · `9404fac` A vertex fix + oracle suite.

| Workstream | State |
|---|---|
| **A** — Hamiltonian correctness | ✅ **DONE.** Vertex fix (3 builders) + physics-oracle suite (`tests/test_vertex_algebra.py`: Kronecker oracle for all 12 σ⊗τ channels, cross-builder equivalence, `[H_WT,n̂_x]=0`) + `a_L**dim`. **Full suite 143 pass.** |
| **B** — Rigorous cutoff | ✅ **CODE DONE.** `boson_cutoff_method='tong_rigorous'` (exact Bogoliubov tail + variational bound), `classical/trimci/gaussian_cutoff.py`, tests. Draft `claude/research/bosonic-encodings/05_*.md` — audited & **rescoped honestly** (2 rigorous results; 2 open gaps: bound `‖Vψ‖` via Gaussian moments, control `|δ_true−δ_Gauss|`). **For user + Codex proof review.** `'heuristic'` stays default. |
| **C3** — PauliLCU anchor | ✅ **DONE.** Genuinely circuit-level on corrected H (L=2,dim=1,n_b=2 → Λ=3077, Walk_T=214724, 31 qubits, 1.3s). Small-L validation anchor; ceiling is QubitOperator memory. |
| **C1** — compiled sparse RE | 📐 **DESIGN DONE** (`docs/sparse_full_bundle_design.md`), **implementation TODO — this is the next major build.** |
| **C2** — amplitude composed encoding | ⬜ scoped, not started (2nd large build). |
| **C4** — runtime bands | ⬜ scoped, not started (medium). |
| **D** — paper cleanup | ⬜ mostly parallel/low-compute; sign-problem writeup done (`docs/sign_problem.md`). |
| **Phase 2/3** — regeneration | ⬜ not started; needs the corrected H + (for quantum) C1/C2. HPC, launch-approval loop. |

### ▶ Recommended action for the C-focused session (start here)

**Implement C1 step-by-step from `docs/sparse_full_bundle_design.md`**, in its gated order —
do **not** build it all at once; each step has a validation gate:

1. `SparseBosonMonomialBlockEncoding` (d=1 atom; cases 3a linear / 3b number-op / 3c-direct
   two-mode) → **gate:** per-atom block-matrix sim (`α_l·⟨0|U_l|0⟩` vs exact monomial matrix,
   n_b=2). Lowest risk; reuses the proven `single_ladder.py` machinery.
2. Fermion atom via off-the-shelf `PauliStringLCU(MyCustomHamiltonian(jordan_wigner(fermion_part)))`
   → **gate:** term-for-term vs standalone PauliLCU `estimate_resources`. This removes the
   fermion LOWER-bound proxy.
3. `BundleSelect(UnaryIterationGate)` + `SparseFullBundleBlockEncoding` → **gates:** α_tot
   invariant (`be.alpha == compute_native_lambda(...)['physical_lambda']`) **and** the scaled-toy
   assembly sim (≤~12 qubits). **These retire the #1 correctness risk (reflection subspace / α).**
4. Compiled-vs-analytical A/B at L=2 dim=1 then dim=3 (n_b=2); report the ratio + boson/fermion/
   mixed breakdown — that A/B *is* a publishable "cost of honest compilation" number.
5. Only then flip the Config default; extend the pyLIQTR cache to the compiled composite.

**Guardrails for the C session:**
- **Highest risk = reflection subspace / α_tot** (design §7.1): the walk reflects about the flag
  subspace; get it wrong and every Λ/T-count is *silently* corrupt. The α_tot invariant + toy
  sim (step 3) are the gates that catch it — do not skip them.
- **Env is pinned:** pyLIQTR 1.3.4 / Qualtran 0.4.0 (no composite block-encoders → all custom
  bloqs from primitives). Do not plan around a Qualtran upgrade.
- **Comparison-switch discipline** (CLAUDE.md): keep the analytical `estimate_sparse_resources`
  alive as the A/B baseline; gate the compiled path behind a `compiled` Config switch, default
  analytical until validated.
- **No heavy local compute** (laptop crashes twice on record): validation sims stay tiny
  (n_b≤2, ≤~12 qubits); anything larger → cluster.
- After C1: **C2** (composed `H_pos+H_mom` encoding + Watson Eqs. 102–104 species-selective QFT),
  then **C4** (runtime bands). Then Phase 2/3 regeneration.

**Not blocking C:** the B proof gaps are the user's/Codex's to review; A is the foundation
(worth a look before Phase-2/3 regeneration burns cluster time).

---

## The bug, precisely (P0-1, verified first-hand)

`_nucleon_transition_fermion` (`fock_native.py:63`), `_nucleon_transition_jw_uncached`
(`Operators.py:58`), and the 1D twin (`Operators1D.py:96`) each return
`a†_α a_β + a†_β a_α`. **Every caller already loops over all 16 ordered (α,β) pairs**
(`fock.py:355`, `amplitude.py:181/234`, `fock_native.py:175/214`). Result:

- real/symmetric channels **doubled**;
- imaginary/antisymmetric channels (τ_y; σ_y·τ_y) **cancel to zero**.

**Fix:** each builder returns only the single ordered `a†_α a_β` (drop `+h.c.`). The
caller loops then reconstruct `Σ χ^I_{αβ} a†_α a_β`, which is automatically Hermitian
(χ = τ_I ⊗ σ_S is a Hermitian Pauli tensor). No caller changes; cache stays valid.

**Contamination:** everything touching H_AV or H_WT — all classical energies/frames/
binding, all λ/T-count/runtime, all cutoff studies (~40 campaign dirs). **Static-only
sector is clean** (free-nucleon hop, H_C, H_CI2, free pion — Codex-verified vs Watson).

---

## Workstream A — Hamiltonian correctness (CRITICAL PATH; gates all data)

Fast (days, local). Nothing downstream is valid until this lands.

- **A1. Vertex fix** in the 3 builders: return single ordered `a†_α a_β`; rename to
  "ordered bilinear"; fix docstrings (`Nucleon_Transition_JW` no longer "+h.c.").
  Decide fix-or-retire on the legacy 1D path (`Lattice1D/`).
- **A2. Physics-oracle test suite** (`tests/`) — the real deliverable (tasks 05, 10):
  - Term-by-term matrix oracle for **every** σ_S ⊗ τ_I channel incl. all imaginary
    ones, vs. an independently built Kronecker matrix. Absorb Codex's
    `test_operator_bilinears.py` (fixed).
  - Full-H Hermiticity; fermion-number conservation on corrected vertices.
  - Zero-coupling limits: g_A→0 kills H_AV; WT prefactor→0 kills H_WT; recover analytic
    free-nucleon + free-pion spectrum.
  - Cross-builder equivalence: `fock.py` (PauliLCU) ≡ `fock_native.py` (sparse) post-fix;
    amplitude ≡ Fock in the limit where they must agree.
  - Small **ED anchor** (L=1–2, tight cutoff) as TrimCI ground truth (task 05).
- **A3. Consistency fixes:** `a_L**3`→`a_L**dim` in the amplitude free-pion prefactor /
  cutoff formulas (or lock dim=3 and document).

**Exit gate:** all A2 tests green; ED anchor matches an independent hand-built matrix.

---

## Workstream B — Rigorous cutoff theory (task 25; parallel, theory)

Deliver an explicit `N_f(ε, t, L, A, couplings)` bound that **resolves the H_WT
obstruction Watson flags after Lemma 5**, with constants retained. Interim: run the
empirical `N_f→N_f+1` energy/weighted-tail convergence study (also needed as the
cross-check) so production isn't blocked while the proof matures. Reconcile the
production N_f against the proven bound once it lands.

**Exit gate:** checked theorem with hypotheses mapped term-by-term to the finite-volume
H; production N_f justified by proof (and consistent with empirical convergence).

### Workstream B — status (2026-08-18)

- **Derivation draft** (`claude/research/bosonic-encodings/05_rigorous_cutoff_persite_number.md`,
  gitignored). Two results rigorous + verified: (1) H_WT conserves per-site-total pion number
  (ε-antisymmetry; `[H_WT,n̂_x]=0`) → dissolves the *specific* Watson H_WT obstruction; (2) the
  exact variational eigenvalue inequality `(★)` (replaces `02`'s 2nd-order-PT). Adversarially
  audited (proof-auditor) → **draft rescoped**: the certificate route is the spectral bound
  `(★)` + exact Gaussian tail (covers all terms incl. H_WT *without* Tong), NOT Tong T6 on the
  full H (the gradient squeezing is a 2nd Δλ=±2 obstruction). **Two gaps to a certified bound:**
  (i) bound `‖Vψ‖` via Gaussian 4-point moments; (ii) control `|δ_true − δ_Gauss|`. User +
  Codex to review the proof.
- **Code:** `classical/trimci/gaussian_cutoff.py` (exact Bogoliubov tail + `(★)` solver);
  `boson_cutoff_method='tong_rigorous'` switch in `estimate_boson_cutoff` (dim-general;
  rigorous-modulo-approx); `'heuristic'` stays default. `n_q=4–5` across L=2…10.
- **A3 remainder (dim consistency):** the `'tong_rigorous'` path is dim-general by construction;
  the Watson-3D `calculate_dynamic_cutoffs` is now **guarded** (warns for `dim≠3`) rather than
  given a wrong mechanical `a_L**3→a_L**dim` substitution.
- **Tests:** `[H_WT,n̂_x]=0` regression; `tong_rigorous` single-digit n_q / ε-monotone /
  dim-general / Gaussian-tail-monotone; Config round-trip.
- **Deferred to Phase 2 (cluster):** empirical `N_f→N_f+1` ED convergence on corrected H.

---

## Workstream C — Quantum resource-estimation validity

- **C1. Compiled sparse full-bundle** (task 26): implement `SparseFullBundleBlockEncoding`
  (per-term encoders for n̂-shaped and multi-mode monomials — the C3d.2/C3d.3 items in
  `sparse_oracle/resources.py`), cost via pyLIQTR `estimate_resources`. Cross-check vs
  PauliLCU on small instances. Retire the mixed-bound analytical proxy as the headline.
- **C2. Amplitude composed encoding** (P0-4): build the `H_pos+H_mom` sum block encoding
  (or a costed product formula with a stated simulation-error budget), incl. the H_WT
  species-selective QFT / term-controlled basis-change circuit (Watson Eqs. 102–104).
  Only then is amplitude-vs-Fock apples-to-apples.
- **C3. PauliLCU anchor** (already circuit-level via pyLIQTR): the small-L *validation
  anchor*, NOT the L=10 vehicle. Fock-basis term count grows ~`O(L³·f(n_b))`, so brute-
  force materialization as openfermion `QubitOperator`s hits a memory/runtime cliff
  (observed locally). The cluster buys only ~linear headroom — it does not reach L=10.
  Use PauliLCU where tractable (pin the empirical ceiling in Phase 2; expect ~L=3–4) to
  cross-check the compiled sparse numbers term-for-term. Two escapes if a larger PauliLCU
  point is wanted: (i) compute λ + PREP/SELECT costs analytically from one unit cell ×
  site multiplicity instead of enumerating terms; (ii) rely on C1 for scaling.
- **C1 carries the L=10 scaling headline** — it composes per-term/per-mode costs without
  materializing a giant Pauli operator, which is exactly why the sparse path was built.
- **C4. Runtime as scenario bands** (task 30) + error budget (task 12): present
  optimistic/pessimistic bands over cycle time, physical error, reaction latency,
  factory rate/count, routing, total failure probability. No single "2-day" number until
  a scheduler/factory model closes it.

**Exit gate:** every reported quantum number is either pyLIQTR-compiled (PauliLCU, sparse)
or a clearly-scoped scenario band; no mixed upper/lower "bounds."

### Workstream C — implementation scoping + status (2026-08-18)

**Current state of the code (audited):**
- The single-mode `(â+â†)` sparse encoder IS a compiled pyLIQTR `BlockEncoding`
  (`SparseSingleLadderBlockEncoding`: real Cirq `decompose_from_registers` + Qualtran
  `_t_complexity_`). ✓
- The **full-bundle** sparse number (`resources.py::estimate_sparse_resources`) is
  **analytical** (Gilyén Lemma 30 + LCU over per-term single-mode costs) and mixes a boson
  UPPER bound with a fermion LOWER bound → the P0-2 defect. This is what C1 must replace.
- PauliLCU (`estimators.py`) IS genuinely circuit-level via pyLIQTR `QubitizedWalkOperator`
  + `estimate_resources`. ✓

**C3 (anchor) — DONE.** Verified on the corrected H: L=2,dim=1,n_b=2 → Λ=3077,
Walk_T=214724, 31 logical qubits (938 Pauli strings, 1.3s). Estimation is fast; the ceiling
is QubitOperator build/memory (term count ~ O(L^d · f(n_b))). Use PauliLCU as the small-L
validation anchor; pin the empirical ceiling in Phase 2 on the cluster.

**C1 (compiled sparse full-bundle) — DESIGN DONE (`docs/sparse_full_bundle_design.md`),
implementation TODO (large build).** Concrete buildable spec from the quantum-algorithms
specialist: pyLIQTR 1.3.4/Qualtran 0.4.0 has no composite block-encoders, so all custom bloqs
from primitives (`UnaryIterationGate`, `SelectPauliLCU`, `StatePreparationAliasSampling`,
`ProgrammableRotationGateArray`, `AddK`). Highest correctness risk = the reflection subspace /
α_tot (retired by the scaled-toy assembly sim). 5 prioritized steps in the design doc. Build
`SparseFullBundleBlockEncoding(BlockEncoding)` with a real PREP/SELECT decomposition:
  1. Per-term boson encoders beyond the linear `(â+â†)`: number-operator-shaped monomials
     (`n̂`, H_grad diagonal), multi-mode products (H_WT's `a^{b†}a^c`), via Qualtran bloq
     composition of the single-mode encoder.
  2. Fermion JW-Pauli factors (H_AV/H_WT nucleon parts) via a pyLIQTR PauliLCU sub-encoder
     (replaces the current fermion LOWER-bound proxy).
  3. LCU PREP (alias-sampling over the `L_eff` term coefficients) + controlled SELECT.
  4. `_t_complexity_` = Qualtran-tracked composite (no mixed bounds); cross-check vs the
     analytical proxy (same order) and vs PauliLCU term-for-term on small instances.

**C2 (amplitude composed encoding) — TODO, large build.** Build the controlled block
encoding of `H_pos + H_mom` (or a costed product-formula with a stated simulation-error
budget), *including* the H_WT species-selective QFT / term-controlled basis-change circuit
(Watson Eqs. 102–104). Only then is amplitude-vs-Fock apples-to-apples.

**C4 (runtime bands) — TODO, medium.** Recast `physical_runtime.py` as optimistic/pessimistic
bands over cycle time, physical error, reaction latency, factory rate/count, routing, total
failure probability (task 30) + error budget (task 12).

**Note:** C1/C2 are the largest, most specialized pyLIQTR/Qualtran builds in the remediation;
they warrant focused execution (fresh context; `quantum-algorithms` agent for the circuit
design). The vertex fix changes the sparse/PauliLCU *numbers* (restored τ_y terms) but not
the encoder *structure*.

---

## Scientific consequences of the fix (surfaced during Workstream A)

- **Both H_AV and H_WT are sign-problem sources, not H_WT alone.** The bug had
  cancelled the imaginary τ_y channels, artificially making H_AV real. Corrected,
  H_AV alone gives ~18% complex off-diagonals (L=2, dim=1, n_b=2, A=2); H_WT alone
  ~21%; full ~27%; static-only 0%. The paper's sign-problem narrative ("H_WT is the
  unique phase source") must be revised: the *dynamical-pion coupling as a whole* is
  the phase source, with the static sector stoquastic-real. (`test_sign_structure.py`
  updated to the corrected claim.)
- Term counts of the assembled vertices grew (H_av 192→384, H_wt 1024→1536 at
  L=2,dim=2,n_b=2) as the previously-cancelled channels reappeared — expect λ, T-count,
  and every energy to shift accordingly.

## Workstream A — status (2026-08-18)

- A1 vertex fix: DONE in all 3 builders (`Operators.py`, `fock_native.py`,
  `Operators1D.py`); callers unchanged.
- A2 oracle suite: DONE — `tests/test_vertex_algebra.py` (Kronecker one-body oracle for
  all 12 σ_S⊗τ_I channels, imaginary-channel guard, native/JW cross-check, Hermiticity,
  full-H fermion-number conservation, zero/linear-coupling limits, and cross-builder
  equivalence: fock.py PauliLCU ≡ fock_native multiplied-out, dim=1 and dim=2). Bug-
  baseline tests updated: `test_fock_basis` counts, `test_sign_structure` claim.
- A3 `a_L**dim` consistency: DONE for the operator builder (`amplitude.py` free-pion
  vol_factor a_L**3 → a_L**dim, matching Fock; identical at dim=3). DEFERRED to
  Workstream B: the Watson cutoff formulas in `EFTParameters.py` (Eqs 75–78) still carry
  fixed a_L**3 — they are entangled with the rigorous-cutoff rework and are 3D-derived, so
  the dim-general form lands with B. `trotter_exact.py`/`trotter_theory.py` a_L**3 are
  Watson Table IX 3D constants — intentionally left.
- Full suite: 143 passed pre-A3; +4 cross-builder tests. Amplitude change is dim=3-
  identical (no production-data impact).
- **Workstream A COMPLETE** (Hamiltonian correctness). Small ED reference energy for
  TrimCI is a Phase-2 validation artifact, not an A code item.

## Workstream D — Paper / claims cleanup (parallel, low compute)

- Document open boundary conditions + forward-difference derivative (matches Watson).
- Frame/LF: quantify transform error vs expansion order & cutoff; carry as error bar;
  call LF a *variational effective* transform, not "exact/isospectral" (task 32).
- `2Theory.tex`: Fock ladder actions need `√n` / `√(n+1)`.
- `QPE.tex`: fix the sign/time/2π convention for `U=exp(iHt/Λ)`; repetition count is
  `~log(1/η)/p0`, not `1/p0`; add a ground-state identification/certification protocol.
- `Qubitization.tex`: λ = block-encoding subnormalization (not spectral norm / not always
  a Pauli one-norm). Derive the QPE walk-count constant from the actual estimator +
  failure probability, not a fixed `√2·π·λ/δE`.
- Density is `A/(4L³)` (filling) / `A/(L a_L)³` (number density), not `A/L`.
- Drop A>4L³ unphysical points from physics plots (label as builder-scaling tests only).
- Binding energy: consistent vacuum subtraction, rest-energy convention, finite-volume
  analysis; absolute energies are not yet nuclei predictions without LEC calibration.
- Manuscript hygiene: abstract, citations, reproducible tables/figures, `4results`→
  `4Results` path (case-sensitive builds).

---

## Phase 2 — Validation runs (small; laptop-safe or one tiny job)

No heavy local compute (laptop memory is bounded — cluster for anything real).

1. All Workstream-A tests green.
2. **Old-vs-new delta** on a small L=2 case: quantify how wrong the buggy data was and
   whether qualitative headlines (frame advantage grows with L; classical feasibility;
   quantum crossover) survive or flip.
3. Small TrimCI convergence probe → pick production core sizes / confirm PT2 protocol.
4. PauliLCU λ/T-count sanity at L=1,2.

**Decision gate:** confirm the story survives before scaling.

---

## Phase 3 — Production regeneration (HPC; via launch-approval loop)

All manifest-versioned: seeds, dependency locks, logs, checksums (task 12). Launch loop:
commit → user pushes → user go-ahead → submit → report batch/cluster ID.

**Classical (TrimCI), corrected H:**
- Deep-core sweeps L=2…6 (3D): matched N_f, repeated random-init seeds (no warm starts),
  PT2 + extrapolation with uncertainty, ED anchor at smallest L (task 31).
- Frame study redo: per-mode squeeze, analytic r*, advantage-vs-L (task 33).
- Binding box-convergence redo: A∈{0,1,2,4}×L∈{2..6}, vacuum subtraction, error bars.
- N_f/n_b convergence study (feeds Workstream B).
- LF with quantified transform error (task 32).

**Quantum RE, corrected H:**
- PauliLCU circuit-level λ/T-count/logical-qubits/walk-count scaling to L=10 (headline).
- Compiled sparse RE (C1) + amplitude composed encoding (C2) comparisons.
- Frame→QPE bridge (⟨n⟩→n_b seam) (task 34).
- Runtime scenario bands (C4).

**Headline figures LAST** — only after all data regenerated with manifests (tasks 07, 08, 14).

---

## Phase 4 — Housekeeping

- Task tracker: adopt research/prototype/validated/production/paper-ready fields; fix
  stale statuses (13, 24, 26, 30, 32, 33) and README Done/Active categorization.
- Archive old buggy data under `data/**/PRE_VERTEX_FIX/` (or delete after extracting the
  old-vs-new comparison). Never let it feed a figure.
- Freeze bibliography snapshot + source-to-claim table; cite primary papers at the exact
  claim, not review notes.

---

## Dependency map (what unblocks what)

```
A (vertex fix + oracle tests)  ── gates ALL data ──┐
                                                   ├─► classical production (Phase 3)
C3 (PauliLCU, already compiled) ───────────────────┤
B (rigorous cutoff)  ─ interim empirical ──────────┤   (final N_f reconciled to proof)
C1 (compiled sparse) ──────────────┐               │
C2 (amplitude encoding) ───────────┼─► quantum production (Phase 3)
C4 (runtime bands) ────────────────┘
D (paper cleanup) ── parallel, low compute
```

Core paper spine (corrected H + classical feasibility + quantum resource scaling +
frame advantage) rides on **A + C1** (C3 = PauliLCU validates at small L; C1 = compiled
sparse carries L=10). B/C2 add the rigorous-cutoff and amplitude-vs-Fock claims.
