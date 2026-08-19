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
`a4a38f9` C1 step 5 (Config switch+cache) · `47c662e` C1 step 4 (A/B) · `91ca3de` C1 step 3
(full-bundle) · `fe7fdf5` C1 step 2 (fermion atom) · `524b1c8` C1 step 1 (boson monomial) ·
`5ec99be`/`9bc2e7a` C1 design · `644d94f` C scoping + C3 anchor · `11beb50` B `tong_rigorous` ·
`d532857` A3 + cross-builder tests · `9404fac` A vertex fix + oracle suite.

| Workstream | State |
|---|---|
| **A** — Hamiltonian correctness | ✅ **DONE.** Vertex fix (3 builders) + physics-oracle suite (`tests/test_vertex_algebra.py`: Kronecker oracle for all 12 σ⊗τ channels, cross-builder equivalence, `[H_WT,n̂_x]=0`) + `a_L**dim`. |
| **B** — Rigorous cutoff | ✅ **CODE DONE.** `boson_cutoff_method='tong_rigorous'` (exact Bogoliubov tail + variational bound), `classical/trimci/gaussian_cutoff.py`, tests. Draft `claude/research/bosonic-encodings/05_*.md` — audited & **rescoped honestly** (2 rigorous results; 2 open gaps: bound `‖Vψ‖` via Gaussian moments, control `|δ_true−δ_Gauss|`). **For user + Codex proof review.** `'heuristic'` stays default. |
| **C3** — PauliLCU anchor | ✅ **DONE.** Genuinely circuit-level on corrected H (L=2,dim=1,n_b=2 → Λ=3077, Walk_T=214724, 31 qubits, 1.3s). Small-L validation anchor; ceiling is QubitOperator memory. |
| **C1** — compiled sparse RE | ⚠ **BLOCK ENCODING DONE; WALK HERMITIZATION PENDING.** `SparseFullBundleBlockEncoding` built (5 steps): the block encoding `α_tot·⟨0|U|0⟩ = H` and `α_tot` (= Λ) are **exact & validated** (α invariant to machine precision incl. fermion sector; heterogeneous toy assembly sim ~1e-15). **BUT** the quantum-algorithms review found the d=1 atoms make `U` non-Hermitian, so the single-reflection `QubitizedWalkOperator` is **not a valid qubitization** (walk spectrum lacks `e^{±i arccos(E/α)}` — confirmed, xfail regression test). The compiled `Walk_T` (0.89×/0.92× the proxy) is a **block-encoding-level estimate, not a genuine walk cost**, until the atoms are Hermitized (re-pair `c·m+c̄·m†`; fermion atoms already Hermitian; α_tot preserved). Also open: mixed-atom operator sim (α is phase-blind), control-overhead + junk undercounts. **See §C1 below — needs a Hermitization pass before it feeds a figure.** |
| **C2** — amplitude composed encoding | ⬜ scoped, not started (next large build). |
| **C4** — runtime bands | ⬜ scoped, not started (medium). |
| **D** — paper cleanup | ⬜ mostly parallel/low-compute; sign-problem writeup done (`docs/sign_problem.md`). |
| **Phase 2/3** — regeneration | ⬜ not started; needs the corrected H + (quantum) C1 ✅/C2. HPC, launch-approval loop. |

### ▶ Recommended action — C1 walk-validity: defect is DEEP; feasibility spike DONE

**The walk-validity defect runs deeper than the C1 bundle.** The quantum-algorithms review
(2026-08-18) found the compiled walk is not a valid qubitization (non-Hermitian `U`); investigating,
**even the pre-existing `single_ladder.py` `(â+â†)` encoder is non-Hermitian** (`‖U−U†‖=5.66`,
`U²≠I`) — so the whole BCK/sparse foundation is affected, not just C1. pyLIQTR's single-reflection
walk needs a **Hermitian** `U`; there is **no off-the-shelf sparse-Hermitian encoder** (pyLIQTR
1.3.4 ships only PauliLCU — Hermitian because Paulis are self-inverse — plus specialized
fermionic/chemistry ones), and the cheap general Hermitization wrapper `(H_b)(|0⟩⟨0|⊗U+|1⟩⟨1|⊗U†)(H_b)`
does **not** yield a Hermitian `U` (verified — walk still fails).

**Feasibility spike (DONE, `tests/test_hermitian_sparse_spike.py`, 7 pass):** a valid sparse-Hermitian
encoder of `(â+â†)` **does** exist — **edge-colour into two 1-sparse Hermitian matchings** `M_a+M_b`;
for a matching `M²` is diagonal, so the contraction dilation `[[M/α,√(I−M²/α²)],[√,−M/α]]` is sparse
+ Hermitian + self-inverse; LCU-combine with a Hermitian SELECT. Verified `U=U†`, `U²=I`,
`α·⟨0|U|0⟩=(â+â†)`, and **walk qubitizes** (`e^{±i arccos(E/α)}`) at n_b=2,3,4. Bonus: **α is tighter**
than single_ladder (3.15 vs 3.46 at n_b=2 → tighter Λ). **Cost of Hermiticity ≈4×** the per-atom
boson SELECT (2 matchings × 2 amplitude oracles vs single_ladder's 1); fermion atoms unchanged;
diagonal `n̂` cheap (diagonal dilation).

**Decision pending (user):** with feasibility + bounded cost + tighter α established, either
**(A) full sparse-Hermitian rebuild** — reimplement the boson encoders on the matching-dilation
(single-mode `(â+â†)`-type, diagonal `n̂`, and the trickier two-mode H_WT Hermitization), re-pair
atoms, flip `test_bundle_walk_qubitizes_hermitian_H` to pass, re-validate α invariant + toy sim +
A/B (~1 session; main risk = two-mode H_WT) — or **(B) pivot** to PauliLCU as the sole *valid* walk
cost + analytic PauliLCU scaling for L=10, demoting sparse to Λ + a labelled block-encoding-level
estimate.

**Do NOT flip `sparse_oracle_mode` to `'compiled'`** until the walk is Hermitized. Analytical stays
default. **Also open regardless of path:** mixed-atom operator sim (α is phase-blind), control-overhead
+ junk undercounts.

**Then C2 — amplitude composed encoding** (real `H_pos+H_mom` + H_WT species-selective QFT, Watson
Eqs. 102–104), **C4** (runtime bands), Phase 2/3 regeneration.

**Guardrails (unchanged):** env pinned pyLIQTR 1.3.4 / Qualtran 0.4.0; comparison-switch discipline;
no heavy local compute. The B proof gaps are the user's/Codex's to review.

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

**C1 (compiled sparse full-bundle) — ✅ DONE (implemented + validated).** Built per
`docs/sparse_full_bundle_design.md` in 5 gated steps; each gate green. New files:
`sparse_oracle/boson_monomial_encoding.py`, `fermion_atom.py`, `bundle_encoding.py`;
`resources.py` (+`estimate_sparse_resources_compiled`, `compiled_vs_analytical`); `sparse.py`
(+`compiled` switch); `tests/test_sparse_full_bundle.py` (66 tests).
  1. **`SparseBosonMonomialBlockEncoding`** — d=1 per-mode encoder (linear `â`/`â†`, number op,
     two-mode products). **Gate:** per-atom block-matrix sim vs exact monomial, ~1e-15 for every
     shape in the real H (n_b=2,3); per-atom α == `lambda_compute._monomial_max_amplitude`.
  2. **Fermion atom** via off-the-shelf pyLIQTR PauliLCU over `jordan_wigner(fermion_factor)`.
     Replaces the `4·weight` LOWER bound (≈32 T) with the genuine PauliLCU cost (≈836 T).
     **Gate:** α == Pauli 1-norm; term-for-term == standalone PauliLCU `estimate_resources`.
     Asserts JW reality (vertex-fixed factors are Hermitian χ-channels).
  3. **`SparseFullBundleBlockEncoding`** — LCU over atoms, `PREP(alias)·D_phase·SELECT·PREP†`.
     Genuine Walk_T = `estimate_resources(QubitizedWalkOperator(be))` (389,496 at L=2 dim=1 n_b=2).
     **Retired risk #1** two ways: (a) **α_tot invariant** `be.alpha == compute_native_lambda
     physical_lambda` to machine precision (L=2 dim=1/2/3, incl. the static-nucleon fermion
     sector); (b) **toy assembly sim** — ideal decompose reproduces H to ~1e-15 on a heterogeneous
     toy (2-mode boson ⇒ multi-qubit shared ancilla + fermion + imaginary phase). All block-flag
     qubits are in `selection_registers`, so `QubitizedWalkOperator` reflects the correct block.
     *Bug found + fixed here:* H_WT's conjugate-momentum Π gives **imaginary** boson coefficients;
     atoms now carry complex coeffs (phase → `D_phase`), not `.real`-projected (had dropped 12/15
     mixed terms).
  4. **A/B:** compiled **0.89×** (389,496 vs 438,208) at L=2 dim=1; **0.92×** (2.27M vs 2.45M) at
     L=2 dim=3 — the honest number lands just under the proxy (the boson upper-bound ceiling had
     dominated the fermion floor). Same α_tot both paths; per-kind SELECT breakdown reported.
  5. **`sparse_oracle_mode`** Config switch (`'analytical'` default / `'compiled'`), A-independent
     walk-estimate cache. Analytical proxy retained as the A/B baseline (comparison-switch
     discipline). Default unflipped so no downstream number moves silently.

  **⚠ Quantum-algorithms review verdict (2026-08-18):** the α_tot invariant and the one-sided
  `D_phase` are correct; the block-flag qubits are all in `selection_registers`; the ideal-sim
  `MatrixGate` decomposition never leaks into the cost path (verified). **But** the review found a
  load-bearing defect: the d=1 atoms encode non-Hermitian monomials → `U` is non-Hermitian → the
  single-reflection `QubitizedWalkOperator` is **not a valid qubitization** (I verified: `‖U−U†‖≈7`
  and the walk spectrum lacks `e^{±i arccos(E/α)}`). So the compiled `Walk_T` costs an object that
  would not run QPE. **Λ/α_tot is exact; the walk is not — Hermitization required (see Recommended
  action).** Secondary (all optimistic undercounts, documented): per-atom SELECT charged
  *uncontrolled*; `LogicalQubits` junk width estimated; **mixed atoms unvalidated at the operator
  level** (the α invariant is phase-blind — the largest untested surface). All still strictly more
  honest than the retired mixed upper/lower bounds, but C1 is **not** publication-ready until the
  walk is Hermitized + the mixed-atom sim lands.

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
