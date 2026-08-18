# NuQu Remediation & Regeneration Plan

Response to the Codex publication-readiness audit (`codex_audit/`, 2026-08-18).
Owner: user (reviewer). Branch: TBD (recommend a dedicated `remediation/vertex-fix`
cut from `main`). All data generated before the vertex fix is **invalid** and must be
regenerated — see Workstream A.

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
production N_f against the proven bound once it lands. Relabel the `'tong'` switch in
`EFTParameters.py` away from "certified" until the bound is proven.

**Exit gate:** checked theorem with hypotheses mapped term-by-term to the finite-volume
H; production N_f justified by proof (and consistent with empirical convergence).

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
  full-H fermion-number conservation, zero/linear-coupling limits). 71 fix-adjacent tests
  green. Bug-baseline tests updated: `test_fock_basis` counts, `test_sign_structure` claim.
- A3 `a_L**dim` amplitude consistency: TODO.
- Remaining: small ED anchor for TrimCI (folds into Phase 2).

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
