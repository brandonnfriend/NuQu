# NuQu — Paper-First Plan (reframed 2026-08-19)

Reframed from the earlier "Remediation & Regeneration Plan" after three Codex audits
(`codex_audit/publication_readiness.md`, `.../c1_publication_decision_2026-08-19.md`,
`.../publication_readiness_reconciliation_2026-08-19.md`). **Goal: publish.** Everything below
is judged by one test — *does it drive or support a paper headline?* If not, it's frozen or cut.
Branch: `remediation/vertex-fix`. Launch discipline: commit → user pushes → user go-ahead → submit.

---

## The paper's three headlines

1. **Classical baseline for the dynamical-pion system.** A corrected lattice EFT Hamiltonian
   (Watson dynamical-pion χEFT) solved classically (TrimCI) with honest convergence + uncertainty.
2. **Classically-informed quantum approach.** Use the classical solution (frame/occupation) to
   *precondition* the quantum problem — lower the boson cutoff `n_b`, shrink the register.
3. **Reduced quantum resources for QPE GSEE** — and *how* the reduction was achieved. The
   quantum-resource anchor is **PauliLCU** (compiler-derived), with the classical preconditioning
   as the lever that reduces the cost.

Everything the paper claims must be traceable to a tested, regenerated output at the level of the
evidence. No claim beyond what's proved/validated.

---

## What the paper NEEDS (the universal gate — do all of this)

| # | Requirement | Status | Owner-notes |
|---|---|---|---|
| **N1** | **Corrected Hamiltonian + independent validation.** Vertex fix + term-by-term algebra tests + cross-builder agreement, frozen at a commit. | ✅ **DONE** (`9404fac` fix; `tests/test_vertex_algebra.py` Kronecker oracle for all 12 σ⊗τ channels, cross-builder equivalence). Keep these in the release suite. | See "The bug" below. |
| **N2** | **Regenerate ALL physical results** after the fix — energies, binding, frames, occupations, λ/resource inputs — with versioned manifests (seeds, deps, logs, checksums). No pre-fix number survives. | ⬜ **NOT STARTED** — the big remaining classical job (HPC). | Phase R below. |
| **N3** | **Numerical convergence + honest uncertainty.** Empirical `n_b` (weighted boundary population, not just ⟨n⟩), solver/core-size, finite-volume/lattice sensitivity → uncertainty bars. Small ED anchors. | ⬜ **NOT STARTED** (design ready). | Replaces the "rigorous cutoff theorem." |
| **N4** | **PauliLCU quantum-resource anchor**, compiled at small–moderate L, with large-L extrapolation *explicitly labeled + validated in its calibration range*. | 🔨 **PARTIAL** — compiled L≤3 dim=3 works; `pauli_lcu_scaling.py` λ-extrapolation validated 0.00% at L=1,2,3. Need: push compiled L higher on the cluster + finalize the labeled extrapolation. | **PauliLCU plan below.** |
| **N5** | **Consistent end-to-end assumptions:** QPE convention (sign/time/2π), λ = block-encoding subnormalization, walk-query constant derived, ΔE target, success/repetition, runtime as *assumption-listed bands* (not one wall-clock number). | ⬜ mostly wording + a few small fixes (`docs/` + `.tex`). | Workstream D below. |
| **N6** | **Claim discipline** — every value labeled: compiled / analytic-extrapolation / provisional-model / empirical / proved. Remove "rigorous cutoff," "compiled sparse walk," asymptotic-advantage language. | ⬜ editorial pass. | |
| **N7** | **Finish the manuscript** — equations/conventions, citations, figures/tables from regenerated data, reproducible build. | ⬜ last. | |

---

## PauliLCU compilation plan (N4 — the load-bearing quantum spine)

**The concern (from months ago): the Fock-basis Pauli expansion is expensive.** Re-measured today
(dim=3, cache off): the operator *build* is cheap; the **pyLIQTR estimate over the materialized term
list is the wall** — ~3 ms and ~18 KB **per Pauli term**, ~720 terms/site (bulk).

| L (dim=3) | sites | # Pauli terms | estimate time | peak mem |
|---|---|---|---|---|
| 1 | 1 | 408 | 2.6 s | 37 MB |
| 2 | 8 | 5,160 | 13 s | 87 MB |
| 3 | 27 | 19,467 | 66 s | 360 MB |
| *10 (proj.)* | *1000* | *~720k* | *~40 min* | *~13 GB* |

**Plan, in priority order:**
1. **Compile brute-force as high as the cluster allows.** Target **L=5–7 dim=3** on a normal node;
   attempt **L=8–10** on a large-memory node (~13 GB / ~40 min is plausible, not a hard wall).
   Each (L, n_b) is one independent job; parallelize across L. HPC via the launch-approval loop.
2. **Calibrated extrapolation as the labeled fallback** to reach L=10 if brute force stalls. This is
   *well-supported*, not casual: `pauli_lcu_scaling.py` reproduces the compiled λ to **0.00%** at
   L=1,2,3 via exact lattice combinatorics (λ = a·S + b·N_bonds; N_walk from λ; qubits exact). The
   `walk_T` prefactor is the only modeled piece (JW-nonlocality → biased low) — anchor it on the
   highest compiled L and quote an honest band. Label it "fitted/analytic scaling, validated to L=k."
3. **Escape hatch (only if 1+2 are insufficient for a required headline):** a translation-invariant /
   Hubbard-style structured PauliLCU encoding (per-unit-cell SELECT + site index) reaches L=10 exactly
   without materializing 720k terms. **Real build — kept OUT of the critical path** unless forced.

**Deliverable:** compiled PauliLCU λ / N_walk / logical-qubits / walk-T at the highest feasible L,
plus the labeled extrapolation to L=10, with ΔE, λ-definition, and walk-count constant stated (N5).

---

## What we are FREEZING / CUTTING (not paper-critical)

Per the audits, none of these is required for the three headlines. Preserve the work as
feasibility/appendix material; **do not put it back on the critical path.**

- **Sparse oracle (C1).** ❄ **FROZEN** as a *validated feasibility construction with a provisional
  compositional resource estimate.* The math is validated (Hermitian matching-dilation qubitizes;
  mixed terms covered; rotations synthesized), but the coherent-control composition is not a
  monolithic compiled walk (Codex `c1_publication_decision`). **PauliLCU is the anchor; sparse is
  NOT a headline.** No further sparse dev unless a sparse-vs-PauliLCU comparison becomes central —
  and then only the *one bounded controlled-composite validation* the memo specifies, then freeze.
- **Rigorous analytic bosonic-cutoff theorem (B).** ❄ **CUT from the critical path.** The paper needs
  *empirical* cutoff convergence (N3), not a theorem. Keep `tong_rigorous`/`gaussian_cutoff.py` as a
  provisional prescription; **stop calling it "rigorous/certified/Tong."** The proof is future work.
- **Amplitude-basis composed encoder (C2).** ❄ **CUT** unless the paper reports an amplitude-vs-Fock
  *quantum* advantage (it should not). Classical amplitude-coordinate calcs may stay *if* independently
  validated + converged; the split-walk quantum cost must not be used.
- **Polylog-in-cutoff oracle, QROM optimization, amplitude encoder, hardware-architecture study.** ❄
  optional enhancements for a later version.

---

## Terminology fix (do during any touch of these files)

- **"atom" → "LCU term" (or "block-encoding term").** "Atom" was borrowed LCU/LOBE jargon for one
  summand of `H = Σ_l c_l U_l`; it has no physical meaning and is confusing in a nuclear-physics paper.
  Rename in `sparse_oracle/*` and docs. ("fermionic atom" → "fermion LCU term," etc.)
- Distinguish **`N_f` (Fock levels)** from **`n_b` (qubits/mode)** consistently (`N_f = 2^{n_b}`).
- **λ = block-encoding subnormalization** (not spectral norm, not always a Pauli 1-norm).

---

## The bug, precisely (N1 — verified first-hand, kept for the record)

`_nucleon_transition_fermion` (`fock_native.py`), `_nucleon_transition_jw_uncached` (`Operators.py`),
and the 1D twin each returned `a†_α a_β + a†_β a_α` while **every caller already loops over all 16
ordered (α,β) pairs** → real channels doubled, imaginary/antisymmetric channels (τ_y; σ_y·τ_y)
cancelled to zero. **Fix:** each builder returns the single ordered `a†_α a_β`; the caller loops
reconstruct the Hermitian `Σ χ^I_{αβ} a†_α a_β` (χ = τ_I ⊗ σ_S Hermitian). Static-only sector was
already clean. **Scientific consequence (paper-relevant):** both H_AV and H_WT are sign-problem
sources (not H_WT alone) — the dynamical-pion coupling *as a whole* is the phase source, static sector
stoquastic-real (`docs/sign_problem.md`). Everything touching H_AV/H_WT must be regenerated (N2).

---

## Workstream D — manuscript/claims corrections (N5/N6, low-compute, parallel)

Do these as editorial passes (mostly `.tex` + `docs/`):
- Boundary conditions (open vs periodic), bond counting, derivative stencil, `a_L**dim` volume —
  **freeze the convention and test it**; match Watson or state the modified finite-volume scope.
- Frames as **numerical preconditioners** with empirical convergence/error — drop "exact
  isospectral / guaranteed upper bound" where the truncated implementation doesn't support it (task 32).
- `√n`/`√(n+1)` Fock ladder actions; QPE sign/time/2π consistency; repetition `~log(1/η)/p0`;
  ground-state identification protocol; walk-count constant *derived* from the estimator + failure prob.
- Density `A/(4L³)` filling / `A/(L a_L)³` number density; drop/label `A>4L³` unphysical points.
- Binding: consistent vacuum subtraction, rest-energy convention, finite-volume — no nuclei-prediction
  language without LEC calibration.
- Runtime as assumption-listed **bands** (factory rate/count, reaction time, routing, failure budget) —
  never one wall-clock forecast.
- Narrow "explicit pions ⇒ more accurate"; fix citations, case-sensitive paths, build.

---

## Phase R — Regeneration (N2/N3, HPC, the big remaining job)

All manifest-versioned (seeds, dep locks, logs, checksums). Launch-approval loop. **No heavy local
compute** (laptop memory-bounded — cluster for anything real).

**Classical (TrimCI, corrected H):**
- Deep-core sweeps L=2…6 (3D), matched `N_f`, repeated random-init seeds (no warm starts), PT2 +
  extrapolation with uncertainty, **ED anchor at smallest L** (independent small-matrix spectra).
- **Cutoff convergence (N3):** energies/observables at adjacent `n_b`; **weighted boundary population**;
  stated cutoff uncertainty. This is the most important numerical task — `n_b=2` drives everything.
- Frame study redo (per-mode squeeze, analytic r*, advantage-vs-L) as *empirical* preconditioners.
- Binding box-convergence redo (A∈{0,1,2,4}×L∈{2..6}, vacuum subtraction, error bars).

**Quantum RE (corrected H):**
- **PauliLCU** λ/N_walk/logical-qubits/walk-T at the highest feasible L (see PauliLCU plan) +
  labeled extrapolation to L=10.
- Frame→QPE bridge (⟨n⟩→n_b reduction) — the headline-3 "reduced resources" lever.
- Runtime bands (N5).

**Headline figures LAST** — only from regenerated, manifest-versioned data.

---

## Phase H — Housekeeping / provenance

- Archive pre-fix data under `data/**/PRE_VERTEX_FIX/`; never let it feed a figure.
- Freeze bibliography snapshot + source-to-claim table; cite primary papers at the exact claim.
- Task-tracker + README status cleanup.

---

## Dependency map

```
N1 corrected H  ── gates ALL data ──┐
                                    ├─► N2 regenerate (classical + quantum)  ─► N7 manuscript
N3 convergence/uncertainty ─────────┤        (Phase R, HPC)                       (figures LAST)
N4 PauliLCU anchor + labeled extrap ┘
N5/N6 conventions + claim discipline ── parallel, low-compute (Workstream D)
FROZEN: sparse C1 · rigorous-cutoff B · amplitude C2   (feasibility/appendix only)
```

Core spine = **N1 (done) → N2/N3 regeneration → N4 PauliLCU anchor → N5/N6 discipline → N7 manuscript.**
The strongest contributions (corrected EFT, converged classical baseline, classically-informed cutoff
reduction, PauliLCU resource scaling) stay; the optional theory/encoder work is off the critical path.
