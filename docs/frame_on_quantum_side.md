# Frames on the quantum side — warm-start vs walking a different frame

**Audience:** the PI, and the agent who will code + test this. This document only *frames
the problem and the experiment*; it does not implement anything.

**One-sentence goal:** decide whether it is worth transforming the Hamiltonian into a
different frame *before* qubitizing + running QPE on the fault-tolerant device — and if so,
which frame — by comparing the total quantum resource cost (logical qubits, T-count, Λ,
QPE repetitions) of three architectures, at fixed target accuracy (~1 MeV).

---

## 0. Background: what a "frame" is and why we care

A frame is a unitary similarity transform `H → U†HU` applied to the mixed
fermion–boson EFT Hamiltonian. We use three kinds (all in `classical/trimci/frame.py`):

- **Gaussian squeeze** (`squeeze_terms`) — a single-mode/Bogoliubov canonical transform on
  the boson sector. **Exactly isospectral** and **finite/closed-form** (degree-≤2 boson
  operators map to a finite polynomial; no truncation).
- **Lang–Firsov (LF) polaron displacement** (`displace_terms`) — a boson displacement
  conditioned on the nucleon density/transition vertex. For the density (Holstein) piece it
  is exact + terminating; for our physical **σ⊗τ transition vertex** it is an *infinite*
  BCH series that we **truncate** (`order` parameter + a leading-order substitution).
- **Combined** `gaussian+lf` = squeeze ∘ truncated-LF.

**Why frames help (classical evidence, `PROJECT_CONTEXT.md` + `data/classical/hpc/`):**
the framed ground state is far more *compact* (fewer boson levels populated). Measured:
per-mode pion occupation is ~0.02 in the bare frame and drops further in the framed basis;
the squeeze frame does the bulk of the compaction, LF adds a filling-dependent increment
(turns on ~A=2–3, grows to −420 MeV at half-filling at L=3). Compaction → a **smaller boson
Fock cutoff `n_b`** and a **better QPE initial state** — both quantum resource levers.

**The crucial architectural fork (this is the whole decision):** a frame can enter the
quantum computation in two fundamentally different places, with completely different proof
obligations.

---

## 1. Architecture A — warm-start only (the current code; SAFE, no proof needed)

**What it is:** qubitize the **bare** H (unchanged), and use the frame unitary `U` only to
prepare the QPE **initial state** `|ψ₀⟩ = U|ref⟩` (ref = boson vacuum ⊗ HF determinant).

**This is what the repo already does.** `classical/trimci/frame_qpe.py` +
`frame_qpe_bridge.py` treat the frame as a warm start; `src_PI/` qubitizes the bare H (no
framed-H path exists there). The classical solve tells us the frame that maximizes the
overlap `p₀ = |⟨ref|U†|g⟩|²`, and that overlap is already recorded per-core in the cluster
runs (`p0` field).

**Why it is unconditionally safe:** QPE returns eigenvalues of whatever operator the walk
encodes — here, the **bare** H. A wrong, truncated, or heuristic `U` **cannot shift the
returned energy**; a bad warm start only lowers `p₀`, costing `~1/p₀` extra QPE
repetitions. So:
- **Accuracy:** exact by construction — the 1 MeV budget is protected by QPE ancilla
  precision on the bare H, independent of the frame. **No isospectrality proof required.**
- **Payoff:** fewer QPE repetitions, `p₀_frame / p₀_bare`, at a one-time state-prep cost
  `T_prep` to synthesize `U|ref⟩`.

**Use the truncated LF here freely** — truncation is harmless to accuracy; it only makes
the warm start slightly less good than the exact frame would. This is the honest headline:
*"the LF frame is a variational warm-start; its only failure mode is reduced overlap, not
spectral error."*

---

## 2. Architecture B — walk the Gaussian-squeeze frame (CERTIFIED n_b win)

**What it is:** qubitize the **squeezed** Hamiltonian `H_sq = squeeze_terms(H)` — the walk
register itself encodes `H_sq`, whose ground state is near-vacuum in the boson sector, so a
**smaller `n_b`** suffices in the walk. QPE returns `spec(H_sq)`.

**Why it is certified:** the squeeze transform is **canonical and exact** (`|μ|²−|ν|²=1`;
degree-≤2 → finite polynomial, no truncation), so `spec(H_sq) = spec(H)` to machine
precision. QPE therefore returns the **exact bare spectrum** while the walk enjoys the
smaller boson register. This is the clean way to buy the in-the-walk `n_b` reduction.

**Caveat to keep honest:** the `n_b` reduction is a *physical* claim about the encoded H's
ground state, and the rigorous cutoff certificate is the **Tong bound** (open homework)
applied to `H_sq`; the measured framed `⟨n⟩` is the *motivation*, not the proof. The
squeeze also changes the LCU coefficients → Λ and per-walk-step T-count shift; those must be
re-estimated, not assumed.

---

## 3. Architecture C — walk the Gaussian + truncated-LF frame (needs an error bound)

**What it is:** qubitize `H̃ = squeeze ∘ LF_trunc (H)`. LF adds a further compaction
increment (the filling-dependent one), potentially a still-smaller `n_b`. QPE returns
`spec(H̃)`.

**Why it needs a bound:** the truncated LF is **not** a similarity transform of H — it
breaks exact isospectrality, so `spec(H̃) ≠ spec(H)`. The error has two parts:
- **FC-dressing truncation** (controlled by `order`): factorially small — measured ~1e-6
  MeV at order 4. **Negligible.**
- **Transition-vertex substitution** (the σ⊗τ piece): **order-independent** (arises because
  the vertex operators don't commute, so the BCH series never terminates). This is the only
  error that matters, and the `order` knob does not reduce it.

**Admissibility test:** `|E₀(H̃) − E₀(H)| ≤ ‖R_trans‖` (Weyl). To use Architecture C, must
show `‖R_trans‖ (per site) × sites < ε_budget` at the production λ. If it fails, C is
inadmissible and you fall back to B (squeeze walk) + A (LF warm-start).

---

## 4. The experiment to run (the actual task)

For each architecture, on the SAME physical system(s), produce the resource estimate and
the accuracy, then compare. Reuse the existing pipeline (`src_PI` resource estimator +
`classical/trimci` for the frames and the classical overlaps).

**Systems:** start where we have classical ground truth — L=2 and L=3, a couple of fillings
(dilute + half-filling), N_f/n_b from the tail study.

**Metrics per architecture (the comparison table):**

| | logical qubits | Λ | T / walk-step | N_walk (∝ Λ/ΔE) | QPE reps (∝ 1/p₀) | spectral error | **total T** |
|---|---|---|---|---|---|---|---|
| A: bare walk + LF warm-start | bare n_b | bare Λ | bare | bare | **reduced by p₀_frame/p₀_bare** | 0 (exact) | ? |
| B: squeeze walk | **reduced n_b** | new Λ | new | new | (+ warm-start optional) | 0 (exact) | ? |
| C: squeeze+LF walk | **smallest n_b?** | new Λ | new | new | **‖R_trans‖ (must be < budget)** | ? |

`total T ≈ (T_prep + N_walk · T_per_step) · QPE_reps`. The winner is the architecture with
the lowest total T **at accuracy within budget**.

**Concrete steps for the coding agent:**
1. **A (baseline + warm-start):** run the existing bare-H resource estimate; fold in the
   measured `p₀_frame/p₀_bare` (already in the cluster JSONs) as a repetition reduction, and
   estimate `T_prep` for the state-prep circuit `U|ref⟩` (squeeze is Gaussian → efficient;
   LF displacement → a controlled-displacement circuit). Net: does the repetition win beat
   `T_prep`?
2. **B (squeeze walk):** build the qubitized walk for `squeeze_terms(H)`; re-estimate n_b
   (Tong bound on `H_sq`), Λ, T/step, N_walk. **Verify isospectral** (spectrum vs bare
   unchanged — it must be, to machine precision; a sanity check, not a result). Compare
   total T to A.
3. **C (squeeze+LF walk):** build the walk for `squeeze ∘ LF_trunc(H)`; same resource
   re-estimate PLUS the **accuracy check**: measure `‖R_trans‖` via the *order-vs-spectral-
   error floor* on ED-able systems (run `isospectral_check` at fixed production λ for
   order ∈ {1..5}; it rolls off factorially then floors — the floor height per site is
   ‖R_trans‖). If floor × sites < budget, C is admissible; report its total T; else mark C
   inadmissible and stop.
4. **Decide:** is switching frames before QPE worth it? I.e. is min(B, C-if-admissible)
   total T meaningfully below A? Report the table + the verdict.

**Expected shape of the answer (hypothesis, to be confirmed):** A is free and always safe
(take the warm-start win regardless). B is the certified structural win (smaller n_b in the
walk with exact spectrum) and is likely the recommendation. C adds a further n_b increment
*if* ‖R_trans‖ is within budget at the L we care about — the filling-dependent LF gain
suggests C helps most at high filling, exactly where ‖R_trans‖ also grows (~λ²), so the
admissibility test is the crux.

---

## 5. Gotchas / corrections to carry

- **A is the status quo** — `src_PI` qubitizes the bare H today; B and C are *new* walk
  operators someone must build. Don't assume the framed-walk path exists.
- The `n_b` reduction is only *rigorous* via the **Tong cutoff bound on the actually-encoded
  H** (bare for A, `H_sq` for B, `H̃` for C); the classical `⟨n⟩`/tail is motivation +
  a numerical corroboration, not the certificate.
- The frame changes the **LCU coefficients**, so Λ and per-step T must be **re-estimated**
  for B and C, not carried over from bare. (Squeeze can move where Λ lives; see the lambda
  audit in `PROJECT_CONTEXT.md`.)
- For C, only the **transition-vertex** error is load-bearing; the FC/`order` truncation is
  factorially negligible. Measure the *floor*, not the roll-off.
- Keep the exact-isospectral frames (`bare`, `gaussian`/`squeeze`, `bogoliubov`) as the
  reference whitelist (already `EXACT_ISOSPECTRAL_FRAMES` in `frame_qpe_bridge.py`); only
  those are safe to *walk* without an error bound.

## 6. Pointers
- Frames: `classical/trimci/frame.py` (`squeeze_terms`, `displace_terms`, `analytic_*`),
  `lf.py` (exact generator, coupling-scale probe).
- Warm-start bridge (Architecture A, as-built): `classical/trimci/frame_qpe.py`,
  `frame_qpe_bridge.py` (`isospectrality_gate`, `EXACT_ISOSPECTRAL_FRAMES`).
- Resource estimator (the walk / Λ / T-count): `src_PI/`.
- Isospectrality error harness: `classical/trimci/frame.py::isospectral_check`,
  `data/classical/hpc/2026-08-04/isospectrality_check.py` (block2 reference).
- Classical evidence (compaction, occupation, tail): `PROJECT_CONTEXT.md` "Live state",
  `data/classical/hpc/2026-08-04/`.
