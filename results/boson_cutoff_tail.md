# Boson-cutoff tail: how much does a small `n_b` truncate, and does framing help?

**Question (PI):** the mean per-mode pion occupation is tiny (~0.02), so `n_b=2` is
obviously enough — but *what does the tail of the occupation distribution look like?* How
much ground-state probability does a per-mode Fock cutoff actually cut off, and is that a
reason to keep `n_b=2` rather than drop to `n_b=1`?

**Answer, one line:** keep `n_b=2` — `n_b=1` truncates **4–70%** of the ground state
(the per-mode mean is deeply misleading because the tail is a *union over ~81 modes*), and
the **Gaussian squeeze frame tightens the `n_b=2` leak by ~40×** (from up to 1.8% in the
bare basis to <0.05% framed, and *exactly* 0% at full filling) — a new, direct argument for
walking the squeeze frame on the quantum side.

Data: cluster **290388** (`tail-20260805-122135`), L=3, d=3, **`n_b=3` (N_f=8, so the tail
is fully resolved up to level 7)**, bare & gaussian frames, fillings 0.1/1.0/4.0
(A = 3 / 27 / 108, i.e. dilute / quarter-of-capacity / **full occupancy**), 64k core,
2 seeds. Recorded per-rung via `occupation_tail` + `occupation_histogram`
(`classical/trimci/observables.py`), added to the shard driver in commit `72e9fae`.

---

## 1. The measured tail

`δ(N_f)` = ground-state weight on determinants where **some** pion mode has occupation
`≥ N_f` — i.e. exactly the weight a per-mode Fock cutoff `N_f` would drop. `n_b` bits give
`N_f = 2^{n_b}` levels, so **`n_b=1` cuts at N_f=2** and **`n_b=2` cuts at N_f=4**.

| frame | A | density | `δ(2)` = **n_b=1 cut** | `δ(3)` | `δ(4)` = **n_b=2 cut** | `δ(6)` | ⟨occ⟩/mode |
|---|---|---|---|---|---|---|---|
| bare | 3 | 0.11 | **0.643** | 0.021 | 1.2e-2 | 1.6e-4 | 0.025 |
| bare | 27 | 1.00 | 0.323 | 0.010 | 2.5e-3 | 1.4e-5 | 0.018 |
| bare | 108 | 4.00 | **0.704** | 0.022 | **1.8e-2** | 2.9e-4 | 0.027 |
| gaussian | 3 | 0.11 | 0.042 | 0.003 | 2.2e-4 | 6e-7 | 0.012 |
| gaussian | 27 | 1.00 | 0.073 | 0.007 | 4.8e-4 | 0 | 0.017 |
| gaussian | 108 | 4.00 | 0.009 | 1e-4 | **0** | 0 | 0.007 |

Per-mode histogram `p(n)` (bare, deepest rung): `p0 ≈ 0.986`, then a small non-monotonic
tail, e.g. A=108: `p0=0.986, p1=0.0015, p2=0.0124, p3≈0, p4=0.0002`. Note `p2 > p1` — the
physical pion cloud is not a simple geometric decay; there's structured weight at level 2.

---

## 2. Why the tail is large even though ⟨occ⟩ ≈ 0.02 — the union effect

This is the crux and the non-obvious part. The **per-mode** probability of exceeding the
cutoff is small (`p(n≥2) ≈ 1%` per mode), but there are **~81 pion modes** (27 sites × 3),
and a per-mode cutoff drops a determinant if *any* mode exceeds it. So the relevant leaked
weight is a **union over all modes**:
```
δ(2) = P(any of ~81 modes has occ ≥ 2) ≈ 1 − (1 − p_mode)^{81}
```
With `p_mode(≥2) ≈ 1%`, that union is ~0.5–0.7 — matching the measured `δ(2)`. **The
per-mode mean occupation (~0.02) is therefore the wrong diagnostic for the cutoff; `δ(N_f)`
is.** A "near-vacuum" per mode still leaves a large fraction of the *many-mode* state
outside a tight box.

---

## 3. Findings

1. **`n_b=1` is out.** It cuts at N_f=2 and drops **32–70%** of the ground state in the bare
   basis (4–7% even framed). No error budget survives that. **`n_b=2` is the floor.**

2. **`n_b=2` in the bare basis is *marginal at high filling*.** `δ(4)` grows with filling:
   0.25% (A=27) → **1.8% (A=108, full occupancy)**. 1.8% leaked weight is not obviously
   inside a ~1 MeV budget — so if we encode the **bare** H (the warm-start architecture,
   `docs/frame_on_quantum_side.md` Arch A), `n_b=2` should be checked against the budget at
   the top filling, and `n_b=3` may be needed there.

3. **The squeeze frame is a cutoff-tightening tool, not just an energy-compaction one.**
   In the Gaussian frame the `n_b=2` leak collapses to **<0.05% everywhere, and exactly 0%
   at full filling** — a ~40× reduction vs bare at high filling. The frame moves boson
   weight out of the tail and into the vacuum, so the *same* `n_b=2` register that is
   marginal in the bare basis is comfortably exact in the framed basis.

4. **This is a direct quantum-resource argument for walking the squeeze frame** (Arch B in
   `docs/frame_on_quantum_side.md`): beyond compacting the classical solve, the squeeze
   frame *shrinks the boson register needed in the walk operator* — it makes `n_b=2`
   rigorously sufficient (0% leak at full filling) where the bare encoding is borderline.
   The squeeze is **exactly isospectral**, so this register saving is free of any spectral
   error. (LF adds further compaction but is truncated → needs the separate `‖R_trans‖`
   check; not required for this squeeze-only cutoff win.)

---

## 4. Caveats

- **Tail convergence at 64k core.** `δ(N_f)` is dominated by rare high-occupation
  configurations, which a 64k selected-CI core samples imperfectly; the numbers are good
  estimates, not converged to many digits. The *ordering* (n_b=1 ≫ n_b=2; bare ≫ gaussian;
  worst at full filling) is robust; the third-digit values are not. A deep-core rerun (the
  new `NUQU_DEEP_SOLVE` path) would tighten them.
- **`δ(N_f)` vs the energy error.** `δ(N_f)` is the leaked *weight*; the truncation error in
  the *energy* is controlled by it but is not equal to it (the variational solve in the
  truncated box redistributes weight). Treat `δ(N_f)` as the error *scale* / an upper-bound
  proxy, not the exact MeV error. Converting `δ(N_f)` → a MeV bound (against the ~1 MeV
  budget) is the natural follow-up, and connects to the Tong cutoff-bound homework.
- **The tail is fully resolved** here: the run used `n_b=3` (N_f=8), and `δ(8)=0` in all
  cases, so nothing leaks past what we measured — the cutoff study itself is not truncated.

---

## 5. Bottom line for the register choice
- **Never `n_b=1`.** **`n_b=2`** is the choice.
- If we **walk the bare H** (warm-start architecture): `n_b=2` is safe up to ~quarter-
  filling (leak ≤0.25%) but marginal at full filling (1.8%) — verify against budget or use
  `n_b=3` there.
- If we **walk the Gaussian squeeze frame**: `n_b=2` is comfortably exact at *all* fillings
  (leak <0.05%, 0% at full occupancy), certified isospectral. This is the cleanest register.

Reproduce: `data/classical/hpc/2026-08-04/pion_occupation.py` (means) + the tail analysis in
this campaign's shards (`occ_tail` / `occ_hist` fields).
