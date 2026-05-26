# NuQu — Project Context

Deep context for the NuQu project. Loaded on demand (not every conversation). For standing rules and commands, see `CLAUDE.md`.

## The reference paper
- Watson, Bringewatt, Shaw, Childs, Gorshkov, Davoudi, *"Quantum Algorithms for Simulating Nuclear Effective Field Theories"* — arXiv:2312.05344 (2023/2025). Local PDF at `claude/Watson2025.pdf`.
- Watson et al. estimate fault-tolerant resources for several nuclear EFTs (pionless, one-pion-exchange / static-pion, dynamical-pion) using **Trotterization** for `e^{iHt}`. Pionless is cheapest; **dynamical pion is by far the most expensive**, and that is the regime this project targets.
- Specific results from Watson reused here:
  - **Lemma 5** → field-cutoff and qubit-count formulas (Eqs. 75–78) for the bosonic pion encoding: `pi_max`, `Pi_max`, `n_b = ceil(log2(...))`. Implemented in `src_PI/hamiltonians/core/EFTParameters.py::calculate_dynamic_cutoffs`.
  - **Lemma 23** → Trotter step T-cost: `g(L,n_b) * (1.15 log10(2 g/δ) + 9.2)` with `g = (45 n_b² + 114 n_b + 76) L³`. Implemented in `src_PI/trotter_theory/trotter_theory.py`.
  - **Lemma 78** → WT–WT commutator bound (used for nested-commutator Trotter error estimates).
- Physical inputs (Table I/IV of Watson) are encoded in `EFTParameters.get_physical_parameters` — `M_N=938`, `m_pi=135`, `f_pi=93`, `g_A=1.26`, `a_L=2.2 fm`, contact terms `C=-51.94`, `C_I=1.73` MeV.

## What this project does differently
- Replaces Trotterization with **Qubitization** (block-encoded walk operator + QPE) using `pyLIQTR`. Initial finding: Qubitization gives **lower T-counts** than Trotterization at the same accuracy — that's the headline result so far and the motivation for the publication push.
- The Hamiltonian is **split-oracle**: position-basis terms (`H_pos`: free pion mass + gradient + axial-vector + Weinberg–Tomozawa + static nucleon hopping/contacts) and momentum-basis terms (`H_mom`: pion conjugate-momentum kinetic) are encoded as separate walks. They run sequentially on the same hardware, so the peak logical-qubit count is `max(pos_walk_qubits, mom_walk_qubits)` (T/Clifford gate counts, by contrast, are summed across the two walks). Each walk step also requires a QFT/IQFT pair to swap bases — counted in `calculate_qft_cost`.
- Qubits per site: `4 + 3*n_b` (4 nucleon modes from spin × isospin, plus 3 pion species × `n_b` bits each). Total qubits: `L^dim * (4 + 3 n_b)`.

## Key code map
- `run_nucleon_sweep.py` — top-level sweep over nucleon number `A` (dense 1–10, sparse 20–100) at fixed `L` (default 2; "crank to 10 for paper-grade HPC"). Calls `evaluate_resources` and dumps JSON via `save_sweep_data`.
- `plot_sweep_data.py` — loaders + plotters: single-L T-counts/Λ/runtime, qubitization-vs-Trotter total cost vs A, total cost vs L for chosen A. All plots saved under `data/<date>/`.
- `src_PI/hamiltonians/`
  - `ConstructEFT.py` — assembles `H_pos` and `H_mom` (Jordan–Wigner on the static fermion sector).
  - `core/DynamicalPion.py` — `H_pion_free`, `H_axial_vector` (H_AV), `H_WT_Logic` (Weinberg–Tomozawa).
  - `core/StaticTerms.py` — nucleon hopping + on-site + contact (`HC`, `HCI2`).
  - `core/Operators.py` — `Pi_Squared`, `Gradient_Squared`, `Momentum_Squared`, `Nucleon_Transition_JW`.
  - `core/EFTParameters.py` — physical constants, dynamic cutoffs, `T_cross` time scale.
  - `Lattice1D/` — older 1D-only implementations; treat as legacy reference.
- `src_PI/estimation/`
  - `EstimateResources.py` — orchestrator: build → normalize → pyLIQTR analysis → QFT overhead → package metrics.
  - `NormalizeHamiltonians.py` — strips identity terms (tracked as classical shift), divides by `Δ = safety * Λ` (default `safety_factor=2.5`) so eigenvalues fit in [0, 0.5] for QPE.
  - `estimators.py` — `run_qubitization_analysis`: builds two `PauliLCU` block encodings, two `QubitizedWalkOperator`s, calls `pyLIQTR.utils.resource_analysis.estimate_resources` on each, and sums.
  - `instances.py` — `MyCustomHamiltonian` adapter from OpenFermion `QubitOperator` to pyLIQTR `ProblemInstance`.
- `src_PI/trotter_theory/` — Watson Trotter-cost formulas for the comparison curves in the plots.
- `src_PI/utils/`
  - `LatticeGeometry.py` — flat-index ↔ coordinates, OBC neighbors, qubit index mapping (`site_to_nucleon_qubit`, `site_to_pion_qubit`).
  - `utils.py` — Pauli matrices, chiral coefficients, `P/Q` and `P'/Q'` encoding params (Watson Eq 29/32–33).
  - `DataIO.py` — JSON dump to `data/<YYYY-MM-DD>/sweep_L{L}_{dim}D_<HHMMSS>.json`.

## Conventions
- Boundary conditions: **Open** (OBC). `get_neighbors` and the gradient terms enforce `coords[d] < L-1` for forward derivatives.
- Energy unit throughout: **MeV**. Length conversions via `hc = 197.327 MeV·fm`.
- Data files are date-stamped under `data/`; raw outputs are gitignored (only `data/.gitkeep` is tracked).

## Bosonic cutoff parameters: π_max, Π_max, δπ, n_b
The Watson bosonic encoding has **two physically meaningful parameters and two derived ones**, despite four numbers appearing in the code (`pi_max`, `Pi_max`, `delta_pi`, `n_b`). Knowing which are independent and which are derived matters when choosing or auditing cutoffs.

### Structural relations (Watson §II.C.2.a, Eqs. 28–33)
- Eq. 28: `n_b = log₂(2·π_max/δπ + 1)` → grid relation: `δπ = 2·π_max/(2^{n_b} − 1)`
- Eq. 32: `Π_max = π / (a_L³ · δπ)` → FFT/Nyquist relation: `Π_max ∝ 1/δπ`
- Combined (Eq. 78): `n_b = log₂(2·a_L³·π_max·Π_max/π + 1)`

So once you pick `π_max` and `Π_max`, both `δπ` and `n_b` are determined. **The two free parameters are (π_max, Π_max)**, not (π_max, δπ, n_b).

### What Lemma 5 actually fixes
Lemma 5 gives **two independent lower bounds** (one per parameter), derived from Watson Appendix D's Chebyshev-style energy-bound argument:
- Eq. 75: `π_max ≥ π_max_bound` (controls field-amplitude aliasing for states with `⟨H⟩ ≤ E_bound`)
- Eq. 76: `Π_max ≥ Π_max_bound` (controls conjugate-field aliasing)

`δπ` does not get a separate error bound in App. D — once `(π_max, Π_max)` are chosen, the FFT is exact and `δπ` is determined.

### The integer-ceiling slack and how to use it
Watson p. 26 explicitly notes: "since `n_b` is the number of qubits, it must be an integer. In practice we do not exactly substitute the bounds for `π_max` and `Π_max` into Eq. (78). Rather, we choose the nearest cutoffs above these bounds to ensure `n_b` is an integer."

Current code (`EFTParameters.calculate_dynamic_cutoffs`) returns `pi_max = π_max_bound` and `Pi_max = Π_max_bound` exactly, then `n_b = ceil(log₂(...))`. Because `n_b` is rounded up, the *effective* `Π_max` used downstream — computed from `δπ = 2·π_max/(2^{n_b}−1)` and `Π_max = π/(a_L³·δπ)` — is generally **larger than the Eq. 76 bound**. The slack is silently absorbed into `Π_max`.

### Why this matters for Λ
Λ has contributions roughly `Λ ≈ c_π · π_max² + c_Π · Π_max² + (small cross terms)`. Under the constraint `π_max · Π_max = const` (set by the chosen integer `n_b`), the Lagrangian optimum is `c_π · π_max² = c_Π · Π_max²` (the two contributions equal).

The Λ audit at L=2, dim=3, A=2 measured **c_Π · Π_max² = 73.25%** of Λ and `c_π · π_max² = 26.69%`. The current implicit allocation (slack absorbed into `Π_max`) makes Π² *larger*, going the **wrong direction**. The optimum at this parameter point is `π_max → 1.29·π_max_bound`, `Π_max → 0.78·Π_max_effective` (still above the Eq. 76 bound), giving roughly **~12% Λ reduction** for free.

(The 73/27 ratio drifts with (A, L) since `π_max²`'s T2/T3 pieces grow like `A`; the optimum shifts but the *principle* — re-allocate slack to equalize contributions — is constant.)

### Bottom line for code authors
- `(π_max, Π_max)` are the meaningful dials. `δπ` and `n_b` are derived.
- Macridin/Klco-Savage's NS prescription replaces Lemma 5's energy-bound `(π_max_bound, Π_max_bound)` with NS-optimal bounds tied to a boson-number cutoff. See `claude/research/bosonic-encodings/01_two_readings_oscillator_basis.md` for the three-reading taxonomy (B = NS-optimal cutoff in field-amplitude register, A = occupation-number register, C = basis-adapted compression on top of B via displaced HO or variational basis à la Li 2023). **Reading B is now implemented** as `calculate_ns_cutoffs` (selected via `config.cutoff_method='ns'`); see Live state.
- The slack-reallocation Λ reduction is independent of those three readings and available *now* — it's a tweak inside `calculate_dynamic_cutoffs`, not a new derivation.

## Known issues / TODOs the user has flagged
- `utils.py` has a TODO block about generalizing strides/PBC, but the dimension-agnostic helpers in `LatticeGeometry.py` already cover most of it; the `_1D` helpers in `utils.py` are legacy duplicates.

## Resolved
- *Logical-qubit count not saving properly* (fixed): two bugs — `EstimateResources.py` was reading the pyLIQTR key as `'Logicalqubits'` while pyLIQTR returns `'LogicalQubits'`; and the `combined // 2` halving was wrong. `estimators.run_qubitization_analysis` now returns `LogicalQubits = max(pos, mom)` plus per-walk diagnostics (`Pos_LogicalQubits`, `Mom_LogicalQubits`); `evaluate_resources` exposes `Logical_Qubits`, `Pos_Walk_Logical_Qubits`, `Mom_Walk_Logical_Qubits`; `run_nucleon_sweep.py` saves all three. Verified by `tests/test_logical_qubits.py` (L=2, dim=2, A=1: pos=245, mom=244, reported=245).

## Live state
- Venv: Python 3.10.11, `pip install -r requirements.txt` clean. `pypdf` installed in the venv for Claude's use (read `claude/Watson2025.pdf` via `from pypdf import PdfReader` — 120 pages).
- `tests/` directory exists with `test_logical_qubits.py`. Run from project root: `python -m tests.test_logical_qubits`.
- `claude/research/VC Encoding/` exists with 9 PDFs and a synthesis `00_literature_review.md` (Verstraete–Cirac and successors). Headline conclusions: VC primarily helps the static-nucleon hopping operator (currently O(L) JW string under the existing implementation); 3D VC adds ≈3× qubits but gives O(1) Pauli weight for nearest-neighbor terms. A useful next deliverable would be a JW-vs-VC comparison run through the existing pyLIQTR pipeline.
- `claude/research/lambda audit/` exists with `00_lambda_analysis.md`. Headline: the LCU Λ values (10⁹–10¹³ MeV) are NOT a bug — they're driven >99.9% by the bosonic free-pion sector (`m_π² π²` and `Π²`), where Λ ~ N_sites × pi_max². The fermion sector is ~ppm of total. Switching units doesn't help (Λ/ΔE is dimensionless). **`E_bound` is *not* a meaningful dial** (the contact and chiral-coupling pieces of `pi_max²` dwarf any reasonable `E_bound` choice — < 1% impact). The only cheap Λ-reduction lever is loosening `epsilon_cut` (currently 0.1; Λ ∝ 1/eps). Bigger structural win would be a Galerkin / oscillator pion encoding (replaces Λ ~ cutoff² with Λ ~ ω · N_basis). Quantum-chemistry's small Λ comes from absence of bosonic fields and from low-rank factorizations (DF/THC) that don't transplant directly to digital binary boson encodings.
- **Energy convention in the EFT (clarified on follow-up):** the EFT Hamiltonian is non-relativistic and rest-mass-stripped, so `⟨H⟩ = 0` corresponds to A free nucleons at zero momentum with no pion excitations. Ground-state eigenvalues are therefore `≈ −BE` for bound nuclei (modulo EFT-truncation and lattice-spacing error). The `E_bound` parameter in Lemma 5 is an upper bound on `⟨ψ|H|ψ⟩` for states the cutoff must represent, *not* a value of any eigenvalue — these are different energies that share the word "bound."
- **Path B / Reading B (NS-optimal amplitude cutoff) implemented.** New `cutoff_method` axis on `Config` ('energy_bound' default = Watson Lemma 5; 'ns' = Nyquist-Shannon). `EFTParameters.calculate_ns_cutoffs` sets `pi_max=√(π·2^n_b/(2·a_L^d·ω_0))`, `Pi_max=√(π·2^n_b·ω_0/(2·a_L^d))` (Macridin Eq. 87, field-theory units; ω_0=m_0=m_π), with `N_b=2^n_q` from the shared `estimate_boson_cutoff` (renamed from `calculate_fock_cutoff`) and `n_b=⌈log₂(2N_b)⌉=n_q+1`. Same amplitude register + operators + split-oracle/QFT as energy_bound — only the windows change. `get_Pp_Qp` now takes `dim` (default 3) and uses `a_L**dim`; the matching ratio `Pi_max/pi_max=ω_0` falls out of the existing conjugate grid automatically (realized `Pi_max` = NS `Pi_max`·(2^n_b−1)/2^n_b, one grid cell). Verified by `tests/test_ns_cutoffs.py`. **A/B at L=2,d=3,A=1:** energy_bound (n_b=19, 502 logical qubits, Λ=3.96e9) vs NS (n_b=6, 189 logical qubits, Λ=6.47e5) — ~2.7× qubits, **~6100× Λ** (the projected "~10×" was the naive (20/6)²; at low A nothing floors the bosonic collapse, so the realized cut is far larger — expect it to shrink at high A as the contact floor (∝A) and the heuristic's A-growth of N_b kick in). Plotting (`plot_sweep_data.py`) now keys color/label on a basis+cutoff "series" so amplitude/energy_bound vs amplitude/ns vs fock are distinguishable.
- Current uncommitted changes — verify with `git diff` before assuming.
