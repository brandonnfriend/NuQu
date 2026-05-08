# NuQu — Working Notes for Claude

This file is the assistant's standing context for the NuQu project. Updated as the project evolves.

## Project at a glance
- **NuQu** ("Nuclear Quantum") — fault-tolerant resource estimation for QPE on a Dynamical Pion Chiral EFT, using **Qubitized walk operators** as the time-evolution subroutine instead of Trotterization.
- Started as a Phys 765 (Quantum Algorithms & QEC, S26) final project. Now being extended toward a publication.
- **Public GitHub:** https://github.com/brandonnfriend/NuQu — assume anything committed is world-readable.
- Current branch: `claude-optimization` (work-in-progress; user reviews before pushing).

## Rules for the assistant

### Scope and boundaries
- **Stay inside the project root** (the directory containing this `CLAUDE.md`). Do not create or modify files outside it.
- **Don't push to remote.** The user reviews changes locally and pushes themselves. Never `git push` without an explicit request.
- Work on a feature branch. Don't commit to `main` directly.

### Security (public repo!)
- **No hardcoded credentials, API keys, or tokens.** Always use environment variables.
- Before adding any file that could contain secrets, verify it's covered by `.gitignore`.
- The `.env/` directory and `claude/` directory are already gitignored — keep it that way. The Watson PDF lives in `claude/` and must not be committed (likely copyrighted).
- When creating new files, decide: should this live on the cloud? If not (data dumps, scratch notes, large outputs), add a `.gitignore` entry.

### Research notes and literature reviews
- Save downloaded papers and lit-review documents under `claude/research/<topic>/` (gitignored along with the rest of `claude/`). Each topic folder should contain:
  - The downloaded PDFs (named `Author_Year_short_topic.pdf` — sortable, no spaces in the leading tokens).
  - A markdown synthesis file `00_literature_review.md` (or similar `00_*` prefix to sort first).
- Existing topic folders: `claude/research/VC Encoding/` (Verstraete–Cirac fermion-to-qubit encoding; 9 papers + review).
- Use `pypdf` (already in the venv) for reading PDFs locally. arXiv papers can be downloaded with `curl -sLo <out> https://arxiv.org/pdf/<arxiv_id>` (the `-L` follows redirects; the server returns a real PDF with `content-type: application/pdf`).

### Dependencies
- **Use the project venv** at `.venv/` — Python 3.10.11, pip working, fresh install of `requirements.txt` confirmed by the user before this phase started.
- Update `requirements.txt` if you introduce a new dependency, but **keep it minimal** — only pin what's actually needed for the public reproduction path. Claude-side tooling that isn't part of the public reproduction (e.g., `pypdf` for reading the Watson PDF) goes into the venv but stays out of `requirements.txt`.

### Code-change discipline
- **Write tests with new code.** The repo currently has ~no verification tests; the user wants tests added alongside any new functionality so changes can be verified. A small `test_<module>.py` next to the module, or a short script that prints expected vs. actual, is fine — match what the user asks for case by case.
- **Print a short result summary whenever you run a script.** The user wants a quick read on what happened (e.g., key resource numbers, runtime, file written). Don't dump giant tables.
- **No infinite loops.** Sweeps and HPC-style runs must have explicit bounds; warn before launching anything that could be long-running. Default to small `L` (≤2) and a short `A` range for local smoke tests.

### Workflow
- The default reviewer is the user. Stage diffs, summarize what changed, and let them push.
- When in doubt about scope, ask before doing.
- **Keep this file in sync.** Whenever a change in the project resolves, contradicts, or extends notes in `CLAUDE.md` (rules, known issues, TODOs, live-state, conventions), update the relevant section in the same change so this file is always a current snapshot — not a stale to-do list.

## Project context (from Watson 2025 + the codebase)

### The reference paper
- Watson, Bringewatt, Shaw, Childs, Gorshkov, Davoudi, *"Quantum Algorithms for Simulating Nuclear Effective Field Theories"* — arXiv:2312.05344 (2023/2025). Local PDF at `claude/Watson2025.pdf`.
- Watson et al. estimate fault-tolerant resources for several nuclear EFTs (pionless, one-pion-exchange / static-pion, dynamical-pion) using **Trotterization** for `e^{iHt}`. Pionless is cheapest; **dynamical pion is by far the most expensive**, and that is the regime this project targets.
- Specific results from Watson reused here:
  - **Lemma 5** → field-cutoff and qubit-count formulas (Eqs. 75–78) for the bosonic pion encoding: `pi_max`, `Pi_max`, `n_b = ceil(log2(...))`. Implemented in `src_PI/hamiltonians/core/EFTParameters.py::calculate_dynamic_cutoffs`.
  - **Lemma 23** → Trotter step T-cost: `g(L,n_b) * (1.15 log10(2 g/δ) + 9.2)` with `g = (45 n_b² + 114 n_b + 76) L³`. Implemented in `src_PI/trotter_theory/trotter_theory.py`.
  - **Lemma 78** → WT–WT commutator bound (used for nested-commutator Trotter error estimates).
- Physical inputs (Table I/IV of Watson) are encoded in `EFTParameters.get_physical_parameters` — `M_N=938`, `m_pi=135`, `f_pi=93`, `g_A=1.26`, `a_L=2.2 fm`, contact terms `C=-51.94`, `C_I=1.73` MeV.

### What this project does differently
- Replaces Trotterization with **Qubitization** (block-encoded walk operator + QPE) using `pyLIQTR`. Initial finding: Qubitization gives **lower T-counts** than Trotterization at the same accuracy — that's the headline result so far and the motivation for the publication push.
- The Hamiltonian is **split-oracle**: position-basis terms (`H_pos`: free pion mass + gradient + axial-vector + Weinberg–Tomozawa + static nucleon hopping/contacts) and momentum-basis terms (`H_mom`: pion conjugate-momentum kinetic) are encoded as separate walks. They run sequentially on the same hardware, so the peak logical-qubit count is `max(pos_walk_qubits, mom_walk_qubits)` (T/Clifford gate counts, by contrast, are summed across the two walks). Each walk step also requires a QFT/IQFT pair to swap bases — counted in `calculate_qft_cost`.
- Qubits per site: `4 + 3*n_b` (4 nucleon modes from spin × isospin, plus 3 pion species × `n_b` bits each). Total qubits: `L^dim * (4 + 3 n_b)`.

### Key code map
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

### Known issues / TODOs the user has flagged
- `utils.py` has a TODO block about generalizing strides/PBC, but the dimension-agnostic helpers in `LatticeGeometry.py` already cover most of it; the `_1D` helpers in `utils.py` are legacy duplicates.

### Resolved
- *Logical-qubit count not saving properly* (fixed): two bugs — `EstimateResources.py` was reading the pyLIQTR key as `'Logicalqubits'` while pyLIQTR returns `'LogicalQubits'`; and the `combined // 2` halving was wrong. `estimators.run_qubitization_analysis` now returns `LogicalQubits = max(pos, mom)` plus per-walk diagnostics (`Pos_LogicalQubits`, `Mom_LogicalQubits`); `evaluate_resources` exposes `Logical_Qubits`, `Pos_Walk_Logical_Qubits`, `Mom_Walk_Logical_Qubits`; `run_nucleon_sweep.py` saves all three. Verified by `tests/test_logical_qubits.py` (L=2, dim=2, A=1: pos=245, mom=244, reported=245).

### Conventions
- Boundary conditions: **Open** (OBC). `get_neighbors` and the gradient terms enforce `coords[d] < L-1` for forward derivatives.
- Energy unit throughout: **MeV**. Length conversions via `hc = 197.327 MeV·fm`.
- Data files are date-stamped under `data/`; raw outputs are gitignored (only `data/.gitkeep` is tracked).

## Live state
- Venv refreshed at the start of this phase: Python 3.10.11, `pip install -r requirements.txt` clean. `pypdf` installed in the venv for Claude's use (read `claude/Watson2025.pdf` via `from pypdf import PdfReader` — 120 pages).
- `tests/` directory exists with `test_logical_qubits.py`. Run from project root: `python -m tests.test_logical_qubits`.
- `claude/research/VC Encoding/` exists with 9 PDFs and a synthesis `00_literature_review.md` (Verstraete–Cirac and successors). Headline conclusions: VC primarily helps the static-nucleon hopping operator (currently O(L) JW string under the existing implementation); 3D VC adds ≈3× qubits but gives O(1) Pauli weight for nearest-neighbor terms. A useful next deliverable would be a JW-vs-VC comparison run through the existing pyLIQTR pipeline.
- `claude/research/lambda audit/` exists with `00_lambda_analysis.md`. Headline: the LCU Λ values (10⁹–10¹³ MeV) are NOT a bug — they're driven >99.9% by the bosonic free-pion sector (`m_π² π²` and `Π²`), where Λ ~ N_sites × pi_max². The fermion sector is ~ppm of total. Switching units doesn't help (Λ/ΔE is dimensionless). **`E_bound` is *not* a meaningful dial** (the contact and chiral-coupling pieces of `pi_max²` dwarf any reasonable `E_bound` choice — < 1% impact). The only cheap Λ-reduction lever is loosening `epsilon_cut` (currently 0.1; Λ ∝ 1/eps). Bigger structural win would be a Galerkin / oscillator pion encoding (replaces Λ ~ cutoff² with Λ ~ ω · N_basis). Quantum-chemistry's small Λ comes from absence of bosonic fields and from low-rank factorizations (DF/THC) that don't transplant directly to digital binary boson encodings.
- **Energy convention in the EFT (clarified on follow-up):** the EFT Hamiltonian is non-relativistic and rest-mass-stripped, so `⟨H⟩ = 0` corresponds to A free nucleons at zero momentum with no pion excitations. Ground-state eigenvalues are therefore `≈ −BE` for bound nuclei (modulo EFT-truncation and lattice-spacing error). The `E_bound` parameter in Lemma 5 is an upper bound on `⟨ψ|H|ψ⟩` for states the cutoff must represent, *not* a value of any eigenvalue — these are different energies that share the word "bound."
- Current uncommitted changes — verify with `git diff` before assuming.
