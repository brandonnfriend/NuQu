# NuQu — Working Notes for Claude

This file is the assistant's standing context for the NuQu project. Keep it short — deeper project context (paper details, code map, conventions, live state) lives in `PROJECT_CONTEXT.md` and should be read on demand when the task needs it.

## Project in one paragraph
**NuQu** ("Nuclear Quantum") — fault-tolerant resource estimation for QPE on a Dynamical Pion Chiral EFT, using **Qubitized walk operators** (via `pyLIQTR`) instead of Trotterization. Started as a Phys 765 (Quantum Algorithms & QEC, S26) final project, now being extended toward a publication. Public GitHub: https://github.com/brandonnfriend/NuQu — assume anything committed is world-readable.

## Rules for the assistant

### Scope and boundaries
- **Stay inside the project root** (the directory containing this `CLAUDE.md`). Do not create or modify files outside it.
- **`human_knowledge/` is read-only.** Files inside `human_knowledge/` are written by the user to record their own understanding, intuition, and direction. Read them freely for context, but never edit, rewrite, rename, delete, or create files inside that folder. If you think something there is wrong or stale, raise it in conversation instead of editing.
- **Don't push to remote.** The user reviews changes locally and pushes themselves. Never `git push` without an explicit request.
- Work on a feature branch. Don't commit to `main` directly.

### Security (public repo!)
- **No hardcoded credentials, API keys, or tokens.** Always use environment variables.
- Before adding any file that could contain secrets, verify it's covered by `.gitignore`.
- The `.env/` directory and `claude/` directory are already gitignored — keep it that way. The Watson PDF lives in `claude/` and must not be committed (likely copyrighted).
- When creating new files, decide: should this live on the cloud? If not (data dumps, scratch notes, large outputs), add a `.gitignore` entry.

### Research notes and literature reviews
- Save downloaded papers and lit-review documents under `claude/research/<topic>/` (gitignored). Each topic folder should contain:
  - PDFs named `Author_Year_short_topic.pdf` — sortable, no spaces in the leading tokens.
  - A markdown synthesis file `00_literature_review.md` (or similar `00_*` prefix to sort first).
- Use `pypdf` (already in the venv) for reading PDFs locally. arXiv papers can be downloaded with `curl -sLo <out> https://arxiv.org/pdf/<arxiv_id>` (the `-L` follows redirects).

### Dependencies
- **Use the project venv** at `.venv/` — Python 3.10.11.
- Update `requirements.txt` if you introduce a new dependency, but **keep it minimal** — only pin what's actually needed for the public reproduction path. Claude-side tooling (e.g., `pypdf`) goes into the venv but stays out of `requirements.txt`.

### Code-change discipline
- **Write tests with new code.** A small `test_<module>.py` in `tests/`, or a short script that prints expected vs. actual, is fine — match what the user asks for case by case.
- **Print a short result summary whenever you run a script.** Quick read on what happened (key resource numbers, runtime, file written). Don't dump giant tables.
- **No infinite loops.** Sweeps and HPC-style runs must have explicit bounds; warn before launching anything that could be long-running. Default to small `L` (≤2) and a short `A` range for local smoke tests.
- **Comparison switches for substantial design changes.** When introducing a substantive design change (different basis, different encoder, different fermion mapping, different cutoff prescription, etc.), **keep the previous path alive as a runtime flag/switch rather than replacing it.** Direct A-vs-B comparison numbers are themselves publishable results — they quantify the gain from each change instead of just asserting it. Both paths must pass their own tests; the dual-maintenance cost is real but acceptable for the duration of the comparison study, and the deprecated path can be retired once the result is published. Examples on the roadmap: Fock vs. field-amplitude basis; sparse-oracle vs. PauliLCU block encoder; Verstraete–Cirac vs. Jordan–Wigner fermion encoding; NS-optimal vs. Watson-Lemma-5 cutoffs.

### Workflow
- The default reviewer is the user. Stage diffs, summarize what changed, and let them push.
- When in doubt about scope, ask before doing.
- **Keep notes in sync.** When a change resolves, contradicts, or extends anything in `CLAUDE.md` or `PROJECT_CONTEXT.md` (rules, known issues, TODOs, live state, conventions), update the relevant section in the same change so these files are always a current snapshot — not a stale to-do list.

## Common commands
- Run the nucleon sweep: `python run_nucleon_sweep.py` (default L=2, dim=3, A∈[1..100] sparse).
- Run the logical-qubit verification test: `python -m tests.test_logical_qubits`.
- Activate the venv: `source .venv/bin/activate`.

## When to read PROJECT_CONTEXT.md
Read it before:
- Modifying Hamiltonian construction, normalization, or resource estimation code.
- Interpreting Λ values or T-counts in the saved JSON output.
- Reasoning about Watson 2025 lemmas, the split-oracle structure, or qubit-counting conventions.
- Picking up follow-up work referenced in "Live state" (VC encoding comparison, lambda-reduction levers, etc.).

## Open homework (user-owned tasks)
Bring up periodically when contextually relevant; don't nag.

- **Rigorous Fock-basis cutoff from Tong et al. 2022.** `calculate_fock_cutoff` in `src_PI/hamiltonians/core/EFTParameters.py` currently uses a hand-rolled heuristic `n_q = max(4, ceil(4 + log₂(1+A)))`. The Tong–Albert–McClean–Preskill–Su 2022 paper (arXiv:2110.06942, PDF in `claude/research/bosonic-encodings/`) proves `N_f = O(polylog(1/ε))` suffices for bosonic Hamiltonians with bounded-degree polynomial coupling (our case: H_AV degree 1, H_WT degree 2). The open task: instantiate their Theorem 6 for our specific Hamiltonian — bound the operator norms of the per-site polynomial pieces, plug the lattice geometry into their multi-site analysis, derive the explicit `N_f(ε, t, L, A, couplings)` formula. Would be a publishable contribution to the NuQu paper. Until then the heuristic is a placeholder.
