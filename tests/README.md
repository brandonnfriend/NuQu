# NuQu test suite — tiers and expected runtimes

Measured 2026-09-05 (laptop, Python 3.10.11, `.venv`). Counts and times drift; the
tiers and the ordering do not.

| command | tests | wall | when |
|---|--:|--:|---|
| `python -m pytest -q -m "not slow"` | 325 | **~6 min 26 s** (measured 2026-10-20) | the edit/commit loop |
| `python -m pytest -q` | 330 | not re-measured since 2026-09-05 | before a commit that touches physics, and before any release |

> **TODO — the `slow` markers are stale; fix soon.** The fast tier was ~1 min 40 s when the five
> tests below were marked. It is now **~6 min 26 s**, i.e. the tier no longer does its job: the
> suite roughly doubled (217 → 330 tests) and the new slow tests were never marked. Re-measure with
> `python -m pytest -q --durations=30` and re-mark whatever now dominates. Until then `-m "not slow"`
> is not a usable edit-loop tier. The five rows below are the *original* measurement, not current.

The default run is **everything** — `slow` is opt-OUT, never opt-in, so nothing
silently stops running. Five tests carry ~70% of the wall time and are marked
`@pytest.mark.slow`:

| test | ~s | what it protects |
|---|--:|---|
| `test_dmrg.py::test_area_law_onset` | 158 | entanglement growth / area-law onset in the DMRG baseline |
| `test_back_evaluate.py::test_eps_leak_reported_and_shrinks_with_reference_cutoff` | 56 | the frame back-evaluation leak bound shrinking with the reference cutoff |
| `test_hij.py::test_frame_projector_lf_variational` | 33 | projector-LF stays variational |
| `test_compiled_resources.py::test_no_dense_matrixgate_at_counting_boundary` | 22 | pyLIQTR does not fall back to a dense MatrixGate at the counting boundary |
| `test_hij.py::test_occupation_A_independence` | 15 | boson occupation is A-independent |

They are physics/regression checks, not scaffolding — run the full suite before a release.

## Long silent stretch
The audit noted the suite "pauses for several minutes without progress output". That is
`test_area_law_onset` (~2.5 min in one call) plus the back-evaluate group. `-m "not slow"`
removes both; `-q --durations=25` shows where time actually goes.

## Known warnings (all expected, none silenced blindly)
- `os.fork()` + JAX in `test_hij.py::test_dets_vs_L` / `test_robustness_guards` — the
  fork-ensemble solver forks while JAX (pulled in by an unrelated import) is loaded.
  Benign here: the forked children are the C++/numpy selected-CI path and never touch
  JAX. See `hpc/HPC_WORKFLOW.md` §5 on the fork ensemble.
- The dim≠3 Watson-Lemma-5 `RuntimeWarning` is asserted with `pytest.warns` in
  `test_tong_cutoff.py`, so it stays a real signal for callers rather than suite noise.

## Standalone runners
Several files (notably `classical/trimci/tests/test_hij.py`, `tests/test_pt2_cap.py`)
also run directly — `python -m tests.test_tong_cutoff`, `python tests/test_hij.py` — and
print a human-readable report. Both entry points work; pytest is the gate.
