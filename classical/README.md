# `classical/` — classical ground-state solvers (Stream A)

The classical baseline the NuQu qubitization resource estimate is meant to beat.
First approach: a **TrimCI-style selected-CI** over the mixed fermion-boson Fock
basis of the dynamical-pion EFT.

## `classical/trimci/`

A self-contained, ED-validated solver that drives the TrimCI expansion–trim cycle
(Zhang & Otten 2025, arXiv:2511.14734) over our *generalized* mixed-state `H_ij`.

| module | what it does |
|---|---|
| `state.py` | `MixedState` = fermion determinant (bitmask) ⊗ boson occupations (tuple) — pion register is an OCCUPATION-NUMBER integer per mode (the "n basis"), so ladder ops are 1-sparse; not the dense bit basis |
| `hamiltonian.py` | builds the full EFT `H` as a ladder-term list, reusing `src_PI`'s `build_eft_hamiltonian` (fock+sparse path) — the *same* Hamiltonian as the quantum side. `N_f` is decoupled from `2**n_b` (classical freedom — pass `build_from_eft(..., N_f=<int>)`) |
| `hij.py` | matrix-free `connections(H, state)` and dense builder; the `H_ij` evaluator |
| `pt2.py` | Epstein-Nesbet PT2 over the mixed selected-CI state (`epstein_nesbet_pt2`, `pt2_from_result`) — re-diagonalizes the saved core for a self-consistent eigenpair, then perturbs |
| `extrapolation.py` | honest `E_infinity` fit (power-law `E∞+a N^-b` + SHCI/PT2 intercept) with uncertainty + guard; `report_energies` = the 3-number report |
| `dump.py` | generalized-FCIDUMP writer/reader (the mixed analogue of an FCI dump) |
| `graph.py` | TrimCI core: `expand` / `local_trim` / `global_trim` / `ground_state` / `ground_state_ensemble` / `exact_ground_state` |
| `lanczos.py` | `lanczos_ground_state` — sparse full-space exact reference (`scipy eigsh`), scales past dense ED |
| `nf_convergence.py` | exact E0 vs Fock cutoff N_f via Lanczos, up to the real heuristic cutoff |
| `backend.py` | hybrid hook to the **official compiled TrimCI** — offloads diagonalization to its C++ Davidson |
| `run_toy.py` | end-to-end toy driver (build → dump → read back → solve → compare to exact) |

### Official TrimCI backend (optional accelerator)
The official TrimCI (`/Users/brandonfriend/Desktop/Projects/TrimCI`) builds into the venv:
```
.venv/bin/python -m ensurepip --upgrade            # venv ships without pip
.venv/bin/python -m pip install /Users/brandonfriend/Desktop/Projects/TrimCI
```
Then `ground_state(..., diag_fn=backend_diagonalize)` runs the official C++ Davidson for
the trim diagonalization (4×–15× faster than `eigh` at N=512–5000). **Only the
diagonalization plugs in** — the selection kernel is hardwired to fermionic Slater–Condon,
so full offload awaits the team's mixed-mode generalization (see `trimci/TODO.md` Route C).
The pure-Python path needs none of this.

A **minimal C++ fork** (`trimci/backend_fork/`, applied + tested) adds a sparse-CSR
Davidson so the official eigensolver runs at O(nnz) memory on our mixed H (no dense wall):
`ground_state(..., diag_fn=backend_diagonalize_sparse)`. Rebuild the fork via
`VIRTUAL_ENV=.venv uv pip install --reinstall-package trimci /path/to/TrimCI`.

### Full C++ path — larger-L solves (fast)
With the connections oracle ported to C++ (`backend_fork/mixed_ci.hpp`, build:
`bash backend_fork/build_mixed_ci.sh`), all three hot kernels (connections, matrix build,
expansion) run in C++ and Python only orchestrates. **55× faster** than pure Python; solves
systems far past exact reach in seconds (L=2 d=2: a 2.7×10⁸-state sector in ~5 s).
```
python -m classical.trimci.run_cpp --L 2 --dim 2 --A 1 --n_b 2 --n_dets 2000 --save
```
Or in code: `cpp_ground_state_ensemble(H, n_elec, n_dets=2000)`.
- **Nucleon number A** is conserved (built in) but not size-free: cost grows mildly with A
  (more nucleons → more active terms).
- **Fock cutoff N_f** is essentially free (selected-CI). C++ path supports **N_f ≤ 65536
  (n_b ≤ 16)**; physically n_b ~ 5–11 and the energy converges by N_f ≈ 16–64. For n_b > 16
  use the pure-Python solver.

## Data pipeline (`io.py`, `analysis.py`, `plotting.py`)
Runs save to `data/classical/<date>/<run-id>/` (gitignored): `metadata.json` (method,
**transform** axis = `COO` bare / `LF` Lang–Firsov, system + solver params, **wall-clock
runtime**, energy, convergence, exact reference), `hamiltonian.mixedfci` (the Hamiltonian
dump), and `groundstate.npz` (the compact wavefunction: fermion dets + boson occ + coeffs).
```python
from classical.io import save_classical_run, load_classical_run
from classical.analysis import index_runs, summary      # tidy DataFrame of all runs
from classical.plotting import plot_convergence, plot_runtime_vs_size
```
`method`/`transform` are recorded per run so future solvers (DMRG/AFQMC) and the Lang–Firsov
frame (task 32) overlay on the same axes. `plot_runtime_vs_size` is the "where does classical
stop being fast" picture for the cost-confront table (task 14).

### Memory safety
Several functions scale explosively (`enumerate_basis` ~ `C(n_ferm,A)·N_f^n_bos`; `build_dense` ~ O(N²)). They now **compute the size up front and raise `MemoryError` past explicit caps** (`MAX_BASIS=1e5`, `MAX_DENSE=6000`) rather than allocating — an unguarded full-sector enumeration previously OOM-crashed a machine. Pass `max_states` / `max_dense` to opt into larger runs deliberately. `lanczos_ground_state` is guarded at every allocation (basis size, projected nnz, projected GB).

### Quick start
```python
from classical.trimci import build_from_eft, ground_state_ensemble, exact_ground_state, write_dump

H = build_from_eft(L=2, dim=1, n_b=1)               # n_b small => N_f=2**n_b small (toy/ED)
E_ed, _ = exact_ground_state(H, n_elec=2)           # exact-diagonalization truth
res = ground_state_ensemble(H, n_elec=2, n_runs=8, n_dets=100, seed=0)
write_dump(H, "system.mixedfci", n_elec=2)          # generalized dump artifact
```

### Energy report — variational / +PT2 / extrapolated (the headline numbers)
`run_cpp.solve_and_report` solves an INDEPENDENT core ladder and prints the three
energies the paper compares to experiment, plus the exact Lanczos reference when the
sector is small enough to validate the extrapolation:
```python
from classical.trimci.run_cpp import solve_and_report
solve_and_report(L=2, dim=1, A=2, n_b=1, cores=(20, 40, 80, 160))
#   variational (best)        E_var           (rigorous upper bound at fixed N)
#   variational + PT2         E_var + dE_PT2   (Epstein-Nesbet; ~always lower)
#   extrapolated (SHCI/PT2)   E∞ ± σ           (intercept at dE_PT2 → 0)  << best
#   exact reference (Lanczos) + (extrapolation − exact) validation gap
```
Three energy notions that must NOT be conflated: **E_var(N)** is the variational
upper bound of the TRUNCATED (fixed L, N_f, frame) H at N determinants; **E_var+PT2**
is a tighter estimate at the same N; **E∞** is the N→∞ (Full CI within that truncated
H) limit, *estimated by extrapolation* — still carrying the L / N_f / EFT-truncation
/ lattice error, so it is neither exact truth nor the experimental binding energy.
The extrapolation is only trustworthy over an INDEPENDENT-solve ladder (a single
growing run's ramp lies for this bosonic system); for slowly-converging 3D systems it
is optimistic from small cores — push the ladder on HPC. The classical boson cutoff
`N_f` need not be a power of two (that is a quantum-only constraint):
`build_from_eft(..., N_f=6)`. Note Fock truncation is *non-variational in N_f* (odd
cutoffs can undershoot) — converge in N_f rather than trusting one point.

### Toy run (the full route, end to end)
```
python -m classical.trimci.run_toy --L 2 --dim 1 --A 2 --n_b 1
```
Builds H, writes the generalized dump, reads it back, runs ensemble TrimCI over a
det-count sweep, and compares to exact diagonalization. Reaches the exact ground
state with a few percent of the sector (e.g. 64/1792 dets at L=2,A=2), variational
throughout — the compact-core effect at toy scale.

> Use `ground_state_ensemble` (not single-run `ground_state`) for production: single
> runs are seed-dependent and can trap in a local basin; the ensemble's min-over-runs
> escapes it, per Zhang & Otten 2025.

### Validate
```
python -m classical.trimci.tests.test_hij
```
Cross-checks the fermion sector vs `openfermion.get_sparse_operator`, the boson sector
vs `openfermion.boson_operator_sparse`, full-`H` hermiticity, the solver vs ED, and the
dump round-trip.

### Exact reference at real N_f
```
python -m classical.trimci.nf_convergence --L 1 --A 1            # single site -> real N_f=32
python -m classical.trimci.nf_convergence --L 2 --n_b_max 2 --trimci
```
Sweeps the Fock cutoff N_f via Lanczos and reports how the exact ground-state energy
converges — the empirical check for the rigorous Tong cutoff (task 25). Reach: a single
site solves the real N_f=32 (131k states, ~14 s); L=2 (with the gradient coupling active)
reaches N_f=4 exactly (32k states) and shows a 4.24 MeV N_f-dependence, while N_f≥8 (2.1M+
states) is TrimCI-only — TrimCI matches Lanczos to 3e-6 MeV at N_f=4 using 1.5% of the dets.

### Fock-cutoff (n_b / N_f) + occupation studies
```
python misc/run_nb_convergence.py A|B|C|D    # n_b convergence + uniform-init control
python misc/run_nb_convergence.py E|F|G      # occupation vs A (SCI d=3 / n_b×A / exact d=1)
python misc/run_nb_convergence.py plot       # all summary plots
```
Certifies the quantum-side Fock cutoff `n_b` classically at ED-impossible sizes (L≥2):
`n_b=3` (N_f=8) converges the L=2 d=3 energy to <0.01 MeV. The occupation studies (E/F/G)
show the GS pion occupation is **A-independent** (set by vacuum squeezing, not nucleon
count) — exact ED at L=2 d=1 gives ⟨N⟩/mode flat to 5 sig figs across A=1→8, and `n_b=3`
stays converged at A=6, A=10. So the cutoff is A-robust; predicted ⟨N⟩ at A=100 (needs
L≥3) ≈ 0.066/mode, near-vacuum. See `claude/research/bosonic-encodings/03,04_*.md`.

### Status & roadmap
Route A (our own solver over mixed `H_ij`) is scaffolded, ED-validated, and pushed to the
real Fock cutoff via sparse Lanczos. Route B (fermionize bosons → standard FCIDUMP → stock
TrimCI) is the path to the fast C++ engine. See `trimci/TODO.md` and `tasks/31`.

> Key finding: released TrimCI is a **purely fermionic** FCIDUMP solver — it computes
> `H_ij` itself from 1-/2-body integrals and has no bosonic modes or custom-`H_ij` hook.
> That's why Route A reimplements the graph algorithm over our mixed `H_ij`.
