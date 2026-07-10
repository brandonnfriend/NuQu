# Minimal C++ fork of official TrimCI — for the NuQu mixed fermion-boson CI

Source-of-truth for the fork patches. The official TrimCI lives at
`/Users/brandonfriend/Desktop/Projects/TrimCI` (a clone). These are the *minimal*
changes that make its fast C++ engine run smoother on our mixed (Anderson–Holstein-like)
Hamiltonian, where the stock engine assumes a purely fermionic Slater–Condon-from-FCIDUMP
form. Two tiers: Tier 1 is implemented + tested; Tier 2 is the design for the full offload.

---

## Tier 1 — sparse (CSR) Davidson  ✅ implemented + tested (2026-06-29)

**Problem it removes.** The stock hook we use, `fast_expansion.davidson_solve_dense(H, diag)`,
needs the full N×N matrix — O(N²) memory. Our hybrid feeds it the projected H of a
selected determinant set, and the complex→real 2N embedding doubles the dimension, so
the dense matrix is the memory wall at scale (50k dets → a 100k×100k dense matrix ≈ 80 GB).

**The change.** One new pybind lambda, `davidson_solve_sparse(data, indices, indptr, diag,
params, guess)`, added to `cpp/trimci_core/bindings/bind_fast_expansion.cpp` right after
`davidson_solve_dense`. It runs the **same** C++ `davidson_solve` over a CSR sparse matvec
(`sigma = H @ v`, OpenMP-parallel) — O(nnz) memory, no other backend change, **no CMake
edit** (same translation unit). The exact code is `bind_sparse_davidson.snippet.cpp`.

**Apply + build (uv):**
```
# (the patch is already applied to the clone; to re-apply elsewhere, paste the
#  snippet after davidson_solve_dense in bind_fast_expansion.cpp)
cd /Users/brandonfriend/Desktop/Projects/NuQu
VIRTUAL_ENV=.venv uv pip install --reinstall-package trimci /Users/brandonfriend/Desktop/Projects/TrimCI
```

**Python side.** `classical/trimci/backend.py::davidson_lowest_sparse` /
`backend_diagonalize_sparse` build the projected H **directly as a scipy CSR** via our
`connections` oracle (never forming the dense matrix), apply the 2N real embedding for the
complex-Hermitian case, and call `davidson_solve_sparse`. Use it as
`ground_state(..., diag_fn=backend_diagonalize_sparse)`.

**What it buys.** The official C++ Davidson, at O(nnz) memory, on our mixed H — so the trim
diagonalization scales to large selected spaces without the dense wall. (The Davidson
preconditioner exploits our diagonal dominance; see `00_literature_review.md` §5.3.)

---

## Tier 2 — provider interface for the SELECTION kernel

Tier 1 offloads only the *diagonalization*. The *selection* (`pool_build`, `run_trim`,
`run_expansion`, `generate_excitations`, `davidson_solve_matfree`) is still hardwired to
`(Determinant, h1, eri)` Slater–Condon — the remaining Python bottleneck at scale. To
offload it, the engine needs a **matrix-element provider** instead of the integral kernel.

### The connections port — DRAFTED + VALIDATED (2026-06-29)

The kernel of the provider — our `hij.connections` (apply every ladder term to a mixed
Fock ket; fermion signs, boson `√` factors, hard cutoff) — is ported to C++ here:

- **`mixed_ci.hpp`** — header-only: `MixedDet`, `Term`, `apply_fermion_ops`,
  `apply_boson_ops`, `connections`, and the **`MixedHijProvider`** (`diagonal` + `neighbors`).
- **`mixed_ci_pybind.cpp`** + **`build_mixed_ci.sh`** — a standalone pybind module (no TrimCI
  dependency) to test/drive it. Build: `bash classical/trimci/backend_fork/build_mixed_ci.sh`
  (needs `uv pip install pybind11`).
- **`test_mixed_ci_cpp.py`** — exhaustive validation vs the Python reference.

**Status: bit-for-bit identical to Python** across the L=1 (complex Weinberg–Tomozawa),
A=2, full-coupling L=2, and 32768-state N_f=4 sectors (`max|ΔH_ij| = 0`), and **~19× faster**
(suite test `[11]`). So the hard, error-prone part of Tier 2 — the matrix-element logic — is
done and proven.

**Representation (arbitrary mode count, 2026-06-30).** The fermion mask is a
`std::vector<uint64_t>` of `W = ceil(n_ferm_modes/64)` little-endian words (was `uint64`,
then `__uint128_t`) — **no cap on lattice size**: L=3 in 3d is 108 modes (W=2), L=4 is 256
(W=4), and larger scales with RAM. Bosons are `vector<uint16_t>` (N_f ≤ 65536). All bit ops
route through the `Ferm` typedef + `fbit`/`fflip`/`fpopcount_below`, so the representation is
the single chokepoint. Masks cross the Python boundary as `(N, W)` uint64 arrays. Validated
vs the Python reference at W = 1,2,3,5 with `max|ΔH| = 0` including masks living entirely in
high words (suite test `[14]`). **RAM safeguard:** the `MixedHijProvider` connection cache is
bounded by a byte budget (`max_cache_bytes`, default 1 GiB) as well as a state count, and
clears wholesale on overflow — at large L each cached state holds a Hamiltonian-sized
`ConnMap`, so the byte budget (not the count) is the real guard. Raise it on big-memory/HPC
hosts via `H._cpp_cache_bytes`. The production form packs both registers into TrimCI's
determinant container.

### The full C++ hot path — WIRED + WORKING (2026-06-29)

Rather than fork TrimCI's `run_trim`/`run_expansion` internals, we offload the three hot
kernels to C++ and let Python only *orchestrate* (random init, top-k ranking, set ops — all
O(N), cheap):

- **`build_coo(states)`** (C++) — the projected complex H over a det set as COO, the matrix
  build inside every subspace diagonalization.
- **`expand(states, coeffs)`** (C++) — neighbor generation + `|H_ij c_j|` scoring for expansion.
- diagonalization via the official **C++ sparse Davidson** (Tier 1).

Python side: `backend.cpp_diagonalize` / `cpp_expand` / `cpp_ground_state[_ensemble]`, wired
through `graph.ground_state`'s `diag_fn`/`expand_fn` hooks. **Result: 55× faster than pure
Python** on the L=2,N_f=4 toy (0.9 s vs 48 s, same energy), and **larger systems in seconds**:
L=2,N_f=8 (2.1e6 states) 1.5 s; L=3,d=1 (3.1e6) 2.5 s; **L=2,d=2 (2.7e8 states) ~5 s**.
Validated to ED where reachable (`run_cpp.py` converges to E0 to 1e-12; suite test `[12]`).

This makes the deep run_trim fork below **unnecessary for our goal** — we get C++ speed
without touching TrimCI's selection internals. The design is kept only as the alternative
"native engine" path (relevant if/when the team's mixed-mode release lands).

### Alternative (deep) integration — design only, not needed

**Design (the "modular update" we discussed):**

**Design (the "modular update" we discussed):**

1. **State encoding.** Generalize the `Determinant` (alpha/beta uint64 bitstrings) to a
   `MixedDeterminant` = fermion bitstring ⊕ packed boson occupations
   (`ceil(log2 N_f)` bits per pion mode). Fermion sign/antisymmetry stays on the fermion
   bits; the boson bits are plain occupation indices (symmetric, no sign).

2. **Provider abstraction** (replaces the hardwired Slater–Condon calls):
   ```cpp
   struct HijProvider {
       virtual double diagonal(const MixedDeterminant& d) const = 0;          // H_ii
       virtual std::vector<std::pair<MixedDeterminant,double>>                 // edges
               neighbors(const MixedDeterminant& d) const = 0;                // = our connections()
       virtual ~HijProvider() = default;
   };
   ```
   `neighbors` IS our `hij.connections`: apply each Hamiltonian term to the ket, return the
   connected states + matrix elements.

3. **Refactor the ~3 call sites** in `run_trim` / `pool_build` / `run_expansion` /
   `compute_diagonals` to dispatch through `HijProvider` instead of `compute_H_ij(...,h1,eri)`
   and `generate_excitations(...,n_orb)`. Davidson, trim, OpenMP, the graph logic — all
   unchanged.

4. **Two provider impls:** the existing fermionic one (default), and a `MixedTermProvider`
   that holds our flat ladder-term list and implements `connections` in C++. The latter is a
   mechanical port of `classical/trimci/hij.py` (apply fermion ops with sign + boson ladder
   factors + cutoff) — bounded work, with the Python as the reference + correctness oracle.

   *(Do NOT use a pybind trampoline that calls Python `connections` per edge — the GIL +
   per-edge Python overhead would erase the C++ speedup. The provider must be C++.)*

**Why not just do Tier 2 now:** it needs the C++ `connections` port and touches the
engine's internals; it overlaps with the Otten group's announced native mixed-mode
generalization, which presumably exposes exactly this provider interface. Best to align
with theirs rather than maintain a parallel fork. Tier 1 is the standalone, low-risk win.

**Asks for the team (cheaper for them than a fork for us):**
- A **complex-Hermitian** Davidson (drops our 2N real embedding → 2× on the eigensolver).
- The **`HijProvider` hook** above — then our `connections` plugs straight into their
  expansion/trim/Davidson at full C++ speed, no fermionization, no integral file.

---

## Files here
- `bind_sparse_davidson.snippet.cpp` — the Tier 1 C++ (canonical copy; applied to the clone).
- `README.md` — this file.
