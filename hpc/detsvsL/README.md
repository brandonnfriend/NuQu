# HPC run: classical dets-vs-L scaling (first real workflow test)

The first end-to-end HPC job on the classical side — the Phase-C **dets-vs-L(V)
exponent** measurement (`N*(L; ε)` → the scaling exponent γ, "does selected-CI
cost grow polynomially or exponentially in volume?"). On the laptop this returns
only *bounds* because it needs cores past ~16k; this run pushes toward **50k dets**
on the cluster. Treat it as a **workflow shakedown that also produces
larger-than-laptop data** — not yet the final science run.

## What it runs

`misc/run_dets_vs_L.py` at **L = 2, 3** (3D, dilute A = 1, N_f = 4), cores up to
**50 000**, `n_runs = 4`, ε ∈ {1.0, 0.1} MeV/site, with a **1 h/rung wall cap** so
the ladder self-limits and stops honestly where the budget runs out (the JSON
records exactly where). Output: `data/classical/detsvsL_hpc_<stamp>.{json,png,log}`,
tagged `"hpc": true` in the JSON for provenance.

- `n_runs = 4` matches the validated laptop config, so this run isolates the one
  variable Phase C is blocked on (deeper cores). **Production** bumps `n_runs ≥ 16`
  (the Phase-D seed-fragility fix) and adds L = 4 — edit the params block in
  `run_detsvsL.sh`.

## Environment (the "packages the computer needs")

Deliberately minimal and verified empirically (everything else was blocked and the
run still completed): **openfermion** (Hamiltonian build, pulls cirq), **numpy /
scipy** (arrays + `scipy.sparse.linalg.eigsh` diagonalization), **matplotlib**
(headless plot), **pybind11** (build-time, compiles the `mixed_ci` C++ hot path).
See `requirements-hpc.txt`.

**Not needed on the cluster** (kept off deliberately): pyLIQTR (quantum estimator),
the official `trimci` pip package and its jax / netket / flax / numba deps
(diagonalization uses scipy eigsh instead of the official C++ Davidson — same
answer, one fewer heavy dependency tree), pyscf, block2.

The venv + compiled backend live on the **shared `/nfs_scratch` checkout**, so qis
execute nodes read them directly — no per-job runtime tarball. (A transferable
tarball would only be needed to run on wide-pool nodes that don't mount
`/nfs_scratch`; not used here.)

## Files

| file | role | run on |
|---|---|---|
| `requirements-hpc.txt` | pinned minimal runtime | — |
| `setup_env.sh` | one-time: venv + install + build C++ + self-test | login (`ssh hep`) |
| `run_detsvsL.sh` | Condor executable (activate venv, run driver) | execute node |
| `submit_detsvsL.sh` | mint rundir + write submit + `condor_submit` | submit (`ssh hep-submit`) |

## Workflow

### 1. One-time environment setup — on a login node (`ssh hep`)
```sh
ssh hep
cd /nfs_scratch/bfriend3/NuQu/NuQu && git pull        # pull the pushed commit
sh hpc/detsvsL/setup_env.sh                            # venv + build + self-test
```
The self-test prints `self-test RAN OK` — that confirms openfermion, the `mixed_ci`
C++ backend, scipy eigsh, and the driver all work on the cluster before any job is
queued. (Requires `uv` on PATH; if the `.so` fails to load on a qis node, rerun the
build step on a qis node — same command.)

### 2. Submit — on the pinned submit node (`ssh hep-submit`)
```sh
ssh hep-submit
cd /nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL && sh submit_detsvsL.sh
```
Prints the rundir name and the cluster id. Resources: 4 CPUs, 24 GB, 4 GB disk,
pinned to qis1–4. (If it sits idle, the qis pin + 24 GB may be matching few slots —
lower `request_memory` in `submit_detsvsL.sh`.)

### 3. Monitor — same pinned node (schedd is per-login-node)
```sh
ssh hep-submit "condor_q <cluster> -af:V ClusterId ProcId JobStatus RemoteHost RemoteUserCpu"
# status: 1=idle 2=running 5=held ;  held/gone -> condor_history <cluster> ... HoldReason
```

### 4. Retrieve — results land in TWO places
- Durable, on the shared FS: `/nfs_scratch/bfriend3/NuQu/NuQu/data/classical/detsvsL_hpc_<stamp>.{json,png,log}`
- Returned in the rundir: `hpc/detsvsL/rundir-*/`

```sh
rsync -av hep:/nfs_scratch/bfriend3/NuQu/NuQu/data/classical/detsvsL_hpc_'*'.{json,png,log} data/classical/
```

## Sanity checks before trusting numbers
- JSON `"hpc": true` and `"host"` name a qis node.
- The `L=2 d=1` trust anchor still holds elsewhere (extrap = exact Lanczos); this run
  is L≥2 3D where there is no exact reference, so per-L points are honest
  point-or-bound (the pipeline never fabricates an exponent).
- `robustness.max_n_ext` below the 50M semistochastic-PT2 trigger. If it trips,
  deterministic PT2 is over budget at these cores — the sampled-tail variant is
  unbuilt, so that's a build-first signal, not a silent approximation.
