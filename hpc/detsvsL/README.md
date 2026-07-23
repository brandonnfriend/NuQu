# HPC run: classical dets-vs-L scaling

The first real HPC job on the classical side — the Phase-C **dets-vs-L(V) exponent**
measurement (`N*(L; ε)` → the scaling exponent γ, "does selected-CI cost grow
polynomially or exponentially in volume?"). On the laptop this returns only *bounds*
because it needs cores past ~16k; this run pushes toward **50k dets** on the cluster.

## Design: the job provisions its own environment on the compute node

There is **no login-node setup step** and **nothing pre-built on shared storage**. A
qis execute node (verified by probe, cluster 289774 on qis3) mounts `/nfs_scratch`,
has outbound internet + `g++ 11.5`, but has **no `uv`, only Python 3.9, and no home
mount**. So `run_detsvsL.sh`, executing on the node, does everything in the Condor
sandbox and throws it away on exit:

1. installs `uv` (into the sandbox — `HOME`/`UV_*` all redirected there),
2. provisions **CPython 3.10** and a venv,
3. `pip`-installs the pinned deps (`requirements-hpc.txt`),
4. compiles the `mixed_ci` C++ hot path,
5. runs the driver — diagonalizing via **scipy eigsh** (no official TrimCI / jax /
   netket / pyLIQTR needed).

The repo is only **read** from `/nfs_scratch`; the venv/interpreter/build live in the
sandbox and are discarded; only the small **rundir** (JSON + PNG + log) transfers back.

## Files

| file | role | run on |
|---|---|---|
| `requirements-hpc.txt` | pinned minimal runtime (installed on the node) | — |
| `run_detsvsL.sh` | Condor executable — self-provisions env + runs driver; `arg1 = test\|full` | execute node |
| `submit_detsvsL.sh` | mint rundir + write submit + `condor_submit`; `sh submit_detsvsL.sh [test\|full]` | submit (`ssh hep-submit`) |

**`test`** = L=2 1D, cores→400 (~30s; a full provisioning + build + solve shakedown).
**`full`** = L=2,3 dilute 3D, N_f=4, cores→50k, `n_runs=4`, 1 h/rung wall cap (self-limits;
the JSON records where it stopped). Production bumps `n_runs`≥16 (Phase-D seed-fragility
fix) and adds L=4 — edit the `full` branch in `run_detsvsL.sh`.

## Workflow

Keep the checkout current, then submit — everything else is on the node:
```sh
ssh hep-submit
cd /nfs_scratch/bfriend3/NuQu/NuQu && git pull
cd hpc/detsvsL
sh submit_detsvsL.sh test     # optional ~30s shakedown
sh submit_detsvsL.sh full     # the real run
```

Monitor (schedd is per-login-node — query from the same pinned node):
```sh
ssh hep-submit "condor_q <cluster> -af:V ClusterId ProcId JobStatus RemoteHost RemoteUserCpu"
# 1=idle 2=running 5=held ;  held/gone -> condor_history <cluster> ... HoldReason
```

Retrieve — results come back inside the rundir:
```sh
rsync -av hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/rundir-<...>/data/classical/ data/classical/
```

## Sanity checks
- JSON `"hpc": true` and `"host"` name a qis node (provenance tag).
- Per-L points are honest point-or-bound; a shallow/wall-capped ladder yields a
  `NO_REFERENCE`/bound entry, never a crash or a fabricated exponent.
- `robustness.max_n_ext` below the 50M semistochastic-PT2 trigger. If it trips,
  deterministic PT2 is over budget at these cores — a build-first signal (the
  sampled-tail variant is unbuilt), not a silent approximation.

## Validation status
- Probe (289774): qis3 mounts /nfs_scratch, internet reachable, g++ present.
- `test` (289775): self-provision + C++ build + solve all green on qis3 in ~30s;
  198.3166 MeV/site (matches laptop), JSON tagged `"hpc": true`.
- `full` (289776): submitted L=2,3 → 50k.
