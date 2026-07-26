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
| `run_detsvsL.sh` | single-job runner (shakedown / one-shot); `arg1 = test\|full` | execute node |
| `submit_detsvsL.sh` | mint rundir + submit the single job; `sh submit_detsvsL.sh [test\|full]` | submit |
| **`run_detsvsL_shard.sh`** | **one (L, seed) shard** of the parallel campaign; `args: L seed campaign` | execute node |
| **`submit_detsvsL_campaign.sh`** | **build the (L,seed) grid + submit all shards**; `sh submit_detsvsL_campaign.sh [n_seeds] "[L list]"` | submit |
| `misc/run_detsvsL_shard.py` | one shard's ladder (n_runs=1, frame) → per-shard JSON | (driver) |
| `misc/combine_detsvsL.py` | min-over-seeds → per-L reference → exponent fit | laptop, post-hoc |

## Production: parallel (L, seed) campaign, in the compacting frame

The n_runs ensemble is parallelised across jobs — **1 shard = 1 (L, seed), n_runs=1,
full ladder** — so wall-clock is the *single slowest shard*, not the ~50 h serial sum.
Default grid = **4 L (L=2–5) × 16 seeds = 64 shards**. Per-L memory + ladder depth are
sized in `submit_detsvsL_campaign.sh`: L=2→128k/8G, L=3→256k/48G, L=4→256k/128G,
L=5→128k/128G (L=5 stays at 128k — it's the load-bearing 4th point and 256k on 125
sites risks an OOM that would lose the whole shard, which saves only at the end).
Combine takes the **min over seeds per (L, core)** to reconstruct the ensemble, then
extrapolates and fits — identical result to n_runs=16 in-process, and it keeps the
independent-random-init requirement (each seed is its own job, no warm starts).

Every shard runs in the **per-mode analytic squeeze frame** (`transform=gaussian`,
auto `-r*`). Measured win (L=2 3D): the framed basis at **1000 dets beats the bare
basis at 16000** (>16× determinant compaction) and has ~⅓ the terms (~3–10× faster
per solve) — so the study **pins the reference at modest cores** where the bare basis
only ever gave lower bounds. Because N\* is now small, the ladder starts low (**250→128k**)
to *bracket* it (low rungs are instant in the frame; high rungs pin E∞).

```sh
# submit (ssh hep-submit), then combine on the laptop when done
cd /nfs_scratch/bfriend3/NuQu/NuQu && git pull && cd hpc/detsvsL
sh submit_detsvsL_campaign.sh 16 "2 3 4 5"        # 64 shards; prints CAMPAIGN id
# monitor:  condor_q <cluster> -af JobStatus | sort | uniq -c
# retrieve + combine (laptop):
rsync -av hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_<id>/shards/ /tmp/sh/
python -m misc.combine_detsvsL --shard-dir /tmp/sh --label detsvsL_hpc_<id>
```

Shards write per-`(L,seed)` JSON directly to `campaign_<id>/shards/` on `/nfs_scratch`
(nothing transfers back). Each shard provisions its **own** env in its sandbox — a
shared NFS uv dir corrupts the managed interpreter under many-way concurrent cold
`uv python install` (NFS locking is too weak), so isolation beats deduplication here.
If a shard fails, just resubmit — combine tolerates missing seeds (min over whatever
completed).

**`test`** = L=2 1D, cores→400 (~30s; a full provisioning + build + solve shakedown).
**`full`** = L=2,3 dilute 3D, N_f=4, **large-core rungs 8k→64k** (×2/rung), `n_runs=4`,
1 h/rung wall cap. HPC does the cores the laptop can't reach; the cheap small-core rungs
run locally and are **combined post-hoc** (must match dim/A/N_f/n_runs) — the 8k/16k
overlap cross-validates the two datasets. Production bumps `n_runs`≥16 (Phase-D
seed-fragility fix) and adds L=4 — edit the `full` branch in `run_detsvsL.sh`.

> The selected-CI subspace is diagonalized with **scipy eigsh over the `mixed_ci` C++
> matvec** (no dense build, no official TrimCI). This path is exact-equal to the official
> C++ Davidson (validated to 9e-13 vs Lanczos) and is what makes the large-core runs
> possible without dragging jax/netket onto the cluster — see `backend.cpp_available` /
> `_diagonalize_arrays_scipy`.

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
