# NuQu HPC Workflow (UW–Madison HEP / qis cluster)

Read this before running **anything** on the cluster — classical *or* quantum. It captures
the user's launch preferences and the hard-won operational lessons. When a request
involves an HPC run, follow the launch loop below exactly.

---

## 1. The launch-approval loop  ← the most important rule

The user reviews and pushes code themselves, and wants to know exactly what is running and
how many batches exist at any time. **Do not run things autonomously.**

1. Make your changes and **commit**. Tell the user precisely **what was committed** (files +
   short hash).
2. **The user pushes.** You do **not** `git push`, and you do **not** rsync code to the server
   to launch around the push — the push *is* the review gate.
3. The user gives an explicit **go-ahead**.
4. **Then** you `condor_submit`, and you **always report the submit string + batch/cluster ID**
   (e.g. `290348`) and the `CAMPAIGN=<timestamp>`.

Corollary: after committing, stop and hand off — "committed `<hash>`: `<files>` — push when
ready, then tell me to launch." Only submit after the go-ahead. Managing an *already-running*
job you launched (hold/release/remove a malfunctioning batch) is fine; starting new ones is not.

---

## 2. Access & data flow

- `ssh hep` — load-balanced login (`login.hep.wisc.edu`). Use for **git, rsync, interactive**.
- `ssh hep-submit` — pinned `login01`. Use for **all Condor commands**. The schedd is
  **per-login-node**, so submit *and* query (`condor_q`/`condor_history`/`condor_rm`) must both
  go through `hep-submit`.
- Server checkout: `/nfs_scratch/bfriend3/NuQu/NuQu`. **Reconcile before every launch:**
  `git fetch origin -q && git reset --hard origin/main`. Untracked `campaign_*/` output dirs
  survive the reset.
- `/nfs_scratch` is shared and readable from the login nodes but quota-sensitive/purge-prone —
  rsync results back promptly; delete big tarballs on the server.
- **Landing convention** for pulled results: `data/{quantum|classical}/<YYYY-MM-DD>/<run-label>/`
  (the `data/` tree is a separate gitignored GitLab repo). Analysis scratch scripts live beside
  the data.

---

## 3. Allocation — qis1–4 ONLY

- **The Otten group's allocation is qis1–4.** The rest of the ~700-machine HEP pool is **CMS
  grid production** (negotiator quota groups are all CMS; a plain `bfriend3` job has
  `AccountingGroup=undefined` and matches **0** non-qis slots — ~327 machines actively reject
  it). **Do not try to "spill" onto the wider pool** — it can't work without a CMS accounting
  group. The `requirements = Machine=="qis1..4"` pin in the submit files is the real allocation,
  not an arbitrary limit.
- **qis node hardware (probed):** AMD EPYC 9254, **48 physical cores + 2-way SMT (96 logical)**,
  2 sockets/NUMA nodes, **~1 TB RAM** each. "Memory" in Condor = RAM (`request_memory`), not disk.
- **qis4 is GPU-gated** (`GPU_JOBS_ONLY` in its `START`) — plain CPU jobs won't land there even
  when it looks idle. Effectively you have qis1–3 for CPU work.
- **Contention is fair-share** against other qis users (e.g. `yshen295`). In HTCondor **lower
  `condor_userprio` = better**, and heavy recent usage degrades *your* priority, so a competing
  user can out-schedule you. Levers when starved:
  - right-size `request_memory` (below) so your jobs pack into fragments and don't inflate your usage;
  - `condor_qedit <cluster> -constraint '...' JobPrio <n>` to front-load your own decision-relevant shards;
  - `condor_hold` / `condor_release` (or `condor_qedit RequestMemory`) the biggest shards to free/reshape.

---

## 4. Self-provisioning jobs (the pattern that works)

qis **compute** nodes have `/nfs_scratch` + outbound internet + `g++ 11`, but **no `uv`, only
`python3.9`, and no home mount**. So the Condor **run script provisions per-sandbox at job
start** (everything discarded on exit): `curl` uv → `uv python install 3.10` → `uv pip install`
the pinned deps → compile the C++ hot path. Key points:

- **Per-sandbox `UV_*` dirs** (`UV_CACHE_DIR`/`UV_PYTHON_INSTALL_DIR` under the sandbox). A
  *shared* NFS uv dir **corrupts** the managed interpreter under concurrent cold installs.
- `should_transfer_files=YES`, `transfer_input_files=<run script>`, `transfer_output_files=""`
  — the job reads code from `/nfs_scratch` and **writes output to `/nfs_scratch`** (so the
  submit node can read it without an output transfer). This means jobs depend on the
  `/nfs_scratch` mount, which qis has (another reason we're pinned to qis).
- **Incremental per-rung save**: write results after every step and atomically rename, so a
  shard that OOMs/times-out at a deep step keeps everything it finished.
- Pin the minimal deps in `hpc/detsvsL/requirements-hpc.txt` (numpy/scipy/openfermion/cirq/
  matplotlib/pybind11). Comparison-only tooling (netket/pyscf/official-trimci) stays out.
- **Pinned tools on `/nfs_scratch` (2026-10-08).** 294238 was a full resubmit because one job
  could not reach astral.sh at start-up. `run_frame_shard.sh` now takes a pinned `uv`, a managed
  CPython 3.10 and a wheelhouse from `/nfs_scratch/bfriend3/NuQu/tools` (`NUQU_TOOLS`), **copying**
  them into the sandbox (nothing on NFS is written by a job), and only falls back to the
  downloads, each wrapped in a 3-attempt retry. Populate the directory **once** on the submit
  node: `cd hpc/detsvsL && sh provision_tools.sh` (re-run after bumping `requirements-hpc.txt`
  or the uv pin; it stages and renames atomically, so it is safe while jobs start). Any
  provisioning failure exits **3**, which the submit files re-queue (below); a C++ build or
  solver failure exits 1 and leaves the queue.
- **Resume (2026-10-08).** The warm-grow ladder checkpoints every finished rung's core beside the
  shard JSON (`<stem>.core.npz`, `classical/trimci/checkpoint.py`), and the wrapper runs the
  python with `--resume` (`NUQU_RESUME=0` to overwrite instead). A released or re-queued job
  continues from the next rung, bit-identical to an uninterrupted run; a finished shard exits 0
  at once; a checkpoint from different settings is refused (exit 2). Pull shards with
  `rsync --exclude '*.npz'`; the checkpoints are a few hundred MB each at 512k dets.

---

## 5. Parallelism — what actually helps

Learned the hard way; don't repeat the detours.

- **`request_cpus` reserves cores; the code has to *use* them.** Reserving 4 and running a
  single serial stream = "one thread on all your jobs." `OMP_NUM_THREADS` is a **no-op** for
  sparse selected-CI (the C++ H-build wasn't threaded and `scipy.eigsh` drives the matvec via a
  single-threaded Python callback).
- **The win is the fork ensemble.** Selected-CI runs `n_runs` **independent random-init solves**
  and keeps the best — embarrassingly parallel. `graph_arrays.ground_state_ensemble_arrays`
  forks `NUQU_NUM_WORKERS` children (H inherited via `fork`, **not** pickled — the C++
  `MixedProvider` is unpicklable; only the int seed crosses the pipe). Bit-identical to serial,
  ~3–4×. The run script sets **all** numeric libs to 1 thread/process (fork-safe: single-thread
  BLAS has no pool to corrupt) and `NUQU_NUM_WORKERS=request_cpus`.
- **Worker count is adaptive**: `_ensemble_workers` reads `NUQU_NUM_WORKERS` → `_CONDOR_REQUEST_CPUS`
  → `sched_getaffinity`, capped at `n_runs`. **qis Condor does NOT export `_CONDOR_REQUEST_CPUS`**
  (found 2026-09-24). Before then, every `run_frame_shard.sh` job ran at `cpus=2` whatever
  `request_cpus` said: 2 fork workers, or 2 OMP threads under `NUQU_DEEP_SOLVE`. That covers
  292477, 293958 and 293959. `detect_cpus` now reads `NUQU_CPUS` → `_CONDOR_REQUEST_CPUS` →
  `OMP_THREAD_LIMIT` → `PYTHON_CPU_COUNT` → `Cpus` in `$_CONDOR_MACHINE_AD` → 2. The value used is
  recorded in the shard manifest (`threads.NUQU_CPUS_RESOLVED`). The other run scripts
  (nb_cutoff, run_detsvsL*, backeval) still read only `_CONDOR_REQUEST_CPUS`. The DMRG scripts
  pass cpus explicitly. Compute-bound solves scale on the **48 physical
  cores**, not the 96 SMT threads — request up to ~48 for a big frame fit.
- **C++ OpenMP** (on the H-build/expand/matvec, behind `-fopenmp`) is in the tree but **dormant**
  (run script pins `OMP=1`): it gave ~1× at L=2 because those solves are *overhead-bound*
  (`eigsh` is ~8 ms; the system is too small to be compute-bound). It only pays off for large
  systems — reconsider it only if profiling shows the C++ connections dominate.
- **spawn** was considered and rejected: expensive per-call (re-imports) for our many-small-calls
  pattern, and needs a picklable H. Fork won once the real blocker (below) was fixed.

---

## 6. Memory sizing

- Selected-CI solves use **far less RAM than you'd guess** (L=2 ~4–10 GB of a 64 GB request).
  **Don't over-request** — it fragments poorly on a contended pool *and* inflates your fair-share
  usage. Right-size, then recover the rare OOM: `condor_qedit <cluster> -constraint '...'
  RequestMemory <MB>` then `condor_release` — incremental save means the resubmit only redoes
  the deep rung. (Current per-L defaults live in the submit scripts; bump only the dense/deep
  corners.)
- **Automatic OOM recovery (2026-10-08, `submit_nb3_Asweep*.sh`).** The 293963 holds were
  transient top-rung spikes 100–250 GB above Condor's last 5-minute `MemoryUsage` sample
  (HoldReasonCode 34, cgroup limit), and took three manual qedit+release rounds. The submit
  files now carry, validated with `condor_submit -dry-run` + `classad_eval` on HTCondor 25.0:
  ```
  MEMGROW_MB       = ($(MEM) * 1024 * (1 + NumHolds))          # MEM column is an integer in GB
  request_memory   = min(MEMGROW_MB, MEMCAP_MB)                # cap 768 GB (MEMCAP_GB env)
  periodic_release = (HoldReasonCode == 34) && (NumHolds < 4)  # MAXHOLDS env
  on_exit_remove   = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))
  ```
  So L=4 goes 384 → 768 GB on the first hold; L=3 128 → 256 → 384 → 512. After the fourth hold
  the job stays held for a person. `NumHolds`, not `NumSystemHolds` (the cgroup holds left it
  at 0 in 293963's history). Paired with resume (§4) a hold costs one rung. Smoke it with
  `sh submit_nb3_Asweep_squeeze.sh oomtest` (one 1 GB L=2 shard: expect holds, doubling, exit 0).
- **Per-rung memory records.** Every rung now stores `mem` = peak RSS of the process and of
  its fork workers plus the solver's pool-size/RSS trace (`graph_arrays.ground_state_arrays`),
  so the spike the samples miss is on record. `python -m misc.analyze_shard_memory --data <dir>`
  prints the peak per shard, the stage it happened in (expand pool vs trim+diag vs select
  worker) and a suggested `MEM` per (frame, L) — use it after the smoke shards to set the
  memory per row rather than per L.

---

## 6b. Disk sizing — the constraint that silently strands jobs (2026-09-15)

`request_disk` is **not** boilerplate on this pool. The three qis nodes have wildly
different free EXECUTE-DIR disk:

| node | free CPUs (idle) | free execute disk |
|---|--:|--:|
| qis1 | 96 | **~3.1 GB** |
| qis2 | 0 (busy) | ~1 TB |
| qis3 | 95 | **~2.0 GB** |

A `request_disk = 8G`/`10G` therefore makes **qis1 and qis3 permanently unmatchable**, and
every job funnels onto qis2 — which is why a campaign can sit fully idle while two
96-core machines show `Unclaimed`. This is invisible in `condor_q` (jobs just stay Idle);
the tell is in `condor_q -better-analyze`, where the `TARGET.Disk >= RequestDisk` row
collapses the match count:

```
[4]  Machine == qis1||qis2||qis3   -> 3 slots
[19] TARGET.Disk >= RequestDisk    -> 702 matched, 336 rejected
[20] [4] && [19]                   -> 1 slot          <-- disk, not cpus/memory
```

**Right-size from measurement, same as memory (§6):**
```sh
condor_status -constraint 'regexp("qis[123]", Machine)' -af Name Cpus Memory Disk
condor_history <cluster> -af DiskUsage MemoryUsage
```
Measured peaks: **nb_cutoff / selected-CI shards ~1.9 GB** (self-provisioned uv env + the
mixed_ci C++ build) → request ~2.5 GB and they reach qis1. **block2-DMRG shards ~4.8 GB**
(block2 + the full pip MKL runtime) → these genuinely do *not* fit qis1/qis3 and are
qis2-only; that is a real constraint, not a misconfiguration, so size the campaign
expecting qis2 serialisation.

Fixing a stranded job in flight: `condor_qedit <cluster> RequestDisk <KB>` (and
`JobPrio <n>` to put it ahead of your own idle shards). Both are reversible and
scientifically neutral — the fork ensemble is bit-identical regardless of core count (§5).

---

## 7. Monitoring — gotchas

- **SSH drops during a `sleep` inside a poll** (idle disconnect). Two robust patterns:
  (a) run the `sleep` **locally** in a background task, then a *fresh* `ssh` for the one-shot
  check; or (b) `ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=20`.
- **Pull the real job accounting — routinely, not just wall-clock.** `condor_history <cluster>
  -af RequestCpus RemoteWallClockTime RemoteUserCpu RemoteSysCpu MemoryUsage NumJobStarts`. From it:
  - **CPU-hours is the true cost metric** (wall-clock is for *your* turnaround). `sum(RemoteUserCpu)/3600`
    ≈ actual CPU-time used; `sum(RemoteWallClockTime × RequestCpus)/3600` is the allocated upper
    bound. Advisor's rule of thumb: **under ~100 CPU-hours is cheap** — so reframe "how far can we
    push" around CPU-h, not wall-clock. (Ref: the whole 60-shard L=3 128k campaign was ~50 CPU-h
    actual; the L=4/16k smoke ~2 CPU-h — we have large headroom to go deeper/bigger.)
  - `RemoteUserCpu` for a *completed* job DOES count the forked children (it can exceed wall when
    >1 core ran); it's only unreliable mid-run or for a *stuck* job (the earlier ~0 reading was
    `random_core` hung). For a clean speedup number, still prefer **wall-time, same config+seed**.
  - `MemoryUsage` = **peak RSS** → right-size `request_memory`. We've been over-requesting badly
    (peaks ~24–46 GB against 128–192 GB requests); trim toward peak + headroom to schedule better.
- Confirm completion with `condor_history <cluster> -af ExitCode` (want all `0`) **and** by
  grepping `'"done": true'` in the shard JSONs — a job can exit 0 while a shard capped early.
- A shard sitting at **0 rungs forever** is usually *stuck*, not slow — historically that was the
  `random_core` bug (below), or an OOM about to hit.

---

## 8. Campaign mechanics

- A "batch" = one `condor_submit` = one **cluster ID** (e.g. `290348`) + a `CAMPAIGN=<timestamp>`
  and its own `campaign_<CAMPAIGN>/shards/` dir. `condor_q` (batch view) on `hep-submit` lists
  all your batches at once. `condor_q -global` spans all login-node schedds (in case something
  was submitted from a different node).
- A multi-shard campaign = a submit script that writes a `shards.txt` grid and a `.sub` with
  `queue <vars> from shards.txt`. Give each launch a distinct campaign dir so re-runs don't collide.
- Retrieve: `rsync -az hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/detsvsL/campaign_<CID>/shards/ <local>/`
  then combine/analyze locally.
- The `hep-condor-jobs` skill (`.claude/skills/`) has generic submit/recovery patterns; this file
  captures the NuQu-specific choices on top of it.

---

## 8b. The control surface — every lever we have (2026-10-08)

What an agent can turn, from the outside, without editing solver code. Specifics of the
classical and quantum drivers live in their submit-file headers; this is the list of **kinds**
of control, with where each one lives.

**Before launch (submit-file env, read at `sh submit_*.sh <mode>` time)**
- **Grid trims** — `AS`, `LS`, `SEEDS`, `REF_LS`, `ARMS` (space-separated lists) shrink or extend
  the grid without editing the script; each submit prints the resulting cell count.
- **Modes** — every submit script has `test` (one shard, the first thing to run on new code),
  `all`, and arm selectors (`squeeze`, `refs`, …); `submit_nb3_Asweep_squeeze.sh oomtest` is the
  1-shard smoke for the memory policy + resume, with `OOM_MEM` (GB, default 2) as its base.
- **Legacy re-runs** — `hpc/detsvsL/submit_seedfix_rerun.sh {test|baseline|gate|volscaling|all}`
  re-runs the three seed-sharing-affected datasets under the production search (`SEEDS`,
  `GATE_SEEDS`, `VOL_SEEDS`, `NB2=1` trims/extends); priorities below the squeeze campaign.
- **Independent reference** — `hpc/dmrg/submit_dmrg_calib.sh {test|all}` runs block2 DMRG on
  the same truncated H the TrimCI shards ran (`AS`, `L`, `DIM`, `NB`, `CHIS`, `MAXCHISEC`,
  `MEM`, `CPUS`, `DISK` env); shards save after every χ but have NO resume, so a memory hold
  restarts the χ ladder. `python -m misc.compare_dmrg_trimci --dmrg <dir> --trimci <dir>`
  reports TrimCI E∞ − DMRG E∞ in units of σ.
- **Memory policy** — `MEMCAP_GB` (default 768) caps the auto-growing request; `MAXHOLDS`
  (default 4) is how many memory holds a job may release itself from. The per-L base lives in
  the `sizing_for_L` table (MEM column, integer GB) and is overridable per row by editing the
  grid file before `condor_submit` (or `NUQU_MEM_OVERRIDE` where a script offers it).
- **Solver knobs via Condor `environment`** — `NUQU_*` variables the run script turns into
  driver flags: depth/cutoff (`NUQU_N_B`, `NUQU_N_RUNGS`, `NUQU_PT2_MAX_CORE`, `NUQU_LADDER_START`),
  search (`NUQU_PHASE0_RUNS`, `NUQU_PHASE0_SEED_STRIDE`, `NUQU_PHASE0_SELECT_CORE`,
  `NUQU_PHASE0_INIT`, `NUQU_PHASE0_WORKERS`, `NUQU_NOVEL_*`), parallelism (`NUQU_DEEP_SOLVE`,
  `NUQU_CPUS`), extras (`NUQU_BACK_EVAL`, `NUQU_EXACT_REF`, `NUQU_DIM`, …). The full set and
  defaults are the `${NUQU_…:-}` lines at the top of `run_frame_shard.sh`; the campaign scripts
  set the validated combination and a shard records what it ran with in its manifest.

**Provisioning (per job, automatic)**
- `NUQU_TOOLS` (default `/nfs_scratch/bfriend3/NuQu/tools`): pinned `uv`, managed Python 3.10
  and a wheelhouse, copied into the sandbox; missing pieces fall back to downloads with a
  3-attempt retry. `NUQU_UV_VERSION` pins uv (keep equal in `provision_tools.sh`). Repopulate
  with `sh hpc/detsvsL/provision_tools.sh` after changing `requirements-hpc.txt` (`FORCE=1`
  redoes the 5-minute interpreter unpack). Provisioning failures exit 3 and self-requeue.

**Resume (per job, automatic)**
- Every finished rung is checkpointed beside the shard JSON; a restarted job continues from it,
  bit-identically. `NUQU_RESUME=0` forces a fresh run that overwrites. A finished shard re-run
  exits 0; a checkpoint from different settings is refused (exit 2) rather than mixed.

**In flight (condor_*, on the pinned submit node)**
- `condor_qedit <id> RequestMemory <MB>` / `JobPrio <n>` / `Environment "<…>"` with `-constraint`
  to reshape held or idle jobs in place (the auto-policy stacks on top of a manual bump).
- `condor_hold` / `condor_release` with `-constraint` to share the pool between campaigns
  (same-user jobs are FIFO by priority, not fair-shared); `condor_rm` only for batches we own.
- Memory holds release themselves with more memory up to `MAXHOLDS`; a job still held with
  `NumHolds` = `MAXHOLDS` is the signal for a person. Exit code 3 in the log = provisioning retry.

**Watching**
- `condor_q -af ClusterId ProcId JobStatus NumHolds RequestMemory NumJobStarts RemoteHost`
  and `condor_q -better-analyze <id>` (disk/memory/cpus matchability, §6b).
- `condor_history <cluster> -af ProcId ExitCode MemoryUsage NumJobStarts LastHoldReason` after.
- Shard JSONs: `done`, `rungs[*].mem` (peak RSS per rung), `resumed` (restart records),
  `checkpoint`. `python -m misc.analyze_shard_memory --data <dir>` summarizes memory per
  (frame, L) and suggests the next MEM.

**After**
- `rsync -az --exclude '*.npz' hep:<campaign>/shards/ data/classical/<date>/<name>_<cluster>/`
  (checkpoints stay on scratch), then the loaders with `--frame` where a campaign mixes frames.

## 9. The meta-lesson

**Profile before optimizing.** The entire "we're only using one thread" thread turned out to be
an **algorithmic bug**, not a missing thread: `random_core` was enumerating the *whole*
C(n_modes, A) determinant space (`itertools.combinations`) to draw a few random inits — O(1e15)
at L=3, which *hung* every L=3/high-filling job. Fixing it (direct O(A) sampling) was a >1000×
speedup and unblocked L=3 entirely — far bigger than any parallelization. When a job is "slow"
or "single-threaded," profile a representative solve first; the bottleneck is often not where you
assume.

Follow-up (2026-08): re-profiling a *large-core* solve after the random_core fix showed **no
second jam** — the cost is the genuine C++ H-build + `eigsh` + `expand` (compute-bound, scales
~linearly with core, per the L=4 ladder). So reaching 1M+ cores is a *time* cost, not a wall,
and it's affordable in CPU-hours. Two levers if we want it faster: (a) `_MATFREE_N=2000` means
the cluster runs **scipy `eigsh` for all real cores** (no official Davidson in the self-provisioned
env) — a C++ Davidson would cut the eigensolve; (b) the dormant C++ OpenMP on the H-build/expand
helps *large* cores (unlike the overhead-bound small L=2 solves where it was ~1×). Neither is
needed for correctness; both are speedups for the deepest convergence tests.

---

## 10. Quantum resource-estimation harness (`hpc/quantum/`) — 2026-08

The quantum counterpart of `hpc/detsvsL/` (submit/run/combine + `run_quantum_shard.py`). It
reuses everything above (launch-approval loop, qis1–4 pin, per-sandbox uv self-provision, write
to `/nfs_scratch`, incremental save) with three deliberate differences and several hard-won
lessons from first light (smoke tests 290356–290358, campaign 290367).

- **No C++ build; threads pinned to 1.** `evaluate_resources` is pure-Python/numpy *symbolic*
  counting (Pauli/T tallies), not a BLAS-bound solve — so the classical `mixed_ci` compile step
  and the OMP/fork-ensemble machinery are dropped. One shard = one `(L, series[, frame_occ])`.
- **Smoke-test the deps on a qis node FIRST.** `sh submit_quantum_sweep.sh test` submits ONE job
  that imports pyLIQTR via a real L=2 estimate. The pyLIQTR tree is heavier/riskier than the
  classical one; run this before any campaign. It caught every issue below in single cheap jobs.
- **Dependency pins are load-bearing** (`requirements-hpc-quantum.txt`) — pyLIQTR 1.3.4 under-pins
  its tree, so an unpinned install silently resolves broken versions:
  - `qualtran==0.4.0` — pyLIQTR doesn't pin qualtran; a newer one has a *slotted* `CtrlSpec`
    (no `__dict__`) that breaks `functools.cached_property` in the walk `t_complexity`
    (`TypeError: No '__dict__' attribute on 'CtrlSpec'`). Λ computes, then the T-count dies.
  - `cirq-core==1.4.0` — NOT the `cirq` metapackage. The metapackage drags in
    `cirq-rigetti → pyquil → attrs>=20,<22`, unsatisfiable against the modern attrs qualtran
    pulls. The laptop env has only cirq-core; match it.
  - `scipy` — the `tong` cutoff series import `classical.trimci.tong_bound`, whose package
    `__init__` needs numpy+scipy at import (the compiled `mixed_ci` backend + official `trimci`
    /jax/netket stay lazy and are never pulled — verified). Without scipy the tong series won't
    import on a quantum-only node.
  - **Validate `requirements-hpc-quantum.txt` with `uv pip compile` locally before pushing** — it
    does a full resolve and surfaces conflicts in seconds, saving a commit→push→submit round-trip.
- **The server reconciles to the CAMPAIGN BRANCH, not `main`.** The quantum harness lives on a
  feature branch (not merged), so §2's `git reset --hard origin/main` is wrong here — use
  `git checkout <branch> && git reset --hard origin/<branch>`. (`git_dirty:true` in a run's
  manifest is usually just the untracked `campaign_*/` dirs, not real changes.)
- **Condor `queue <vars> from file` splits columns on COMMAS as well as whitespace.** A comma-list
  in a column (e.g. an A-grid `1,2,4,8`) is silently tokenized across the queue variables →
  `RequestMemory = 4,8,16,...` → submit parse error (transactional: it rolls back, 0 jobs queued).
  Use a comma-free separator (`+`) in the shards file and normalize it back (`tr '+' ','`) in the
  run script.
- **Calibrate per-series cost locally before sizing the grid** (profile-before-scaling, again).
  Key facts: `sparse`/fock/tong is cheap and **A-flat** (tong → n_b A-independent → the whole
  A-sweep is *one* unique estimate; one representative A suffices), reaching L=10 in ~20–30 min;
  `ns`/amplitude ~L=6; `watson` (Watson Lemma-5 baseline, n_b 19–25, Λ~10¹⁰–10¹¹) is the expensive
  one, ~L=4–5. A only moves the estimate for the n_b-growing series (watson, sparse_heuristic).

---

*This file is the standing HPC reference. The classical dets-vs-L / frame-crossover and the
quantum resource-estimation campaign specifics (scripts, campaign IDs, results) are tracked in
the agent's project memory and the `data/` repo.*
