# hpc/quantum — quantum resource-estimation HPC harness (task 34, I4)

The quantum counterpart of `hpc/detsvsL/` (the classical dets-vs-L harness). One
shard = one `(L, series)` with an A-sweep; the shard writes a self-describing JSON
(with a commit-pinned manifest) straight to `/nfs_scratch`. Read
[`../HPC_WORKFLOW.md`](../HPC_WORKFLOW.md) first — the launch-approval loop, the
qis1–4 allocation, and the self-provisioning pattern all apply here unchanged.

## What's different from the classical harness
- **No C++ build.** `evaluate_resources` is pure Python/numpy symbolic counting —
  the run script drops the `mixed_ci` compile step entirely.
- **Heavier deps.** `requirements-hpc-quantum.txt` pins pyLIQTR (→ cirq/qualtran,
  plus `juliacall`/`juliapkg` and `gmpy2`). Those can try to auto-provision a Julia
  runtime / need GMP inside a fresh sandbox — **smoke-test first** (below).
- **Symbolic, not BLAS-bound.** `request_cpus=1`; threads pinned to 1. RAM is
  modest (a few GB), growing with the operator size.

## Series (design-axis columns)
| series | basis | encoder | cutoff | role |
|---|---|---|---|---|
| `fock_pauli` | fock | **pauli_lcu** | `--n-b` / frame | **THE compiled PauliLCU anchor (N4)** |
| `watson` | amplitude | pauli_lcu | energy_bound | Watson Lemma-5 baseline (Tier 0) |
| `ns` | amplitude | pauli_lcu | ns / tong | Nyquist-Shannon |
| `sparse` | fock | sparse | tong | FROZEN feasibility path (not a headline) |
| `sparse_heuristic` | fock | sparse | heuristic | tong-vs-heuristic comparison |

**`fock_pauli` is the paper's quantum anchor** (REMEDIATION_PLAN N4): the Fock-basis
Hamiltonian materialized as a Pauli sum and block-encoded by pyLIQTR's PauliLCU
(genuinely compiler-derived). It is **A-independent at a fixed n_b** — the block
encoding encodes the *operator*, not the A-nucleon state; A enters only via the
cutoff. So the deep-L anchor is one estimate per `(L, n_b)`: pass `--n-b` (anchor +
convergence sweep) or `--frame-occupation <n>` (the seam picks n_b from a measured
⟨n⟩ via the 5σ rule). `--n-b` wins over both the series cutoff and the frame seam.

## Launch (on the pinned submit node)
```sh
ssh hep-submit
# Reconcile to the ACTIVE CAMPAIGN BRANCH (currently remediation/vertex-fix — the
# corrected-Hamiltonian regeneration). Untracked campaign_*/ output dirs survive.
cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
  && git checkout remediation/vertex-fix \
  && git reset --hard origin/remediation/vertex-fix
cd hpc/quantum

# --- Vertex-fix regeneration, round 1 (the current campaign) ------------------
# 1) SMOKE TEST FIRST — ONE real fock_pauli L=2 n_b=2 anchor estimate on a qis node
#    (validates the pyLIQTR/Julia/GMP deps AND the actual compiled anchor path).
sh submit_vertexfix_quantum.sh test
#    then check: grep '[qshard] done status=0' campaign_<CID>/logs/smoketest.out
# 2) the full Q1-Q4 grid (28 shards) after the smoke test passes:
sh submit_vertexfix_quantum.sh

# --- generic ad-hoc sweeps (older harness, still valid) -----------------------
sh submit_quantum_sweep.sh "2 3 4 6 8" "fock_pauli"    # deep-L, one column
sh submit_quantum_sweep.sh "2 3 4" "watson ns fock_pauli"  # A/B across columns
```
Each submit prints `CAMPAIGN=<ts>`, the job count, and the combine command. Report
the cluster/batch ID back per the launch-approval loop.

## Retrieve + combine
```sh
rsync -az hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/quantum/campaign_<CID>/shards/ <local>/
python -m misc.combine_quantum_shards --shard-dir <local> --out combined.json
```
`combine_quantum_shards` flags any shard still `done: false` (never silently treats
a truncated shard as complete). Plot with `plot_sweep_data.py` (the `series` tag on
each row keys the basis/cutoff/encoder column).

## Local dry-run (no cluster)
```sh
python -m misc.run_quantum_shard --test                       # the same smoke test
python -m misc.run_quantum_shard --L 4 --series sparse --A-values 1,2,4 --out /tmp/L4.json
```
