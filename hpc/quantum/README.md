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
| `watson` | amplitude | pauli_lcu | energy_bound | Watson Lemma-5 baseline (Tier 0) |
| `ns` | amplitude | pauli_lcu | ns / tong | Nyquist-Shannon |
| `sparse` | fock | sparse | tong | deep-L workhorse; Tier-1 realistic |
| `sparse_heuristic` | fock | sparse | heuristic | tong-vs-heuristic comparison |

## Launch (on the pinned submit node)
```sh
ssh hep-submit
cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q && git reset --hard origin/main
cd hpc/quantum

# 1) SMOKE TEST FIRST — one job that imports pyLIQTR via a real L=2 A=1 estimate.
#    This is what surfaces any Julia/GMP provisioning problem before a campaign.
sh submit_quantum_sweep.sh test
#    then check: grep '[qshard:test] OK' campaign_<CID>/logs/smoketest.out

# 2) the real campaign (after the smoke test passes)
sh submit_quantum_sweep.sh "2 3 4 6 8" "sparse"        # deep-L, one column
sh submit_quantum_sweep.sh "2 3 4" "watson ns sparse"  # A/B across columns
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
