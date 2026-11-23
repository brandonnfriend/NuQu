#!/bin/sh
# ONE analysis job: the basin-collapse rule A/B (misc/ab_collapse_rule.py, 2026-11-22).
#
# Pools every classical shard group under both collapse rules (largest drop vs an outlier drop
# >= 3x the next) for nuqu-2026-10 and trimci-coo, and checks the legacy nuqu-2026-09 path is
# bit-identical. CPU-bound bootstrap fits (500 replicates x 5000-point R^2 scans per seed),
# parallel over groups: ~15-20 min per pass on 2 laptop cores -> minutes on 32.
#
# The data is a SNAPSHOT of the local data/classical tree (data/ is its own private repo and is
# not in the server checkout), synced to DATA_ROOT on scratch before submitting:
#     rsync -az --delete data/classical/ hep:/nfs_scratch/bfriend3/NuQu/data_snapshot/classical/
#
# Run from $REPO/hpc/detsvsL/ on hep-submit, after the push:
#     sh submit_ab_collapse_rule.sh
# Output: $OUT_DIR/ab_collapse_rule.{json,md}; pull with
#     rsync -az hep:<OUT_DIR>/ docs/classical/ab_collapse_rule_<date>/
set -eu
DATA_ROOT="${DATA_ROOT:-/nfs_scratch/bfriend3/NuQu/data_snapshot/classical}"
OUT_DIR="${OUT_DIR:-/nfs_scratch/bfriend3/NuQu/analysis/ab_collapse_rule_$(date +%Y%m%d)}"
CPUS="${CPUS:-32}"
MEM="${MEM:-32}"
BASE="$(date +%Y%m%d-%H%M%S)-abcollapse"
[ -d "$DATA_ROOT" ] || { echo "ERROR: $DATA_ROOT missing -- rsync data/classical first" >&2; exit 2; }
mkdir -p "campaign_${BASE}/logs" "$OUT_DIR"
cat > "campaign_${BASE}/ab.sub" <<EOF
Executable              = ./run_analysis.sh
arguments               = misc.ab_collapse_rule --data-root ${DATA_ROOT} --out-dir ${OUT_DIR} --workers $(( CPUS - 1 ))
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_analysis.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = ${CPUS}
request_memory          = ${MEM}G
request_disk            = 2560M
on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))
JobPrio                 = 100
Output                  = campaign_${BASE}/logs/ab.out
Error                   = campaign_${BASE}/logs/ab.err
Log                     = campaign_${BASE}/logs/ab.log
queue
EOF
condor_submit "campaign_${BASE}/ab.sub"
echo "A/B job submitted: LOGS=campaign_${BASE}  OUT=${OUT_DIR}"
