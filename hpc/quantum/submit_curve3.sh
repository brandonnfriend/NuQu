#!/bin/sh
# Curve 3 of the headline figure: QUBITIZATION AT WATSON'S HIGH n_b (task 11).
# Amplitude / energy_bound / PauliLCU qubitization at the Option-A budget-derived
# field cutoff ε_cut = 6.275e-6 (the QPE-task value: 2√(2ε_cut)=√3π/(3·2^m), m=8 for
# ΔE=1 MeV, E_max=140), so the amplitude n_b (39–44 for L=2–6) MATCHES the
# Watson-Trotter baseline exactly. This is the "qubitization beats Trotter at the
# same Hilbert space" curve; it runs only to L≈5 (amplitude PauliLCU can't scale —
# the motivation for the Fock/low-n_b encoding).
#
# The old `watson`-series campaign data used ε_cut=0.1 (n_b~19–25) — the WRONG
# convention for this comparison; this job supersedes it.
#
# Run from $REPO/hpc/quantum/ on the pinned submit node after reconciling to the
# campaign branch (see submit_overnight.sh header).
set -eu
EPS="6.275e-6"
CAMPAIGN="$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"
mkdir -p "$DIR/logs"

# "L series avals frameocc mem" (ε_cut is the fixed EPS for all rows).
# L=2,3 carry an A-sweep (also feeds the appendix cost-vs-A); L=4,5,6 are A=100 only
# (amplitude high-n_b is too slow for a full A-sweep there). Amplitude PauliLCU at
# high n_b is RAM-hungry -> generous memory; 6h cap guards the deep/slow shards.
PLAN="
2 watson 1+4+16+64+100 - 8G
3 watson 1+4+16+64+100 - 12G
4 watson 100 - 16G
5 watson 100 - 32G
6 watson 100 - 64G
"

: > "$DIR/shards.txt"
echo "$PLAN" | while read -r L SERIES AVALS FOCC MEM; do
  [ -z "${L:-}" ] && continue
  echo "$L $SERIES $AVALS $FOCC $EPS $MEM" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_quantum_shard.sh
arguments               = \$(L) \$(SERIES) ${CAMPAIGN} \$(AVALS) run \$(FOCC) \$(EPS)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_quantum_shard.sh
transfer_output_files   = ""
Output                  = ${DIR}/logs/L\$(L)_\$(SERIES).out
Error                   = ${DIR}/logs/L\$(L)_\$(SERIES).err
Log                     = ${DIR}/logs/campaign.log
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = 1
request_memory          = \$(MEM)
request_disk            = 10G
periodic_remove         = (JobStatus == 2) && (time() - JobCurrentStartDate > 21600)
queue L,SERIES,AVALS,FOCC,EPS,MEM from ${DIR}/shards.txt
EOF

condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "combine when done: python -m misc.combine_quantum_shards --shard-dir ${DIR}/shards --out ${DIR}/combined.json"
