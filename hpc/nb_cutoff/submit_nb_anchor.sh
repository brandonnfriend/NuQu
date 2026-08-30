#!/bin/sh
# ED exact anchor (study_A), REPLANNED — the only shard that never finished.
#
# study_A does an EXACT Lanczos on L=2 d=1 to validate TrimCI+PT2 against truth. Its sector is
# N_f^(n_bos=6): N_f=6→373k (reachable in minutes) but N_f=8→2.1M (hung 68h twice) and N_f=16→134M.
# So it is now capped at N_f=(2,3,4,5,6) — covering n_b=1 (N_f=2) and n_b=2 (N_f=4) exactly plus
# convergence points. The exact ENERGY convergence complements studyG's exact OCCUPATION validation
# and the selected-CI tail grid (which already made the n_b=2 claim across density and volume).
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_anchor.sh        # single study_A shard (capped N_f=6; should finish in minutes)
set -eu
CAMPAIGN="$(date +%Y%m%d-%H%M%S)-nbAnchor"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = A ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 12G
request_disk            = 10G
JobPrio                 = 20
Output                  = ${DIR}/logs/A.out
Error                   = ${DIR}/logs/A.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  (study_A ED anchor, capped N_f=6 -> max sector 373k; expect minutes)"
echo "pull -> data/classical/nb_convergence/ ; then re-run misc.make_nb_figure (adds exact energy)"
