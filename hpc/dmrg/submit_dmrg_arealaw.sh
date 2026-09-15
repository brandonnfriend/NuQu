#!/bin/sh
# Submitter for the DMRG AREA-LAW campaign (post-vertex-fix).
#
# THE CLAIM THIS EXISTS TO SOLIDIFY
#   "DMRG is limited by the area law, since the bond dimension needed for a fixed
#    truncation error grows with the area of the cut, which is L^2 in three dimensions,
#    and the explicit pion modes make the local dimension larger on top of that."
#
# Three measurable clauses, one grid axis each:
#   (1) chi(fixed truncation error) grows with CUT AREA  -> vary (L, dim); area = L^(dim-1)
#   (2) that area is L^2 in 3D                           -> the 3D rungs, plus the
#       AREA-vs-VOLUME control below
#   (3) explicit pions raise the LOCAL DIMENSION          -> vary N_f at fixed geometry;
#       per-site dim = 2^4 (fermion) x N_f^3 (three pion species). N_f=1 is the
#       PION-FREE control (boson space = vacuum only, no new code needed).
#
# THE CONTROL THAT MAKES CLAUSE (1) A TEST, NOT AN ILLUSTRATION
#   L=4 dim=2 (16 sites, cut area 4) and L=2 dim=3 (8 sites, cut area 4) have the SAME
#   cut area and DIFFERENT volume. If chi tracks area they agree; if it tracks volume or
#   L they do not. Keep both rungs -- that pair is the actual evidence.
#
# Each shard writes E, S_max_bond and the block2 DISCARDED WEIGHT per chi, re-saving
# after every chi (cost ~chi^3, so the top rung is where a run gets cut off). The
# discarded weight is what turns "fixed truncation error" into a measurement.
#
#   sh submit_dmrg_arealaw.sh LABEL "L:dim:A:N_f:bond_dims:mem:cpus ..." [NSWEEPS] [MAXCHISEC]
#
# Run from $REPO/hpc/dmrg/ on `ssh hep-submit`. Analyze with misc/make_dmrg_arealaw.py.
set -eu
LABEL="$1"; POINTS="$2"; NSWEEPS="${3:-6}"; MAXCHISEC="${4:-18000}"
CAMPAIGN="${LABEL}-$(date +%Y%m%d-%H%M%S)-$$"; DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"

# Condor splits `queue ... from file` on COMMAS as well as whitespace (HPC_WORKFLOW section 10),
# so the chi schedule travels with '+' and is normalised back to ',' in the run script.
: > "$DIR/shards.txt"
for P in $POINTS; do
  IFS=: read -r L DIM A N_F BDIMS MEM CPUS <<EOF
$P
EOF
  echo "$L $DIM $A $N_F $(echo "$BDIMS" | tr ',' '+') $MEM $CPUS" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_dmrg_arealaw.sh
arguments               = \$(L) \$(DIM) \$(A) \$(N_F) \$(BDIMS) ${CAMPAIGN} ${NSWEEPS} \$(CPUS) ${MAXCHISEC}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_dmrg_arealaw.sh
transfer_output_files   = ""
requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")
request_cpus            = \$(CPUS)
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 20
Output                  = ${DIR}/logs/dmrg_L\$(L)d\$(DIM)_A\$(A)_Nf\$(N_F).out
Error                   = ${DIR}/logs/dmrg_L\$(L)d\$(DIM)_A\$(A)_Nf\$(N_F).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,A,N_F,BDIMS,MEM,CPUS from ${DIR}/shards.txt
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  label=${LABEL}  jobs=${NJOBS}  nsweeps=${NSWEEPS}  max_chi_sec=${MAXCHISEC}"
echo "shards:"; cat "$DIR/shards.txt"
echo "pull: rsync -az 'hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/dmrg/${DIR}/shards/' data/classical/<date>/dmrg_arealaw_<cluster>/"
