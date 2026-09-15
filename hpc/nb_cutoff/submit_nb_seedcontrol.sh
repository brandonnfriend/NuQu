#!/bin/sh
# SEED CONTROL (F-009): is the low-occupation prior a circular argument?
#
# Every admissible occupation measurement (studyB*/studyC*/studyHist*) seeds the core
# near vacuum (truncated-geometric, mean 0.5). If that prior were doing the work we would
# be planting the low-occupation answer we then report. The original control (study_D)
# removed the prior, but its only data is PRE-VERTEX-FIX and retired
# (data/RETIRED_pre_vertex_fix_20260818/.../studyD_L2d3A1_uniforminit.json), so the
# control is currently UNEVIDENCED. This re-runs it on the corrected Hamiltonian.
#
# TWO ARMS, MATCHED. Identical settings (L=2 d=3 A=1, N_f=2,4,8,16, core=4000, seed=0,
# PT2 off, n_runs=16) except the initialization:
#   Dprior    boson_init_mean=0.5   near-vacuum prior  (what every current study uses)
#   Duniform  boson_init_mean=None  uniform over [0,N_f), NO vacuum anchor
# Only the init differs, so any difference is attributable to it alone. Re-running just
# the uniform arm against the existing n_runs=3 prior data would confound the init with
# the ensemble breadth -- hence both arms here.
#
# WHY n_runs=16. The block2 cross-check showed selected CI is basin-trapped at n_runs<=6
# and escapes at n_runs>=16. With too few runs a uniform arm that fails to reach the
# prior arm is UNINTERPRETABLE (search failure vs seeding bias). 16 makes a negative
# result mean something. WHY N_f up to 16: a seeding bias would act hardest at the
# LARGEST cutoff, where the prior most disagrees with the available space; capping at
# N_f=8 would let the control pass trivially.
#
# READ THE PAIR ON BOTH observables the original control named: E_var and N_per_mode.
# If the uniform arm reproduces the prior arm on both, the prior is a convergence
# accelerator, not an imposed answer. (E_var is a valid Ritz upper bound under EITHER
# init, so what is at stake is a search-basin risk, not correctness of the bound.)
#
# REQUEST_DISK IS A REAL ALLOCATION CONSTRAINT HERE, not boilerplate. qis1 and qis3
# advertise only ~3.1 GB and ~2.0 GB of free EXECUTE-DIR disk (qis2 has ~1 TB), so a
# 10G request makes two of our three machines permanently unmatchable -- every job
# serialises onto qis2 while 96- and 95-CPU nodes sit Unclaimed. Measured usage for this
# executable (self-provisioned uv env + the mixed_ci C++ build) is ~1.9 GB, so 2.5 GB is
# peak + ~30% headroom and fits qis1. Check with:
#   condor_status -constraint 'regexp("qis[123]", Machine)' -af Name Cpus Disk
#   condor_history <cluster> -af DiskUsage
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#   cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#       && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#   cd hpc/nb_cutoff
#   sh submit_nb_seedcontrol.sh test   # 1 shard, prior arm only -- validates the env path
#   sh submit_nb_seedcontrol.sh        # both arms at L=2, n_runs=16
#   sh submit_nb_seedcontrol.sh grid   # multi-L x breadth campaign (14 shards)
set -eu
MODE="${1:-run}"
CAMPAIGN="seedctl-$(date +%Y%m%d-%H%M%S)"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'

# MODE=grid: the multi-L + ensemble-BREADTH campaign. Breadth is a first-class axis, not a
# knob: the 2026-09-15 L=2 run showed the n_runs=16 prior arm sitting ~11.8 MeV ABOVE what
# n_runs=48 reaches at N_f=8 -- and above a bound the uniform N_f=4 run already proves
# (E_0(N_f=8) <= E_var(N_f=4), since the N_f=4 space is a SUBSPACE of N_f=8 and the term
# list is N_f-independent). So the reported "n_b=3 == n_b=4 to 0.001 MeV" was CORRELATED
# SEARCH ERROR, not convergence. <N> moved only 0.04038 -> 0.04010 across that same change,
# i.e. the occupation is robust while the energy was not -- which is why both observables
# must be read separately.
SH="$DIR/shards.txt"; : > "$SH"
if [ "$MODE" = "test" ]; then
  printf 'Dprior 48G 16\n' > "$SH"
elif [ "$MODE" = "grid" ]; then
  # STUDY MEM CPUS -- L=2 cheap enough for a deep breadth ladder; L=3/4 get the
  # decision-relevant 16-vs-64 contrast.
  for R in 16 64 256; do
    printf 'Dctl_prior_L2_r%s 32G 24\nDctl_uniform_L2_r%s 32G 24\n' "$R" "$R" >> "$SH"
  done
  for R in 16 64; do
    printf 'Dctl_prior_L3_r%s 64G 24\nDctl_uniform_L3_r%s 64G 24\n' "$R" "$R" >> "$SH"
    printf 'Dctl_prior_L4_r%s 96G 24\nDctl_uniform_L4_r%s 96G 24\n' "$R" "$R" >> "$SH"
  done
else
  printf 'Dprior 48G 16\nDuniform 48G 16\n' > "$SH"
fi
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_nb_shard.sh
arguments               = \$(STUDY) ${CAMPAIGN}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = \$(CPUS)
request_memory          = \$(MEM)
request_disk            = 2560M
JobPrio                 = 20
Output                  = ${DIR}/logs/\$(STUDY).out
Error                   = ${DIR}/logs/\$(STUDY).err
Log                     = ${DIR}/logs/campaign.log
queue STUDY,MEM,CPUS from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  mode=${MODE}  shard_dir=${DIR}/shards"
echo "shards:"; cat "$SH"
echo "pull: rsync -az 'hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/nb_cutoff/${DIR}/shards/' data/classical/<date>/seed_control_<cluster>/"
echo "then: python -m misc.compare_seed_control --data <that dir>"
