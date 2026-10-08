#!/bin/sh
# Submitter for the DMRG EXTRAPOLATION-CALIBRATION campaign (classical_next_studies A1, 2026-10-08).
#
# WHAT IT ANSWERS. The TrimCI sigma (TrimCI/COO literature convention, bootstrap of the
# PT2-linear fit) measures the scatter of ONE ladder about ONE estimator. It says nothing
# about whether that estimator lands on the true ground energy. block2 DMRG on the SAME
# truncated Hamiltonian (bare, n_b=3 -> N_f=8, L=2, 3D) is variational and independent of the
# selected-CI search, so E_DMRG(chi -> inf) is the anchor: misc/compare_dmrg_trimci.py reports
# TrimCI E_inf - E_DMRG in units of TrimCI's sigma, per A. (The COO paper did the same check
# against UDMRG: gap 0.19 mHa vs sigma 0.26 mHa.)
#
# GRID. A in AS at one (L, dim, n_b); chi schedule CHIS warm-started in order; NSWEEPS sweeps
# per chi; the per-chi wall cap MAXCHISEC stops the ladder before a chi that cannot finish
# (block2 cost ~chi^3). 293938 at this cutoff: chi=200 took 4100 s on 16 cpus (4 driver
# threads), discarded weight 3e-7, so chi=400 is ~hours and chi=800 is the stretch rung.
#
#   sh submit_dmrg_calib.sh test        # one shard, A=2, chi 50,100 (minutes): the smoke
#   sh submit_dmrg_calib.sh all         # AS x {L=2, 3D, n_b=3}, chi 100..800
#   AS="2 6 10" CHIS="100,200,400" sh submit_dmrg_calib.sh all
#   L=3 AS="2" MEM=192 MAXCHISEC=43200 sh submit_dmrg_calib.sh all   # the decisive L=3 point
#
# Run from $REPO/hpc/dmrg/ on `ssh hep-submit`, AFTER the launch-approval loop (HPC_WORKFLOW).
# Memory/holds: the same automatic recovery as the A-sweep submits (request_memory grows with
# NumHolds, self-release on a cgroup hold, exit 3 = provisioning re-queued). DMRG has NO
# checkpoint, so a hold restarts the ladder from chi[0]; the per-chi JSON keeps what finished.
set -eu
MODE="${1:-all}"
BASE="$(date +%Y%m%d-%H%M%S)-dmrgCalib"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
AS="${AS:-2 4 6 8 10}"
L="${L:-2}"; DIM="${DIM:-3}"; NB="${NB:-3}"
CHIS="${CHIS:-100,200,400,800}"
NSWEEPS="${NSWEEPS:-6}"
MAXCHISEC="${MAXCHISEC:-14400}"     # a chi slower than 4 h means the next (8x) never finishes
MEM="${MEM:-48}"                    # GB; 293938 measured 4.9 GB at chi=200 N_f=8, 12 GB at chi=400 (2D)
CPUS="${CPUS:-16}"
DISK="${DISK:-48G}"                 # 293938 used 5-14 GB of scratch against an 8 GB request
MEMCAP_GB="${MEMCAP_GB:-768}"
MAXHOLDS="${MAXHOLDS:-4}"
PRIO="${PRIO:-20}"

case "$MODE" in
  test) AS="$(echo "$AS" | awk '{print $1}')"; CHIS="50,100"; MAXCHISEC=1800 ;;
  all) ;;
  *) echo "usage: sh submit_dmrg_calib.sh {test|all}" >&2; exit 1 ;;
esac

DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
: > "$DIR/shards.txt"
for A in $AS; do
  # chi schedule travels '+'-separated: Condor splits `queue ... from file` on commas
  printf '%s %s %s %s %s %s %s\n' "$L" "$DIM" "$A" "$NB" "$(echo "$CHIS" | tr ',' '+')" "$MEM" "$CPUS" >> "$DIR/shards.txt"
done
NJOBS=$(wc -l < "$DIR/shards.txt" | tr -d " ")

cat > "$DIR/campaign.sub" <<SUB
Executable              = ./run_dmrg_calib.sh
arguments               = \$(L) \$(DIM) \$(A) \$(NB) \$(BDIMS) ${BASE} ${NSWEEPS} \$(CPUS) ${MAXCHISEC}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_dmrg_calib.sh
transfer_output_files   = ""
${QIS}
request_cpus            = \$(CPUS)
MEMCAP_MB               = $(( MEMCAP_GB * 1024 ))
NHOLDS                  = ifThenElse(isUndefined(NumHolds), 0, NumHolds)
MEMGROW_MB              = (\$(MEM) * 1024 * (1 + \$(NHOLDS)))
request_memory          = ifThenElse(\$(MEMGROW_MB) < \$(MEMCAP_MB), \$(MEMGROW_MB), \$(MEMCAP_MB))
periodic_release        = (HoldReasonCode == 34) && (NumHolds < ${MAXHOLDS})
on_exit_remove          = !((ExitBySignal == False) && (ExitCode == 3) && (NumJobStarts < 3))
request_disk            = ${DISK}
JobPrio                 = ${PRIO}
Output                  = ${DIR}/logs/dmrg_L\$(L)d\$(DIM)_A\$(A)_nb\$(NB).out
Error                   = ${DIR}/logs/dmrg_L\$(L)d\$(DIM)_A\$(A)_nb\$(NB).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,A,NB,BDIMS,MEM,CPUS from ${DIR}/shards.txt
SUB
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${BASE}  mode=${MODE}  jobs=${NJOBS}  L=${L} dim=${DIM} n_b=${NB}  chi=${CHIS}  nsweeps=${NSWEEPS}  max_chi_sec=${MAXCHISEC}  mem=${MEM}G cpus=${CPUS}"
echo "shards:"; cat "$DIR/shards.txt"
echo "pull: rsync -az 'hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/dmrg/${DIR}/shards/' data/classical/<date>/dmrg_calib_<cluster>/"
echo "compare: python -m misc.compare_dmrg_trimci --dmrg data/classical/<date>/dmrg_calib_<cluster> --trimci data/classical/2026-09-25/bare_Asweep_nb3_293963"
