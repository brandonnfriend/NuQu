#!/bin/sh
# NESTED boson-cutoff A-sweep — task 35 T3, audit P0-3.
#
# WHY. The volume-scaling arm that carries the L=10 cutoff conditional has two defects the
# 2026-09-05 audit names: it tests A=1 ONLY, and it measures Delta34 with two INDEPENDENTLY
# selected solves — leaving a selection residual that shows up as Delta34 oscillating in sign
# down the core ladder at ~0.001–0.003 MeV/site, i.e. the size of the signal itself.
#
# THIS CAMPAIGN FIXES BOTH.
#   (a) A SWEEP — dilute to fully-filled, at three volumes (grid below).
#   (b) NESTING — the n_b=4 solve is warm-started FROM the converged n_b=3 core, so the n_b=4
#       space is a strict superset explored from the n_b=3 solution and
#           Delta34 = E_3 - E_4 >= 0  by construction (sign-definite).
#       Verified locally: the nested Delta is exactly 0 at most rungs where the legacy
#       independent arm returns small values of BOTH signs at the same points.
#   (c) BOUNDARY AUGMENTATION — nesting alone is NOT enough. The n_b=3 core tops out around
#       occupation 4 (near-vacuum pion sector), one expansion step raises it by one, and
#       high-occupation determinants are trimmed on amplitude before they can climb. Measured:
#       the un-augmented nested solve reached occupation 6 and put EXACTLY ZERO weight on the
#       n_b=4-only region — so Delta=0 would have meant "the search never looked", and the
#       decision rule below would have passed for the wrong reason. The nested solve is now
#       seeded with ~19k n_b=4-only determinants (the dominant n_b=3 determinants with one
#       boson mode raised into levels 8..11), so the variational solve is SHOWN the new states
#       and keeps them only if they lower the energy. `n_hi_only_seeded` / `n_hi_only_kept` /
#       `hi_only_weight` per rung are the evidence that it looked.
#   Every energy is a Rayleigh quotient over an explicitly named determinant set (NOT
#   GroundStateResult.energy, which is the larger survivor POOL's energy — mixing the two
#   produced a spurious ~39 MeV artifact during development; see misc/run_nb_nested_shard.py).
#   `--also-independent` runs the legacy arm alongside, so nested-vs-independent is measured.
#
#   WHAT DELTA IS AND IS NOT. Both sides are variational upper bounds, so their difference is
#   NOT a bound on the true cutoff effect E_3^exact - E_4^exact (no variational-difference
#   method can be — this is the audit's objection 7 and it stands). The augmented n_b=4 side
#   gets the richer round-0 pool, so Delta is if anything an OVER-estimate of the nested
#   effect — the conservative direction for a smallness claim. The defensible statement is:
#   "at equal core budget, with the n_b=4-only states explicitly offered, raising the cutoff
#   buys at most Delta of variational improvement."
#
# THE GRID (A = nucleon count; fermion modes = 4 x sites, so L=2/A=32 is fully filled)
#   L=2 (8 sites, 32 modes):  A = 1, 2, 4, 8, 16, 32     x seeds{0,1,2}   = 18 shards
#   L=3 (27 sites, 108 modes): A = 1, 8, 27              x seeds{0,1,2}   =  9 shards
#   L=4 (64 sites, 256 modes): A = 1, 8                  x seeds{0,1,2}   =  6 shards
#   A=1 is kept at every L so the new data is directly comparable to the existing arm.
#
# PRE-SPECIFIED DECISION RULE (written down BEFORE the data — task 35 T3/D3). Discharge the
# large-volume conditional over the TESTED RANGE only if BOTH:
#   (a) nested Delta34/site < 0.001 at every sampled (L, A) WITH n_hi_only_seeded > 0 at
#       every rung (i.e. the search demonstrably looked at the new states), AND
#
# ADDENDUM (2026-09-05, after the first cluster run — cluster 292481, removed and relaunched).
# A SECOND, CLEANER observable exists and is now recorded per rung: `delta_shared` =
# E_4(core_3) - E_3(core_3), the two cutoffs on the IDENTICAL determinant set. It was expected
# to be identically zero; it is not, and the reason is physics. The Hamiltonian has 252 terms
# with `a a^dagger` ordering, and a^dagger|N_f-1> = 0 in a truncated Fock space, so a
# determinant sitting at the low cutoff's TOP level (occupation 7 at n_b=3) is scored ~357.6 MeV
# LOWER by H_3 than by H_4 (measured directly; the cutoffs agree exactly at occupations 4 and 6).
# So the low cutoff does not merely OMIT the boundary states, it MIS-SCORES them — in the
# direction that flatters the low cutoff. `delta_shared` carries NO selection noise at all,
# which makes it the cleanest cutoff measurement available and a genuine shared-basis
# comparison of the kind the audit named. `lo_boundary_weight` x ~358 MeV is the leading
# estimate of the truncation error at fixed basis.
# The pre-specified rule above is UNCHANGED (it was fixed before the data and stays that way);
# delta_shared is reported ALONGSIDE it as additional evidence, not swapped in.
# Consequence: `delta_nested` is NOT sign-definite once boundary population exists — E_4 can
# exceed E_3 on the same core. Only delta_shared >= 0 is expected, and it is checked.
#   (b) the core-ladder residual is also < 0.001 MeV/site (the measurement can resolve the
#       target — this is the condition that FAILS today, at +-0.0028).
# If either fails: the conditional is RETAINED and we widen the empirical claim instead,
# reporting which condition failed. 0.001 MeV/site = 1 MeV / 1000 sites, a deliberately
# conservative uniform per-site slice of the L=10 GSEE target.
#
# Runs at JobPrio 10 — BELOW the live n_b=3 baseline campaign (292477/292478, JobPrio 20-25),
# so it queues behind the headline regeneration rather than competing with it.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/nb_cutoff
#     sh submit_nb_nested.sh test   # 1 cheap L=2 A=8 shard — env + sign-definiteness check
#     sh submit_nb_nested.sh        # the 33-shard A-sweep
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbNested"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="./run_nb_nested_shard.sh"

# args: L dim A seed campaign max_core n_rungs phase0_runs max_rung_seconds also_independent
ARGS='$(L) 3 $(A) $(SEED) '"${BASE}"' $(MAXCORE) 11 32 $(MAXRUNGSEC) 1'

emit_sub() {   # $1=name $2=grid $3=prio
  cat > "$DIR/$1.sub" <<EOF
Executable              = ${RUNNER}
arguments               = ${ARGS}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_nb_nested_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = \$(CPUS)
request_memory          = \$(MEM)
request_disk            = 10G
JobPrio                 = $3
Output                  = ${DIR}/logs/$1_L\$(L)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/$1_L\$(L)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,A,SEED,MAXCORE,MAXRUNGSEC,MEM,CPUS from $2
EOF
  condor_submit "$DIR/$1.sub"
}

# cols: L A SEED MAXCORE MAXRUNGSEC MEM CPUS
# The nested shard runs TWO solves per rung (plus the legacy arm = three), so cores are held
# one notch below the baseline campaign's and the rung budget does the real stopping.
row() { printf '%s %s %s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7"; }

if [ "$MODE" = "test" ]; then
  G="$DIR/smoke.txt"; row 2 8 0 32000 7200 32G 16 > "$G"
  emit_sub smoke "$G" 12
  echo "CAMPAIGN=${BASE}  SMOKE (L=2 A=8 to 32k). Check in the shard JSON:"
  echo "  * embed_gap == 0 at every rung (the embedding identity)"
  echo "  * delta_nested >= 0 at every rung (sign-definiteness)"
  echo "  * delta_independent changes sign somewhere (the artifact being removed)"
  exit 0
fi

G="$DIR/grid.txt"; : > "$G"
for S in 0 1 2; do
  for A in 1 2 4 8 16 32;  do row 2 "$A" "$S" 262144 14400 48G 16 >> "$G"; done
  for A in 1 8 27;         do row 3 "$A" "$S" 262144 21600 96G 16 >> "$G"; done
  for A in 1 8;            do row 4 "$A" "$S" 131072 21600 96G 24 >> "$G"; done
done
emit_sub nested "$G" 10
echo "CAMPAIGN=${BASE}  jobs=$(wc -l < "$G")  (nested n_b=3->4 A-sweep: L=2 A{1..32}, L=3 A{1,8,27}, L=4 A{1,8} x seeds{0,1,2})"
echo "Retrieve: rsync -az hep:/nfs_scratch/bfriend3/NuQu/NuQu/hpc/nb_cutoff/campaign_${BASE}/shards/ \\"
echo "              data/classical/\$(date +%F)/nb_nested_<cluster>/"
