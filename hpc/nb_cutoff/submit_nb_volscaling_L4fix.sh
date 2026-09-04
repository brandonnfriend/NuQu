#!/bin/sh
# VOLUME-SCALING L=4 FIX + overnight deep-core robustness probe (re-audit P0-4).
#
# Grid 291048's L=4 n_b=3 cells OOM'd at the 49.5G cgroup limit and stalled at a 128k core, while the
# L=4 n_b=4 cells reached 256k at 96G — so the L=4 DEEPEST COMMON core is capped at 128k (vs L=3's
# 256k), making the L=4 Δ34 point unreliable and the L=10 projection (CONDITIONAL, +1.78 MeV) shaky.
# n_b=3 (N_f=8) is SMALLER than n_b=4 (N_f=16), so the OOM was purely under-provisioned memory.
#
# PRIMARY (the fix): re-run L=4 n_b=3 at 128G, SAME 256k ceiling as n_b=4 → a clean, consistent 256k
#   common core at L=4 (matches L=3). A={0,1}, 3 seeds. 6 shards. High success (n_b=4 did 256k @96G).
# DEEP PROBE (overnight bonus, cancellable): push L=3 AND L=4, n_b={3,4}, A=1, seed 0 to a 512k core
#   (NUQU_N_RUNGS=10, ceiling 524288). Tests whether Δ34/site is CORE-CONVERGED (512k vs 256k) — the
#   per-site shift is currently RISING (L2 .00017 → L3 .00238 → L4 .00280), and this says whether that
#   is real volume scaling or shallow-core drift. 4 shards, heavier memory; cancel if not done by morning.
#
# Separate campaign dirs (…-nb{3,4} vs …-deep-nb{3,4}) so nothing collides with 291048 or each other.
#
# Run from $REPO/hpc/nb_cutoff/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/nb_cutoff
#     sh submit_nb_volscaling_L4fix.sh test   # 1 cell (L=4 n_b=3 A=1 s0 @256k/128G) — de-risk the fix
#     sh submit_nb_volscaling_L4fix.sh        # primary (6) + deep probe (4) = 10 shards
set -eu
MODE="${1:-run}"
BASE="$(date +%Y%m%d-%H%M%S)-nbVolFix"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
DIR="campaign_${BASE}"; mkdir -p "$DIR/logs"
RUNNER="../detsvsL/run_frame_shard.sh"
ENV9="NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_RUNGS=9 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"
ENV10="NUQU_DEEP_SOLVE=1 NUQU_WARM_GROW=1 NUQU_LADDER_NRUNS=1 NUQU_N_RUNGS=10 NUQU_PT2_MAX_CORE=1 NUQU_PHASE0_RUNS=64"

if [ "$MODE" = "test" ]; then
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ${RUNNER}
arguments               = 4 0 ${BASE}-nb3 bare 1 none 262144 independent 4 3 1000 21600
environment             = "${ENV9} NUQU_N_B=3"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = 128G
request_disk            = 12G
Output                  = ${DIR}/logs/test.out
Error                   = ${DIR}/logs/test.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${BASE}  SMOKE (L=4 n_b=3 A=1 s0 @256k/128G): de-risk the memory fix."
  exit 0
fi

# --- PRIMARY: L=4 n_b=3 @256k, 128G (the fix) — A={0,1}, 3 seeds ---
SH1="$DIR/primary.txt"; : > "$SH1"
for A in 0 1; do for S in 0 1 2; do printf '4 3 %s %s 128G\n' "$A" "$S" >> "$SH1"; done; done
cat > "$DIR/primary.sub" <<EOF
Executable              = ${RUNNER}
arguments               = \$(L) \$(SEED) ${BASE}-nb\$(NB) bare \$(A) none 262144 independent 4 3 1000 21600
environment             = "${ENV9} NUQU_N_B=\$(NB)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 12G
JobPrio                 = 18
Output                  = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,A,SEED,MEM from ${SH1}
EOF
condor_submit "$DIR/primary.sub"

# --- DEEP PROBE: 512k robustness, L={3,4} × n_b={3,4}, A=1, seed 0 (cancellable) ---
SH2="$DIR/deep.txt"; : > "$SH2"
printf '3 3 1 0 96G\n3 4 1 0 96G\n4 3 1 0 192G\n4 4 1 0 256G\n' >> "$SH2"
cat > "$DIR/deep.sub" <<EOF
Executable              = ${RUNNER}
arguments               = \$(L) \$(SEED) ${BASE}-deep-nb\$(NB) bare \$(A) none 524288 independent 4 3 1000 21600
environment             = "${ENV10} NUQU_N_B=\$(NB)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = ${RUNNER}
transfer_output_files   = ""
${QIS}
request_cpus            = 16
request_memory          = \$(MEM)
request_disk            = 12G
JobPrio                 = 14
Output                  = ${DIR}/logs/deep_L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).out
Error                   = ${DIR}/logs/deep_L\$(L)_nb\$(NB)_A\$(A)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,NB,A,SEED,MEM from ${SH2}
EOF
condor_submit "$DIR/deep.sub"

echo "CAMPAIGN=${BASE}"
echo "  PRIMARY 6 shards: L=4 n_b=3 @256k/128G, A={0,1}, 3 seeds -> fixes the L=4 common core (128k->256k)"
echo "  DEEP    4 shards: L={3,4} n_b={3,4} @512k, A=1 s0 -> core-convergence robustness (cancel if not done)"
echo "pull primary -> data/classical/nb_volscaling/nb{3,4}/ ; deep -> data/classical/nb_volscaling_deep/nb{3,4}/"
echo "then: python -m misc.make_nb_volscaling   (256k verdict) ; --src …_deep for the 512k robustness read"
