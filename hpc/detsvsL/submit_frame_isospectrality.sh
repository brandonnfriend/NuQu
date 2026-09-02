#!/bin/sh
# FRAME ISOSPECTRALITY / "similar-enough n_b" campaign (Track A + B).
# Quantifies the frame non-isospectrality error and the frame-adjusted boson cutoff, per
# docs/lf_backevaluation.md + the theory verdict (squeeze/COO = operator-identity -> E_frame
# already variational; LF = the only frame needing back-evaluation). One shard = one
# (L, dim, frame, A/filling, n_b, seed) over geometric cores; per core it records E_frame,
# E_orig (LF back-eval, Ritz-valid), eps_leak = 1-||P_Nf U|psi~>||^2 (the exact tightness
# measure), the framed occupation tail/histogram, support/wall, and -- on the small ED anchor --
# the exact E_0/E_1 -> gap_orig and the Kato-Temple certified interval.
#
# Three groups:
#   ANCHOR : L=2 dim=1 (ED-feasible) -- validates E_orig>=E_exact, eps_leak->0, Kato-Temple
#            brackets, and the operator-identity claim (COO/gaussian E_frame variational). Cheap.
#   TRACK A: L={2,3} dim=3 DILUTE, n_b sweep {2,3,4} x all frames -- the error-vs-n_b headline
#            ("what n_b is similar-enough for the ground state in each frame").
#   TRACK B: L={2,3} dim=3, filling {0.5,1.0}, n_b=2 -- matched-cost frame comparison reporting
#            E_orig (LF via back-eval) vs E_var(bare); regenerates the retired "wins grow with L".
#
# Run from $REPO/hpc/detsvsL/ ON THE PINNED SUBMIT NODE (ssh hep-submit):
#     cd /nfs_scratch/bfriend3/NuQu/NuQu && git fetch origin -q \
#         && git checkout remediation/vertex-fix && git reset --hard origin/remediation/vertex-fix
#     cd hpc/detsvsL
#     sh submit_frame_isospectrality.sh test    # 1 tiny gaussian+lf L=2 d=1 shard (path + anchor)
#     sh submit_frame_isospectrality.sh         # full grid (~72 shards)
#
# Combine: rsync campaign_<CID>/shards, then per (frame,L,n_b) plot E_orig / eps_leak / occ vs
# n_b (Track A) and E_orig(frame) vs E_var(bare) at matched core (Track B).
set -eu
MODE="${1:-run}"
CAMPAIGN="frameiso-$(date +%Y%m%d-%H%M%S)-$$"
DIR="campaign_${CAMPAIGN}"; mkdir -p "$DIR/logs"
QIS='requirements = (Machine == "qis1.hep.wisc.edu") || (Machine == "qis2.hep.wisc.edu") || (Machine == "qis3.hep.wisc.edu")'
FRAMES="bare gaussian lf gaussian+lf coo gaussian+coo"

if [ "$MODE" = "test" ]; then
  # gaussian+lf L=2 d=1 n_b=2 (N_f=4): reference = solve cutoff so exact-ref Lanczos (32768
  # states) is ED-feasible -> validates C++ build, LF back-eval path, eps_leak, and the anchor.
  cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = 2 gaussian+lf ${CAMPAIGN} none 250+1000 2 1 1 16 0
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=- NUQU_EXACT_REF=1"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = 8G
request_disk            = 8G
Output                  = ${DIR}/logs/smoketest.out
Error                   = ${DIR}/logs/smoketest.err
Log                     = ${DIR}/logs/campaign.log
queue
EOF
  condor_submit "$DIR/campaign.sub"
  echo "CAMPAIGN=${CAMPAIGN}  SMOKE TEST (gaussian+lf L=2 d=1 n_b=2, back-eval-nf=4, exact-ref)"
  echo "check: grep '\[bkshard\] done status=0' ${DIR}/logs/smoketest.out"
  exit 0
fi

# --- shards.txt : columns = L DIM FRAME A FILLING CORES NB BKNF EXREF MEM SEED ---
# CORES uses '+' (Condor 'queue from file' splits on commas too; run_backeval_shard.sh maps back).
SH="$DIR/shards.txt"; : > "$SH"
row() { printf '%s %s %s %s %s %s %s %s %s %s %s\n' "$@" >> "$SH"; }

for fr in $FRAMES; do
  # ANCHOR: L=2 dim=1, n_b=2 (N_f=4). Reference = solve cutoff (BKNF off) so the exact-ref
  # Lanczos (32768 states) is ED-feasible -> validates E_orig>=E_exact, gap_orig, Kato-Temple,
  # eps_leak, and the operator-identity claim (COO/gaussian E_frame == E_exact).
  row 2 1 "$fr" 1 none "250+1000+4000" 2 - 1 8G 0
  # TRACK A: L={2,3} dim=3 dilute (A=1), n_b sweep {2,3,4}; LF map-back scored at ref N_f=16
  # (BKNF=4 raises the low-n_b solves; n_b=4 is already there). exact-ref OFF (d=3 is ED-impossible).
  for nb in 2 3 4; do
    row 2 3 "$fr" 1 none "250+1000+4000+16000" "$nb" 4 0 24G 0
    row 3 3 "$fr" 1 none "250+1000+4000"       "$nb" 4 0 48G 0
  done
  # TRACK B: L={2,3} dim=3, matched-cost comparison at n_b=2 (ref N_f=16), fillings 0.5 & 1.0.
  for fill in 0.5 1.0; do
    row 2 3 "$fr" 1 "$fill" "250+1000+4000+16000" 2 4 0 24G 0
    row 3 3 "$fr" 1 "$fill" "250+1000+4000"       2 4 0 48G 0
  done
done
NJOBS=$(wc -l < "$SH")

cat > "$DIR/campaign.sub" <<EOF
Executable              = ./run_backeval_shard.sh
arguments               = \$(L) \$(FRAME) ${CAMPAIGN} \$(FILLING) \$(CORES) \$(NB) \$(DIM) \$(A) 16 \$(SEED)
environment             = "NUQU_NUM_WORKERS=8 NUQU_BACKEVAL_NF=\$(BKNF) NUQU_EXACT_REF=\$(EXREF)"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_backeval_shard.sh
transfer_output_files   = ""
${QIS}
request_cpus            = 8
request_memory          = \$(MEM)
request_disk            = 8G
JobPrio                 = 15
Output                  = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_A\$(A)f\$(FILLING)_nb\$(NB)_s\$(SEED).out
Error                   = ${DIR}/logs/L\$(L)d\$(DIM)_\$(FRAME)_A\$(A)f\$(FILLING)_nb\$(NB)_s\$(SEED).err
Log                     = ${DIR}/logs/campaign.log
queue L,DIM,FRAME,A,FILLING,CORES,NB,BKNF,EXREF,MEM,SEED from ${SH}
EOF
condor_submit "$DIR/campaign.sub"
echo "CAMPAIGN=${CAMPAIGN}  jobs=${NJOBS}  shard_dir=${DIR}/shards"
echo "grid: {${FRAMES}} x [anchor L2d1 nb2 exact-ref; TrackA L{2,3}d3 dilute nb{2,3,4}; TrackB L{2,3}d3 fill{0.5,1.0} nb2]"
echo "analyze: E_orig/eps_leak/occ vs n_b (Track A); E_orig(frame) vs E_var(bare) at matched core (Track B)"
