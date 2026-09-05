"""The NESTED boson-cutoff comparison (`misc.run_nb_nested_shard`) — task 35, T3 / audit P0-3.

The volume-scaling arm that carries the L=10 cutoff conditional measures Delta34 with two
INDEPENDENTLY selected solves, and the audit's objections 3 and 4 (independent spaces; shifts
oscillating in sign at the size of the signal) are consequences of that. Nesting the high-cutoff
solve inside the low-cutoff solution is supposed to remove both. This test pins the three facts
the method rests on, because if any of them is false the sign-definiteness claim is empty:

  1. THE EMBEDDING IDENTITY. H(n_b=hi) and H(n_b=lo) have identical term lists, and a low-cutoff
     determinant transfers to the high-cutoff space unchanged, so evaluating the low-cutoff core
     under H_hi returns the low-cutoff energy EXACTLY. (This is also why a same-basis comparison
     is trivially null and nesting is required: P.H_hi.P == P.H_lo.P on a shared core.)
  2. THE POOL/CORE TRAP. `GroundStateResult.energy` is the energy of the survivor POOL, not of the
     saved top-k core, so it must never enter the difference. Both are valid Ritz bounds -- of
     different spaces. Getting this wrong produced a spurious ~39 MeV "gap" during development.
  3. SIGN-DEFINITENESS. With every energy taken over a named determinant set and the high-cutoff
     energy defined as the better of (embedded core, nested solve), Delta = E_lo - E_hi >= 0 holds
     exactly, not just usually.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "classical", "trimci", "backend_fork"))

from classical.trimci import build_from_eft                                    # noqa: E402
from classical.trimci.graph_arrays import ground_state_ensemble_arrays          # noqa: E402
from misc.run_nb_nested_shard import _core_energy                               # noqa: E402

L, DIM, A, NB_LO, NB_HI = 2, 1, 2, 3, 4      # 1D, 2 sites -- tiny and fast


def main():
    fails = []
    H_lo = build_from_eft(L, DIM, NB_LO, transform="bare")
    H_hi = build_from_eft(L, DIM, NB_HI, transform="bare")

    # --- 1. the two Hamiltonians are the same operator, differing only in Fock dimension
    if H_lo.N_f != 2 ** NB_LO or H_hi.N_f != 2 ** NB_HI:
        fails.append(f"N_f wrong: {H_lo.N_f}, {H_hi.N_f}")
    if len(H_lo.terms) != len(H_hi.terms):
        fails.append(f"term counts differ: {len(H_lo.terms)} vs {len(H_hi.terms)}")
    k = lambda t: (tuple(t.ferm_ops), tuple(t.bos_ops))
    d_lo = {k(t): t.coeff for t in H_lo.terms}
    d_hi = {k(t): t.coeff for t in H_hi.terms}
    if set(d_lo) != set(d_hi):
        fails.append("term keys differ between cutoffs")
    elif any(abs(d_lo[key] - d_hi[key]) > 1e-12 for key in d_lo):
        fails.append("term coefficients differ between cutoffs")

    # --- 2. the embedding identity, on a real selected core
    res = ground_state_ensemble_arrays(H_lo, n_elec=A, n_runs=4, n_dets=400, seed=0)
    core = (res.ferm_arr, res.bos_arr)
    e_lo = _core_energy(H_lo, core)
    e_hi = _core_energy(H_hi, core)
    if abs(e_hi - e_lo) > 1e-9 * max(1.0, abs(e_lo)):
        fails.append(f"embedding identity broken: E_hi(core)={e_hi:.9f} != E_lo(core)={e_lo:.9f}")
    if int(np.asarray(res.bos_arr).max()) >= H_lo.N_f:
        fails.append("core carries an occupation outside the low cutoff -- arrays did not transfer")

    # --- 3. the pool/core trap: res.energy is NOT the saved core's energy
    if abs(float(res.energy) - e_lo) < 1e-12:
        fails.append("res.energy == core energy; the pool/core distinction this method depends on "
                     "has changed -- re-check run_nb_nested_shard before trusting delta")
    elif float(res.energy) > e_lo + 1e-9:
        fails.append(f"pool energy {res.energy:.6f} ABOVE core energy {e_lo:.6f} "
                     "(the pool is the larger space; it cannot be higher)")

    # --- 4. end-to-end: sign-definiteness and embed_gap==0 over a real ladder
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "nested.json")
        p = subprocess.run(
            [sys.executable, "-m", "misc.run_nb_nested_shard", "--L", str(L), "--dim", str(DIM),
             "--A", str(A), "--seed", "0", "--n-b-lo", str(NB_LO), "--n-b-hi", str(NB_HI),
             "--ladder-start", "200", "--n-rungs", "3", "--max-core", "800",
             "--phase0-runs", "4", "--also-independent", "--out", out],
            cwd=_ROOT, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                [_ROOT, os.path.join(_ROOT, "classical", "trimci", "backend_fork")])))
        if p.returncode != 0:
            fails.append(f"nested shard failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        else:
            j = json.load(open(out))
            if not j.get("done") or not j.get("rungs"):
                fails.append("nested shard produced no rungs")
            for r in j.get("rungs", []):
                if abs(r["embed_gap"]) > 1e-6 * max(1.0, abs(r["E_lo"])):
                    fails.append(f"core {r['core']}: embed_gap {r['embed_gap']:.3e} != 0")
                if r["delta_nested"] < -1e-9:
                    fails.append(f"core {r['core']}: NESTED delta {r['delta_nested']:.3e} < 0 -- "
                                 "sign-definiteness violated")
                if "delta_independent" not in r:
                    fails.append(f"core {r['core']}: --also-independent produced no column")
                if r["E_hi_nested"] > r["E_lo"] + 1e-9:
                    fails.append(f"core {r['core']}: E_hi above E_lo despite nesting")
            if "manifest" not in j or not j["manifest"].get("git_commit"):
                fails.append("nested shard wrote no provenance manifest")

    if fails:
        print("test_nb_nested: FAILED")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"test_nb_nested: PASS  (identical term lists; embedding identity exact "
          f"(|E_hi-E_lo| < 1e-9 on a real core); pool energy {float(res.energy):.4f} distinct from "
          f"core energy {e_lo:.4f}; nested delta >= 0 and embed_gap == 0 at every rung; "
          f"independent switch and provenance manifest present)")


_SUBMIT = os.path.join(_ROOT, "hpc", "nb_cutoff", "submit_nb_nested.sh")
_QUEUE_VARS = ["L", "A", "SEED", "MAXCORE", "MAXRUNGSEC", "MEM", "CPUS"]


def test_nb_nested_submit_grid():
    """The T3 A-sweep grid. A wrong column count in a Condor `queue <vars> from file` binds
    MEM to a core count and the submit either fails or runs nonsense, which is expensive to
    discover on the cluster -- so the grid is checked against a stubbed condor_submit."""
    import shutil
    import tempfile
    fails = []
    tmp = tempfile.mkdtemp(prefix="nbnestedsub_")
    try:
        shutil.copy(_SUBMIT, tmp)
        shutil.copy(os.path.join(_ROOT, "hpc", "nb_cutoff", "run_nb_nested_shard.sh"), tmp)
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "condor_submit")
        open(stub, "w").write("#!/bin/sh\necho fake $*\n")
        os.chmod(stub, 0o755)
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"])
        p = subprocess.run(["sh", os.path.basename(_SUBMIT)], cwd=tmp, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, f"submit script failed:\n{p.stdout}\n{p.stderr}"
        cdir = os.path.join(tmp, [d for d in os.listdir(tmp) if d.startswith("campaign_")][0])
        rows = [ln.split() for ln in open(os.path.join(cdir, "grid.txt")).read().splitlines() if ln.strip()]
        sub = open(os.path.join(cdir, "nested.sub")).read()

        col = {v: i for i, v in enumerate(_QUEUE_VARS)}
        qline = [ln for ln in sub.splitlines() if ln.startswith("queue ")]
        if len(qline) != 1 or qline[0].split()[1] != ",".join(_QUEUE_VARS):
            fails.append(f"queue line does not name {_QUEUE_VARS}: {qline}")
        for i, r in enumerate(rows):
            if len(r) != len(_QUEUE_VARS):
                fails.append(f"row {i}: {len(r)} columns, expected {len(_QUEUE_VARS)}: {r}")
        if len(rows) != 33:
            fails.append(f"{len(rows)} shards, expected 33 (18 at L=2 + 9 at L=3 + 6 at L=4)")
        got = {(r[col["L"]], r[col["A"]]) for r in rows if len(r) == len(_QUEUE_VARS)}
        want = ({("2", a) for a in ("1", "2", "4", "8", "16", "32")}
                | {("3", a) for a in ("1", "8", "27")} | {("4", a) for a in ("1", "8")})
        if got != want:
            fails.append(f"(L, A) grid {sorted(got)} != agreed {sorted(want)}")
        if sorted({r[col["SEED"]] for r in rows if len(r) == len(_QUEUE_VARS)}) != ["0", "1", "2"]:
            fails.append("expected seeds {0,1,2}")
        # A=1 must be present at every L so the sweep stays comparable to the existing arm
        for L_ in ("2", "3", "4"):
            if (L_, "1") not in got:
                fails.append(f"A=1 missing at L={L_} -- breaks comparability with the current arm")
        if "qis1.hep.wisc.edu" not in sub or "qis4" in sub:
            fails.append("qis1-3 allocation pin missing/incorrect")
        if "JobPrio                 = 10" not in sub:
            fails.append("JobPrio should be 10 (below the live baseline campaign)")
        if "--also-independent" not in open(os.path.join(tmp, "run_nb_nested_shard.sh")).read():
            fails.append("runner never passes --also-independent (the comparison switch)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not fails, "T3 submit-grid problems:\n  - " + "\n  - ".join(fails)


def test_nb_nested():
    main()


if __name__ == "__main__":
    main()
