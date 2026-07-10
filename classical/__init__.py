"""Classical ground-state solvers for the NuQu dynamical-pion EFT (Stream A).

The classical baseline the qubitization resource estimate is meant to beat.
First approach: TrimCI-style selected CI over the mixed fermion-boson Fock
basis (`classical.trimci`).
"""

# --- Concurrency / thermal safety (do this BEFORE numpy/scipy/the C++ solver
# load; this package init runs first). Every math backend (OpenBLAS/MKL/Accelerate
# and the OpenMP TrimCI Davidson) otherwise grabs ALL cores, so running the solver
# or the test suite in parallel — e.g. two agents at once — oversubscribes the CPU:
# N runs x C cores of threads thrash the scheduler, make no progress, peg every
# core at 100%, and overheat the machine (this bit us: two agents "stuck" for 40
# CPU-minutes). Cap to half the cores by default so a single run stays fast (the
# hot path is single-threaded C++ connection recompute, so threads barely help
# anyway) while several concurrent runs still fit without oversubscribing.
# Override with NUQU_NUM_THREADS (=<ncpu> for a dedicated single run, =1 for many).
import os as _os

_cap = _os.environ.get("NUQU_NUM_THREADS") or str(max(1, (_os.cpu_count() or 4) // 2))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_v, _cap)
del _os, _cap, _v
