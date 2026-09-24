"""`detect_cpus` in hpc/detsvsL/run_frame_shard.sh (2026-09-24).

qis Condor does not export _CONDOR_REQUEST_CPUS, so the old `${_CONDOR_REQUEST_CPUS:-2}` ran
every job at 2 cpus. The chain is NUQU_CPUS -> _CONDOR_REQUEST_CPUS -> OMP_THREAD_LIMIT ->
PYTHON_CPU_COUNT -> `Cpus` in $_CONDOR_MACHINE_AD -> 2, skipping empty/non-numeric values.
"""
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SH = os.path.join(_ROOT, "hpc", "detsvsL", "run_frame_shard.sh")


def _fn():
    lines, on = [], False
    for ln in open(_SH):
        if ln.startswith("detect_cpus()"):
            on = True
        if on:
            lines.append(ln)
            if ln.startswith("}"):
                break
    return "".join(lines)


def _run(env):
    p = subprocess.run(["sh", "-c", _fn() + "\ndetect_cpus"], env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_detect_cpus_chain():
    ad = tempfile.NamedTemporaryFile("w", suffix=".ad", delete=False)
    ad.write("DetectedCpus = 96\nCpus = 6\n")
    ad.close()
    try:
        cases = [({}, "2"),
                 ({"_CONDOR_REQUEST_CPUS": "16"}, "16"),
                 ({"OMP_THREAD_LIMIT": "4"}, "4"),                    # what qis actually sets
                 ({"PYTHON_CPU_COUNT": "8"}, "8"),
                 ({"_CONDOR_MACHINE_AD": ad.name}, "6"),
                 ({"NUQU_CPUS": "3", "OMP_THREAD_LIMIT": "4"}, "3"),
                 ({"OMP_THREAD_LIMIT": "abc", "PYTHON_CPU_COUNT": "5"}, "5"),
                 ({"_CONDOR_REQUEST_CPUS": "0", "OMP_THREAD_LIMIT": "4"}, "4")]
        for env, want in cases:
            got = _run(dict(env, PATH=os.environ["PATH"]))
            assert got == want, f"{env} -> {got}, expected {want}"
    finally:
        os.unlink(ad.name)


if __name__ == "__main__":
    test_detect_cpus_chain()
    print("test_detect_cpus: PASS  (NUQU_CPUS > request > OMP_THREAD_LIMIT > PYTHON_CPU_COUNT "
          "> machine ad > 2; junk skipped)")
    sys.exit(0)
