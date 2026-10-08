"""run_frame_shard.sh provisioning that survives a network blip (infrastructure C4, 2026-10-08).

Extracts the `retry` / `provision_*` functions from the job wrapper and runs them with stub
tools in a sandbox:
  * `retry` returns 0 as soon as the command succeeds and 1 after N failures (sleep stubbed);
  * with a pinned uv in $NUQU_TOOLS, `provision_uv` copies it and never calls curl;
    without one, it falls back to the versioned installer and gives up with exit 1 after
    3 failed attempts (-> the script exits 3, which the submit files re-queue);
  * with $NUQU_TOOLS/uvpy holding a cpython-3.10 install, `provision_python` copies it and
    forbids downloads (UV_PYTHON_DOWNLOADS=never); otherwise `uv python install` runs;
  * with $NUQU_TOOLS/wheels, `provision_wheels` installs offline (--no-index) and only
    falls back to PyPI when that fails;
  * the wrapper passes --resume to both ladder modes unless NUQU_RESUME=0, and every
    provisioning failure exits with PROVISION_FAIL (3), while the C++ build failure stays 1.
"""
import os
import re
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SH = os.path.join(_ROOT, "hpc", "detsvsL", "run_frame_shard.sh")
_SRC = open(_SH).read()


def _fn(name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", _SRC, re.S | re.M)
    assert m, name
    return m.group(0)


_FUNCS = "\n".join(_fn(n) for n in ("retry", "provision_uv", "provision_python", "provision_wheels"))


def _sandbox():
    tmp = tempfile.mkdtemp(prefix="prov_")
    binp = os.path.join(tmp, "bin")
    os.makedirs(binp)
    return tmp, binp


def _stub(binp, name, body):
    p = os.path.join(binp, name)
    open(p, "w").write("#!/bin/sh\n" + body + "\n")
    os.chmod(p, 0o755)


def _run(tmp, binp, script, tools=None):
    env = {"PATH": binp + os.pathsep + "/usr/bin:/bin", "SANDBOX": tmp,
           "UV_INSTALL_DIR": os.path.join(tmp, "uvbin"),
           "UV_PYTHON_INSTALL_DIR": os.path.join(tmp, "uvpy"),
           "TOOLS": tools or os.path.join(tmp, "no-tools"), "UV_PIN": "0.12.23",
           "REQ": os.path.join(tmp, "req.txt"), "LOG": os.path.join(tmp, "calls.log")}
    open(env["REQ"], "w").write("numpy==1.26.4\n")
    for stale in (env["LOG"], os.path.join(tmp, "uvbin", "uv")):   # one scenario per call
        if os.path.exists(stale):
            os.remove(stale)
    pre = 'sleep() { echo "sleep $1" >> "$LOG"; }\nPATH="$UV_INSTALL_DIR:$PATH"\n'
    p = subprocess.run(["sh", "-c", pre + _FUNCS + "\n" + script], env=env,
                       capture_output=True, text=True)
    calls = open(env["LOG"]).read() if os.path.exists(env["LOG"]) else ""
    return p, calls


def test_retry():
    tmp, binp = _sandbox()
    try:
        _stub(binp, "flaky", 'n=$(cat "$SANDBOX/n" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$SANDBOX/n"; [ $n -ge 3 ]')
        p, calls = _run(tmp, binp, "retry 5 flaky; echo rc=$?")
        assert "rc=0" in p.stdout and calls.count("sleep") == 2, (p.stdout, calls)
        p, calls = _run(tmp, binp, "retry 2 false; echo rc=$?")
        assert "rc=1" in p.stdout and calls.count("sleep") == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uv_from_tools_or_download():
    tmp, binp = _sandbox()
    try:
        tools = os.path.join(tmp, "tools")
        os.makedirs(tools)
        _stub(tools, "uv", 'echo "uv-from-tools $*"')
        _stub(binp, "curl", 'echo "curl $*" >> "$LOG"; exit 28')
        p, calls = _run(tmp, binp, "provision_uv; echo rc=$? src=$UV_SRC; uv --version", tools)
        assert "rc=0 src=tools" in p.stdout and "uv-from-tools" in p.stdout and "curl" not in calls
        # no pinned binary: the versioned installer is tried 3 times, then it gives up
        p, calls = _run(tmp, binp, "provision_uv; echo rc=$?")
        assert "rc=1" in p.stdout and calls.count("curl") == 3 and "0.12.23/install.sh" in calls
        assert calls.count("sleep") == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_python_and_wheels():
    tmp, binp = _sandbox()
    try:
        tools = os.path.join(tmp, "tools")
        os.makedirs(os.path.join(tools, "uvpy", "cpython-3.10.18-linux-x86_64-gnu", "bin"))
        open(os.path.join(tools, "uvpy", "cpython-3.10.18-linux-x86_64-gnu", "bin", "python3.10"), "w").write("")
        os.makedirs(os.path.join(tools, "wheels"))
        _stub(binp, "uv", 'echo "uv $* downloads=${UV_PYTHON_DOWNLOADS:-unset}" >> "$LOG"; '
                          'case "$*" in *--no-index*) exit "${OFFLINE_RC:-0}" ;; esac; exit 0')
        p, calls = _run(tmp, binp, "provision_python; echo rc=$? src=$PY_SRC; "
                                   "provision_wheels; echo rc=$? src=$WHL_SRC", tools)
        assert "rc=0 src=tools" in p.stdout and "rc=0 src=tools" in p.stdout.splitlines()[-1], p.stdout
        assert os.path.exists(os.path.join(tmp, "uvpy", "cpython-3.10.18-linux-x86_64-gnu", "bin", "python3.10"))
        assert "python install" not in calls and "venv --python 3.10" in calls
        assert "downloads=never" in calls and "--no-index --find-links" in calls
        assert "pip install -q -r" not in calls, "offline install succeeded: no PyPI call"
        # offline wheelhouse broken -> PyPI fallback with retries
        p, calls = _run(tmp, binp, "export OFFLINE_RC=1; provision_wheels; echo rc=$? src=$WHL_SRC", tools)
        assert "rc=0 src=pypi" in p.stdout and "pip install -q -r" in calls
        # no tools at all: managed python is downloaded
        p, calls = _run(tmp, binp, "provision_python; echo rc=$? src=$PY_SRC")
        assert "rc=0 src=download" in p.stdout and "python install 3.10" in calls
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wrapper_wiring():
    assert 'PROVISION_FAIL=3' in _SRC
    assert _SRC.count('|| exit "$PROVISION_FAIL"') >= 3
    assert 'ERROR: C++ build failed" >&2; exit 1' in _SRC, "a build failure is not a provisioning retry"
    assert 'RESUME_ARG="--resume"; [ "${NUQU_RESUME:-1}" = "0" ] && RESUME_ARG=""' in _SRC
    assert _SRC.count("$RESUME_ARG --out") == 2, "both ladder modes pass --resume"
    assert "NUQU_TOOLS:-/nfs_scratch/bfriend3/NuQu/tools" in _SRC
    pin = re.search(r'UV_PIN="\$\{NUQU_UV_VERSION:-([\d.]+)\}"', _SRC).group(1)
    prov = open(os.path.join(_ROOT, "hpc", "detsvsL", "provision_tools.sh")).read()
    assert f'UV_VERSION="${{NUQU_UV_VERSION:-{pin}}}"' in prov, "pin the same uv in both scripts"


def main():
    test_retry()
    test_uv_from_tools_or_download()
    test_python_and_wheels()
    test_wrapper_wiring()
    print("test_run_frame_shard_provisioning: PASS  (retry/backoff; pinned uv, python, wheels "
          "from $NUQU_TOOLS with download fallbacks; exit 3 on provisioning failure; --resume wired)")


if __name__ == "__main__":
    main()
