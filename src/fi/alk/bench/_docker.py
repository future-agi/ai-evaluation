"""Docker code-exec lane — run held-out checks against untrusted candidate code
in a per-task, network-isolated, resource-capped, ephemeral container.

This is the harder-isolated sibling of the subprocess verifier in ``_codeexec``.
For **untrusted agent output** it adds real OS-level isolation the subprocess lane
cannot: no network, no host writes, dropped capabilities + no-new-privileges, a
nosuid tmpfs, capped CPU/memory/PIDs, killed + removed after the run.

Honesty: a Docker run of untrusted candidate code is a genuine **live** event,
so :func:`fi.alk.bench._coding.run_coding_artifact_in` stamps these rows
``evidence_class=live_lane`` (never ``captured_fixture``) — see that module.

This lane is **opt-in** (``sandbox="docker"``) and is NEVER a release-gate
prerequisite: the credential-free ``bench_contract_readiness`` gate runs the
subprocess lane on trusted shipped code so it works anywhere with no Docker. The
import is lazy (only resolved when ``sandbox="docker"`` is requested), so the kit
imports fine on a machine with no Docker.

KNOWN LIMITATION (oracle hold-out): hold-out here is only *structural* — the
checks file is not part of the candidate's source, but it is materialised into the
same container the candidate runs in, and the candidate's module body executes
(at import) before the ``check_*`` functions. A deliberately adversarial candidate
could therefore read the checks file at runtime and reflect the expected values,
or print a forged result line. This lane defends against *accidental* gaming
(no-op / fake-success / wrong answer all fail) and gives strong OS isolation, but
it is NOT yet a hardened defence against a candidate that actively attacks the
harness protocol. Closing that requires process/UID separation of the oracle from
the candidate (the inject-tests-only-after-the-agent-finishes topology), tracked
as the live-agent-in-container step. Do not treat a passing score from an
untrusted, adversarial candidate as authoritative until that lands.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import uuid
from typing import Any

from ._codeexec import (
    SUPPORTED_LANGUAGES,
    _empty_result,
    _parse_runner_stdout,
    _tail,
)

# In-container bootstrap: materialise the candidate + held-out checks from base64
# into the writable tmpfs (no host bind-mount — works identically on macOS Docker
# Desktop and Linux), then run each check_* in isolation and emit one JSON line.
# Doubled braces are literal dict syntax preserved through ``str.format``.
_DOCKER_BOOTSTRAP = (
    "import base64,importlib,json,sys,traceback\n"
    "open('/tmp/solution.py','wb').write(base64.b64decode('{cand_b64}'))\n"
    "open('/tmp/bench_checks.py','wb').write(base64.b64decode('{checks_b64}'))\n"
    "sys.path.insert(0,'/tmp')\n"
    "results={{}}\n"
    "try:\n"
    "    checks=importlib.import_module('bench_checks')\n"
    "except Exception:\n"
    "    print(json.dumps({{'results':{{}},'fatal':'checks_import_failed: '"
    "+traceback.format_exc(limit=2).strip().replace(chr(10),' | ')}}));sys.exit(1)\n"
    "names=sorted(n for n in dir(checks) if n.startswith('check_') and callable(getattr(checks,n)))\n"
    "if not names:\n"
    "    print(json.dumps({{'results':{{}},'fatal':'no check_* callables found'}}));sys.exit(1)\n"
    "for name in names:\n"
    "    try:\n"
    "        getattr(checks,name)();results[name]=True\n"
    "    except Exception:\n"
    "        results[name]=False\n"
    "print(json.dumps({{'results':results,'fatal':None}}))\n"
    "sys.exit(0 if results and all(results.values()) else 1)\n"
)

# Default base image. Production should pin by digest (image@sha256:...) for
# determinism; the tag default keeps the example/proof portable.
DEFAULT_IMAGE = "python:3.11-slim"

_DEFAULT_MEMORY = "256m"
_DEFAULT_CPUS = "1.0"
_DEFAULT_PIDS = 128


def docker_available() -> bool:
    """True if a working Docker daemon is reachable (cheap, no pull)."""

    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _build_docker_argv(
    name: str, image: str, memory: str, cpus: str, bootstrap: str
) -> list[str]:
    """Build the hardened ``docker run`` argv (pure; unit-testable without a daemon).

    Defense-in-depth for the untrusted lane: no network, read-only rootfs, a
    non-root user, ALL capabilities dropped (the bounding set too — uid 65534 only
    clears effective/permitted, but the base image ships setuid-root binaries that
    could otherwise re-escalate), no new privileges, a nosuid size-capped tmpfs as
    the only writable surface, and PID/memory/CPU caps. Per-task, ephemeral
    (``--rm``); args passed as a list (never a shell).
    """

    return [
        "docker", "run", "--rm",
        "--name", name,
        "--network", "none",            # the real win the subprocess lane can't give
        "--memory", memory,
        "--cpus", cpus,
        "--pids-limit", str(_DEFAULT_PIDS),
        "--user", "65534:65534",        # nobody: no in-container root
        "--cap-drop", "ALL",            # drop the bounding set (block setuid re-escalation)
        "--security-opt", "no-new-privileges",
        "--read-only",                   # rootfs read-only; only the tmpfs is writable
        "--tmpfs", "/tmp:size=16m,nosuid",  # nosec B108 — container-internal path, not a host temp
        "--entrypoint", "python",
        image,
        "-B", "-c", bootstrap,           # -B: no .pyc writes under the read-only fs
    ]


def run_code_tests_docker(
    candidate_code: str,
    checks_code: str,
    *,
    language: str = "python",
    timeout_s: float = 10.0,
    image: str = DEFAULT_IMAGE,
    memory: str = _DEFAULT_MEMORY,
    cpus: str = _DEFAULT_CPUS,
) -> dict[str, Any]:
    """Run ``checks_code`` against ``candidate_code`` inside an isolated container.

    Returns the same ``{"result", "raw"}`` shape as the subprocess verifier; the
    ``raw`` block records ``sandbox="docker"``, the image, and isolation flags.
    Never raises for an unavailable daemon or a hostile candidate — both surface
    as an honest failing Result.
    """

    if language not in SUPPORTED_LANGUAGES:
        return _empty_result(
            f"unsupported language {language!r}; supported: {SUPPORTED_LANGUAGES}",
            {"sandbox": "docker", "language": language, "infra_error": True},
        )
    if not docker_available():
        return _empty_result(
            "docker unavailable (no daemon / not installed)",
            {"sandbox": "docker", "language": language,
             "docker_available": False, "infra_error": True},
        )

    name = f"agent-learn-bench-{uuid.uuid4().hex[:12]}"
    raw: dict[str, Any] = {
        "sandbox": "docker",
        "language": language,
        "image": image,
        "network": "none",
        "memory": memory,
        "cpus": cpus,
        "cap_drop": "all",
        "no_new_privileges": True,
        "container": name,
        "timed_out": False,
        "exit_code": None,
    }

    bootstrap = _DOCKER_BOOTSTRAP.format(
        cand_b64=base64.b64encode(candidate_code.encode("utf-8")).decode("ascii"),
        checks_b64=base64.b64encode(checks_code.encode("utf-8")).decode("ascii"),
    )
    argv = _build_docker_argv(name, image, memory, cpus, bootstrap)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s + 20.0
        )
    except subprocess.TimeoutExpired as exc:
        raw["timed_out"] = True
        raw["stdout_tail"] = _tail(exc.stdout if isinstance(exc.stdout, str) else "")
        raw["stderr_tail"] = _tail(exc.stderr if isinstance(exc.stderr, str) else "")
        _force_kill(name)
        return _empty_result(f"timed out after {timeout_s}s (container killed)", raw)

    raw["exit_code"] = proc.returncode
    raw["stdout_tail"] = _tail(proc.stdout)
    raw["stderr_tail"] = _tail(proc.stderr)

    # A daemon/image error (e.g. image not pulled) is infra, not an agent fail.
    if proc.returncode not in (0, 1) and "{" not in (proc.stdout or ""):
        raw["infra_error"] = True
        return _empty_result(
            f"docker run failed (exit {proc.returncode}): {_tail(proc.stderr, 300)}",
            raw,
        )

    parsed = _parse_runner_stdout(proc.stdout)
    if parsed is None:
        return _empty_result(
            f"runner produced no parseable result (exit {proc.returncode})", raw
        )
    fatal = parsed.get("fatal")
    results = {str(k): bool(v) for k, v in (parsed.get("results") or {}).items()}
    if fatal:
        return _empty_result(str(fatal), raw)
    if not results:
        return _empty_result("no checks executed", raw)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    return {
        "result": {
            "scalar": round(passed / total, 6),
            "components": {
                "checks_passed": float(passed),
                "checks_total": float(total),
            },
            "pass_fail": results,
            "explanation": f"{passed}/{total} checks passed",
        },
        "raw": raw,
    }


def _force_kill(name: str) -> None:
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)
    except Exception:  # nosec B110 — best-effort cleanup; a kill failure must never propagate
        pass
