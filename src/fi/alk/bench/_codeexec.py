"""Code-tests verifier — run held-out tests against candidate code in isolation.

This is the coding-modality verifier for the bench harness. It implements the
trustworthiness rule every serious coding benchmark uses: **the oracle is held
out** — the held-out checks live in a separate file the candidate code never
imports or sees, and they are executed by a harness-written runner, not by the
candidate.

Sandboxes:
  * ``subprocess`` (default) — a fresh interpreter in a throwaway tempdir, with a
    scrubbed environment (no harness secrets) and a hard wall-clock timeout. This
    is the sandbox used by the credential-free release gate, which only ever runs
    trusted, shipped reference code. It is **not** a security boundary against
    deliberately hostile code (no real filesystem/network isolation); for
    untrusted agent output use the Docker lane (bench step 15E).
  * ``docker`` — per-task container isolation with a no-network default
    (bench step 15E).

The convention for a checks file: it defines one or more ``check_*`` callables
that import the candidate module (``import solution``) and ``assert`` the
expected behaviour. The harness discovers them, runs each, and reports per-check
pass/fail — so a candidate that no-ops, prints a fake "success" message, returns
wrong answers, or fails to define the entrypoint is failed deterministically.

THREAT-MODEL NOTE: the runner and the candidate share one process, so the
deterministic-failure guarantee covers *accidental* gaming, not an *adversarial*
candidate. A candidate that knows this runner's protocol could, during its import
body (which runs before ``check_*``), print a forged ``{"results": ...}`` line and
exit 0, or read the checks file to reflect expected values. Hardening that
(process/UID separation of the oracle + an authenticated out-of-band verdict
channel — the inject-tests-after-the-agent-finishes topology) is tracked separate
work; until then do not treat a passing score from an untrusted adversarial
candidate as authoritative. The release gate is unaffected: it runs only trusted
shipped reference code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..live._runner import scrubbed_lane_env

#: Entry module the candidate code is written to (checks ``import`` this name).
ENTRY_MODULE = "solution"
_CHECKS_MODULE = "bench_checks"
_DEFAULT_TIMEOUT_S = 10.0

SUPPORTED_LANGUAGES = ("python",)

# The harness-written runner: discovers ``check_*`` callables in the checks
# module, runs each, and emits a single JSON line of per-check results to stdout.
# The candidate (``solution.py``) is imported only by the checks module — never
# by this runner directly — so the oracle stays out of the candidate's reach.
_PYTHON_RUNNER = """\
import importlib, json, sys, traceback
results = {}
fatal = None
try:
    checks = importlib.import_module("%(checks)s")
except Exception:
    fatal = "checks_import_failed: " + traceback.format_exc(limit=2).strip().replace(chr(10), " | ")
    print(json.dumps({"results": {}, "fatal": fatal}))
    sys.exit(1)
names = sorted(n for n in dir(checks) if n.startswith("check_") and callable(getattr(checks, n)))
if not names:
    print(json.dumps({"results": {}, "fatal": "no check_* callables found"}))
    sys.exit(1)
for name in names:
    try:
        getattr(checks, name)()
        results[name] = True
    except Exception:
        results[name] = False
print(json.dumps({"results": results, "fatal": None}))
sys.exit(0 if results and all(results.values()) else 1)
"""


def _tail(text: str, limit: int = 2000) -> str:
    text = text or ""
    return text[-limit:]


def _empty_result(explanation: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "result": {
            "scalar": 0.0,
            "components": {"checks_passed": 0.0, "checks_total": 0.0},
            "pass_fail": {},
            "explanation": explanation,
        },
        "raw": raw,
    }


def run_code_tests(
    candidate_code: str,
    checks_code: str,
    *,
    language: str = "python",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    sandbox: str = "subprocess",
) -> dict[str, Any]:
    """Run ``checks_code`` (the held-out oracle) against ``candidate_code``.

    Returns ``{"result": <unified Result>, "raw": <execution evidence>}``. The
    unified ``Result`` carries a scalar (fraction of checks passed), components
    (passed/total), per-check ``pass_fail`` booleans, and an explanation.
    """

    if language not in SUPPORTED_LANGUAGES:
        return _empty_result(
            f"unsupported language {language!r}; supported: {SUPPORTED_LANGUAGES}",
            {"sandbox": sandbox, "language": language, "infra_error": True},
        )
    if sandbox == "docker":
        # The Docker lane is opt-in; never silently fall back to a weaker sandbox
        # (that would mislabel isolation).
        from ._docker import run_code_tests_docker  # local import: optional lane

        return run_code_tests_docker(
            candidate_code, checks_code, language=language, timeout_s=timeout_s
        )
    if sandbox != "subprocess":
        return _empty_result(
            f"unknown sandbox {sandbox!r}; expected 'subprocess' or 'docker'",
            {"sandbox": sandbox, "language": language, "infra_error": True},
        )

    return _run_subprocess(candidate_code, checks_code, timeout_s=timeout_s)


def _run_subprocess(
    candidate_code: str, checks_code: str, *, timeout_s: float
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agent-learn-bench-") as tmp:
        root = Path(tmp)
        (root / f"{ENTRY_MODULE}.py").write_text(candidate_code, encoding="utf-8")
        (root / f"{_CHECKS_MODULE}.py").write_text(checks_code, encoding="utf-8")
        (root / "_runner.py").write_text(
            _PYTHON_RUNNER % {"checks": _CHECKS_MODULE}, encoding="utf-8"
        )

        raw: dict[str, Any] = {
            "sandbox": "subprocess",
            "language": "python",
            "timed_out": False,
            "exit_code": None,
        }
        try:
            proc = subprocess.run(
                [sys.executable, "_runner.py"],
                cwd=str(root),
                env=scrubbed_lane_env(()),  # no harness secrets cross into the run
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raw["timed_out"] = True
            raw["stdout_tail"] = _tail(exc.stdout if isinstance(exc.stdout, str) else "")
            raw["stderr_tail"] = _tail(exc.stderr if isinstance(exc.stderr, str) else "")
            return _empty_result(f"timed out after {timeout_s}s", raw)

        raw["exit_code"] = proc.returncode
        raw["stdout_tail"] = _tail(proc.stdout)
        raw["stderr_tail"] = _tail(proc.stderr)

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


def _parse_runner_stdout(stdout: str) -> dict[str, Any] | None:
    # The runner prints exactly one JSON line; tolerate trailing candidate prints
    # by scanning for the last decodable JSON object.
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
