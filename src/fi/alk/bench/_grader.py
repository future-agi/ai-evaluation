"""Command/artifact-graded coding lane — the hardened coding tier.

This resolves the in-process forge/oracle-read weakness of the ``check_*`` lane by
changing the *model*, not bolting on isolation. A task gives the candidate a
working directory + a way to RUN it; a **held-out grader** runs AFTERWARD and
emits the verdict via its **exit code + a reward file in a grader-controlled
path** — never parsed from candidate-shared stdout.

Why this is robust (the two vulns from the PR review, structurally closed):

* **No verdict forgery** — the verdict is the grader's exit code (and an optional
  ``reward.json`` the grader writes), not anything the candidate prints.
* **No oracle read** — the grader files (held-out expected values / tests) are
  written ONLY after the candidate command has finished and its processes are
  killed. The candidate never co-runs with the grader, so it cannot read the
  expected values; and in the Docker lane the grader files are owned by a
  different user the candidate uid cannot read.

It also gives **multi-language for free**: the candidate ``build`` command and the
``grader`` command are arbitrary shell, so the same lane grades Python, bash,
Node, compiled languages, etc.

Two sandboxes share one flow (temporal separation candidate→grader):
  * ``subprocess`` — candidate + grader run as host subprocesses in *separate*
    temp dirs; the grader dir path is given only to the grader. Credential-free,
    Docker-free; used by the release gate on trusted shipped tasks.
  * ``docker`` — a per-task, network-off, capped, ephemeral container; candidate
    runs as an unprivileged uid, grader files land in a root-owned dir the
    candidate cannot read. The hardened lane for untrusted agent output.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from ._codeexec import _empty_result, _tail

GRADING_COMMAND = "command"  # suite/task grading mode discriminator
_DEFAULT_TIMEOUT_S = 20.0
_REWARD_FILE = "reward.json"


def _files(value: Any) -> dict[str, str]:
    """Coerce a {path: content} mapping to str->str (defensive)."""

    if not isinstance(value, Mapping):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def _result_from_grader(
    rc: int | None, reward: Mapping[str, Any] | None, raw: dict[str, Any]
) -> dict[str, Any]:
    """Build the unified Result from the grader's exit code + optional reward.json.

    Verdict authority: the grader's exit code (0 = pass). If the grader also wrote
    a ``reward.json`` with a numeric ``score`` in [0,1], that becomes the scalar;
    otherwise the scalar is 1.0/0.0 from the exit code. Sub-check booleans, if the
    grader reports a ``checks`` map, flow into ``pass_fail``.
    """

    passed = rc == 0
    scalar: float
    pass_fail: dict[str, bool] = {}
    explanation = f"grader exit {rc}"
    if isinstance(reward, Mapping):
        score = reward.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scalar = round(float(score), 6)
        else:
            scalar = 1.0 if passed else 0.0
        checks = reward.get("checks")
        if isinstance(checks, Mapping):
            pass_fail = {str(k): bool(v) for k, v in checks.items()}
        if reward.get("explanation"):
            explanation = str(reward["explanation"])
    else:
        scalar = 1.0 if passed else 0.0
    if not pass_fail:
        pass_fail = {"grader": passed}
    return {
        "result": {
            "scalar": scalar,
            "components": {"grader_exit_ok": 1.0 if passed else 0.0},
            "pass_fail": pass_fail,
            "explanation": explanation,
        },
        "raw": raw,
    }


def run_command_graded(
    task: Mapping[str, Any],
    candidate_files: Mapping[str, str],
    *,
    sandbox: str = "subprocess",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Grade ``candidate_files`` for a command-graded ``task``.

    Returns ``{"result", "raw"}``. Never raises for infra problems (missing Docker,
    grader crash) — those surface as a failing/infra Result, tagged
    ``raw["infra_error"]`` when the lane could not run at all.
    """

    if sandbox == "docker":
        from ._docker import docker_available

        if not docker_available():
            return _empty_result(
                "docker unavailable (no daemon / not installed)",
                {"sandbox": "docker", "grading": GRADING_COMMAND, "infra_error": True},
            )
        return _run_docker_graded(task, candidate_files, timeout_s=timeout_s)
    if sandbox != "subprocess":
        return _empty_result(
            f"unknown sandbox {sandbox!r}; expected 'subprocess' or 'docker'",
            {"sandbox": sandbox, "grading": GRADING_COMMAND, "infra_error": True},
        )
    return _run_subprocess_graded(task, candidate_files, timeout_s=timeout_s)


def _run_subprocess_graded(
    task: Mapping[str, Any], candidate_files: Mapping[str, str], *, timeout_s: float
) -> dict[str, Any]:
    build = task.get("build")
    grader_cmd = str(task.get("grader_cmd") or "")
    if not grader_cmd:
        return _empty_result("task has no grader_cmd", {"sandbox": "subprocess", "infra_error": True})

    raw: dict[str, Any] = {"sandbox": "subprocess", "grading": GRADING_COMMAND, "timed_out": False}
    with tempfile.TemporaryDirectory(prefix="bench-work-") as work_s, \
            tempfile.TemporaryDirectory(prefix="bench-grader-") as grader_s:
        work = Path(work_s)
        grader = Path(grader_s)
        _write_tree(work, {**_files(task.get("files")), **dict(candidate_files)})

        # PHASE 1 — candidate runs with NO grader present (temporal hold-out).
        if build:
            try:
                proc = subprocess.run(
                    ["sh", "-c", str(build)], cwd=str(work), capture_output=True,
                    text=True, timeout=timeout_s,
                )
                raw["build_exit"] = proc.returncode
                raw["build_stdout_tail"] = _tail(proc.stdout)
            except subprocess.TimeoutExpired:
                raw["timed_out"] = True
                return _empty_result(f"candidate build timed out after {timeout_s}s", raw)

        # PHASE 2 — grader written AFTER, in a dir the candidate phase never knew.
        _write_tree(grader, _files(task.get("grader_files")))
        # The subprocess lane is the trusted/gate tier (not a security boundary —
        # the Docker lane is), so the grader gets a usable PATH to resolve
        # interpreters; GRADER_DIR points it at its held-out files.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "GRADER_DIR": str(grader),
            "HOME": os.environ.get("HOME", str(work)),
        }
        try:
            gproc = subprocess.run(
                ["sh", "-c", grader_cmd], cwd=str(work), capture_output=True,
                text=True, timeout=timeout_s, env=env,
            )
        except subprocess.TimeoutExpired:
            raw["timed_out"] = True
            return _empty_result(f"grader timed out after {timeout_s}s", raw)
        raw["grader_exit"] = gproc.returncode
        raw["grader_stdout_tail"] = _tail(gproc.stdout)
        raw["grader_stderr_tail"] = _tail(gproc.stderr)

        reward = _read_reward(grader / _REWARD_FILE)
        return _result_from_grader(gproc.returncode, reward, raw)


def _read_reward(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


# ---- Docker command-graded lane (hardened, opt-in) ----

_DOCKER_IMAGE = "python:3.11-slim"


def _docker(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def _run_docker_graded(
    task: Mapping[str, Any], candidate_files: Mapping[str, str], *, timeout_s: float
) -> dict[str, Any]:
    build = task.get("build")
    grader_cmd = str(task.get("grader_cmd") or "")
    image = str(task.get("image") or _DOCKER_IMAGE)
    if not grader_cmd:
        return _empty_result("task has no grader_cmd", {"sandbox": "docker", "infra_error": True})

    name = f"agent-learn-grade-{uuid.uuid4().hex[:12]}"
    raw: dict[str, Any] = {
        "sandbox": "docker", "grading": GRADING_COMMAND, "image": image,
        "network": "none", "container": name, "timed_out": False,
    }
    with tempfile.TemporaryDirectory(prefix="bench-docker-") as host_s:
        host = Path(host_s)
        work_host = host / "work"
        grader_host = host / "grader"
        _write_tree(work_host, {**_files(task.get("files")), **dict(candidate_files)})
        _write_tree(grader_host, _files(task.get("grader_files")))

        started = _docker(
            "run", "-d", "--rm", "--name", name, "--network", "none",
            "--memory", "512m", "--cpus", "1.0", "--pids-limit", "256",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            image, "sleep", str(int(timeout_s * 2 + 60)),
        )
        if started.returncode != 0:
            raw["infra_error"] = True
            return _empty_result(
                f"docker run failed: {_tail(started.stderr, 300)}", raw
            )
        try:
            # candidate user + dirs; /work candidate-writable, /grader root-only.
            _docker("exec", name, "sh", "-c",
                    "id cand 2>/dev/null || useradd -M -s /usr/sbin/nologin cand; "
                    "mkdir -p /work /grader")
            _docker("cp", f"{work_host}/.", f"{name}:/work")
            _docker("exec", name, "sh", "-c", "chown -R cand:cand /work && chmod 700 /grader")

            # PHASE 1 — candidate runs as `cand`, no grader files present yet.
            if build:
                try:
                    b = _docker("exec", "-u", "cand", "-w", "/work", name,
                                "sh", "-c", str(build), timeout=timeout_s + 15)
                    raw["build_exit"] = b.returncode
                    raw["build_stdout_tail"] = _tail(b.stdout)
                except subprocess.TimeoutExpired:
                    raw["timed_out"] = True
                    return _empty_result(f"candidate build timed out after {timeout_s}s", raw)

            # kill any lingering candidate processes before grading.
            _docker("exec", name, "sh", "-c", "pkill -u cand 2>/dev/null || true")

            # PHASE 2 — inject grader (root-owned, unreadable to cand), run as root.
            _docker("cp", f"{grader_host}/.", f"{name}:/grader")
            _docker("exec", name, "sh", "-c", "chown -R root:root /grader && chmod -R go-rwx /grader")
            try:
                g = _docker("exec", "-w", "/work", name, "sh", "-c",
                            f"export GRADER_DIR=/grader; {grader_cmd}", timeout=timeout_s + 15)
            except subprocess.TimeoutExpired:
                raw["timed_out"] = True
                return _empty_result(f"grader timed out after {timeout_s}s", raw)
            raw["grader_exit"] = g.returncode
            raw["grader_stdout_tail"] = _tail(g.stdout)
            raw["grader_stderr_tail"] = _tail(g.stderr)

            cat = _docker("exec", name, "sh", "-c", "cat /grader/reward.json 2>/dev/null || true")
            reward: dict[str, Any] | None = None
            if cat.stdout.strip():
                try:
                    reward = json.loads(cat.stdout)
                except json.JSONDecodeError:
                    reward = None
            return _result_from_grader(g.returncode, reward, raw)
        finally:
            _docker("rm", "-f", name)
