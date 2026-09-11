"""Bench sandboxes — the two code-exec lanes that run candidate code for you.

The unified bench harness scores coding suites by *actually executing* candidate
code against a held-out oracle. Where that execution happens is the ``sandbox``
dimension:

* ``subprocess`` (default) — a fresh interpreter in a throwaway tempdir with a
  scrubbed environment and a hard timeout. Credential-free, works anywhere, and
  is the lane the release gate uses on trusted shipped code. NOT a security
  boundary against deliberately hostile code.
* ``docker`` (opt-in) — per-task, ephemeral, OS-level isolation for UNTRUSTED
  agent output: ``--network none``, ``--cap-drop ALL``, ``no-new-privileges``,
  a read-only rootfs, a nosuid size-capped tmpfs, a non-root (65534) user, and
  PID / memory / CPU caps. A real Docker run of untrusted code is a genuine LIVE
  event, so its rows are stamped ``evidence_class=live_lane`` (never a fixture).

This example proves the convenience lane end to end and *reports* the hardened
lane without launching a container:

  (a) runs the ``subprocess`` lane on the shipped ``coding_starter`` suite (real
      execution, deterministic, credential-free);
  (b) reports ``bench._docker.docker_available()`` (cheap probe, no image pull);
  (c) captures the exact hardened ``docker run`` argv via the PURE
      ``bench._docker._build_docker_argv(...)`` so the artifact shows the
      isolation flags — with no daemon and no container ever started.

Run it::

    python examples/bench_sandboxes.py artifacts/bench-sandboxes.json

Docker is never a requirement here: the artifact records the argv a Docker run
*would* use plus whether a daemon is reachable, so the page stays offline. To
score real untrusted output under isolation, pass ``sandbox="docker"`` to
``bench.run_bench`` on a host with Docker (see ``docs/eval/benchmark-sandboxes.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench
from fi.alk.bench import _docker

SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_starter.json"
OUTPUT_KIND = "agent-learning.bench-sandboxes.v1"

# The flags we assert the hardened argv carries. These are the OS-level isolation
# controls the subprocess lane cannot give; reported so the artifact is auditable.
_HARDENING_FLAGS = (
    ("--network", "none"),
    ("--cap-drop", "ALL"),
    ("--security-opt", "no-new-privileges"),
    ("--read-only", None),
    ("--user", "65534:65534"),
    ("--rm", None),
)


def _flag_present(argv: list[str], flag: str, value: str | None) -> bool:
    """True if ``flag`` appears in ``argv`` (followed by ``value`` when given)."""

    if value is None:
        return flag in argv
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv) and argv[i + 1] == value:
            return True
    return False


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    # (a) The convenience lane, for real: score the shipped suite's gold
    # references in a scrubbed subprocess. local_gate = trusted, shipped code.
    # emit_telemetry=False keeps the docs/release fresh-lane silent (no ledger).
    suite = bench.load_coding_suite(SUITE_PATH)
    submission = bench.reference_submission(suite)
    subprocess_result = bench.run_bench(
        SUITE_PATH,
        control_mode="artifact_in",
        submission=submission,
        sandbox="subprocess",
        evidence_class="local_gate",
        emit_telemetry=False,
    )

    # (b) Report whether a Docker daemon is reachable — a cheap probe, no pull,
    # no container. The page stays valid whether or not Docker is installed.
    docker_ready = _docker.docker_available()

    # (c) Capture the EXACT hardened argv the Docker lane would launch, using the
    # PURE builder (no daemon contacted, no container started). The bootstrap is a
    # placeholder string here — we are inspecting the isolation flags, not running.
    hardened_argv = _docker._build_docker_argv(
        name="agent-learn-bench-example",
        image=_docker.DEFAULT_IMAGE,
        memory=_docker._DEFAULT_MEMORY,
        cpus=_docker._DEFAULT_CPUS,
        bootstrap="<materialised at run time; not executed here>",
    )
    isolation = {
        f"{flag}{(' ' + value) if value else ''}": _flag_present(hardened_argv, flag, value)
        for flag, value in _HARDENING_FLAGS
    }

    agg = subprocess_result["aggregate"]
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "suite_name": subprocess_result["dataset_name"],
        "suite_version": subprocess_result["dataset_version"],
        "subprocess_lane": {
            "sandbox": "subprocess",
            # The subprocess lane on trusted shipped code: a local gate event.
            "evidence_class": "local_gate",
            "aggregate": agg,
            "per_task": subprocess_result["per_task"],
        },
        "docker_lane": {
            "sandbox": "docker",
            # A real Docker run of untrusted code is a genuine live event.
            "evidence_class": "live_lane",
            "docker_available": docker_ready,
            "launched_container": False,
            "image": _docker.DEFAULT_IMAGE,
            "memory": _docker._DEFAULT_MEMORY,
            "cpus": _docker._DEFAULT_CPUS,
            "pids_limit": _docker._DEFAULT_PIDS,
            # The pure builder's output: the exact argv, plus a flag-by-flag audit.
            "hardened_argv": hardened_argv,
            "isolation_flags": isolation,
            "all_hardening_present": all(isolation.values()),
        },
    }

    if output_path is not None:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    # Never print inside run(): the docs fresh-lane exec-loads this module and the
    # release-check asserts empty stdout. Printing is __main__-only.
    return payload


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    destination = args[0] if args else None
    print(json.dumps(run(destination), indent=2, sort_keys=True, default=str))
