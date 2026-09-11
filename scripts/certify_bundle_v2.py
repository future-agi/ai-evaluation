#!/usr/bin/env python3
"""Exercise a real V2 bundle through authoring and the hosted process runtime.

This intentionally stops before simulation.  It is the fast, deterministic certification gate
for source inspection, build, start, readiness, reset, isolation and cleanup.  Provider secrets
are read from this process's environment and are written only to the runtime's one-shot secrets
file; neither values nor resolved endpoints are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from fi.alk.harness.bundle_author_v2 import author_bundle_v2
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_runtime import ProcessRuntimeProvider, RuntimeState


_SECRET_NAMES = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "OPENAI_API_KEY",
)
_SOURCE_EXCLUDES = {".git", ".venv", "__pycache__", "node_modules"}


def _job(args: argparse.Namespace, secret_names: list[str]) -> HarnessJob:
    return HarnessJob.model_validate(
        {
            "job_id": f"certify-{args.name}",
            "run_id": f"certify-{args.name}",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": args.name},
            "agent": {
                "connector": args.connector,
                "secret_refs": {
                    name: {
                        "manager": "platform-vault",
                        "key": name,
                        "purpose": "target_provider",
                    }
                    for name in secret_names
                },
            },
            "scenario_count": args.instances,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": max(2, args.instances * 2),
                "memory_mb": max(4096, args.instances * 2048),
                "parallelism": args.instances,
            },
            "metadata": {"name": args.name},
        }
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    submitted_source = args.source.resolve()
    authoring = args.authoring.resolve()
    work = args.work.resolve()
    if work.exists():
        for child in work.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        work.mkdir(parents=True)
    source = work / "source"
    shutil.copytree(
        submitted_source,
        source,
        ignore=shutil.ignore_patterns(*sorted(_SOURCE_EXCLUDES)),
    )
    bundle_dir = work / "bundle"
    secret_values = {
        name: os.environ[name] for name in _SECRET_NAMES if os.environ.get(name)
    }
    job = _job(args, list(secret_values))
    bundle = author_bundle_v2(
        source=source,
        authoring=authoring,
        output=bundle_dir,
        job=job,
    )
    secrets_path = work / "secrets.json"
    secrets_path.write_text(json.dumps(secret_values), encoding="utf-8")
    provider = ProcessRuntimeProvider(
        secrets_path=secrets_path,
        secret_purpose_map={name: "target_provider" for name in secret_values},
        require_declared_user=False,
        user_resolver=lambda _name: None,
    )
    runtimes = []
    try:
        runtimes = await provider.provision(
            bundle,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=work,
            instances=args.instances,
        )
        initial = [runtime.state.value for runtime in runtimes]
        healthy_before = [
            await provider.healthy(runtime, work_directory=work) for runtime in runtimes
        ]
        for runtime in runtimes:
            await provider.reset(runtime, work_directory=work)
        healthy_after = [
            await provider.healthy(runtime, work_directory=work) for runtime in runtimes
        ]
        if not all(healthy_before) or not all(healthy_after):
            raise RuntimeError(
                f"runtime_not_healthy: before={healthy_before}, after={healthy_after}"
            )
        if not all(runtime.state is RuntimeState.READY for runtime in runtimes):
            raise RuntimeError("runtime_not_ready_after_reset")
        return {
            "name": args.name,
            "packaging": bundle.metadata.get("packaging"),
            "bundle_digest": bundle.digest,
            "instances": len(runtimes),
            "initial_states": initial,
            "healthy_before_reset": healthy_before,
            "healthy_after_reset": healthy_after,
            "secrets_file_consumed": not secrets_path.exists(),
            "status": "passed",
        }
    finally:
        await provider.close(work_directory=work)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--connector", choices=("http", "livekit"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
