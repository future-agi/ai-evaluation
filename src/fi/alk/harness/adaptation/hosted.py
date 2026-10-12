"""Adaptation inside the hosted sandbox (`environment-releases.md` §7.5, Launch).

Credentials arrive the way they do for hosted authoring: the target's read-only provider key in
`/run/futureagi/authoring-target-secrets.json` or `secrets.json`, the harness model's Vertex
credentials through the platform simulator channel. The provider key is used for the one
read-only inspection and never placed in the process environment. The conversation channel is
the only way questions are answered; there is no coordinator.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from ..hosted_authoring_entrypoint import (
    _SECRETS_PATH,
    _TARGET_CREDENTIAL_NAMES,
    _TARGET_SECRETS_PATH,
    _configure_generation_environment,
    _load_values,
    _platform_simulator_values,
)
from ..hosted_entrypoint import SIMULATOR_SECRETS_PATH, load_simulator_secret_values
from ..releases import RELEASES_SUITE, environment_directory
from .askers import ConversationAsker
from .errors import AdaptationFailure


def _values(path: Path) -> dict[str, str]:
    try:
        return _load_values(path)
    except (OSError, ValueError):
        return {}


def _target_credentials() -> dict[str, str]:
    values = {**_values(_SECRETS_PATH), **_values(_TARGET_SECRETS_PATH)}
    _TARGET_SECRETS_PATH.unlink(missing_ok=True)
    return {name: values[name] for name in _TARGET_CREDENTIAL_NAMES if values.get(name)}


def _configure_harness_model() -> None:
    os.environ.update(load_simulator_secret_values(SIMULATOR_SECRETS_PATH))
    _configure_generation_environment(
        _platform_simulator_values(_values(_SECRETS_PATH))
    )


def _environments(job: Any, releases: Path) -> tuple[list[Path], Path]:
    adaptation = (job.metadata or {}).get("adaptation")
    if not isinstance(adaptation, dict):
        raise AdaptationFailure(
            "adaptation_job_invalid", "job metadata.adaptation is missing"
        )
    chain = adaptation.get("environment")
    suite = adaptation.get("suite")
    if not isinstance(chain, list) or not chain or not isinstance(suite, dict):
        raise AdaptationFailure(
            "adaptation_job_invalid",
            "metadata.adaptation needs an environment chain and a suite",
        )
    directories = [
        environment_directory(releases, str(ref.get("digest") or "")) for ref in chain
    ]
    suite_root = releases / RELEASES_SUITE
    from ..releases import load_suite_release

    delivered = load_suite_release(suite_root).digest
    if delivered != suite.get("digest"):
        raise AdaptationFailure(
            "release_invalid",
            f"delivered suite {delivered} is not the job's {suite.get('digest')}",
        )
    return directories, suite_root


def run_hosted(args: argparse.Namespace) -> int:
    from ..hosted_conversation import (
        DEFAULT_CAPABILITIES_PATH,
        ConversationClient,
        ConversationTransportError,
        load_conversation_capabilities,
    )
    from ..job import HarnessJob
    from ..releases import ReleaseError
    from . import __main__ as cli
    from .pipeline import AdaptationInputs
    from .profile import profile_from_job

    for name in _TARGET_CREDENTIAL_NAMES:
        os.environ.pop(name, None)
    credentials = _target_credentials()
    _configure_harness_model()
    output = Path(args.output)
    try:
        job = HarnessJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdaptationFailure("adaptation_job_invalid", str(exc)) from exc
    try:
        environments, suite_root = _environments(job, Path(args.releases))
    except ReleaseError as exc:
        raise AdaptationFailure("release_invalid", f"{exc.code}: {exc.detail}") from exc
    try:
        capabilities = load_conversation_capabilities(
            args.capabilities or DEFAULT_CAPABILITIES_PATH
        )
    except ConversationTransportError as exc:
        raise AdaptationFailure("conversation_unavailable", str(exc)) from exc
    asker = ConversationAsker(ConversationClient(capabilities))

    def load_profile() -> Any:
        return profile_from_job(job, credentials=credentials)

    with contextlib.ExitStack() as stack:
        if args.postgres_dsn:
            dsn = args.postgres_dsn
        else:
            from .scratch_postgres import scratch_cluster

            try:
                dsn = stack.enter_context(scratch_cluster())
            except (OSError, RuntimeError, KeyError) as exc:
                raise AdaptationFailure("postgres_unavailable", str(exc)) from exc
        inputs = AdaptationInputs(
            environments=environments,
            suite_root=suite_root,
            output=output,
            postgres_dsn=dsn,
            load_profile=load_profile,
            drafter_for=cli.drafter_factory(None),
        )
        try:
            code = asyncio.run(cli.adapt(inputs, asker))
        except ConversationTransportError as exc:
            # The channel is how answers arrive; without it the job cannot finish.
            raise AdaptationFailure("conversation_unavailable", str(exc)) from exc
    print(json.dumps({"status": "completed" if code == 0 else "failed", "exit": code}))
    return code
