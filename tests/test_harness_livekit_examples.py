"""Opt-in conformance tests against unchanged official LiveKit example agents.

Set ``LIVEKIT_EXAMPLES_ROOT`` to the ``examples`` directory of
https://github.com/livekit/agents. These tests deliberately use the upstream Dockerfiles and
source trees directly; a passing result must never depend on copying or adapting agent code.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fi.alk.harness.bundle import RuntimeKind, export_session_bundle, load_bundle
from fi.alk.harness.contract import AgentContract, Runtime, ToolSpec
from fi.alk.harness.credentials import discover_credentials
from fi.alk.harness.provision import (
    healthy,
    provision,
    source_fingerprint,
    start_runtime,
    stop,
)


EXAMPLES_ROOT = Path(os.environ.get("LIVEKIT_EXAMPLES_ROOT", "__not_configured__"))
CASES = ("drive_thru", "frontdesk", "hotel_receptionist")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1" or not EXAMPLES_ROOT.is_dir(),
    reason="requires Docker and LIVEKIT_EXAMPLES_ROOT",
)
@pytest.mark.parametrize("example", CASES)
def test_official_livekit_dockerfile_only_agent_builds_unchanged(tmp_path, example):
    source = (EXAMPLES_ROOT / example).resolve()
    assert (source / "Dockerfile").is_file()
    before = source_fingerprint(source)
    contract = AgentContract(
        agent=f"livekit-{example}",
        modality="voice",
        tools=[ToolSpec(name="upstream_agent_tools")],
        real_use_cases=["run the upstream voice agent"],
        runtime=Runtime(dockerfile="Dockerfile"),
    )
    destination = tmp_path / example

    try:
        environment = provision(source, destination, contract)
        assert environment.managed
        assert environment.services == []
        assert environment.runtime_services == ["agent-runtime"]
        assert environment.running and healthy(destination)
        bundle_root, bundle = export_session_bundle(
            source, destination, name=f"livekit-{example}"
        )
        assert bundle.runtime.kind is RuntimeKind.COMPOSE
        assert bundle.runtime.document == "compose.json"
        assert bundle.services == []
        assert load_bundle(bundle_root).digest == bundle.digest
        assert source_fingerprint(source) == before
    finally:
        stop(destination)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LIVEKIT_EXAMPLES_START_WORKERS") != "1"
    or not EXAMPLES_ROOT.is_dir()
    or not all(
        os.environ.get(name)
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    ),
    reason="requires Docker, the examples checkout, and LiveKit worker credentials",
)
@pytest.mark.parametrize("example", CASES)
def test_official_livekit_agent_worker_starts_with_discovered_credentials(
    tmp_path, example
):
    source = (EXAMPLES_ROOT / example).resolve()
    before = source_fingerprint(source)
    supplied = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    credentials = discover_credentials(source, provided_environment=supplied)
    assert credentials.ready
    contract = AgentContract(
        agent=f"livekit-{example}",
        modality="voice",
        tools=[ToolSpec(name="upstream_agent_tools")],
        real_use_cases=["run the upstream voice agent"],
        runtime=Runtime(dockerfile="Dockerfile"),
    )
    destination = tmp_path / f"{example}-worker"
    overrides = {name: os.environ[name] for name in supplied}

    try:
        provision(source, destination, contract)
        environment = start_runtime(destination, overrides=overrides)
        assert environment.runtime_container
        assert healthy(destination)
        assert source_fingerprint(source) == before
    finally:
        stop(destination)
