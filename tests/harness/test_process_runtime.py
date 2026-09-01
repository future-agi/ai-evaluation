"""The execution half of the provisioner (`process_runtime.py`), per `hosted-execution-seams.md`
v1.12 §2b/§3/§4/§5. Manifests here are built directly through `EnvironmentBundleV2.model_validate` —
no digest sealing, no on-disk bundle — since every rule under test needs nothing but the parsed
model's own field values plus the job-supplied inputs (instances, secrets, credentials). Digest
and file-content verification are `test_process_preflight.py`'s job, not this module's.

No Docker, no real postgres/redis/rabbitmq anywhere: managed-engine spawn is exercised only
through fakes (`ProcessRunner`, `sync_run`), since §0 assumes those binaries are on the snapshot's
PATH and this test lane must not require that. Real subprocess spawn IS exercised, through tiny
`python3 -c ...` stand-in scripts — proving `default_process_runner` genuinely spawns, captures
output, and reports liveness, without any engine-specific plumbing.

P5's fix pass (F1/F2/F3/F7) made real filesystem operations security-relevant, so those are
exercised directly against real symlinks/tmp dirs where a fake would hide the exact class of bug
being fixed. `os.chown`/`Popen(user=...)` to a FOREIGN uid needs root, which this lane does not
have and must not require — those paths are verified STRUCTURALLY: a fake `chown`/`user_resolver`
is injected and the call it WOULD make is asserted, rather than performing the real privileged
syscall.

Phase 6 (seed/baseline/worlds/reset/conformance/provision/close) adds `SqlSpy` below: a fake
`SqlRunner` that records every `(dbname, statement)` call in order and simulates just enough
postgres semantics (CREATE/DROP/ALTER DATABASE, `to_regclass`, row counts) for the baseline-freeze
and reset state machines to run against — no real postgres anywhere, same rule as everything
above. `asyncio.run` drives every `async def` seam (`ProcessRuntimeProvider.provision`/`reset`/
`close`) directly; there is no event loop fixture here, matching this repo's existing async test
style elsewhere in the harness suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import types
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Callable

import pytest

from fi.alk.harness.bundle import CapabilityProtocol
from fi.alk.harness.bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    CapabilityV2,
    EnvironmentBundleV2,
    ProcessUser,
    SecretPurpose,
    SourceProcess,
)
from fi.alk.harness import process_runtime as pr
from fi.alk.harness.job import FailureDomain
from fi.alk.harness.provider_lifecycle import (
    ProviderCleanupReceipt,
    ProviderProvisionReceipt,
    ProviderResource,
    ProviderTarget,
)

# --- shared manifest builder -----------------------------------------------------------------
#
# One postgres store (job-shared, `template_database`), one `tools-api` http-capability source
# process, one `agent` control-service source process claiming the job's only `target_provider`
# secret — the same shape `test_process_preflight.py` uses, so port/ordinal numbers a reader
# already knows from that file carry over here.


def _body(
    mutate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": BUNDLE_V2_SCHEMA_VERSION,
        "name": "demo",
        "digest": "sha256:" + "0" * 64,
        "runtime": {
            "kind": "process",
            "control_service": "agent",
            "evidence_seam": "http_tool",
        },
        "processes": [
            {
                "name": "postgres",
                "kind": "managed",
                "engine": "postgres",
                "version": "16",
                "user": "svc-data",
                "depends_on": [],
            },
            {
                "name": "tools-api",
                "kind": "source",
                "working_directory": "services/tools-api",
                "build_commands": [["npm", "ci"]],
                "run_command": ["node", "server.js"],
                "environment": {
                    "DATABASE_URL": "{{DATABASE_URL}}",
                    "PORT": "{{PORT_tools-api}}",
                    "TMPDIR": "{{WORLD_DIR}}",
                },
                "secret_purposes": [],
                "user": "svc-tools",
                "depends_on": ["postgres"],
            },
            {
                "name": "agent",
                "kind": "source",
                "working_directory": ".",
                "build_commands": [["pip", "install", "-r", "requirements.txt"]],
                "run_command": ["python3", "agent.py"],
                "environment": {
                    "DATABASE_URL": "{{DATABASE_URL}}",
                    "TOOLS_API_URL": "{{TOOLS_API_URL}}",
                    "NAME": "agent-w{{WORLD_INDEX}}",
                },
                "secret_purposes": ["target_provider"],
                "user": "svc-agent",
                "depends_on": ["postgres", "tools-api"],
            },
        ],
        "capabilities": {
            "database": {
                "protocol": "postgres",
                "service": "postgres",
                "configuration_name": "DATABASE_URL",
            },
            "tools": {
                "protocol": "http",
                "service": "tools-api",
                "configuration_name": "TOOLS_API_URL",
            },
        },
        "readiness": [
            {
                "capability": "tools",
                "path": "/health",
                "timeout_seconds": 5,
                "interval_seconds": 0.1,
            },
        ],
        "seed": {
            "stores": [
                {
                    "capability": "database",
                    "migrations": [],
                    "seed_files": [],
                    "baseline": {
                        "strategy": "template_database",
                        "inputs_digest": "sha256:" + "a" * 64,
                    },
                    "sentinel": {"query": "SELECT 1", "expected": "1"},
                }
            ]
        },
        "provenance": {
            "source_kind": "repository",
            "repository": "org/repo",
            "source_digest": "c" * 64,
        },
        "metadata": {},
    }
    return mutate(body) if mutate is not None else body


def _manifest(
    mutate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> EnvironmentBundleV2:
    return EnvironmentBundleV2.model_validate(_body(mutate))


def _source_process(**overrides: Any) -> SourceProcess:
    fields: dict[str, Any] = {
        "name": "svc",
        "kind": "source",
        "working_directory": ".",
        "build_commands": [],
        "run_command": ["python3", "-c", "pass"],
        "environment": {},
        "secret_purposes": [],
        "user": ProcessUser.SVC_TOOLS,
        "depends_on": [],
    }
    fields.update(overrides)
    return SourceProcess(**fields)


def _solo_port_plan(process_name: str, *, ordinal: int = 0) -> pr.PortPlan:
    """A single-process `PortPlan` for tests that spawn a `_source_process()` standalone, off the
    shared `_manifest()` topology — `spawn_source_process` always looks its own process up in the
    plan it is given, so the plan must actually know that process's name."""
    return pr.PortPlan(
        ordinals={process_name: ordinal},
        job_shared=frozenset(),
        fixed_ports={},
        effective_instances=1,
        degraded_reason=None,
    )


class FakeHandle:
    """A `SpawnedProcess` fake: no real subprocess, just a captured-output buffer. `wait()`
    reports the fake as exited the instant `terminate()`/`interrupt()`/`kill()` has been called —
    no real process to actually wait on — which is enough for `_terminate_and_wait` (M7) to take
    its happy path without ever escalating to `kill()` in a test. `StubbornHandle` (below,
    N13-specific) is the deliberately-does-not-cooperate counterpart used to reach the kill branch."""

    def __init__(self, output: str = "") -> None:
        self._output = output
        self.terminated = False
        self.interrupted = False
        self.killed = False

    def is_running(self) -> bool:
        return not self.terminated

    def captured_output(self) -> str:
        return self._output

    def terminate(self) -> None:
        self.terminated = True

    def interrupt(self) -> None:
        # N12, p6-review-r2: same instant-exit fake shape as `terminate()` — this fake exists to
        # prove WHICH signal a caller preferred (`self.interrupted`), not to model postgres's own
        # shutdown semantics.
        self.interrupted = True
        self.terminated = True

    def wait(self, timeout: float) -> bool:
        return self.terminated

    def kill(self) -> None:
        self.killed = True
        self.terminated = True


class _FakePasswd:
    """Stands in for `pwd.struct_passwd` — only `pw_uid`/`pw_gid` are ever read by this module."""

    def __init__(self, uid: int, gid: int) -> None:
        self.pw_uid = uid
        self.pw_gid = gid


def _fake_user_resolver(
    known: dict[str, tuple[int, int]],
) -> Callable[[str], _FakePasswd | None]:
    def resolver(username: str) -> _FakePasswd | None:
        if username in known:
            uid, gid = known[username]
            return _FakePasswd(uid, gid)
        return None

    return resolver


def _reap(handle: Any, *, attempts: int = 50, interval: float = 0.05) -> None:
    """Waits for a real spawned subprocess to exit. `pytest.fail`s loudly on exhaustion rather
    than falling through to an assertion that would blame the wrong thing — a hung reap here means
    the child never exited, not that its output was wrong."""
    for _ in range(attempts):
        if not handle.is_running():
            return
        time.sleep(interval)
    else:
        pytest.fail(f"subprocess did not exit within {attempts * interval}s")


# --- §2b port allocation ------------------------------------------------------------------------


def test_ordinals_follow_the_authored_processes_array_order() -> None:
    plan = pr.plan_ports(_manifest(), instances=3)
    assert plan.ordinals == {"postgres": 0, "tools-api": 1, "agent": 2}


def test_a_template_database_managed_engine_is_job_shared() -> None:
    """§2b: `template_database` -> once per job; `14000 + ordinal`, the same in every world."""
    plan = pr.plan_ports(_manifest(), instances=3)
    assert plan.is_job_shared("postgres")
    assert plan.port_for("postgres", 0) == 14000
    assert plan.port_for("postgres", 2) == 14000


def test_source_processes_use_the_per_world_stride_formula() -> None:
    """§2b: `15000 + 100*world_index + ordinal`."""
    plan = pr.plan_ports(_manifest(), instances=3)
    assert plan.port_for("tools-api", 0) == 15001
    assert plan.port_for("tools-api", 1) == 15101
    assert plan.port_for("agent", 2) == 15202


def test_a_datadir_copy_managed_engine_is_per_world_not_job_shared() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "a" * 64,
                        },
                    },
                ]
            },
        }
    )
    plan = pr.plan_ports(manifest, instances=2)
    assert not plan.is_job_shared("postgres")
    assert plan.port_for("postgres", 0) == 15000
    assert plan.port_for("postgres", 1) == 15100


def test_a_managed_engine_with_no_seed_entry_at_all_defaults_to_per_world() -> None:
    """§2b: "per-world is the only safe default" when a managed engine has no `seed.stores`
    entry — a shared engine one world reset could otherwise corrupt for the others.

    T4, p5-round1-review: a REDIS capability with no store entry (§2c: implicitly `empty`, safe
    with no `seed` block at all) rather than the old postgres one — a postgres capability with no
    store entry is preflight-*invalid* (`seed_missing`), so the old fixture proved this branch on
    a manifest that could never reach the provisioner for real. `postgres`'s own `seed.stores`
    entry (`template_database`) is left in place, unrelated to the branch under test.
    """
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                *body["processes"],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": None,
                },
            },
        }
    )
    plan = pr.plan_ports(manifest, instances=2)
    assert not plan.is_job_shared("cache")
    assert plan.port_for("cache", 0) != plan.port_for("cache", 1)


def test_the_empty_baseline_strategy_managed_engine_is_per_world() -> None:
    """T5, p5-round1-review: `empty` is one of §2b's four instancing branches
    (`template_database` job-shared; `datadir_copy`, `empty`, and no-seed-entry-at-all all
    per-world) and had no direct test — only `datadir_copy` and the no-entry case were covered.
    Redis is the only catalog engine whose strategies include `empty`
    (`bundle_v2._ENGINE_STRATEGIES`)."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                *body["processes"],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": [],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "_seeded", "expected": "1"},
                    },
                ]
            },
        }
    )
    plan = pr.plan_ports(manifest, instances=2)
    assert not plan.is_job_shared("cache")
    assert (
        plan.port_for("cache", 0) == 15003
    )  # ordinal 3 (postgres,tools-api,agent,cache), world 0
    assert plan.port_for("cache", 1) == 15103  # ordinal 3, world 1


def test_fixed_port_is_honored_exactly_and_forces_effective_instances_to_one() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {**body["processes"][1], "fixed_port": 8081},
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=5)
    assert plan.effective_instances == 1
    assert plan.degraded_reason == "fixed_port"
    # Honored exactly, in every world index — there is only ever world 0 once degraded, but the
    # allocator itself must not silently fall back to the formula for any index it is asked for.
    assert plan.port_for("tools-api", 0) == 8081
    assert plan.port_for("tools-api", 3) == 8081


def test_no_fixed_port_leaves_parallelism_undegraded() -> None:
    plan = pr.plan_ports(_manifest(), instances=5)
    assert plan.effective_instances == 5
    assert plan.degraded_reason is None


def test_port_band_boundary_ordinal_99_world_7_equals_15799() -> None:
    """T6, p5-round1-review: the tiling is exact with zero slack (ordinal in [0,99], world in
    [0,7], per §2e item 7's caps) — nothing exercised the actual boundary before."""
    plan = pr.PortPlan(
        ordinals={"last": 99},
        job_shared=frozenset(),
        fixed_ports={},
        effective_instances=8,
        degraded_reason=None,
    )
    assert plan.port_for("last", 7) == 15799


def test_job_shared_and_per_world_port_bands_are_disjoint() -> None:
    """T6: 900 ports of slack between [14000,14099] (job-shared) and [15000,15799] (per-world) —
    no ordinal/world combination within the contract's own caps can alias across bands."""
    plan = pr.PortPlan(
        ordinals={"shared": 99, "world": 99},
        job_shared=frozenset({"shared"}),
        fixed_ports={},
        effective_instances=8,
        degraded_reason=None,
    )
    job_shared_ports = {plan.port_for("shared", w) for w in range(8)}
    per_world_ports = {plan.port_for("world", w) for w in range(8)}
    assert job_shared_ports == {14099}  # ordinal 99, the same port in every world
    assert per_world_ports == {15099, 15199, 15299, 15399, 15499, 15599, 15699, 15799}
    assert job_shared_ports.isdisjoint(per_world_ports)


# --- §2b placeholder renderer --------------------------------------------------------------------


def test_configuration_name_placeholder_renders_the_capabilitys_address(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=2)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    endpoints = pr.build_endpoints(
        manifest, world_index=1, port_plan=plan, credentials=credentials
    )
    addresses = pr.configuration_addresses_from_endpoints(endpoints)
    agent = manifest.processes[2]
    rendered = pr.render_environment(
        agent,
        world_index=1,
        world_dir=tmp_path / "worlds" / "w1" / "agent",
        port_plan=plan,
        configuration_addresses=addresses,
    )
    assert rendered["DATABASE_URL"] == "postgresql://harness:PW@localhost:14000/w1"
    assert rendered["TOOLS_API_URL"] == "http://localhost:15101"
    assert rendered["NAME"] == "agent-w1"


def test_world_dir_db_name_host_and_port_placeholders(tmp_path: Path) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    **body["processes"][1],
                    "environment": {
                        **body["processes"][1]["environment"],
                        "SCRATCH": "{{WORLD_DIR}}",
                        "DB": "{{DB_NAME}}",
                        "PEER": "{{HOST_postgres}}:{{PORT_postgres}}",
                    },
                },
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    endpoints = pr.build_endpoints(
        manifest, world_index=0, port_plan=plan, credentials=credentials
    )
    addresses = pr.configuration_addresses_from_endpoints(endpoints)
    world_dir = tmp_path / "worlds" / "w0" / "tools-api"
    tools_api = manifest.processes[1]
    rendered = pr.render_environment(
        tools_api,
        world_index=0,
        world_dir=world_dir,
        port_plan=plan,
        configuration_addresses=addresses,
    )
    assert rendered["SCRATCH"] == str(world_dir)
    assert rendered["DB"] == "w0"
    assert rendered["PEER"] == "localhost:14000"


def test_world_index_placeholder_renders_the_bare_integer() -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=1)
    rendered = pr.render_template(
        "world-{{WORLD_INDEX}}",
        process_name="agent",
        world_index=3,
        world_dir=Path("/x"),
        port_plan=plan,
        configuration_addresses={},
    )
    assert rendered == "world-3"


def test_job_id_placeholder_scopes_external_dispatch_identity() -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=1)
    rendered = pr.render_template(
        "agent-{{JOB_ID}}-w{{WORLD_INDEX}}",
        process_name="agent",
        job_id="job-123",
        world_index=2,
        world_dir=Path("/x"),
        port_plan=plan,
        configuration_addresses={},
    )
    assert rendered == "agent-job-123-w2"


def test_an_unresolvable_token_is_an_internal_error_not_a_preflight_code() -> None:
    """`preflight_bundle` already validated every token against the closed vocabulary before this
    ever runs — a token this function cannot resolve is a bug, not a bundle defect, so it is NOT
    one of `PreflightError`'s §2e codes."""
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=1)
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.render_template(
            "{{NOT_A_REAL_TOKEN}}",
            process_name="agent",
            world_index=0,
            world_dir=Path("/x"),
            port_plan=plan,
            configuration_addresses={},
        )
    assert excinfo.value.code == "internal_unknown_placeholder"
    assert excinfo.value.stage == "render"


def test_a_port_placeholder_naming_an_unknown_process_is_also_an_internal_error() -> (
    None
):
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=1)
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.render_template(
            "{{PORT_ghost}}",
            process_name="agent",
            world_index=0,
            world_dir=Path("/x"),
            port_plan=plan,
            configuration_addresses={},
        )
    assert excinfo.value.code == "internal_unknown_placeholder"


def test_build_environment_is_merged_raw_and_never_templated(tmp_path: Path) -> None:
    """§2b: `build_environment` "takes NO placeholders at all" — proven end to end through
    `spawn_source_process`'s env, since `render_environment`/`render_template` are never even
    called on it."""
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process(build_environment={"TMPDIR": "{{WORLD_DIR}}"})
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["env"] = env
        return FakeHandle()

    plan = _solo_port_plan("svc")
    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert captured["env"]["TMPDIR"] == "{{WORLD_DIR}}"


def test_render_capability_address_raises_for_an_unsupported_protocol() -> None:
    """F10, p5-round1-review: a `mongodb`/`s3`/`kafka`/... capability has no defined address
    shape at this seam (§3 names exactly two worked examples) — this used to silently render a
    bare `<scheme>://host:port`, which is not a working address for any of them."""
    capability = CapabilityV2(
        protocol="mongodb", service="svc", configuration_name="MONGO_URL"
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.render_capability_address(
            capability, port=15005, world_index=0, credentials=None
        )
    assert excinfo.value.code == "unsupported_capability_protocol"


def test_render_capability_address_raises_for_missing_postgres_credentials() -> None:
    """S11, p5-round1-review: a typed raise for this precondition, not a bare `assert` (stripped
    under `python -O`) — the same defect class as P4's N8(2)."""
    capability = CapabilityV2(
        protocol="postgres", service="postgres", configuration_name="DATABASE_URL"
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.render_capability_address(
            capability, port=14000, world_index=0, credentials=None
        )
    assert excinfo.value.code == "internal_missing_credentials"


# --- secrets: purpose filtering (F13) -----------------------------------------------------------


def test_select_process_secrets_matches_only_the_claimed_purpose() -> None:
    process = _source_process(secret_purposes=["target_provider"])
    selected = pr.select_process_secrets(
        process,
        secret_values={"A": "1", "B": "2"},
        secret_purposes={"A": "target_provider", "B": "source_checkout"},
    )
    assert selected == {"A": "1"}


def test_spawn_source_process_keeps_capability_endpoint_over_pasted_env_secret(
    tmp_path: Path,
) -> None:
    """A customer .env value must not replace Dockerless process-runtime wiring."""
    process = _source_process(secret_purposes=["target_provider"])
    process = process.model_copy(
        update={"environment": {"TOOLS_API_URL": "{{TOOLS_API_URL}}"}}
    )
    captured: dict[str, str] = {}

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        del argv, cwd, log_path, user, group
        captured.update(env)
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=tmp_path / "build",
        world_dir=tmp_path / "world",
        world_index=0,
        port_plan=_solo_port_plan(process.name),
        configuration_addresses={"TOOLS_API_URL": "http://localhost:15101"},
        secret_values={"TOOLS_API_URL": "http://harness:8787"},
        secret_purposes={"TOOLS_API_URL": "target_provider"},
        runner=runner,
    )

    assert captured["TOOLS_API_URL"] == "http://localhost:15101"


def test_select_process_secrets_hard_excludes_source_checkout_even_if_claimed() -> None:
    """F13, p5-round1-review: §1 states `source_checkout` is gateway-only and never uploaded to
    the guest — preflight's `secret_unclaimed`/`secret_missing` pair is scoped to
    `target_provider` only, so a process CAN legally claim `source_checkout` too. This function
    must not depend on the gateway alone to keep that promise."""
    process = _source_process(secret_purposes=["target_provider", "source_checkout"])
    selected = pr.select_process_secrets(
        process,
        secret_values={
            "A": "target-provider-value",
            "B": "checkout-value-must-not-move",
        },
        secret_purposes={"A": "target_provider", "B": "source_checkout"},
    )
    assert selected == {"A": "target-provider-value"}


def test_select_process_secrets_hard_excludes_simulator_provider_even_if_claimed() -> (
    None
):
    """Untrusted bundle processes must never receive Future AGI simulator credentials."""
    process = _source_process(secret_purposes=["target_provider", "simulator_provider"])
    selected = pr.select_process_secrets(
        process,
        secret_values={"AGENT_KEY": "customer", "SIM_KEY": "futureagi"},
        secret_purposes={
            "AGENT_KEY": "target_provider",
            "SIM_KEY": "simulator_provider",
        },
    )
    assert selected == {"AGENT_KEY": "customer"}


# --- env construction (F12/F14) -----------------------------------------------------------------


def test_allowlisted_ambient_env_keeps_only_the_fixed_set() -> None:
    source = {
        "PATH": "/bin",
        "HOME": "/root",
        "LANG": "C",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "LC_ALL": "C",
        "SECRET_TOKEN": "leak-me",
        "AWS_SECRET_ACCESS_KEY": "leak-me-too",
    }
    env = pr._allowlisted_ambient_env(source)
    assert env == {
        "PATH": "/bin",
        "HOME": "/root",
        "LANG": "C",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "LC_ALL": "C",
    }


def test_ambient_env_preserves_egress_trust_without_provider_secrets():
    trust = {
        name: "/etc/daytona/netleash/ca.crt"
        for name in (
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            "PIP_CERT",
            "NODE_EXTRA_CA_CERTS",
        )
    }
    assert (
        pr._allowlisted_ambient_env(
            {
                **trust,
                "GOOGLE_APPLICATION_CREDENTIALS": "/private/adc.json",
                "SIMULATOR_API_KEY": "private",
                "HTTPS_PROXY": "https://user:password@proxy",
            }
        )
        == trust
    )


def test_spawn_source_process_env_does_not_inherit_an_arbitrary_ambient_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F12, p5-round1-review: §2b enumerates exactly what a process receives — the ambient
    `svc-control` environment (a future bearer token, a `FUTUREAGI_*` marker) is not on that
    list."""
    monkeypatch.setenv("FUTUREAGI_PROVISIONER_MARKER", "leak-me")
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["env"] = env
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert "FUTUREAGI_PROVISIONER_MARKER" not in captured["env"]


def test_base_process_env_path_prepend_has_no_empty_trailing_element(
    tmp_path: Path,
) -> None:
    """F14, p5-round1-review: an unset/empty inherited `PATH` used to leave a trailing `:`, and
    POSIX `execvp` reads an empty PATH element as "current directory" — cwd for build/run is the
    customer's own tree, so a repo shipping an executable literally named `ls`/`git`/`sh` could
    shadow the real one for any bare-name argv[0]."""
    env = pr._base_process_env(tmp_path / "build" / "svc", base={})
    assert "" not in env["PATH"].split(":")


# --- §2b copy-based build trees -------------------------------------------------------------------


def test_build_process_tree_copies_the_working_directory(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    (tmp_path / "source" / "svc" / "main.py").write_text("print(1)\n")
    (tmp_path / "source" / "svc" / "sub").mkdir()
    (tmp_path / "source" / "svc" / "sub" / "helper.py").write_text("print(2)\n")
    process = _source_process(working_directory="svc", build_commands=[])
    build_dir = pr.build_process_tree(
        process, source_root=tmp_path / "source", build_root=tmp_path / "build"
    )
    assert build_dir == tmp_path / "build" / "svc"
    assert (build_dir / "main.py").read_text() == "print(1)\n"
    assert (build_dir / "sub" / "helper.py").read_text() == "print(2)\n"


def test_build_commands_get_the_venv_and_node_modules_path_prepend(
    tmp_path: Path,
) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(step, *, cwd, env, **kwargs):
        calls.append(env["PATH"].split(":")[:2])
        return subprocess.CompletedProcess(step, 0)

    process = _source_process(
        working_directory="svc", build_commands=[["npm", "ci"], ["npm", "build"]]
    )
    build_dir = pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        run=fake_run,
    )
    expected = [
        str(build_dir / ".venv" / "bin"),
        str(build_dir / "node_modules" / ".bin"),
    ]
    assert len(calls) == 2  # every step, not just the first
    assert calls[0] == expected
    assert calls[1] == expected


def test_build_environment_is_merged_into_the_build_step_env(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_run(step, *, cwd, env, **kwargs):
        captured["FOO"] = env.get("FOO")
        return subprocess.CompletedProcess(step, 0)

    process = _source_process(
        working_directory="svc",
        build_commands=[["true"]],
        build_environment={"FOO": "bar"},
    )
    pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        run=fake_run,
    )
    assert captured["FOO"] == "bar"


def test_build_runs_every_step_once_per_call_in_order(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    order: list[list[str]] = []

    def fake_run(step, *, cwd, env, **kwargs):
        order.append(step)
        return subprocess.CompletedProcess(step, 0)

    process = _source_process(
        working_directory="svc", build_commands=[["step-a"], ["step-b"], ["step-c"]]
    )
    pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        run=fake_run,
    )
    assert order == [["step-a"], ["step-b"], ["step-c"]]


def test_build_process_trees_builds_each_source_process_exactly_once(
    tmp_path: Path,
) -> None:
    """No world loop exists inside `build_process_trees` at all — structural proof of "once per
    job," not "once per world.\""""
    (tmp_path / "source" / "services" / "tools-api").mkdir(parents=True)
    (tmp_path / "source" / ".").mkdir(exist_ok=True)
    manifest = _manifest()
    calls: list[str] = []

    def fake_run(step, *, cwd, env, **kwargs):
        calls.append(str(cwd))
        return subprocess.CompletedProcess(step, 0)

    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=pr.plan_ports(manifest, instances=4),
        credentials={},
        secret_values={},
        secret_purposes={},
        sync_run=fake_run,
    )
    build_dirs = pr.build_process_trees(
        manifest, source_root=tmp_path / "source", context=context
    )
    assert set(build_dirs) == {
        "tools-api",
        "agent",
    }  # every `source` process, no managed ones
    # tools-api has 1 build step, agent has 1 build step — exactly 2 calls total, never per-world.
    assert len(calls) == 2


def test_a_nonzero_build_step_is_build_failed(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)

    def fake_run(step, *, cwd, env, **kwargs):
        return subprocess.CompletedProcess(step, 1, stdout="", stderr="npm ERR! boom")

    process = _source_process(working_directory="svc", build_commands=[["npm", "ci"]])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            run=fake_run,
        )
    assert excinfo.value.stage == "build"
    assert excinfo.value.code == "build_failed"
    assert "boom" in str(excinfo.value)


def test_transient_package_network_failure_is_retried_in_place(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    calls: list[list[str]] = []
    delays: list[float] = []

    def fake_run(step, *, cwd, env, **kwargs):
        calls.append(step)
        if len(calls) < 3:
            return subprocess.CompletedProcess(
                step,
                2,
                stdout="",
                stderr="Failed to fetch package: client error (Connect): operation timed out",
            )
        return subprocess.CompletedProcess(step, 0, stdout="", stderr="")

    process = _source_process(
        working_directory="svc", build_commands=[["uv", "sync", "--no-cache"]]
    )
    pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        run=fake_run,
        sleep=delays.append,
    )

    assert calls == [
        ["uv", "sync", "--no-cache"],
        ["uv", "sync", "--no-cache"],
        ["uv", "sync", "--no-cache"],
    ]
    assert delays == [1.0, 2.0]


def test_deterministic_build_failure_is_not_retried(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    calls = 0

    def fake_run(step, *, cwd, env, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(step, 1, stdout="", stderr="syntax error")

    process = _source_process(working_directory="svc", build_commands=[["npm", "ci"]])
    with pytest.raises(pr.ProcessRuntimeError):
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            run=fake_run,
            sleep=lambda _: None,
        )

    assert calls == 1


def test_a_missing_interpreter_is_runtime_unsupported(tmp_path: Path) -> None:
    """§0 (v1.8): "a repo needing an interpreter the snapshot lacks fails at BUILD time... the
    build step's failure is reported `runtime_unsupported`.\""""
    (tmp_path / "source" / "svc").mkdir(parents=True)

    def fake_run(step, *, cwd, env, **kwargs):
        raise FileNotFoundError(f"no such file: {step[0]!r}")

    process = _source_process(
        working_directory="svc", build_commands=[["python3.13", "-m", "venv", ".venv"]]
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            run=fake_run,
        )
    assert excinfo.value.stage == "build"
    assert excinfo.value.code == "runtime_unsupported"


def test_a_missing_non_interpreter_build_tool_is_build_failed_not_runtime_unsupported(
    tmp_path: Path,
) -> None:
    """The missing-interpreter detection is scoped to python*/node* argv[0] patterns (§0 only
    promises those two families) — a missing custom build tool is a `build_failed`, not a
    `runtime_unsupported`, since the snapshot never promised it in the first place."""
    (tmp_path / "source" / "svc").mkdir(parents=True)

    def fake_run(step, *, cwd, env, **kwargs):
        raise FileNotFoundError(f"no such file: {step[0]!r}")

    process = _source_process(
        working_directory="svc", build_commands=[["some-custom-tool", "x"]]
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            run=fake_run,
        )
    assert excinfo.value.code == "build_failed"


def test_a_build_step_that_times_out_is_build_failed(tmp_path: Path) -> None:
    """F15, p5-round1-review: an install step wedged on a private registry with no DNS answer
    used to block the provisioner forever."""
    (tmp_path / "source" / "svc").mkdir(parents=True)

    def fake_run(step, *, cwd, env, **kwargs):
        raise subprocess.TimeoutExpired(cmd=step, timeout=kwargs.get("timeout", 0))

    process = _source_process(working_directory="svc", build_commands=[["npm", "ci"]])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            run=fake_run,
            build_step_timeout_seconds=5,
        )
    assert excinfo.value.code == "build_failed"


# --- F5: typed copy-phase failures ----------------------------------------------------------------


def test_build_process_tree_missing_working_directory_is_source_tree_unavailable(
    tmp_path: Path,
) -> None:
    (tmp_path / "source").mkdir()
    process = _source_process(working_directory="does-not-exist", build_commands=[])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process, source_root=tmp_path / "source", build_root=tmp_path / "build"
        )
    assert excinfo.value.code == "source_tree_unavailable"
    assert excinfo.value.stage == "build"


def test_build_process_tree_working_directory_naming_a_file_is_source_tree_unavailable(
    tmp_path: Path,
) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "svc").write_text("not a directory")
    process = _source_process(working_directory="svc", build_commands=[])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process, source_root=tmp_path / "source", build_root=tmp_path / "build"
        )
    assert excinfo.value.code == "source_tree_unavailable"


def test_build_process_tree_wraps_a_copy_failure_as_source_tree_unavailable(
    tmp_path: Path,
) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    process = _source_process(working_directory="svc", build_commands=[])

    def failing_copy(src: Path, dst: Path) -> None:
        raise PermissionError("no")

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            copy=failing_copy,
        )
    assert excinfo.value.code == "source_tree_unavailable"


# --- F2 (BLOCKER): symlink safety, exercised against a real filesystem ---------------------------


def test_build_process_tree_rejects_a_symlink_escaping_the_source_root(
    tmp_path: Path,
) -> None:
    """F2, p5-round1-review — BLOCKER. A repo can ship a symlink pointing outside `/work/source`
    (e.g. at `/run/futureagi/capabilities.json`); `copytree`'s default (`symlinks=False`) used to
    dereference it and copy the TARGET's bytes into the customer's own build tree, read as
    svc-control. Verified against a REAL symlink and a REAL filesystem, not a fake copier — this
    is exactly the class of bug a fake would hide."""
    outside_secret = tmp_path / "outside_secret.txt"
    outside_secret.write_text("TOP SECRET")
    svc_dir = tmp_path / "source" / "svc"
    svc_dir.mkdir(parents=True)
    (svc_dir / "evil_link").symlink_to(outside_secret)

    process = _source_process(working_directory="svc", build_commands=[])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process, source_root=tmp_path / "source", build_root=tmp_path / "build"
        )
    assert excinfo.value.code == "source_tree_unavailable"
    # Rejected BEFORE the copy, not after — nothing was ever materialized.
    assert not (tmp_path / "build" / "svc").exists()


def test_build_process_tree_preserves_a_within_tree_symlink_as_a_symlink(
    tmp_path: Path,
) -> None:
    """A link that stays inside the tree is legitimate and must keep working — `symlinks=True`
    copies it AS a link, it does not forbid it."""
    svc_dir = tmp_path / "source" / "svc"
    svc_dir.mkdir(parents=True)
    (svc_dir / "real.txt").write_text("hello")
    (svc_dir / "link.txt").symlink_to(svc_dir / "real.txt")

    process = _source_process(working_directory="svc", build_commands=[])
    build_dir = pr.build_process_tree(
        process, source_root=tmp_path / "source", build_root=tmp_path / "build"
    )
    assert (build_dir / "link.txt").is_symlink()
    assert (build_dir / "link.txt").read_text() == "hello"


def test_build_process_tree_rejects_a_symlinked_working_directory_path_component(
    tmp_path: Path,
) -> None:
    """F2: the resolve-then-`is_relative_to` check catches a symlinked PATH COMPONENT too, not
    just a symlinked leaf file — `working_directory: "services/tools-api"` passes the model
    layer's `_safe_relative` (no `..`, not absolute) even when `/work/source/services` is ITSELF
    a symlink pointing outside the checkout."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "services").symlink_to(tmp_path)  # escapes source_root entirely

    process = _source_process(working_directory="services/tools-api", build_commands=[])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process, source_root=source_root, build_root=tmp_path / "build"
        )
    assert excinfo.value.code == "source_tree_unavailable"


# --- F3 (BLOCKER): path-containment defense in depth ----------------------------------------------


def test_build_tree_dir_rejects_a_name_that_escapes_the_work_directory(
    tmp_path: Path,
) -> None:
    """F3, p5-round1-review — BLOCKER. Defense in depth, independent of the model-layer regex
    (`bundle_v2.SourceProcess`/`ManagedProcess.name`'s pattern) — these helpers take a plain
    `str`, so a caller bypassing model validation still cannot walk a name-derived directory
    outside `work_directory`. `pathlib` makes an absolute name a one-field catastrophe otherwise:
    `Path("/work/build") / "/etc"` IS `Path("/etc")`, which the old code then `rmtree`'d."""
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_tree_dir(tmp_path, "/etc")
    assert excinfo.value.code == "process_name_invalid"


def test_world_scratch_dir_rejects_a_traversal_name_that_truly_escapes(
    tmp_path: Path,
) -> None:
    # `world_scratch_dir` prepends TWO fixed levels ("worlds", "w<N>") before the name — 3 levels
    # of ".." is needed to escape `work_directory` itself (2 would only cancel the two prepended
    # levels and land back inside it, which is not an escape).
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.world_scratch_dir(tmp_path, 0, "../../../etc")
    assert excinfo.value.code == "process_name_invalid"


def test_managed_engine_data_dir_rejects_a_traversal_name_in_both_branches(
    tmp_path: Path,
) -> None:
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.managed_engine_data_dir(tmp_path, "/etc", world_index=None)
    assert excinfo.value.code == "process_name_invalid"
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.managed_engine_data_dir(tmp_path, "../../../etc", world_index=0)
    assert excinfo.value.code == "process_name_invalid"


def test_legitimate_process_names_are_unaffected_by_the_containment_check(
    tmp_path: Path,
) -> None:
    assert pr.build_tree_dir(tmp_path, "tools-api") == tmp_path / "build" / "tools-api"
    assert (
        pr.world_scratch_dir(tmp_path, 2, "agent")
        == tmp_path / "worlds" / "w2" / "agent"
    )
    assert pr.managed_engine_data_dir(tmp_path, "postgres", world_index=None) == (
        tmp_path / "managed" / "postgres"
    )


# --- F1 (BLOCKER): `user` honored — chown + privilege drop, verified structurally -----------------


def test_build_process_tree_chowns_the_copied_tree_to_the_resolved_user(
    tmp_path: Path,
) -> None:
    """F1, p5-round1-review — BLOCKER. The build tree (root AND every copied file, not just the
    directory) is chowned to the process's declared `user` after copy. `os.chown` to a foreign uid
    needs root, which this lane does not have and must not require — `chown` is faked here and the
    calls it WOULD make are asserted; see the module docstring."""
    (tmp_path / "source" / "svc").mkdir(parents=True)
    (tmp_path / "source" / "svc" / "main.py").write_text("print(1)\n")
    process = _source_process(working_directory="svc", build_commands=[])
    chowned: list[tuple[str, int, int]] = []

    pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        user_resolver=_fake_user_resolver({"svc-tools": (1234, 5678)}),
        chown=lambda path, uid, gid: chowned.append((str(path), uid, gid)),
    )
    build_dir = tmp_path / "build" / "svc"
    assert (str(build_dir), 1234, 5678) in chowned
    assert (str(build_dir / "main.py"), 1234, 5678) in chowned


def test_build_commands_run_under_the_resolved_user(tmp_path: Path) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_run(step, *, cwd, env, **kwargs):
        captured["user"] = kwargs.get("user")
        captured["group"] = kwargs.get("group")
        return subprocess.CompletedProcess(step, 0)

    process = _source_process(working_directory="svc", build_commands=[["true"]])
    pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        run=fake_run,
        user_resolver=_fake_user_resolver({"svc-tools": (1234, 5678)}),
        # `chown` faked too — real `os.chown` to a fake uid needs root, which this lane does not
        # have; the chown call itself is asserted separately (`test_build_process_tree_chowns_...`).
        chown=lambda path, uid, gid: None,
    )
    assert captured["user"] == 1234
    assert captured["group"] == 5678


def test_build_process_tree_falls_back_unprivileged_when_user_is_not_resolvable(
    tmp_path: Path,
) -> None:
    """The local test lane's own shape: no `svc-*` accounts on a dev box.
    `require_declared_user=False` (the default) must NOT raise — it runs unprivileged instead."""
    (tmp_path / "source" / "svc").mkdir(parents=True)
    process = _source_process(working_directory="svc", build_commands=[])
    build_dir = pr.build_process_tree(
        process,
        source_root=tmp_path / "source",
        build_root=tmp_path / "build",
        user_resolver=lambda name: None,
    )
    assert build_dir.is_dir()  # did not raise


def test_build_process_tree_raises_when_user_is_required_but_not_resolvable(
    tmp_path: Path,
) -> None:
    (tmp_path / "source" / "svc").mkdir(parents=True)
    process = _source_process(working_directory="svc", build_commands=[])
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.build_process_tree(
            process,
            source_root=tmp_path / "source",
            build_root=tmp_path / "build",
            user_resolver=lambda name: None,
            require_declared_user=True,
        )
    assert excinfo.value.code == "spawn_failed"


def test_spawn_source_process_applies_the_resolved_user_to_the_runner(
    tmp_path: Path,
) -> None:
    """Tests: assert the resolved user IS applied — the fake runner records `user=`."""
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")
    captured: dict[str, Any] = {}
    chowned: list[tuple[str, int, int]] = []

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["user"] = user
        captured["group"] = group
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        user_resolver=_fake_user_resolver({"svc-tools": (1234, 5678)}),
        chown=lambda path, uid, gid: chowned.append((str(path), uid, gid)),
    )
    assert captured["user"] == 1234
    assert captured["group"] == 5678
    # `{{WORLD_DIR}}` is chowned too — otherwise a process running as anyone but svc-control could
    # not write into its own per-world scratch directory at all.
    assert (str(world_dir), 1234, 5678) in chowned


def test_spawn_source_process_falls_back_unprivileged_when_user_is_not_resolvable(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["user"] = user
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        user_resolver=lambda name: None,
    )
    assert captured["user"] is None


def test_spawn_source_process_raises_when_user_is_required_but_not_resolvable(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        return FakeHandle()

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=world_dir,
            world_index=0,
            port_plan=plan,
            configuration_addresses={},
            secret_values={},
            secret_purposes={},
            runner=fake_runner,
            user_resolver=lambda name: None,
            require_declared_user=True,
        )
    assert excinfo.value.code == "spawn_failed"


def test_spawn_managed_process_chowns_data_dir_to_the_resolved_user(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    data_dir = tmp_path / "pg"
    chowned: list[tuple[str, int, int]] = []

    def fake_sync_run(argv, **kwargs):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "PG_VERSION").write_text("16\n")
        return subprocess.CompletedProcess(argv, 0)

    pr.spawn_managed_process(
        postgres,
        port=14000,
        data_dir=data_dir,
        credentials=creds,
        runner=lambda *a, **k: FakeHandle(),
        sync_run=fake_sync_run,
        user_resolver=_fake_user_resolver({"svc-data": (2222, 3333)}),
        chown=lambda path, uid, gid: chowned.append((str(path), uid, gid)),
    )
    assert (str(data_dir), 2222, 3333) in chowned
    # The pwfile is chowned to the same user too — `initdb` runs as that user (below) and must be
    # able to read its own 0600 file.
    assert any(
        uid == 2222 and gid == 3333 and path.endswith(".pwfile")
        for path, uid, gid in chowned
    )


def test_spawn_managed_process_chowns_a_pre_populated_data_dir_recursively(
    tmp_path: Path,
) -> None:
    """B6, p6-review-r1: `data_dir` is empty on the very FIRST spawn (`initdb` itself creates
    everything under it as the already-correct user), but `freeze_baseline`'s `datadir_copy`
    snapshot and `_seal_world_store`'s restore both populate it via a COPY that runs as the
    provisioner — every file underneath is provisioner-owned, and postgres refuses to start
    unless the data directory AND ITS CONTENTS are owned by the effective user. Simulates that by
    pre-populating nested content before the spawn call, same as a restored `datadir_copy` world
    would look like at this point, and asserts every path underneath was chowned, not only the
    top-level directory a single non-recursive `chown` call would have caught."""
    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    data_dir = tmp_path / "pg"
    (data_dir / "base" / "1").mkdir(parents=True)
    (data_dir / "PG_VERSION").write_text(
        "16\n"
    )  # already looks bootstrapped; initdb is skipped.
    (data_dir / "base" / "1" / "1234").write_text("data")
    chowned: list[tuple[str, int, int]] = []

    pr.spawn_managed_process(
        postgres,
        port=14000,
        data_dir=data_dir,
        credentials=creds,
        runner=lambda *a, **k: FakeHandle(),
        sync_run=_fake_sync_run,
        user_resolver=_fake_user_resolver({"svc-data": (2222, 3333)}),
        chown=lambda path, uid, gid: chowned.append((str(path), uid, gid)),
    )
    chowned_paths = {path for path, _, _ in chowned}
    assert str(data_dir) in chowned_paths
    assert str(data_dir / "base") in chowned_paths
    assert str(data_dir / "base" / "1") in chowned_paths
    assert str(data_dir / "base" / "1" / "1234") in chowned_paths
    assert all(uid == 2222 and gid == 3333 for _, uid, gid in chowned)
    assert (data_dir.stat().st_mode & 0o777) == 0o700


def test_default_process_runner_forwards_user_and_group_to_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: `subprocess.Popen(user=, group=)` requires root/CAP_SETUID to actually switch identity
    — this lane does not have it and must not require it. Verified structurally: `Popen` itself is
    faked, and the kwargs it would have been called with are asserted."""
    captured: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def poll(self) -> None:
            return None

    monkeypatch.setattr(pr.subprocess, "Popen", FakePopen)
    chowned: list[tuple[str, int, int]] = []
    pr.default_process_runner(
        ["true"],
        cwd=tmp_path,
        env={},
        log_path=tmp_path / "x.log",
        user=1234,
        group=5678,
        chown=lambda path, uid, gid: chowned.append((str(path), uid, gid)),
    )
    assert captured["user"] == 1234
    assert captured["group"] == 5678
    assert captured["start_new_session"] is True
    # The log file the harness creates (before the child's privilege drop) is chowned to match —
    # otherwise nothing running as the child's own user could ever open it fresh afterward.
    assert chowned == [(str(tmp_path / "x.log"), 1234, 5678)]


def test_popen_process_signals_the_owned_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePopen:
        pid = 4321

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise AssertionError("must signal the complete process group")

        def send_signal(self, _signum: int) -> None:
            raise AssertionError("must signal the complete process group")

        def kill(self) -> None:
            raise AssertionError("must signal the complete process group")

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(pr.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    handle = pr.PopenProcess(popen=FakePopen(), log_path=tmp_path / "process.log")

    handle.interrupt()
    handle.terminate()
    handle.kill()

    assert signals == [
        (4321, signal.SIGINT),
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]


def test_default_process_runner_does_not_chown_the_log_when_no_user_is_given(
    tmp_path: Path,
) -> None:
    chowned: list[Any] = []
    handle = pr.default_process_runner(
        ["python3", "-c", "pass"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        log_path=tmp_path / "logs" / "proc.log",
        chown=lambda *args: chowned.append(args),
    )
    _reap(handle)
    assert chowned == []


# --- process spawn: real subprocesses via `python3 -c`, secret injection ------------------------


def test_default_process_runner_spawns_a_real_subprocess_and_captures_output(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "logs" / "proc.log"
    handle = pr.default_process_runner(
        ["python3", "-c", "print('hello-from-child')"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        log_path=log_path,
    )
    _reap(handle)
    assert "hello-from-child" in handle.captured_output()


def test_a_process_claiming_the_purpose_receives_its_secret(tmp_path: Path) -> None:
    build_dir = tmp_path / "build" / "agent"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "agent"
    script = "import os,sys; sys.stdout.write('SECRET=' + os.environ.get('LIVEKIT_API_KEY', '<none>'))"
    process = _source_process(
        run_command=["python3", "-c", script], secret_purposes=["target_provider"]
    )
    plan = _solo_port_plan("svc")
    handle = pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={"LIVEKIT_API_KEY": "abc123", "OTHER": "zzz"},
        secret_purposes={
            "LIVEKIT_API_KEY": "target_provider",
            "OTHER": "source_checkout",
        },
    )
    _reap(handle.handle)
    assert handle.handle.captured_output() == "SECRET=abc123"


def test_a_process_not_claiming_the_purpose_does_not_receive_the_secret(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    script = "import os,sys; sys.stdout.write('SECRET=' + os.environ.get('LIVEKIT_API_KEY', '<none>'))"
    process = _source_process(run_command=["python3", "-c", script], secret_purposes=[])
    plan = _solo_port_plan("svc")
    handle = pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={"LIVEKIT_API_KEY": "abc123"},
        secret_purposes={"LIVEKIT_API_KEY": "target_provider"},
    )
    _reap(handle.handle)
    assert handle.handle.captured_output() == "SECRET=<none>"


def test_spawn_source_process_materializes_uploaded_google_credentials(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "agent"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "agent"
    process = _source_process(
        environment={"GOOGLE_APPLICATION_CREDENTIALS": "/etc/vertex/creds.json"},
        secret_purposes=["target_provider"],
    )
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["env"] = env
        return FakeHandle()

    raw = '{"type":"service_account","project_id":"example"}'
    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=_solo_port_plan("svc"),
        configuration_addresses={},
        secret_values={"GOOGLE_APPLICATION_CREDENTIALS_JSON": raw},
        secret_purposes={"GOOGLE_APPLICATION_CREDENTIALS_JSON": "target_provider"},
        runner=fake_runner,
    )

    credential_path = Path(captured["env"]["GOOGLE_APPLICATION_CREDENTIALS"])
    assert credential_path.read_text(encoding="utf-8") == raw
    assert credential_path.stat().st_mode & 0o777 == 0o600
    assert credential_path.is_relative_to(world_dir)
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in captured["env"]


def test_spawn_source_process_rematerializes_google_credentials_after_world_reset(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "agent"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "agent"
    process = _source_process(secret_purposes=["target_provider"])
    captured: list[dict[str, str]] = []

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured.append(env)
        return FakeHandle()

    for project_id in ("first", "second"):
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=world_dir,
            world_index=0,
            port_plan=_solo_port_plan("svc"),
            configuration_addresses={},
            secret_values={
                "GOOGLE_APPLICATION_CREDENTIALS_JSON": json.dumps(
                    {"type": "service_account", "project_id": project_id}
                )
            },
            secret_purposes={"GOOGLE_APPLICATION_CREDENTIALS_JSON": "target_provider"},
            runner=fake_runner,
        )

    credential_path = Path(captured[-1]["GOOGLE_APPLICATION_CREDENTIALS"])
    assert (
        json.loads(credential_path.read_text(encoding="utf-8"))["project_id"]
        == "second"
    )
    assert credential_path.stat().st_mode & 0o777 == 0o600


def test_spawn_source_process_rejects_invalid_uploaded_google_credentials(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "agent"
    build_dir.mkdir(parents=True)
    process = _source_process(secret_purposes=["target_provider"])

    with pytest.raises(pr.ProcessRuntimeError, match="not valid JSON") as excinfo:
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=tmp_path / "worlds" / "w0" / "agent",
            world_index=0,
            port_plan=_solo_port_plan("svc"),
            configuration_addresses={},
            secret_values={"GOOGLE_APPLICATION_CREDENTIALS_JSON": "not-json"},
            secret_purposes={"GOOGLE_APPLICATION_CREDENTIALS_JSON": "target_provider"},
            runner=lambda *args, **kwargs: FakeHandle(),
        )
    assert excinfo.value.code == "spawn_failed"
    assert excinfo.value.domain is FailureDomain.AGENT


def test_spawn_source_process_creates_the_per_world_scratch_directory(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w2" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=2,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert world_dir.is_dir()


def test_spawn_source_process_cwd_is_the_build_tree_not_the_world_scratch_dir(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    process = _source_process()
    plan = _solo_port_plan("svc")
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured["cwd"] = cwd
        return FakeHandle()

    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert captured["cwd"] == build_dir


def test_spawn_managed_process_requires_credentials_for_postgres(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    postgres = manifest.processes[0]
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.spawn_managed_process(
            postgres,
            port=14000,
            data_dir=tmp_path / "pg",
            credentials=None,
            runner=lambda *a, **k: FakeHandle(),
        )
    assert excinfo.value.code == "spawn_failed"


def test_spawn_managed_process_bootstraps_postgres_once_via_sync_run(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    data_dir = tmp_path / "pg"
    bootstrap_calls: list[list[str]] = []
    run_calls: list[list[str]] = []

    def fake_sync_run(argv, **kwargs):
        bootstrap_calls.append(argv)
        (data_dir).mkdir(parents=True, exist_ok=True)
        (data_dir / "PG_VERSION").write_text("16\n")
        return subprocess.CompletedProcess(argv, 0)

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        run_calls.append(argv)
        return FakeHandle()

    pr.spawn_managed_process(
        postgres,
        port=14000,
        data_dir=data_dir,
        credentials=creds,
        runner=fake_runner,
        sync_run=fake_sync_run,
    )
    assert len(bootstrap_calls) == 1
    assert bootstrap_calls[0][0] == "initdb"
    assert "--encoding=UTF8" in bootstrap_calls[0]
    assert "--locale=C.UTF-8" in bootstrap_calls[0]
    assert run_calls[0][0] == "postgres"
    # No pwfile left behind after bootstrap.
    assert not any(p.name.endswith(".pwfile") for p in data_dir.parent.glob(".*"))

    # A second spawn against the same already-initialized data_dir must not bootstrap again.
    pr.spawn_managed_process(
        postgres,
        port=14000,
        data_dir=data_dir,
        credentials=creds,
        runner=fake_runner,
        sync_run=fake_sync_run,
    )
    assert len(bootstrap_calls) == 1


# --- F7 (MAJOR): the pwfile is created 0600 atomically, not write-then-chmod ---------------------


def test_spawn_managed_process_creates_the_pwfile_with_o_excl_and_0600_atomically(
    tmp_path: Path,
) -> None:
    """F7, p5-round1-review. `write_text` then `os.chmod` creates the file at `0666 & ~umask`
    (typically 0644) and only narrows it afterward — a classic create-then-chmod TOCTOU that
    leaves the generated superuser password world-readable for the window in between.
    `os.open`'s own arguments are captured directly (the mode at CREATION, not observed after the
    fact, which cannot distinguish "created narrow" from "created wide, narrowed a moment later")."""
    import os as os_module

    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="s3cr3t")
    data_dir = tmp_path / "pg"
    captured: dict[str, Any] = {}
    real_open = os_module.open

    def spy_open(path, flags, mode=0o777, *args, **kwargs):
        if str(path).endswith(".pwfile"):
            captured["flags"] = flags
            captured["mode"] = mode
        return real_open(path, flags, mode, *args, **kwargs)

    def fake_sync_run(argv, **kwargs):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "PG_VERSION").write_text("16\n")
        return subprocess.CompletedProcess(argv, 0)

    pr.os.open = spy_open
    try:
        pr.spawn_managed_process(
            postgres,
            port=14000,
            data_dir=data_dir,
            credentials=creds,
            runner=lambda *a, **k: FakeHandle(),
            sync_run=fake_sync_run,
        )
    finally:
        pr.os.open = real_open

    assert captured["mode"] == 0o600
    assert captured["flags"] & os_module.O_EXCL
    assert captured["flags"] & os_module.O_CREAT


# --- §2b depends_on wait -----------------------------------------------------------------------


def test_depends_on_with_neither_readiness_nor_started_check_returns_immediately() -> (
    None
):
    manifest = _manifest(lambda body: {**body, "readiness": []})
    plan = pr.plan_ports(manifest, instances=1)
    spawned = pr.SpawnedWorldProcess("postgres", FakeHandle(), 14000, None)
    calls = {"n": 0}

    def fake_prober(**kwargs):
        calls["n"] += 1
        return True

    pr.wait_for_dependency(
        manifest,
        "postgres",
        world_index=0,
        port_plan=plan,
        spawned=spawned,
        prober=fake_prober,
    )
    assert (
        calls["n"] == 0
    )  # no readiness declared for `database` in this manifest -> immediate.


def test_depends_on_waits_for_the_capabilitys_readiness_probe() -> None:
    manifest = _manifest()  # `tools` capability has a declared readiness probe.
    plan = pr.plan_ports(manifest, instances=1)
    spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), 15001, 0)
    seen: list[tuple[str, int]] = []
    calls = {"n": 0}

    def fake_prober(
        *, protocol, host, port, path, user=None, password=None, dbname=None
    ):
        calls["n"] += 1
        seen.append((host, port))
        return calls["n"] >= 3

    ticks = {"t": 0.0}
    pr.wait_for_dependency(
        manifest,
        "tools-api",
        world_index=0,
        port_plan=plan,
        spawned=spawned,
        prober=fake_prober,
        clock=lambda: ticks["t"],
        sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s),
    )
    assert calls["n"] == 3
    assert seen[0] == ("localhost", 15001)


def test_wait_for_dependency_requires_every_readiness_probe_backing_the_process() -> (
    None
):
    """F8, p5-round1-review: a process backing two capabilities is only "ready" once ALL of its
    declared readiness probes pass — matching `healthy()`'s own all-probes semantics (§4 point 3).
    A process used to be treated as ready by `depends_on` the moment the FIRST-listed probe
    passed, then immediately reported unhealthy by `healthy()` the instant the second had not."""
    manifest = _manifest(
        lambda body: {
            **body,
            "capabilities": {
                **body["capabilities"],
                "tools_admin": {
                    "protocol": "http",
                    "service": "tools-api",
                    "configuration_name": None,
                },
            },
            "readiness": [
                *body["readiness"],
                {
                    "capability": "tools_admin",
                    "path": "/admin/health",
                    "timeout_seconds": 5,
                    "interval_seconds": 0.1,
                },
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), 15001, 0)
    ready = {"/health": False, "/admin/health": False}
    seen_paths: list[str | None] = []

    def fake_prober(
        *, protocol, host, port, path, user=None, password=None, dbname=None
    ):
        seen_paths.append(path)
        return ready[path]

    ticks = {"t": 0.0}

    def sleep(seconds: float) -> None:
        ticks["t"] += seconds
        if ticks["t"] >= 0.2:
            ready["/health"] = True
        if ticks["t"] >= 0.4:
            ready["/admin/health"] = True

    pr.wait_for_dependency(
        manifest,
        "tools-api",
        world_index=0,
        port_plan=plan,
        spawned=spawned,
        prober=fake_prober,
        clock=lambda: ticks["t"],
        sleep=sleep,
    )
    assert "/admin/health" in seen_paths
    assert ready["/health"] and ready["/admin/health"]


def test_depends_on_readiness_probe_timeout_is_a_typed_error() -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=1)
    spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), 15001, 0)
    ticks = {"t": 0.0}
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.wait_for_dependency(
            manifest,
            "tools-api",
            world_index=0,
            port_plan=plan,
            spawned=spawned,
            prober=lambda **kwargs: False,
            clock=lambda: ticks["t"],
            sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s),
        )
    assert excinfo.value.stage == "depends_on"
    assert excinfo.value.code == "depends_on_timeout"


def test_started_check_log_marker_variant_waits_for_the_marker() -> None:
    """T2, p5-round1-review: the marker now APPEARS after N polls (a mutating fake, same idiom as
    `test_depends_on_waits_for_the_capabilitys_readiness_probe`'s `calls['n'] >= 3`) rather than
    being seeded into `captured_output()` up front — the old fixture could not distinguish "polls
    until the marker appears" from "checks once and returns.\""""
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [],
            "processes": [
                body["processes"][0],
                {
                    **body["processes"][1],
                    "started_check": {"log_marker": "listening", "timeout_seconds": 5},
                },
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    handle = FakeHandle("booting...\n")
    spawned = pr.SpawnedWorldProcess("tools-api", handle, 15001, 0)
    calls = {"n": 0}

    def mutating_captured_output() -> str:
        calls["n"] += 1
        if calls["n"] >= 3:
            handle._output = "booting...\nlistening on :9\n"
        return handle._output

    handle.captured_output = mutating_captured_output
    ticks = {"t": 0.0}
    pr.wait_for_dependency(
        manifest,
        "tools-api",
        world_index=0,
        port_plan=plan,
        spawned=spawned,
        clock=lambda: ticks["t"],
        sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s),
    )
    assert calls["n"] >= 3


def test_started_check_log_marker_timeout_is_a_typed_error() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [],
            "processes": [
                body["processes"][0],
                {
                    **body["processes"][1],
                    "started_check": {"log_marker": "listening", "timeout_seconds": 1},
                },
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    handle = FakeHandle("booting...\n")  # marker never appears
    spawned = pr.SpawnedWorldProcess("tools-api", handle, 15001, 0)
    ticks = {"t": 0.0}
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.wait_for_dependency(
            manifest,
            "tools-api",
            world_index=0,
            port_plan=plan,
            spawned=spawned,
            clock=lambda: ticks["t"],
            sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s),
        )
    assert excinfo.value.code == "depends_on_timeout"


def test_started_check_port_variant_probes_the_dependencys_own_allocated_port() -> None:
    """F4, p5-round1-review — MAJOR. §2b (v1.8): `started_check.port` only SELECTS the port-probe
    variant — the dialed port is always the dependency's OWN allocated port (honoring
    `fixed_port`), never a literal. Exercised against a REAL bound socket on the port
    `port_plan.port_for` computes (via `fixed_port`, so the exact port is deterministic rather
    than depending on the formula's value not colliding with something else already listening)."""
    import socket as socket_module

    server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        manifest = _manifest(
            lambda body: {
                **body,
                "readiness": [],
                "processes": [
                    body["processes"][0],
                    {
                        **body["processes"][1],
                        "fixed_port": port,
                        "started_check": {"port": True, "timeout_seconds": 5},
                    },
                    body["processes"][2],
                ],
            }
        )
        plan = pr.plan_ports(manifest, instances=1)
        assert plan.port_for("tools-api", 0) == port
        spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), port, 0)
        pr.wait_for_dependency(
            manifest, "tools-api", world_index=0, port_plan=plan, spawned=spawned
        )
    finally:
        server.close()


def test_started_check_port_variant_dials_the_world_specific_formula_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F4: without `fixed_port`, the dialed port must be world-`world_index`'s OWN formula port —
    proven by monkeypatching the underlying TCP probe and asserting the exact port dialed, which a
    real-server-bind test cannot show as directly (it can only prove "some port worked")."""
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [],
            "processes": [
                body["processes"][0],
                {
                    **body["processes"][1],
                    "started_check": {"port": True, "timeout_seconds": 5},
                },
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=3)
    expected_port = plan.port_for("tools-api", 2)
    assert (
        expected_port == 15201
    )  # ordinal 1, world 2 — NOT the same as world 0's 15001.
    dialed: list[tuple[str, int]] = []

    def fake_tcp_probe(host: str, port: int, *, timeout: float = 0.75) -> bool:
        dialed.append((host, port))
        return True

    monkeypatch.setattr(pr, "_tcp_probe", fake_tcp_probe)
    spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), expected_port, 2)
    pr.wait_for_dependency(
        manifest, "tools-api", world_index=2, port_plan=plan, spawned=spawned
    )
    assert dialed == [("localhost", expected_port)]


def test_started_check_port_variant_timeout_when_nothing_listens() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [],
            "processes": [
                body["processes"][0],
                {
                    **body["processes"][1],
                    "fixed_port": 1,
                    "started_check": {"port": True, "timeout_seconds": 1},
                },
                body["processes"][2],
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    spawned = pr.SpawnedWorldProcess("tools-api", FakeHandle(), 1, 0)
    ticks = {"t": 0.0}
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.wait_for_dependency(
            manifest,
            "tools-api",
            world_index=0,
            port_plan=plan,
            spawned=spawned,
            clock=lambda: ticks["t"],
            sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s),
        )
    assert excinfo.value.code == "depends_on_timeout"


def test_spawn_world_spawns_in_dependency_order_and_waits_between(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        lambda body: {**body, "readiness": []}
    )  # skip real readiness waits
    plan = pr.plan_ports(manifest, instances=2)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    spawn_order: list[str] = []

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        spawn_order.append(argv[0])
        return FakeHandle()

    def fake_sync_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=plan,
        credentials=credentials,
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        sync_run=fake_sync_run,
    )
    result = pr.spawn_world(manifest, world_index=0, context=context)
    assert set(result.handles) == {"postgres", "tools-api", "agent"}
    # postgres has no dependency; tools-api depends on postgres; agent depends on both.
    assert spawn_order.index("postgres") < spawn_order.index(
        "node"
    )  # tools-api's run_command[0]
    assert spawn_order.index("node") < spawn_order.index(
        "python3"
    )  # agent's run_command[0]


def test_spawn_world_waits_for_terminal_source_started_check(tmp_path: Path) -> None:
    """A terminal control process has no dependent edge to wait on its started_check."""
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [],
            "processes": [
                body["processes"][0],
                body["processes"][1],
                {
                    **body["processes"][2],
                    "started_check": {
                        "log_marker": "registered worker",
                        "timeout_seconds": 5,
                    },
                },
            ],
        }
    )
    plan = pr.plan_ports(manifest, instances=1)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    marker_reads = {"count": 0}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        handle = FakeHandle("registered worker\n" if argv[0] == "python3" else "")
        if argv[0] == "python3":
            original = handle.captured_output

            def captured_output() -> str:
                marker_reads["count"] += 1
                return original()

            handle.captured_output = captured_output
        return handle

    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=plan,
        credentials=credentials,
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        sync_run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
    )

    pr.spawn_world(manifest, world_index=0, context=context)

    assert marker_reads["count"] == 1


def test_spawn_world_waits_for_terminal_source_process_started_check(
    tmp_path: Path,
) -> None:
    def terminal_check(body: dict[str, Any]) -> dict[str, Any]:
        body["readiness"] = []
        body["processes"][2]["started_check"] = {
            "log_marker": "registered worker",
            "timeout_seconds": 5,
        }
        return body

    manifest = _manifest(terminal_check)
    plan = pr.plan_ports(manifest, instances=1)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    marker_reads = 0

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        handle = FakeHandle("registered worker\n" if argv[0] == "python3" else "")
        if argv[0] == "python3":
            original = handle.captured_output

            def captured_output() -> str:
                nonlocal marker_reads
                marker_reads += 1
                return original()

            handle.captured_output = captured_output
        return handle

    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=plan,
        credentials=credentials,
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        sync_run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
    )
    pr.spawn_world(manifest, world_index=0, context=context)
    assert marker_reads >= 1


def test_spawn_world_reuses_job_shared_handles_across_worlds(tmp_path: Path) -> None:
    manifest = _manifest(lambda body: {**body, "readiness": []})
    plan = pr.plan_ports(manifest, instances=2)
    credentials = pr.generate_engine_credentials(manifest, token=lambda: "PW")
    spawned_argv0s: list[str] = []

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        spawned_argv0s.append(argv[0])
        return FakeHandle()

    def fake_sync_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=plan,
        credentials=credentials,
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
        sync_run=fake_sync_run,
    )
    world0 = pr.spawn_world(manifest, world_index=0, context=context)
    shared = {
        name: handle
        for name, handle in world0.handles.items()
        if plan.is_job_shared(name)
    }
    assert set(shared) == {"postgres"}
    spawned_argv0s.clear()
    world1 = pr.spawn_world(
        manifest, world_index=1, context=context, shared_handles=shared
    )
    assert "postgres" not in spawned_argv0s  # not respawned for world 1
    assert (
        world1.handles["postgres"] is shared["postgres"]
    )  # the very same handle, reused


# --- §4.3 healthy() ------------------------------------------------------------------------------


def test_healthy_dispatches_to_the_probe_for_each_declared_readiness_entry() -> None:
    manifest = _manifest()  # one readiness entry, for `tools`
    runtime = pr.EnvironmentRuntime(
        runtime_id="r1",
        world_index=0,
        bundle_digest="sha256:" + "0" * 64,
        state=pr.RuntimeState.PREPARING,
        endpoints={
            "tools": pr.RuntimeEndpoint(
                capability="tools",
                protocol="http",
                address="http://localhost:15001",
                configuration_name="TOOLS_API_URL",
            ),
        },
    )
    seen: list[tuple[str, int, str | None]] = []

    def fake_prober(
        *, protocol, host, port, path, user=None, password=None, dbname=None
    ):
        seen.append((host, port, path))
        return True

    assert pr.probe_runtime_health(manifest, runtime, prober=fake_prober) is True
    assert seen == [("localhost", 15001, "/health")]


def test_healthy_is_false_when_a_declared_probe_fails() -> None:
    manifest = _manifest()
    runtime = pr.EnvironmentRuntime(
        runtime_id="r1",
        world_index=0,
        bundle_digest="sha256:" + "0" * 64,
        state=pr.RuntimeState.PREPARING,
        endpoints={
            "tools": pr.RuntimeEndpoint(
                capability="tools",
                protocol="http",
                address="http://localhost:15001",
                configuration_name="TOOLS_API_URL",
            ),
        },
    )
    assert (
        pr.probe_runtime_health(manifest, runtime, prober=lambda **kwargs: False)
        is False
    )


def test_healthy_is_false_when_a_declared_probes_capability_has_no_endpoint() -> None:
    manifest = _manifest()
    runtime = pr.EnvironmentRuntime(
        runtime_id="r1",
        world_index=0,
        bundle_digest="sha256:" + "0" * 64,
        state=pr.RuntimeState.PREPARING,
        endpoints={},
    )
    assert (
        pr.probe_runtime_health(manifest, runtime, prober=lambda **kwargs: True)
        is False
    )


def test_probe_runtime_health_parses_postgres_credentials_out_of_the_endpoint_address() -> (
    None
):
    """F9, p5-round1-review: `probe_runtime_health` only ever sees `EnvironmentRuntime`, not the
    job's credential map — the rendered
    `postgresql://harness:<pw>@localhost:<port>/w<N>` address already carries everything a real
    probe needs, so it is parsed back out rather than threaded separately."""
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [
                {
                    "capability": "database",
                    "timeout_seconds": 5,
                    "interval_seconds": 0.1,
                },
            ],
        }
    )
    runtime = pr.EnvironmentRuntime(
        runtime_id="r1",
        world_index=0,
        bundle_digest="sha256:" + "0" * 64,
        state=pr.RuntimeState.PREPARING,
        endpoints={
            "database": pr.RuntimeEndpoint(
                capability="database",
                protocol="postgres",
                address="postgresql://harness:s3cr3t@localhost:14000/w0",
                configuration_name="DATABASE_URL",
            ),
        },
    )
    seen: dict[str, Any] = {}

    def fake_prober(
        *, protocol, host, port, path, user=None, password=None, dbname=None
    ):
        seen.update(user=user, password=password, dbname=dbname)
        return True

    assert pr.probe_runtime_health(manifest, runtime, prober=fake_prober) is True
    assert seen == {"user": "harness", "password": "s3cr3t", "dbname": "w0"}


async def _run_healthy(*args: Any, **kwargs: Any) -> bool:
    return await pr.healthy(*args, **kwargs)


def test_healthy_transitions_follow_section_3_and_never_promote() -> None:
    """F6, p5-round1-review — MAJOR (T3). §3's `state` row fixes the legal transitions:
    `preparing->ready`, `ready->unhealthy`, and `unhealthy->ready` ONLY via re-provision reconcile
    (§4 point 1). `healthy()` is not a reconcile, so it may only ever DEMOTE — this replaces the
    old test, which asserted only `preparing->ready` then `ready->unhealthy` in sequence and never
    exercised `unhealthy->?` or `stopped->?` at all, so it could not have caught the promotion bug
    the old implementation actually had (`RuntimeState.READY if is_healthy else UNHEALTHY`, which
    unconditionally promotes ANY prior state to READY on a passing probe)."""
    import asyncio

    manifest = _manifest()
    endpoints = {
        "tools": pr.RuntimeEndpoint(
            capability="tools",
            protocol="http",
            address="http://localhost:15001",
            configuration_name="TOOLS_API_URL",
        ),
    }

    def make(state: pr.RuntimeState) -> pr.EnvironmentRuntime:
        return pr.EnvironmentRuntime(
            runtime_id="r1",
            world_index=0,
            bundle_digest="sha256:" + "0" * 64,
            state=state,
            endpoints=endpoints,
        )

    preparing = make(pr.RuntimeState.PREPARING)
    assert (
        asyncio.run(_run_healthy(manifest, preparing, prober=lambda **kwargs: True))
        is True
    )
    assert preparing.state is pr.RuntimeState.READY  # preparing + healthy -> ready

    still_ready = make(pr.RuntimeState.READY)
    assert (
        asyncio.run(_run_healthy(manifest, still_ready, prober=lambda **kwargs: True))
        is True
    )
    assert still_ready.state is pr.RuntimeState.READY  # ready stays ready

    demoted = make(pr.RuntimeState.READY)
    assert (
        asyncio.run(_run_healthy(manifest, demoted, prober=lambda **kwargs: False))
        is False
    )
    assert (
        demoted.state is pr.RuntimeState.UNHEALTHY
    )  # ready + unhealthy probe -> demote

    stays_unhealthy = make(pr.RuntimeState.UNHEALTHY)
    ok = asyncio.run(
        _run_healthy(manifest, stays_unhealthy, prober=lambda **kwargs: True)
    )
    assert ok is True  # the PROBE passed...
    assert (
        stays_unhealthy.state is pr.RuntimeState.UNHEALTHY
    )  # ...but state is NOT promoted

    stays_stopped = make(pr.RuntimeState.STOPPED)
    asyncio.run(_run_healthy(manifest, stays_stopped, prober=lambda **kwargs: True))
    assert (
        stays_stopped.state is pr.RuntimeState.STOPPED
    )  # only provision()/close() clear this


# --- default probers: real postgres/http exercise, no fakes --------------------------------------


def test_default_capability_prober_falls_back_to_tcp_when_psycopg_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The postgres branch must fall back to a bare TCP probe rather than raising
    `ImportError` on hosts without `psycopg`. Absence is simulated by blocking the
    module in `sys.modules` — the prober imports at call time, so the import genuinely
    fails and the real fallback branch runs regardless of what this venv has installed
    (an environment premise check would flip whenever another lane needs `psycopg`
    present)."""
    monkeypatch.setitem(sys.modules, "psycopg", None)
    import socket as socket_module

    server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert (
            pr.default_capability_prober(
                protocol=CapabilityProtocol.POSTGRES,
                host="localhost",
                port=port,
                path=None,
            )
            is True
        )
    finally:
        server.close()


def test_default_capability_prober_tcp_probe_is_false_when_nothing_listens() -> None:
    assert (
        pr.default_capability_prober(
            protocol=CapabilityProtocol.TCP, host="localhost", port=1, path=None
        )
        is False
    )


def test_probe_postgres_runs_a_real_select_1_when_credentials_are_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F9, p5-round1-review — MAJOR. Previously connected as `user="postgres", dbname="postgres"`
    (neither exists — the catalog's `initdb -U harness` creates only `harness`) and treated almost
    any resulting `OperationalError` as "answered, therefore ready." With real credentials, this
    must run an actual query — proven with a fake `psycopg` module, since no real server is
    available in this lane."""
    executed: list[str] = []

    class FakeConnection:
        def execute(self, query: str) -> None:
            executed.append(query)

        def close(self) -> None:
            pass

    def fake_connect(**kwargs: Any) -> FakeConnection:
        assert kwargs["user"] == "harness"
        assert kwargs["dbname"] == "w0"
        return FakeConnection()

    fake_module = types.ModuleType("psycopg")
    fake_module.connect = fake_connect
    fake_module.OperationalError = type("OperationalError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "psycopg", fake_module)

    assert (
        pr._probe_postgres(
            "localhost", 14000, user="harness", password="pw", dbname="w0"
        )
        is True
    )
    assert executed == ["SELECT 1"]


def test_probe_postgres_treats_a_starting_up_server_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old substring-match heuristic treated `FATAL: the database system is starting up`
    (postgres's OWN response while still in recovery — it binds its listen socket early) as
    "answered, therefore ready." A real `SELECT 1` cannot be fooled by it."""
    fake_module = types.ModuleType("psycopg")
    operational_error = type("OperationalError", (Exception,), {})
    fake_module.OperationalError = operational_error

    class FakeConnection:
        def execute(self, query: str) -> None:
            raise operational_error("the database system is starting up")

        def close(self) -> None:
            pass

    fake_module.connect = lambda **kwargs: FakeConnection()
    monkeypatch.setitem(sys.modules, "psycopg", fake_module)

    assert (
        pr._probe_postgres(
            "localhost", 14000, user="harness", password="pw", dbname="w0"
        )
        is False
    )


def test_probe_postgres_falls_back_to_tcp_when_credentials_are_not_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when `psycopg` IS available, no generated credentials means no real query is
    possible — falls back to the same TCP-only check as the psycopg-absent case, never fabricates
    a connection attempt without something to authenticate with."""
    import socket as socket_module

    def fake_connect(**kwargs: Any) -> None:
        raise AssertionError("must not attempt to connect without credentials")

    fake_module = types.ModuleType("psycopg")
    fake_module.connect = fake_connect
    fake_module.OperationalError = Exception
    monkeypatch.setitem(sys.modules, "psycopg", fake_module)

    server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    try:
        port = server.getsockname()[1]
        assert pr._probe_postgres("localhost", port) is True  # no user/dbname given
    finally:
        server.close()


def test_probe_http_against_a_real_server_reports_2xx_as_ready() -> None:
    """T7, p5-round1-review: `_probe_http` — the default prober for the one capability the shared
    manifest actually declares a readiness probe for — had zero direct coverage; every readiness
    test in this file injects a fake `prober` instead."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: Any) -> None:  # silence stderr noise
            pass

    server = http.server.HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert (
            pr.default_capability_prober(
                protocol=CapabilityProtocol.HTTP,
                host="localhost",
                port=port,
                path="/health",
            )
            is True
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_probe_http_reports_not_ready_when_nothing_listens() -> None:
    assert (
        pr.default_capability_prober(
            protocol=CapabilityProtocol.HTTP, host="localhost", port=1, path="/health"
        )
        is False
    )


# =================================================================================================
# Phase 6: seed application, baseline freeze, world clone, reset, conformance gate, provision/close
# =================================================================================================
#
# `_manifest()` above already carries one postgres store (`template_database`, job-shared,
# sentinel `SELECT 1` = `"1"`) — reused everywhere below unless a test needs a different strategy,
# in which case a `mutate` callback swaps `seed.stores[0].baseline` (and, where the row content
# under test matters, `migrations`/`seed_files`). `fake_sync_run`/`SqlSpy` below are this section's
# own `ProcessRunner`/`CapabilityProber`-equivalent fakes — no real postgres/redis/rabbitmq/docker
# anywhere, per the module docstring.


def _fake_sync_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def _recording_sync_run(calls: list[Any]) -> Callable[..., subprocess.CompletedProcess]:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return run


def _pg_spy_row(value: Any) -> list[tuple[Any, ...]]:
    return [(value,)]


def _recording_prober(*, fail_times: int = 0) -> Callable[..., bool]:
    """t1/N2, p6-review-r2: the promote-path fixtures used to inject `prober=lambda **kwargs:
    True` — a fast pass that made `provision()`'s PREPARING->READY promotion and `reset()`'s
    post-sentinel probe structurally unable to observe whether either one actually POLLED (N2) or
    merely sampled once, the exact class of gap t1 was raised about for `_wait_for_store_ready`.
    `fail_times=0` (the default) keeps every EXISTING test's prior instant-pass behavior
    unchanged; a test proving the poll passes `fail_times=N` and asserts `len(prober.calls) > 1`
    or that state ends up correct only after N+1 calls."""
    calls: list[dict[str, Any]] = []
    remaining = [fail_times]

    def prober(**kwargs: Any) -> bool:
        calls.append(kwargs)
        if remaining[0] > 0:
            remaining[0] -= 1
            return False
        return True

    prober.calls = calls  # type: ignore[attr-defined]
    return prober


class SqlSpy:
    """A fake `SqlRunner`: records every `(dbname, statement)` call, in order — the "SQL spy" the
    task calls for — and simulates just enough real postgres semantics for the baseline-freeze /
    world-clone / reset / conformance-gate state machines to run against without a real server:
    `CREATE`/`DROP`/`ALTER ... TEMPLATE` database bookkeeping, `to_regclass` existence checks,
    `pg_tables`/`COUNT(*)` row counts, and a fixed answer for any `sentinel.query` a test seeds via
    `answers`. `canary_leaks` deliberately makes the conformance canary visible across databases,
    for the gate's FAIL-path test.
    """

    def __init__(
        self,
        *,
        answers: dict[str, list[tuple[Any, ...]]] | None = None,
        canary_leaks: bool = False,
        seeded_tables: dict[str, set[str]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.databases: dict[str, set[str]] = {}
        self.row_data: dict[tuple[str, str], int] = {}
        self.answers = answers or {}
        self.canary_leaks = canary_leaks
        # What a freshly `CREATE DATABASE`'d name starts with — simulates "the seed already
        # landed" (a real `psql -f ...` would have populated it; the fake `sync_run` this spy
        # runs alongside is a structural no-op, per the module docstring).
        self.seeded_tables = seeded_tables or {}

    def __call__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        dbname: str,
        statement: str,
        read_only: bool = False,
    ) -> list[tuple[Any, ...]]:
        self.calls.append((dbname, statement))
        body = statement.strip()
        if read_only:
            # N8, p6-review-r2: simulates postgres's own `SET default_transaction_read_only = on`
            # rejecting a non-SELECT statement on a read-only session — the exact class of driver
            # error `_call_sql` converts to `store_statement_failed`. Checked PER STATEMENT, not
            # just the string's own prefix — psycopg3's simple-query protocol runs every `;`-
            # separated statement in one unparameterized `execute()` call, which is exactly how a
            # sentinel like `"SELECT 1; DROP TABLE riders"` smuggles a write past a naive
            # starts-with-SELECT check while still LOOKING like a read at a glance.
            for sub_statement in body.split(";"):
                sub = sub_statement.strip()
                if sub and not sub.upper().startswith(("SELECT", "WITH")):
                    raise RuntimeError(
                        f"cannot execute {sub!r} in a read-only transaction"
                    )
        if body in self.answers:
            return self.answers[body]
        if body == "SELECT 1":
            return [(1,)]  # `_manifest()`'s own default sentinel query.
        if body.startswith("CREATE DATABASE") and "TEMPLATE" not in body:
            name = body.split('"')[1]
            self.databases[name] = set(self.seeded_tables.get(name, set()))
            return []
        if "TEMPLATE" in body and body.startswith("CREATE DATABASE"):
            source, target = body.split('"')[1], body.split('"')[3]
            self.databases[source] = set(self.databases.get(target, set()))
            return []
        if body.startswith("ALTER DATABASE") and "RENAME TO" in body:
            old, new = body.split('"')[1], body.split('"')[3]
            self.databases[new] = self.databases.pop(old, set())
            return []
        if body.startswith("DROP DATABASE"):
            name = body.split('"')[-2]
            self.databases.pop(name, None)
            return []
        if body.startswith("CREATE TABLE") and "_alk_conformance" in body:
            self.databases.setdefault(dbname, set()).add("_alk_conformance")
            if self.canary_leaks:
                leaked = "w1" if dbname == "w0" else "w0"
                self.databases.setdefault(leaked, set()).add("_alk_conformance")
            return []
        if body.startswith("SELECT to_regclass"):
            present = "_alk_conformance" in self.databases.get(dbname, set())
            return [(present,)]
        if body.startswith("SELECT tablename FROM pg_tables"):
            return [(table,) for table in sorted(self.databases.get(dbname, set()))]
        if body.startswith("SELECT COUNT(*)"):
            table = body.split('"')[1]
            return [(self.row_data.get((dbname, table), 0),)]
        return []


def _postgres_creds(m: EnvironmentBundleV2) -> dict[str, pr.EngineCredentials]:
    return pr.generate_engine_credentials(m, token=lambda: "PW")


def _spawn_context(
    manifest: EnvironmentBundleV2,
    *,
    instances: int = 2,
    bundle_dir: Path,
    sql_runner: Any = None,
    redis_runner: Any = None,
    rabbitmq_inspector: Any = None,
    rabbitmq_declare: Any = None,
    rabbitmq_delete: Any = None,
    rabbitmq_import: Any = None,
    runner: Any = None,
    sync_run: Any = None,
    prober: Any = None,
    work_directory: Path,
) -> pr.SpawnContext:
    port_plan = pr.plan_ports(manifest, instances=instances)
    credentials = _postgres_creds(manifest)
    return pr.SpawnContext(
        work_directory=work_directory,
        port_plan=port_plan,
        credentials=credentials,
        secret_values={},
        secret_purposes={},
        runner=runner or (lambda *a, **k: FakeHandle()),
        sync_run=sync_run or _fake_sync_run,
        sql_runner=sql_runner or SqlSpy(),
        redis_runner=redis_runner or (lambda **kwargs: None),
        rabbitmq_inspector=rabbitmq_inspector or (lambda **kwargs: 0),
        # m8, p6-review-r1: `SpawnContext`'s real defaults for these two now make an actual HTTP
        # call — every test lane here must fake them explicitly, same rule as every other engine
        # seam in this fixture, so a test that never overrides them never risks a real network
        # call regardless of which store the conformance canary ends up preferring.
        rabbitmq_declare=rabbitmq_declare or (lambda **kwargs: None),
        rabbitmq_delete=rabbitmq_delete or (lambda **kwargs: None),
        # N17, p6-review-r2: same rule — seeding now goes over HTTP too.
        rabbitmq_import=rabbitmq_import or (lambda **kwargs: None),
        bundle_dir=bundle_dir,
        # Declared `readiness` probes are P5's own concern (`wait_for_dependency`/`healthy`
        # already cover them elsewhere in this file) — a fast-pass fake here so Phase 6's own
        # tests, which reuse `_manifest()`'s `tools` readiness entry incidentally, never block on
        # a real HTTP probe against a `FakeHandle` that never actually opened a port. Also what
        # B4's own `_wait_for_store_ready` polls — fast-pass here means every EXISTING freeze/
        # world-clone test keeps its prior (instant, no-wait) behavior unless it opts into a
        # slower fake explicitly (t1's own observability test does exactly that; N2's new
        # observability tests do the same one layer up, at the promote sites).
        prober=prober or _recording_prober(),
    )


# --- §2c seed application ------------------------------------------------------------------------


def test_postgres_seed_argv_shape() -> None:
    argv = pr.postgres_seed_argv(
        port=14000,
        dbname="alk_baseline_postgres",
        user="harness",
        file=Path("db/schema.sql"),
    )
    assert argv == [
        "psql",
        "-h",
        "localhost",
        "-p",
        "14000",
        "-U",
        "harness",
        "-d",
        "alk_baseline_postgres",
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        "db/schema.sql",
    ]


def test_redis_seed_argv_has_no_file_flag_stdin_carries_the_commands() -> None:
    assert pr.redis_seed_argv(port=15003) == [
        "redis-cli",
        "-h",
        "localhost",
        "-p",
        "15003",
    ]


def test_default_rabbitmq_definitions_importer_posts_the_file_to_the_definitions_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """N17, p6-review-r2: `rabbitmqadmin import` is dropped entirely — seeding now POSTs the raw
    definitions file straight to the management API's own `/api/definitions` endpoint, the same
    HTTP-only seam every other rabbitmq call in this module already uses. No `rabbitmqadmin`
    argv/binary anywhere in this path."""
    definitions_file = tmp_path / "mq" / "defs.json"
    definitions_file.parent.mkdir(parents=True)
    definitions_file.write_bytes(b'{"queues": []}')
    creds = pr.EngineCredentials(username="harness", password="pw")
    requests: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        requests.append(request)

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

        return _Response()

    monkeypatch.setattr(pr.urllib.request, "urlopen", fake_urlopen)
    pr.default_rabbitmq_definitions_importer(
        host="localhost",
        port=14002,
        credentials=creds,
        file=definitions_file,
    )
    assert len(requests) == 1
    assert requests[0].full_url == "http://localhost:24002/api/definitions"
    assert requests[0].data == b'{"queues": []}'
    assert requests[0].get_header("Authorization") is not None


def test_apply_store_seed_runs_migrations_then_seed_files_in_listed_order(
    tmp_path: Path,
) -> None:
    calls: list[Any] = []
    store = pr.StoreEntry.model_validate(
        {
            "capability": "database",
            "migrations": ["db/001.sql", "db/002.sql"],
            "seed_files": ["db/seed_a.sql", "db/seed_b.sql"],
            "baseline": {
                "strategy": "template_database",
                "inputs_digest": "sha256:" + "a" * 64,
            },
            "sentinel": {"query": "SELECT 1", "expected": "1"},
        }
    )
    creds = pr.EngineCredentials(username="harness", password="pw")
    pr.apply_store_seed(
        store,
        engine=pr.ManagedEngine.POSTGRES,
        bundle_dir=tmp_path,
        port=14000,
        dbname="x",
        credentials=creds,
        process_name="postgres",
        sync_run=_recording_sync_run(calls),
    )
    files_in_order = [argv[argv.index("-f") + 1] for argv, _ in calls]
    assert files_in_order == [
        str(tmp_path / "db/001.sql"),
        str(tmp_path / "db/002.sql"),
        str(tmp_path / "db/seed_a.sql"),
        str(tmp_path / "db/seed_b.sql"),
    ]


def test_apply_seed_file_postgres_env_keeps_path_and_adds_pgpassword(
    tmp_path: Path,
) -> None:
    """A real bug caught by manual exercise before this test existed: `env=` REPLACES a child's
    environment, not extends it — passing bare `{"PGPASSWORD": ...}` would drop `PATH` and make
    `psql` unfindable outside `subprocess`'s narrow POSIX fallback."""
    calls: list[Any] = []
    creds = pr.EngineCredentials(username="harness", password="s3cr3t")
    pr.apply_seed_file(
        pr.ManagedEngine.POSTGRES,
        tmp_path / "db/schema.sql",
        port=14000,
        dbname="x",
        credentials=creds,
        process_name="postgres",
        sync_run=_recording_sync_run(calls),
    )
    _, kwargs = calls[0]
    assert kwargs["env"]["PGPASSWORD"] == "s3cr3t"
    assert "PATH" in kwargs["env"]


def test_apply_seed_file_redis_pipes_file_content_over_stdin(tmp_path: Path) -> None:
    seed_file = tmp_path / "cache" / "seed.txt"
    seed_file.parent.mkdir(parents=True)
    seed_file.write_text("SET greeting hi\n")
    calls: list[Any] = []
    pr.apply_seed_file(
        pr.ManagedEngine.REDIS,
        seed_file,
        port=15003,
        dbname="",
        credentials=None,
        process_name="cache",
        sync_run=_recording_sync_run(calls),
    )
    argv, kwargs = calls[0]
    assert argv == ["redis-cli", "-h", "localhost", "-p", "15003"]
    assert kwargs["input"] == "SET greeting hi\n"


def test_apply_seed_file_raises_seed_failed_on_nonzero_exit(tmp_path: Path) -> None:
    def failing_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="syntax error")

    creds = pr.EngineCredentials(username="harness", password="pw")
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.apply_seed_file(
            pr.ManagedEngine.POSTGRES,
            tmp_path / "db/broken.sql",
            port=14000,
            dbname="x",
            credentials=creds,
            process_name="postgres",
            sync_run=failing_run,
        )
    assert excinfo.value.code == "seed_failed"
    assert excinfo.value.stage == "seed"
    assert "syntax error" in str(excinfo.value)


def test_apply_seed_file_rabbitmq_import_failure_is_seed_failed(tmp_path: Path) -> None:
    """rabbitmq's own definitions-import failure is customer-authored seed CONTENT
    (§2f `seed_failed`, deterministic, NOT retried), never `store_statement_failed` (the
    harness's own statement seam, `infrastructure`, retryable) — the two domains carry opposite
    retry semantics, so a code swap here either retries a broken definitions file forever or
    gives up on what might have been a transient import blip."""
    creds = pr.EngineCredentials(username="harness", password="pw")

    def failing_import(*, host: str, port: int, credentials: Any, file: Path) -> None:
        raise RuntimeError("malformed definitions")

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.apply_seed_file(
            pr.ManagedEngine.RABBITMQ,
            tmp_path / "mq" / "defs.json",
            port=14002,
            dbname="",
            credentials=creds,
            process_name="mq",
            sync_run=_fake_sync_run,
            rabbitmq_import=failing_import,
        )
    assert excinfo.value.code == "seed_failed"


# --- §2c sentinel checking ------------------------------------------------------------------------


def test_check_sentinel_postgres_pass_and_fail() -> None:
    store = _manifest().seed.stores[0]  # {query: "SELECT 1", expected: "1"}
    creds = pr.EngineCredentials(username="harness", password="pw")
    passing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.POSTGRES,
        host="localhost",
        port=14000,
        dbname="w0",
        credentials=creds,
        sql_runner=lambda **kwargs: _pg_spy_row(1),
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert passing is True
    failing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.POSTGRES,
        host="localhost",
        port=14000,
        dbname="w0",
        credentials=creds,
        sql_runner=lambda **kwargs: _pg_spy_row(0),
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert failing is False


def test_check_sentinel_passes_read_only_true_to_the_sql_runner() -> None:
    """N8, p6-review-r2 (MAJOR): the sentinel is customer-authored content from an untrusted repo,
    executed as `harness` — the role `initdb -U harness` makes the postgres SUPERUSER — over a
    plain autocommit session. `check_sentinel` must mark this call `read_only=True` so `default_
    sql_runner` puts the session into a read-only transaction before running it."""
    store = _manifest().seed.stores[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    seen: dict[str, Any] = {}

    def recording_sql_runner(**kwargs: Any) -> list[tuple[Any, ...]]:
        seen.update(kwargs)
        return _pg_spy_row(1)

    pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.POSTGRES,
        host="localhost",
        port=14000,
        dbname="w0",
        credentials=creds,
        sql_runner=recording_sql_runner,
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert seen["read_only"] is True


def test_a_sentinel_query_attempting_a_write_fails(tmp_path: Path) -> None:
    """N8, p6-review-r2 (MAJOR, task-required verification): the door B2 closed for `psql -f`
    (dropping privilege for customer-authored SQL) was still open for `sentinel.query` — arbitrary
    multi-statement SQL executed as the postgres superuser on a read-write session. `SqlSpy`
    (`read_only=True`) rejects a non-`SELECT` statement exactly as a real `SET default_
    transaction_read_only = on` session would; a sentinel that tries to write must surface as a
    typed `store_statement_failed`, not silently succeed.
    """
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "sentinel": {
                            "query": "SELECT 1; DROP TABLE riders",
                            "expected": "1",
                        },
                    }
                ]
            },
        }
    )
    store = manifest.seed.stores[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    sql_spy = SqlSpy()
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.check_sentinel(
            store,
            engine=pr.ManagedEngine.POSTGRES,
            host="localhost",
            port=14000,
            dbname="w0",
            credentials=creds,
            sql_runner=sql_spy,
            redis_runner=lambda **kwargs: None,
            rabbitmq_inspector=lambda **kwargs: 0,
        )
    assert excinfo.value.code == "store_statement_failed"


def test_call_redis_driver_exception_is_store_statement_failed() -> None:
    """`_call_redis` wraps a provisioner-ISSUED command (sentinel/canary probes —
    never customer seed content, which goes through `apply_seed_file` instead) — a driver
    exception there is the harness's own statement seam, §2f `store_statement_failed`
    (`infrastructure`, retryable), never `seed_failed` (`environment`, NOT retried). Swapped, a
    transient redis blip during a sentinel check would never retry."""

    def failing_redis_runner(*, host: str, port: int, command: Any) -> Any:
        raise RuntimeError("connection reset")

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._call_redis(
            failing_redis_runner,
            stage="conformance",
            process_name="cache",
            host="localhost",
            port=15003,
            command=["GET", "x"],
        )
    assert excinfo.value.code == "store_statement_failed"


def test_call_rabbitmq_driver_exception_is_store_statement_failed() -> None:
    """`_call_rabbitmq` wraps a provisioner-ISSUED queue inspection (sentinel/canary probes —
    never customer seed content) — a driver exception there is the harness's own statement seam,
    §2f `store_statement_failed` (`infrastructure`, retryable), never `seed_failed`
    (`environment`, NOT retried). Swapped, a transient rabbitmq blip during a sentinel check would
    never retry."""

    def failing_inspector(*, host: str, port: int, credentials: Any, queue: str) -> int:
        raise RuntimeError("connection reset")

    creds = pr.EngineCredentials(username="harness", password="pw")
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._call_rabbitmq(
            failing_inspector,
            stage="conformance",
            process_name="mq",
            host="localhost",
            port=14002,
            credentials=creds,
            queue="canary",
        )
    assert excinfo.value.code == "store_statement_failed"


def test_call_rabbitmq_action_driver_exception_is_store_statement_failed() -> None:
    """`_call_rabbitmq_action` wraps the write-side canary declare/publish call — same B5 typing
    as `_call_rabbitmq`, against a store that has already passed readiness. A driver exception
    there is also §2f `store_statement_failed`, never `seed_failed`."""

    def failing_action(**kwargs: Any) -> None:
        raise RuntimeError("connection reset")

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._call_rabbitmq_action(
            failing_action,
            stage="conformance",
            process_name="mq",
            action="declare",
        )
    assert excinfo.value.code == "store_statement_failed"


def test_measure_postgres_row_counts_passes_read_only_true(tmp_path: Path) -> None:
    """N8, p6-review-r2: the baseline row-count reads (`pg_tables`, `COUNT(*)`) are the other
    read call site named in the fix — pinned directly rather than only indirectly through a
    freeze-baseline integration test."""
    seen: list[bool] = []

    def recording_sql_runner(**kwargs: Any) -> list[tuple[Any, ...]]:
        seen.append(kwargs["read_only"])
        if kwargs["statement"].startswith("SELECT tablename"):
            return [("riders",)]
        return [(3,)]

    creds = pr.EngineCredentials(username="harness", password="pw")
    counts = pr._measure_postgres_row_counts(
        host="localhost",
        port=14000,
        credentials=creds,
        dbname="w0",
        sql_runner=recording_sql_runner,
        process_name="postgres",
    )
    assert counts == {"riders": 3}
    assert seen == [True, True]  # both the table listing AND the COUNT(*) itself.


def test_check_sentinel_redis_pass_and_fail() -> None:
    store = pr.StoreEntry.model_validate(
        {
            "capability": "cache",
            "migrations": [],
            "seed_files": [],
            "baseline": {"strategy": "empty", "inputs_digest": "sha256:" + "b" * 64},
            "sentinel": {"key": "greeting", "expected": "hi"},
        }
    )
    passing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.REDIS,
        host="localhost",
        port=15003,
        dbname=None,
        credentials=None,
        sql_runner=lambda **kwargs: [],
        redis_runner=lambda **kwargs: b"hi",
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert passing is True
    failing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.REDIS,
        host="localhost",
        port=15003,
        dbname=None,
        credentials=None,
        sql_runner=lambda **kwargs: [],
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert failing is False


def test_check_sentinel_rabbitmq_pass_and_fail() -> None:
    store = pr.StoreEntry.model_validate(
        {
            "capability": "queue",
            "migrations": [],
            "seed_files": [],
            "baseline": {
                "strategy": "datadir_copy",
                "inputs_digest": "sha256:" + "c" * 64,
            },
            "sentinel": {"queue": "jobs", "expected_depth": 5},
        }
    )
    creds = pr.EngineCredentials(username="harness", password="pw")
    passing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.RABBITMQ,
        host="localhost",
        port=15003,
        dbname=None,
        credentials=creds,
        sql_runner=lambda **kwargs: [],
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 5,
    )
    assert passing is True
    failing = pr.check_sentinel(
        store,
        engine=pr.ManagedEngine.RABBITMQ,
        host="localhost",
        port=15003,
        dbname=None,
        credentials=creds,
        sql_runner=lambda **kwargs: [],
        redis_runner=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 0,
    )
    assert failing is False


# --- §5.3 baseline freeze -------------------------------------------------------------------------


def test_freeze_baseline_requires_bundle_dir() -> None:
    manifest = _manifest()
    ctx = pr.SpawnContext(
        work_directory=Path("/x"),
        port_plan=pr.plan_ports(manifest, instances=1),
        credentials={},
        secret_values={},
        secret_purposes={},
        bundle_dir=None,
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)
    assert excinfo.value.code == "internal_invariant_violated"


def test_freeze_baseline_waits_for_readiness_before_the_first_statement(
    tmp_path: Path,
) -> None:
    """t1, p6-review-r1: B4's fix is structurally UNOBSERVABLE with a prober that always answers
    `True` on the first call (`_spawn_context`'s own fast-pass default) — this pins that the wait
    is REAL. A prober that fails twice before succeeding must be polled through all of them; the
    manifest declares a `database` readiness entry with a tiny interval so the real `time.sleep`
    between polls stays in the low milliseconds rather than the 30s/0.25s fallback default.
    """
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [
                *body["readiness"],
                {
                    "capability": "database",
                    "timeout_seconds": 5,
                    "interval_seconds": 0.01,
                },
            ],
        }
    )
    probe_calls: list[dict[str, Any]] = []
    remaining_failures = [2]

    def flaky_prober(**kwargs: Any) -> bool:
        probe_calls.append(kwargs)
        if remaining_failures[0] > 0:
            remaining_failures[0] -= 1
            return False
        return True

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        prober=flaky_prober,
        work_directory=tmp_path,
    )
    pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)
    # 2 failures + the passing call — proves `_wait_for_store_ready` actually looped rather than
    # accepting the first `False` as good enough, or skipping the probe entirely.
    assert len(probe_calls) >= 3
    assert remaining_failures[0] == 0
    assert all(
        call["protocol"] is pr.CapabilityProtocol.POSTGRES for call in probe_calls
    )


def test_freeze_baseline_readiness_timeout_is_a_typed_depends_on_timeout(
    tmp_path: Path,
) -> None:
    """B4, p6-review-r1: a store that never becomes ready must fail typed (`depends_on_timeout`,
    §2f — `infrastructure`, retryable), not hang forever or let the next statement run anyway."""
    manifest = _manifest(
        lambda body: {
            **body,
            "readiness": [
                *body["readiness"],
                {
                    "capability": "database",
                    "timeout_seconds": 0.05,
                    "interval_seconds": 0.01,
                },
            ],
        }
    )
    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        prober=lambda **kwargs: False,
        work_directory=tmp_path,
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)
    assert excinfo.value.code == "depends_on_timeout"


def test_freeze_baseline_template_database_seeds_seals_and_measures_row_counts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "migrations": ["db/schema.sql"],
                        "seed_files": ["db/seed.sql"],
                    }
                ]
            },
        }
    )
    # `seeded_tables` tells the spy what `CREATE DATABASE "alk_baseline_postgres"` should start
    # with — simulating "the seed already landed," since the fake `sync_run` for `psql -f ...`
    # below is a structural no-op (same rule as every other engine-binary call in this file).
    sql_spy = SqlSpy(seeded_tables={"alk_baseline_postgres": {"riders"}})
    sql_spy.row_data[("alk_baseline_postgres", "riders")] = 3

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sql_runner=sql_spy,
        sync_run=_fake_sync_run,
        work_directory=tmp_path,
    )
    result = pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)

    assert (
        "postgres" in result.job_shared_handles
    )  # stays running — job-shared for the job.
    record = result.build_output.stores[0]
    assert record.strategy is pr.BaselineStrategy.TEMPLATE_DATABASE
    assert record.baseline_reference == "alk_baseline_postgres"
    assert record.row_counts == {"riders": 3}

    statements = [statement for _, statement in sql_spy.calls]
    assert any(
        s.startswith("CREATE DATABASE") and "TEMPLATE" not in s for s in statements
    )
    # migrations/seed_files apply via `sync_run`, not `sql_runner` — nothing to find in `statements`
    # for them; asserting the CREATE precedes the ALTER is what is left to check here.
    create_index = next(
        i for i, s in enumerate(statements) if s.startswith("CREATE DATABASE")
    )
    alter_index = next(i for i, s in enumerate(statements) if "IS_TEMPLATE" in s)
    assert create_index < alter_index
    assert statements[alter_index] == (
        'ALTER DATABASE "alk_baseline_postgres" WITH IS_TEMPLATE true ALLOW_CONNECTIONS false'
    )


def test_freeze_baseline_seed_commands_run_under_the_stores_declared_user(
    tmp_path: Path,
) -> None:
    """t4 / B2, p6-review-r1: migration/seed files are customer-authored content applied through
    `psql -f`, which honors backslash meta-commands (`\\!`, `\\copy ... program`) — must run under
    the store's declared `svc-data` identity, never the provisioner's own uid, the same privilege
    drop every other untrusted-content execution path in this module already gets
    (`build_commands`, the managed-engine daemon itself)."""
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "migrations": ["db/schema.sql"],
                        "seed_files": [],
                    }
                ]
            },
        }
    )
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "schema.sql").write_text("-- schema\n")
    seed_calls: list[Any] = []

    def recording_sync_run(
        argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        seed_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sync_run=recording_sync_run,
        work_directory=tmp_path,
    )
    from dataclasses import replace as dc_replace

    ctx = dc_replace(
        ctx,
        user_resolver=_fake_user_resolver({"svc-data": (4444, 5555)}),
        chown=lambda path, uid, gid: (
            None
        ),  # a fake uid needs no real os.chown to prove the point.
    )

    pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)

    psql_calls = [
        (argv, kwargs) for argv, kwargs in seed_calls if argv and argv[0] == "psql"
    ]
    assert psql_calls, "expected a psql -f seed invocation"
    _, kwargs = psql_calls[0]
    assert kwargs.get("user") == 4444
    assert kwargs.get("group") == 5555


def test_freeze_baseline_datadir_copy_terminates_the_bootstrap_and_snapshots_the_datadir(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "a" * 64,
                        },
                    }
                ]
            },
        }
    )
    handles: list[FakeHandle] = []

    def runner(
        argv: list[str], *, cwd: Path, env: dict, log_path: Path, user=None, group=None
    ):
        handle = FakeHandle()
        handles.append(handle)
        return handle

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        runner=runner,
        work_directory=tmp_path,
    )
    result = pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)

    assert result.job_shared_handles == {}  # datadir_copy is never job-shared.
    assert len(handles) == 1
    assert (
        handles[0].terminated is True
    )  # the bootstrap instance is stopped after the copy.
    record = result.build_output.stores[0]
    assert record.strategy is pr.BaselineStrategy.DATADIR_COPY
    assert Path(record.baseline_reference) == tmp_path / "managed" / "postgres.baseline"


def test_freeze_baseline_datadir_copy_redis_issues_save_before_terminating(
    tmp_path: Path,
) -> None:
    """Q1, p6-review-r3 (MAJOR): `redis_daemon_argv` disables save points (`--save ""`) — a bare
    SIGTERM shutdown persists nothing, so `_freeze_one_store`'s `DATADIR_COPY` branch must issue
    an explicit synchronous `SAVE` immediately before terminating, or the copied data dir every
    world clones/resets from is an empty baseline."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": [],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    events: list[str] = []

    class RecordingHandle(FakeHandle):
        def terminate(self) -> None:
            events.append("TERMINATE")
            super().terminate()

    def runner(
        argv: list[str], *, cwd: Path, env: dict, log_path: Path, user=None, group=None
    ):
        return RecordingHandle()

    def redis_runner(*, host: str, port: int, command: Any) -> Any:
        events.append(command[0])
        return "hi" if command[0] == "GET" else None

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        runner=runner,
        redis_runner=redis_runner,
        work_directory=tmp_path,
    )
    pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)

    # only "cache" (redis, datadir_copy) is ever terminated at freeze time — postgres stays
    # running (template_database, job-shared), so this is unambiguously the redis command order.
    assert events[-2:] == ["SAVE", "TERMINATE"]


def test_freeze_baseline_empty_strategy_is_a_no_op_capture(tmp_path: Path) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": ["cache/seed.txt"],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    spawned_names: list[str] = []

    def runner(
        argv: list[str], *, cwd: Path, env: dict, log_path: Path, user=None, group=None
    ):
        return FakeHandle()

    def sync_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        spawned_names.append(argv[0])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        runner=runner,
        sync_run=sync_run,
        work_directory=tmp_path,
    )
    result = pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)
    cache_record = next(
        r for r in result.build_output.stores if r.process_name == "cache"
    )
    assert cache_record.strategy is pr.BaselineStrategy.EMPTY
    assert cache_record.baseline_reference == ""
    assert cache_record.row_counts == {}
    assert (
        "redis-cli" not in spawned_names
    )  # nothing seeded at freeze time for `empty`.
    assert "cache" not in result.job_shared_handles


def test_write_build_output_shape(tmp_path: Path) -> None:
    build_output = pr.BuildOutput(
        bundle_digest="sha256:" + "0" * 64,
        stores=[
            pr.StoreBaselineRecord(
                capability="database",
                process_name="postgres",
                engine=pr.ManagedEngine.POSTGRES,
                strategy=pr.BaselineStrategy.TEMPLATE_DATABASE,
                inputs_digest="sha256:" + "a" * 64,
                baseline_reference="alk_baseline_postgres",
                row_counts={"riders": 3},
            )
        ],
        conformance=True,
        conformance_reason=None,
    )
    target = pr.write_build_output(tmp_path, build_output)
    assert target == tmp_path / "artifacts" / "build.json"
    payload = json.loads(target.read_text())
    assert payload["bundle_digest"] == build_output.bundle_digest
    assert payload["conformance"] is True
    assert payload["stores"] == [
        {
            "capability": "database",
            "process_name": "postgres",
            "engine": "postgres",
            "strategy": "template_database",
            "inputs_digest": "sha256:" + "a" * 64,
            "baseline_reference": "alk_baseline_postgres",
            "row_counts": {"riders": 3},
        }
    ]


# --- §4.2 world clone + reset ----------------------------------------------------------------------


def _frozen_template_database(
    manifest: EnvironmentBundleV2,
    tmp_path: Path,
) -> tuple[pr.FreezeResult, pr.SpawnContext, SqlSpy]:
    sql_spy = SqlSpy()
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    result = pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)
    sql_spy.calls.clear()
    return result, ctx, sql_spy


def test_world_clone_template_database_issues_terminate_drop_create_template(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    freeze_result, ctx, sql_spy = _frozen_template_database(manifest, tmp_path)

    result = pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    assert set(result.handles) == {"postgres", "tools-api", "agent"}
    assert (
        result.handles["postgres"] is freeze_result.job_shared_handles["postgres"]
    )  # reused.

    statements = [statement for _, statement in sql_spy.calls]
    assert statements == [
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = 'w0' AND pid <> pg_backend_pid()",
        'DROP DATABASE IF EXISTS "w0"',
        'CREATE DATABASE "w0" TEMPLATE "alk_baseline_postgres"',
    ]


def test_world_clone_datadir_copy_copies_the_baseline_then_renames_to_wn(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "a" * 64,
                        },
                    }
                ]
            },
        }
    )
    sql_spy = SqlSpy()
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    baseline_dir = Path(freeze_result.build_output.stores[0].baseline_reference)
    assert baseline_dir.is_dir()
    sql_spy.calls.clear()

    copy_calls: list[tuple[Path, Path]] = []
    real_copy = pr._copytree_preserving_symlinks

    def recording_copy(src: Path, dst: Path) -> None:
        copy_calls.append((src, dst))
        real_copy(src, dst)

    ctx2 = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sql_runner=sql_spy,
        work_directory=tmp_path,
    )
    from dataclasses import replace as dc_replace

    ctx2 = dc_replace(ctx2, copy=recording_copy)

    result = pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx2,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    assert copy_calls == [(baseline_dir, tmp_path / "worlds" / "w0" / "postgres")]
    statements = [statement for _, statement in sql_spy.calls]
    assert statements == ['ALTER DATABASE "alk_baseline_postgres" RENAME TO "w0"']
    assert "postgres" in result.handles


def test_world_clone_empty_strategy_reseeds_on_every_clone(tmp_path: Path) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": ["cache/seed.txt"],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache" / "seed.txt").write_text("SET greeting hi\n")
    seed_calls: list[Any] = []

    def sync_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seed_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sync_run=sync_run,
        work_directory=tmp_path,
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    assert not any(
        "redis-cli" in argv for argv in seed_calls
    )  # nothing at freeze time.

    pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    assert any(
        "redis-cli" in argv for argv in seed_calls
    )  # (re)established at clone time.


def test_world_clone_empty_strategy_reseed_runs_under_the_stores_declared_user(
    tmp_path: Path,
) -> None:
    """t4 / B2, p6-review-r1: the `empty`-strategy reseed path (`_seal_world_store`'s fallback
    branch) is a SEPARATE call site from `freeze_baseline`'s own seed application — both must
    drop privilege, not just the one exercised by the freeze-time test above."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": ["cache/seed.txt"],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache" / "seed.txt").write_text("SET greeting hi\n")
    seed_calls: list[Any] = []

    def sync_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seed_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sync_run=sync_run,
        work_directory=tmp_path,
    )
    from dataclasses import replace as dc_replace

    ctx = dc_replace(
        ctx,
        user_resolver=_fake_user_resolver({"svc-data": (6666, 7777)}),
        chown=lambda path, uid, gid: (
            None
        ),  # a fake uid needs no real os.chown to prove the point.
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )

    pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    redis_calls = [(argv, kwargs) for argv, kwargs in seed_calls if "redis-cli" in argv]
    assert redis_calls
    _, kwargs = redis_calls[-1]
    assert kwargs.get("user") == 6666
    assert kwargs.get("group") == 7777


def test_world_clone_empty_strategy_requires_bundle_dir_to_reseed(
    tmp_path: Path,
) -> None:
    """Mirrors `freeze_baseline`'s own `bundle_dir` guard: an empty-strategy store re-seeded on
    every clone must fail typed if the context somehow carries no bundle directory, never silently
    resolve `seed_files` against the wrong (cwd) directory."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": ["cache/seed.txt"],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache" / "seed.txt").write_text("SET greeting hi\n")
    ctx = _spawn_context(manifest, bundle_dir=tmp_path, work_directory=tmp_path)
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )

    from dataclasses import replace as dc_replace

    ctx_no_bundle_dir = dc_replace(ctx, bundle_dir=None)
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._clone_or_reset_world(
            manifest,
            0,
            context=ctx_no_bundle_dir,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        )
    assert excinfo.value.code == "internal_invariant_violated"


def test_reset_world_terminates_only_per_world_handles(tmp_path: Path) -> None:
    manifest = _manifest()
    freeze_result, ctx, sql_spy = _frozen_template_database(manifest, tmp_path)
    world0 = pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    old_tools_api = world0.handles["tools-api"]
    old_agent = world0.handles["agent"]
    shared_postgres = world0.handles["postgres"]

    handles, healthy = pr.reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles=world0.handles,
    )
    assert old_tools_api.handle.terminated is True
    assert old_agent.handle.terminated is True
    assert (
        shared_postgres.handle.terminated is False
    )  # job-shared — stays up across a reset.
    assert (
        healthy is True
    )  # SqlSpy answers `SELECT 1` -> 1 by default (no `answers` override).


def test_reset_world_sentinel_failure_reports_unhealthy_without_raising(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    sql_spy = (
        SqlSpy()
    )  # passes at freeze time (default "SELECT 1" -> 1) — m3's own freeze-time
    # sentinel check must NOT be what fails this test; only `reset_world`'s is under test here.
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world0 = pr._clone_or_reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    # Corrupted AFTER the (passing) freeze-time check, simulating something breaking the world's
    # own state before its reset — never matches the sentinel's "1" from here on.
    sql_spy.answers["SELECT 1"] = [(0,)]
    handles, healthy = pr.reset_world(
        manifest,
        0,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles=world0.handles,
    )
    assert healthy is False  # §4.2: a sentinel failure is reported, never raised.


# --- §4 conformance gate -------------------------------------------------------------------------


def test_first_canary_store_prefers_postgres_over_redis_and_rabbitmq() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": [],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "k", "expected": "v"},
                    },
                    body["seed"]["stores"][0],
                ]
            },
        }
    )
    store = pr._first_canary_store(manifest)
    assert store is not None
    assert (
        store.capability == "database"
    )  # postgres, even though redis is listed first.


def test_first_canary_store_is_none_with_no_seed_block() -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "runtime": {**body["runtime"], "kind": "external"},
            "processes": [],
            "seed": None,
        }
    )
    assert pr._first_canary_store(manifest) is None


def _manifest_with_rabbitmq_store() -> EnvironmentBundleV2:
    return _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "queue",
                    "kind": "managed",
                    "engine": "rabbitmq",
                    "version": "3.13",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "queue": {
                    "protocol": "amqp",
                    "service": "queue",
                    "configuration_name": "QUEUE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "queue",
                        "migrations": [],
                        "seed_files": [],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "d" * 64,
                        },
                        "sentinel": {"queue": "jobs", "expected_depth": 0},
                    },
                ]
            },
        }
    )


def test_run_canary_probe_rabbitmq_declares_publishes_and_inspects_without_deleting(
    tmp_path: Path,
) -> None:
    """m8, p6-review-r1: previously only ever INSPECTED a queue nobody had declared — a vacuous
    pass regardless of real isolation. Pins the real sequence: declare+publish in world 0, inspect
    world 1 — real HTTP calls faked, never skipped.

    N15, p6-review-r2 (MINOR): no delete anymore. The delete used to run HERE, before `run_
    conformance_gate`'s own `reset_world` calls for both worlds — by the time `_verify_canary_
    absent` later re-checked world 0, the queue was already gone from THIS unrelated step,
    proving nothing about whether the reset itself actually worked. Cleanup is now the reset's
    job (world 0's rabbitmq is always `datadir_copy`, wiped and restarted from the pristine
    baseline snapshot on every reset), the same mechanism postgres/redis's own canary already
    relied on."""
    manifest = _manifest_with_rabbitmq_store()
    store = manifest.seed.stores[1]  # the rabbitmq store.
    declare_calls: list[dict[str, Any]] = []
    delete_calls: list[dict[str, Any]] = []
    inspected_ports: list[int] = []

    def rabbitmq_declare(**kwargs: Any) -> None:
        declare_calls.append(kwargs)

    def rabbitmq_delete(**kwargs: Any) -> None:
        delete_calls.append(kwargs)

    def rabbitmq_inspector(**kwargs: Any) -> int:
        inspected_ports.append(kwargs["port"])
        return 0  # world 1 never saw the canary — real isolation.

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        work_directory=tmp_path,
        rabbitmq_declare=rabbitmq_declare,
        rabbitmq_delete=rabbitmq_delete,
        rabbitmq_inspector=rabbitmq_inspector,
    )
    result = pr._run_canary_probe(
        manifest, store, pr.ManagedEngine.RABBITMQ, context=ctx
    )
    assert result is True
    assert len(declare_calls) == 1
    assert declare_calls[0]["queue"] == "_alk_conformance"
    assert (
        delete_calls == []
    )  # N15: cleanup deferred to the subsequent reset, not done here.
    port0 = ctx.port_plan.port_for("queue", 0)
    port1 = ctx.port_plan.port_for("queue", 1)
    assert declare_calls[0]["port"] == port0  # declared in world 0.
    assert inspected_ports == [port1]  # inspected in world 1.


def test_run_canary_probe_rabbitmq_fails_when_the_queue_leaks_to_world_1(
    tmp_path: Path,
) -> None:
    """m8: a leaked canary (world 1's inspector reports a message that should not be there) must
    fail the probe, proving the check is not vacuous in the other direction either."""
    manifest = _manifest_with_rabbitmq_store()
    store = manifest.seed.stores[1]
    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        work_directory=tmp_path,
        rabbitmq_declare=lambda **kwargs: None,
        rabbitmq_delete=lambda **kwargs: None,
        rabbitmq_inspector=lambda **kwargs: 1,  # world 1 sees a message: a real leak.
    )
    result = pr._run_canary_probe(
        manifest, store, pr.ManagedEngine.RABBITMQ, context=ctx
    )
    assert result is False


def test_default_rabbitmq_queue_inspector_treats_404_as_zero_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m8, p6-review-r1: a 404 for a nonexistent queue IS "empty" for the canary/sentinel's own
    purposes — the management API's way of saying so, not an error the gate's 'never raises'
    promise should have had to survive only by luck (unreachable via postgres-first preference)."""
    import urllib.error

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(pr.urllib.request, "urlopen", fake_urlopen)
    creds = pr.EngineCredentials(username="harness", password="pw")
    depth = pr.default_rabbitmq_queue_inspector(
        host="localhost",
        port=15003,
        credentials=creds,
        queue="_alk_conformance",
    )
    assert depth == 0


def test_conformance_gate_passes_when_worlds_are_really_isolated(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    sql_spy = (
        SqlSpy()
    )  # `canary_leaks=False` — the default; a real isolation bug would leak.
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world_handles = {
        index: pr._clone_or_reset_world(
            manifest,
            index,
            context=ctx,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        ).handles
        for index in (0, 1)
    }
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        world_handles=world_handles,
    )
    assert (passed, reason) == (True, None)


def test_conformance_gate_fails_and_never_raises_when_isolation_is_broken(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    sql_spy = SqlSpy(canary_leaks=True)  # simulates a real cross-world isolation bug.
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world_handles = {
        index: pr._clone_or_reset_world(
            manifest,
            index,
            context=ctx,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        ).handles
        for index in (0, 1)
    }
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        world_handles=world_handles,
    )
    assert (passed, reason) == (False, "conformance_gate_failed")


def test_conformance_gate_is_vacuously_true_with_no_canary_store(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        lambda body: {
            **body,
            "runtime": {**body["runtime"], "kind": "external"},
            "processes": [],
            "seed": None,
        }
    )
    ctx = _spawn_context(manifest, bundle_dir=tmp_path, work_directory=tmp_path)
    build_output = pr.BuildOutput(bundle_digest=manifest.digest, stores=[])
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=ctx,
        baseline=build_output,
        job_shared_handles={},
        world_handles={},
    )
    assert (passed, reason) == (True, None)


class _StickyCanarySqlSpy(SqlSpy):
    """Simulates a RESET that runs (DROP/CREATE TEMPLATE are still recorded
    normally, so the isolation probe's own w0-vs-w1 read is unaffected) but fails to actually
    clear the conformance canary from the world it was planted in — `_alk_conformance` stays
    `to_regclass`-visible in that one database no matter how many times it gets dropped and
    recreated. Proves `_verify_canary_absent` is load-bearing, not redundant with the per-world
    declared-sentinel check (`SELECT 1`), which never queries the canary table at all."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sticky_dbname: str | None = None

    def __call__(self, **kwargs: Any) -> list[tuple[Any, ...]]:
        statement = kwargs["statement"].strip()
        if statement.startswith("CREATE TABLE") and "_alk_conformance" in statement:
            self._sticky_dbname = kwargs["dbname"]
        rows = super().__call__(**kwargs)
        if (
            statement.startswith("SELECT to_regclass")
            and kwargs["dbname"] == self._sticky_dbname
        ):
            return [(True,)]
        return rows


def test_conformance_gate_fails_when_reset_leaves_the_canary_behind(
    tmp_path: Path,
) -> None:
    """A reset that (for whatever reason) fails to actually clear the reserved
    `_alk_conformance` object must fail the gate via `_verify_canary_absent` — this is the
    project's own named CRITICAL calibration example, "vacuous canary pass" (severity-grading.md).
    Isolation and both worlds' own declared sentinels pass; only the post-reset canary-absence
    check catches it."""
    manifest = _manifest()
    sql_spy = _StickyCanarySqlSpy()
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world_handles = {
        index: pr._clone_or_reset_world(
            manifest,
            index,
            context=ctx,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        ).handles
        for index in (0, 1)
    }
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        world_handles=world_handles,
    )
    assert (passed, reason) == (False, "conformance_gate_failed")


class _BadResetSentinelSqlSpy(SqlSpy):
    """The store's OWN declared sentinel (`SELECT 1`) answers correctly at freeze
    time (queried against the baseline database) but wrong against either world database
    (`w0`/`w1`) — simulating a reset whose reseal left the declared state broken, which
    `reset_world`'s `_check_all_sentinels` is supposed to catch and the gate is supposed to
    escalate via `sentinel_ok`, distinct from (and reached before) the canary-absence check."""

    def __call__(self, **kwargs: Any) -> list[tuple[Any, ...]]:
        if kwargs["statement"].strip() == "SELECT 1" and kwargs["dbname"] in (
            "w0",
            "w1",
        ):
            return [(0,)]
        return super().__call__(**kwargs)


def test_conformance_gate_fails_when_a_worlds_own_sentinel_fails_after_reset(
    tmp_path: Path,
) -> None:
    """`sentinel_ok` from the gate's own per-world `reset_world` calls must actually
    gate the result — a broken reset that fails the store's declared sentinel must degrade
    parallelism, not be silently overridden by an otherwise-clean canary-absence check."""
    manifest = _manifest()
    sql_spy = _BadResetSentinelSqlSpy()
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world_handles = {
        index: pr._clone_or_reset_world(
            manifest,
            index,
            context=ctx,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        ).handles
        for index in (0, 1)
    }
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        world_handles=world_handles,
    )
    assert (passed, reason) == (False, "conformance_gate_failed")


# --- §4 provision / reset / close: ProcessRuntimeProvider -------------------------------------------


def _sql_spy_provider(**overrides: Any) -> pr.ProcessRuntimeProvider:
    kwargs: dict[str, Any] = dict(
        runner=lambda *a, **k: FakeHandle(),
        sync_run=_fake_sync_run,
        sql_runner=SqlSpy(),
        prober=_recording_prober(),  # N2, p6-review-r2: see `_spawn_context`'s own comment.
        rabbitmq_declare=lambda **kwargs: None,
        rabbitmq_delete=lambda **kwargs: None,
        rabbitmq_import=lambda **kwargs: None,
    )
    kwargs.update(overrides)
    return pr.ProcessRuntimeProvider(**kwargs)


def _provision_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """B1, p6-review-r1: `source` (the untrusted checkout) and `bundle_dir` (the verified bundle
    root) are now GENUINELY DISTINCT directories, never the same path passed twice — a regression
    back to conflating them would be invisible if every test fixture kept them identical (t2)."""
    source = tmp_path / "source"
    (source / "services" / "tools-api").mkdir(parents=True)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    return source, bundle_dir


def test_provision_reconciles_to_exactly_w_ready_worlds(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [runtime.world_index for runtime in runtimes] == [0, 1, 2]
    assert len({runtime.runtime_id for runtime in runtimes}) == 3  # never duplicated.
    # t3, p6-review-r1: §4.1 says `provision` "reconciles to exactly `instances` READY worlds" —
    # previously true only of the COUNT, never checked that a returned world is actually `ready`
    # (it was `preparing`, forever, until some OTHER caller happened to call `healthy()`).
    assert all(runtime.state is pr.RuntimeState.READY for runtime in runtimes)
    ports = {runtime.endpoints["tools"].address for runtime in runtimes}
    assert len(ports) == 3  # each world's own tools-api port, all distinct.
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["conformance"] is True
    # m1, p6-review-r1: both degrade causes now land on build.json, not just the gate's own.
    assert build_output["requested_parallelism"] == 3
    assert build_output["effective_parallelism"] == 3
    assert build_output["degrade_reason"] is None


def test_environment_backed_provider_is_created_exposed_and_destroyed(
    tmp_path: Path,
) -> None:
    lifecycle_calls: list[str] = []

    def lifecycle_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if "provider.py" in argv:
            assert kwargs["cwd"] == tmp_path / "build" / "agent"
            assert (kwargs["cwd"] / "provider.py").is_file()
        if argv[-1] == "provision":
            lifecycle_calls.append("provision")
            context = json.loads(
                Path(kwargs["env"]["ALK_PROVIDER_CONTEXT"]).read_text()
            )
            Path(kwargs["env"]["ALK_PROVIDER_OUTPUT"]).write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "provider": "vapi",
                        "attempt_id": context["attempt_id"],
                        "world_id": context["world_id"],
                        "target": {"kind": "assistant", "id": "asst-test"},
                        "resources": [
                            {"kind": "assistant", "id": "asst-test", "owned": True}
                        ],
                        "cleanup": {
                            "receipt_version": "1",
                            "idempotency_key": context["idempotency_key"],
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif argv[-1] == "destroy":
            lifecycle_calls.append("destroy")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    manifest = _manifest(
        lambda body: {
            **body,
            "metadata": {
                "provider_lifecycle": {
                    "type": "vapi",
                    "scope": "world",
                    "process": "agent",
                    "public_capability": "tools",
                    "provision": {"command": ["python", "provider.py", "provision"]},
                    "destroy": {"command": ["python", "provider.py", "destroy"]},
                    "required_secrets": ["VAPI_API_KEY"],
                }
            },
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    (source / "provider.py").write_text("# lifecycle", encoding="utf-8")
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(
        json.dumps({"VAPI_API_KEY": "test-secret"}), encoding="utf-8"
    )
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"VAPI_API_KEY": "target_provider"},
        sync_run=lifecycle_run,
        public_url_resolver=lambda port, _ttl: f"https://signed.example/{port}",
    )

    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert runtimes[0].metadata["provider_target_id"] == "asst-test"
    assert lifecycle_calls == ["provision"]

    asyncio.run(provider.close(work_directory=tmp_path))
    assert lifecycle_calls == ["provision", "destroy"]


def test_provider_import_is_cloned_after_readiness_and_destroyed_from_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def clone(spec, *, context, api_key):
        assert api_key == "test-secret"
        assert context.tool_base_url.startswith("https://signed.example/")
        calls.append(("clone", spec.source_target_id))
        return ProviderProvisionReceipt(
            schema_version="1",
            provider="vapi",
            attempt_id=context.attempt_id,
            world_id=context.world_id,
            target=ProviderTarget(kind="assistant", id="cloned-assistant"),
            resources=[
                ProviderResource(kind="assistant", id="cloned-assistant", owned=True)
            ],
            cleanup=ProviderCleanupReceipt(idempotency_key=context.idempotency_key),
        )

    def destroy(spec, *, receipt, api_key):
        assert api_key == "test-secret"
        calls.append(("destroy", receipt.target.id))

    monkeypatch.setattr(pr, "clone_provider_target", clone)
    monkeypatch.setattr(pr, "destroy_imported_target", destroy)
    manifest = _manifest(
        lambda body: {
            **body,
            "metadata": {
                "provider_import": {
                    "type": "vapi",
                    "source_target_id": "source-assistant",
                    "public_capability": "tools",
                }
            },
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(
        json.dumps({"VAPI_API_KEY": "test-secret"}), encoding="utf-8"
    )
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"VAPI_API_KEY": "target_provider"},
        public_url_resolver=lambda port, _ttl: f"https://signed.example/{port}",
    )

    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )

    assert runtimes[0].metadata["provider_target_id"] == "cloned-assistant"
    assert (
        runtimes[0].metadata["provider_import_source_target_id"] == "source-assistant"
    )
    assert calls == [("clone", "source-assistant")]
    asyncio.run(provider.close(work_directory=tmp_path))
    assert calls == [
        ("clone", "source-assistant"),
        ("destroy", "cloned-assistant"),
    ]


def test_provision_fixed_port_at_w1_records_no_degrade(tmp_path: Path) -> None:
    """`fixed_port` forces `effective_instances=1` regardless of `instances` —
    at `instances=1` that's not a degrade, since requested==effective already. A prior bug copied
    `PortPlan.degraded_reason` verbatim onto `build.json` here, so a job that asked for W=1 and
    got W=1 reported a parallelism degrade that never happened (crashes `hosted_entrypoint`'s
    `parallelism_degraded` emission downstream, whose payload requires `effective < requested`)."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {**body["processes"][1], "fixed_port": 8081},
                body["processes"][2],
            ],
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert [runtime.world_index for runtime in runtimes] == [0]
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["requested_parallelism"] == 1
    assert build_output["effective_parallelism"] == 1
    assert build_output["degrade_reason"] is None


def test_provision_fixed_port_above_w1_records_degrade(tmp_path: Path) -> None:
    """Companion to the test above: at `instances>1` the same `fixed_port` constraint IS a real
    degrade (`effective=1 < requested=3`), so `build.json` must still report it."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {**body["processes"][1], "fixed_port": 8081},
                body["processes"][2],
            ],
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [runtime.world_index for runtime in runtimes] == [0]
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["requested_parallelism"] == 3
    assert build_output["effective_parallelism"] == 1
    assert build_output["degrade_reason"] == "fixed_port"


def test_provision_reads_seed_files_from_bundle_dir_not_source(tmp_path: Path) -> None:
    """B1, p6-review-r1: §2c seed/migration paths are bundle-relative and must resolve against
    the VERIFIED bundle directory, never the untrusted checkout — a bundle declares `migrations:
    ["db/schema.sql"]`; preflight hashes/scans `<bundle_dir>/db/schema.sql`, so reading the same
    relative path from `source` instead would execute bytes nothing ever verified. `source` and
    `bundle_dir` are genuinely different directories and the seed file exists ONLY under
    `bundle_dir` (t2) — a regression back to `bundle_dir=source` is caught by asserting the
    RECORDED `-f` argument, not merely that the fake `sync_run` returned success (which it always
    does regardless of the path)."""
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "migrations": ["db/schema.sql"],
                        "seed_files": [],
                    }
                ]
            },
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    (bundle_dir / "db").mkdir(parents=True)
    (bundle_dir / "db" / "schema.sql").write_text("-- schema\n")

    calls: list[Any] = []
    provider = _sql_spy_provider(
        sync_run=_recording_sync_run(calls),
        secrets_path=tmp_path / "secrets.json",
    )
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    seed_calls = [argv for argv, _ in calls if argv and argv[0] == "psql"]
    assert seed_calls, "expected a psql -f seed invocation"
    applied_file = seed_calls[0][seed_calls[0].index("-f") + 1]
    assert applied_file == str(bundle_dir / "db" / "schema.sql")


def test_provision_empty_strategy_first_clone_reads_seed_files_from_bundle_dir_not_work_directory(
    tmp_path: Path,
) -> None:
    """Mirrors the test above for the OTHER `apply_store_seed` call site — `_seal_world_store`'s
    `empty`-strategy branch, reached on every world clone/reset, not just `freeze_baseline`'s
    once-per-job seed. `strategy: empty` never seeds at freeze (§5.3's own no-op capture; postgres
    itself does not support it, so a redis `cache` store — the same shape the pre-existing
    empty-strategy fixtures use — isolates this second call site cleanly). Every existing
    `empty`-strategy fixture used `bundle_dir == work_directory`, so a regression swapping the two
    there was invisible — genuinely distinct directories here, same proof technique as the
    freeze-path test above. A decoy file of the SAME NAME is planted under `work_directory` too,
    with different content — asserting content, not just presence, is what actually proves the
    code read from `bundle_dir` rather than merely finding a same-named file wherever it looked."""
    manifest = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "cache",
                    "kind": "managed",
                    "engine": "redis",
                    "version": "7",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "cache": {
                    "protocol": "redis",
                    "service": "cache",
                    "configuration_name": "CACHE_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "cache",
                        "migrations": [],
                        "seed_files": ["cache/seed.txt"],
                        "baseline": {
                            "strategy": "empty",
                            "inputs_digest": "sha256:" + "b" * 64,
                        },
                        "sentinel": {"key": "greeting", "expected": "hi"},
                    },
                ]
            },
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    (bundle_dir / "cache").mkdir(parents=True)
    (bundle_dir / "cache" / "seed.txt").write_text("SET greeting hi\n")
    (tmp_path / "cache").mkdir(parents=True)
    (tmp_path / "cache" / "seed.txt").write_text("SET greeting DECOY\n")

    calls: list[Any] = []
    provider = _sql_spy_provider(
        sync_run=_recording_sync_run(calls),
        secrets_path=tmp_path / "secrets.json",
    )
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    redis_calls = [
        (argv, kwargs) for argv, kwargs in calls if argv and "redis-cli" in argv
    ]
    assert redis_calls, (
        "expected a redis-cli seed invocation from the empty-strategy RESET path"
    )
    assert redis_calls[0][1]["input"] == "SET greeting hi\n"


def test_provision_is_idempotent_for_the_same_job_identity(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    spawn_count: list[int] = [0]

    def counting_runner(argv, *, cwd, env, log_path, user=None, group=None):
        spawn_count[0] += 1
        return FakeHandle()

    provider = _sql_spy_provider(
        runner=counting_runner, secrets_path=tmp_path / "secrets.json"
    )
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    count_after_first = spawn_count[0]
    second = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [r.runtime_id for r in first] == [r.runtime_id for r in second]
    assert (
        spawn_count[0] == count_after_first
    )  # nothing re-spawned; every world already `ready`.


def test_provision_hosted_mode_require_declared_user_true_fails_typed_when_unresolvable(
    tmp_path: Path,
) -> None:
    """P5 ledger carry-forward, exercised end to end through `provision()` rather than only at
    `build_process_tree`/`spawn_source_process` directly: the hosted path passes
    `require_declared_user=True`, and a snapshot that somehow lacks a declared `svc-*` user must
    fail typed, never silently fall back to running unprivileged. M2, p6-review-r1: `True` is now
    `provision()`'s own DEFAULT (fail-closed) — passed explicitly here anyway so the test still
    reads as pinning the behavior on its own, not merely relying on the default not having moved."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        user_resolver=lambda name: None,
        secrets_path=tmp_path / "secrets.json",
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
                require_declared_user=True,
            )
        )
    assert excinfo.value.code == "spawn_failed"


def test_provision_require_declared_user_defaults_to_true(tmp_path: Path) -> None:
    """M2, p6-review-r1: `ProcessRuntimeProvider` is hosted-only — a caller that omits the keyword
    entirely must still fail closed, not silently fall back to running everything unprivileged as
    the harness's own `svc-control`. Mirrors the test above but never passes the argument."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        user_resolver=lambda name: None,
        secrets_path=tmp_path / "secrets.json",
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
            )
        )
    assert excinfo.value.code == "spawn_failed"


def test_provision_conformance_degrade_persists_across_a_later_reconcile_call(
    tmp_path: Path,
) -> None:
    """A real bug caught by manual exercise before this test existed: `effective` is recomputed
    fresh from `port_plan` on every call, so a gate failure decided on call 1 was silently
    forgotten on call 2 — worlds beyond index 0 came back even though the gate never re-ran."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        sql_runner=SqlSpy(canary_leaks=True),
        secrets_path=tmp_path / "secrets.json",
    )
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0]
    second = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0]
    assert first[0].runtime_id == second[0].runtime_id
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["degrade_reason"] == "conformance_gate_failed"  # m1


def test_provision_reconcile_at_w1_after_gate_failure_records_no_degrade(
    tmp_path: Path,
) -> None:
    """A sick-world recovery re-call can legitimately pass a smaller `instances` than the job's
    original request (the module's own comment above the reconcile branch). The sticky
    conformance-degrade branch does not know the current call's `requested` — at `instances=1`,
    `effective == requested == 1` already, so re-surfacing the earlier gate failure there would
    write `build.json` a `parallelism_degraded`-shaped record with no valid payload
    (`effective < requested` is required, and 1 < 1 is false)."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        sql_runner=SqlSpy(canary_leaks=True),
        secrets_path=tmp_path / "secrets.json",
    )
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0]
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["degrade_reason"] == "conformance_gate_failed"

    second = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0]
    build_output = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_output["requested_parallelism"] == 1
    assert build_output["effective_parallelism"] == 1
    assert build_output["degrade_reason"] is None


def test_provision_tears_down_before_rebuilding_on_a_bundle_digest_change(
    tmp_path: Path,
) -> None:
    """M6, p6-review-r1: a bundle-digest change (a re-sealed bundle mid-attempt) used to reassign
    this instance's own identity straight over the PREVIOUS job's still-running processes and
    still-allocated ports — §4.1's "never duplicates" broken in the one case this branch exists
    for. Every previously-live handle (job-shared and per-world) must be terminated BEFORE the new
    manifest's own build/freeze ever runs."""
    manifest_a = _manifest()
    manifest_b = _manifest(lambda body: {**body, "digest": "sha256:" + "9" * 64})
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")

    first = asyncio.run(
        provider.provision(
            manifest_a,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    old_world_handles = [
        h for world in provider._world_handles.values() for h in world.values()
    ]
    old_shared_handles = list(provider._job_shared_handles.values())
    assert (
        old_world_handles or old_shared_handles
    )  # something is actually running to tear down.

    second = asyncio.run(
        provider.provision(
            manifest_b,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert all(h.handle.terminated for h in old_world_handles)
    assert all(h.handle.terminated for h in old_shared_handles)
    assert second[0].bundle_digest == manifest_b.digest
    assert second[0].bundle_digest != first[0].bundle_digest


def test_provision_recovers_a_sick_world_via_re_call(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = first[0]
    old_agent_handle = provider._world_handles[0]["agent"]
    old_runtime_id = (
        runtime.runtime_id
    )  # N1, p6-review-r2: snapshotted BEFORE mutation — after
    # N1, `second[0]` and `runtime` are the SAME object, so comparing `second[0].runtime_id !=
    # runtime.runtime_id` post-rebuild would compare an attribute against itself.
    runtime.state = pr.RuntimeState.UNHEALTHY

    second = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert second[0] is runtime  # N1: mutated in place, never replaced.
    assert second[0].runtime_id != old_runtime_id  # rebuilt, not left unhealthy.
    # t3, p6-review-r1: `provision()` now promotes a freshly-(re)built world out of `PREPARING`
    # before returning (the same declared-readiness probe `healthy()` uses) — was `PREPARING`
    # before this fix pass, unconditionally.
    assert second[0].state is pr.RuntimeState.READY
    assert old_agent_handle.handle.terminated is True


def test_reset_transitions_ready_or_unhealthy_per_sentinel(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = runtimes[0]
    asyncio.run(provider.reset(runtime, work_directory=tmp_path))
    assert runtime.state is pr.RuntimeState.READY


def test_ensure_world_mutates_the_same_environment_runtime_object_across_rebuilds(
    tmp_path: Path,
) -> None:
    """N1, p6-review-r2 (BLOCKER), superseding m5's own test: v1.12 §4.5b's live-object model
    ("providers hand out live `EnvironmentRuntime` objects") reads as ONE object per world for
    the provider's whole life. `_ensure_world` used to mint a brand-new `EnvironmentRuntime` on
    every rebuild (m5's own fixture exercised exactly that, holding a deliberately STALE,
    replaced object) — under that shape, a caller holding an EARLIER reference (e.g. `hosted_
    scheduler.py`'s own pool entry, captured right after the first `provision()` call) could never
    see a later rebuild's state land anywhere it could observe. Pins that a sick-world recovery
    rebuild MUTATES the object a caller is already holding, in place — fresh `runtime_id`/
    `endpoints`/`state`, same Python object, visible through the ORIGINAL reference with no
    re-fetch required. `reset()`'s own state write (m5) is a trivial consequence once there is
    only one object to write to.
    """
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    held = first[0]
    old_runtime_id = held.runtime_id
    held.state = (
        pr.RuntimeState.UNHEALTHY
    )  # simulate the scheduler demoting it (§4.5b).

    second = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert second[0] is held  # the SAME object — never replaced.
    assert held.runtime_id != old_runtime_id  # rebuilt: a fresh identity...
    assert (
        held.state is pr.RuntimeState.READY
    )  # ...and promoted — both visible through `held`.

    asyncio.run(provider.reset(held, work_directory=tmp_path))
    assert provider._runtimes[0] is held  # reset() writes through the SAME object too.
    assert held.state is pr.RuntimeState.READY


def test_close_is_idempotent_and_removes_secrets_and_data_directories(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "run-secrets.json"
    secrets_path.write_text("{}")
    provider = _sql_spy_provider(secrets_path=secrets_path)
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    handles = [
        handle
        for world in provider._world_handles.values()
        for handle in world.values()
    ]
    assert handles  # something is actually running before close().
    # B3, p6-review-r1: the secrets file is gone the moment `provision()` loaded it — long before
    # `close()` ever runs (t5). `close()`'s own unlink is the "if still present" backstop only.
    assert not secrets_path.exists()

    asyncio.run(provider.close(work_directory=tmp_path))
    assert all(handle.handle.terminated for handle in handles)
    assert not secrets_path.exists()
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "worlds").exists()
    assert not (tmp_path / "managed").exists()
    for runtime in runtimes:
        assert runtime.state is pr.RuntimeState.STOPPED

    asyncio.run(
        provider.close(work_directory=tmp_path)
    )  # must not raise the second time.


def test_close_removes_a_secrets_file_left_behind_by_a_failed_load(
    tmp_path: Path,
) -> None:
    """`_load_and_delete_secrets` only unlinks `secrets.json` AFTER `json.loads` succeeds — a
    malformed file raises `secrets/spawn_failed` with the file still on disk. Every OTHER test
    that reaches `close()` does so via a successful load, where the file is already gone by the
    time `close()` runs, so its own unlink line is otherwise a no-op across the whole suite;
    plaintext secrets surviving on the sandbox filesystem after a failed provision is exactly what
    that unlink exists to prevent."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text("{not valid json")
    provider = _sql_spy_provider(secrets_path=secrets_path)

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
                require_declared_user=False,
            )
        )
    assert excinfo.value.code == "spawn_failed"
    assert secrets_path.exists()  # the failed load raised before its own unlink ran.

    asyncio.run(provider.close(work_directory=tmp_path))
    assert not secrets_path.exists()


# --- B3/§0.3, p6-review-r1: secrets lifetime and injection through provision() -----------------


def test_secrets_are_loaded_and_the_file_is_gone_before_the_first_process_spawns(
    tmp_path: Path,
) -> None:
    """t5: pins §0.3's lifetime rule end to end — "the provisioner loads this file into memory at
    startup and deletes it immediately after loading, BEFORE ANY CUSTOMER PROCESS STARTS." Records
    whether the secrets file still existed at the moment of the FIRST spawn call (build step,
    managed engine, or source process — whichever the provisioner reaches first)."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "abc123"}))
    existed_at_first_spawn: list[bool] = []

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        existed_at_first_spawn.append(secrets_path.exists())
        return FakeHandle()

    def sync_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        existed_at_first_spawn.append(secrets_path.exists())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    provider = _sql_spy_provider(
        runner=runner, sync_run=sync_run, secrets_path=secrets_path
    )
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert existed_at_first_spawn, "expected at least one spawn/sync_run call"
    assert not any(existed_at_first_spawn), (
        "secrets.json must be gone before the FIRST spawn"
    )
    assert not secrets_path.exists()


def test_secrets_are_re_injected_from_memory_on_reset(tmp_path: Path) -> None:
    """t5: §0.3 — "the in-memory map lives for the whole job — `reset` restarts... re-inject from
    memory." The file is deleted after the first `provision()`, so a `reset()` that still lands
    the secret in the agent's env can only be reading it from the in-memory map, not the disk."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "abc123"}))
    envs_by_argv0: dict[str, dict[str, str]] = {}

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        envs_by_argv0[tuple(argv)] = env
        return FakeHandle()

    provider = _sql_spy_provider(
        runner=runner,
        secrets_path=secrets_path,
        # N10, p6-review-r2: no `/work/job.json` exists in this fixture's `tmp_path` — the
        # constructor override stands in for it, the same way a local/test lane would.
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert (
        not secrets_path.exists()
    )  # gone after the first provision() — nothing left to re-read.

    envs_by_argv0.clear()
    asyncio.run(provider.reset(runtimes[0], work_directory=tmp_path))
    agent_envs = [
        env for argv, env in envs_by_argv0.items() if argv == ("python3", "agent.py")
    ]
    assert agent_envs, "expected the agent process to be respawned by reset()"
    assert agent_envs[0].get("LIVEKIT_API_KEY") == "abc123"


def test_secret_injection_end_to_end_through_provision(tmp_path: Path) -> None:
    """t6: no prior test covered secret injection through `provision()` at all — `select_process_
    secrets` was only ever tested in isolation. A real (on-disk) secrets.json, loaded by a real
    `provision()` call, must land the claimed alias in the claiming process's env and must NOT
    land in a process that never claimed `target_provider`."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "abc123"}))
    envs_by_argv0: dict[tuple[str, ...], dict[str, str]] = {}

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        envs_by_argv0[tuple(argv)] = env
        return FakeHandle()

    provider = _sql_spy_provider(
        runner=runner,
        secrets_path=secrets_path,
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},  # N10, p6-review-r2.
    )
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    agent_env = envs_by_argv0[
        ("python3", "agent.py")
    ]  # claims `target_provider` in `_manifest()`.
    assert agent_env.get("LIVEKIT_API_KEY") == "abc123"
    tools_env = envs_by_argv0[
        ("node", "server.js")
    ]  # claims no purpose in `_manifest()`.
    assert "LIVEKIT_API_KEY" not in tools_env


# --- N10, p6-review-r2 (MAJOR): secret purposes come from the job, not invented ------------------


def test_secrets_with_no_matching_job_purpose_are_dropped_and_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N10, p6-review-r2: every alias used to be relabelled `target_provider` unconditionally,
    which silently defeats `select_process_secrets`'s own `SOURCE_CHECKOUT` exclusion (F13) the
    moment such an alias ever reaches `secrets.json`. An alias with no matching `job.json`/
    `secret_purpose_map` entry must be DROPPED (never injected under a guessed purpose), not
    silently relabelled — and the drop must be logged so a real injection gap is diagnosable."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(
        json.dumps({"LIVEKIT_API_KEY": "abc123", "UNCLAIMED_ALIAS": "xyz"})
    )
    envs_by_argv0: dict[tuple[str, ...], dict[str, str]] = {}

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        envs_by_argv0[tuple(argv)] = env
        return FakeHandle()

    provider = _sql_spy_provider(
        runner=runner,
        secrets_path=secrets_path,
        secret_purpose_map={
            "LIVEKIT_API_KEY": "target_provider"
        },  # UNCLAIMED_ALIAS has no entry.
    )
    with caplog.at_level("WARNING", logger="fi.alk.harness.process_runtime"):
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
                require_declared_user=False,
            )
        )
    agent_env = envs_by_argv0[("python3", "agent.py")]
    assert (
        agent_env.get("LIVEKIT_API_KEY") == "abc123"
    )  # the matched alias still lands.
    assert (
        "UNCLAIMED_ALIAS" not in agent_env
    )  # the unmatched one is never injected anywhere.
    assert any("UNCLAIMED_ALIAS" in record.message for record in caplog.records)


def test_read_job_secret_purposes_reads_agent_secret_refs_from_job_json(
    tmp_path: Path,
) -> None:
    """N10, p6-review-r2: the REAL hosted-path source — `/work/job.json`'s own `agent.secret_refs`
    (§1: `{alias: {manager, key, version, purpose}}`) — not just the constructor override."""
    (tmp_path / "job.json").write_text(
        json.dumps(
            {
                "agent": {
                    "secret_refs": {
                        "LIVEKIT_API_KEY": {
                            "manager": "platform-vault",
                            "key": "k",
                            "version": None,
                            "purpose": "target_provider",
                        },
                    }
                },
            }
        )
    )
    purposes = pr._read_job_secret_purposes(tmp_path)
    assert purposes == {"LIVEKIT_API_KEY": "target_provider"}


def test_read_job_secret_purposes_absent_file_returns_empty_and_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="fi.alk.harness.process_runtime"):
        purposes = pr._read_job_secret_purposes(tmp_path)
    assert purposes == {}
    assert any("job.json" in record.message for record in caplog.records)


def test_load_and_delete_secrets_malformed_job_json_is_typed_not_attribute_error(
    tmp_path: Path,
) -> None:
    """N10, p6-review-r2: `read_text`/`json.loads` failures and a non-dict payload used to escape
    `provision()` untyped (N9's own class of gap) — a non-dict `job.json` used to raise
    `AttributeError` from `raw.items()` deep inside, not the typed failure this pins."""
    (tmp_path / "job.json").write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._read_job_secret_purposes(tmp_path)
    assert excinfo.value.code == "spawn_failed"


def test_load_and_delete_secrets_malformed_secrets_json_is_typed(
    tmp_path: Path,
) -> None:
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text("not valid json{{{")
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr._load_and_delete_secrets(secrets_path, work_directory=tmp_path)
    assert excinfo.value.code == "spawn_failed"


# --- N2, p6-review-r2 (BLOCKER): promote-path polling ---------------------------------------------


def _manifest_with_fast_readiness_poll(
    *, timeout_seconds: float = 0.3
) -> EnvironmentBundleV2:
    """A tiny declared readiness timeout/interval so the poll tests below run in milliseconds, not
    up to `_DEFAULT_STORE_READY_TIMEOUT_SECONDS`'s 30s floor."""
    return _manifest(
        lambda body: {
            **body,
            "readiness": [
                {
                    **body["readiness"][0],
                    "timeout_seconds": timeout_seconds,
                    "interval_seconds": 0.01,
                },
            ],
        }
    )


def _recording_selective_prober(
    *, protocol: CapabilityProtocol, fail_times: int
) -> Any:
    """Like `_recording_prober`, but only ever fails for calls matching `protocol` — every other
    protocol passes immediately. Isolates which call site actually needed to retry, since a bare
    `_recording_prober` shared across the whole `SpawnContext` would also be consumed by `_wait_
    for_store_ready`'s own postgres readiness check at freeze time."""
    calls: list[dict[str, Any]] = []
    remaining = [fail_times]

    def prober(**kwargs: Any) -> bool:
        calls.append(kwargs)
        if kwargs.get("protocol") is protocol and remaining[0] > 0:
            remaining[0] -= 1
            return False
        return True

    prober.calls = calls  # type: ignore[attr-defined]
    return prober


def test_provision_promotion_polls_the_declared_probe_rather_than_sampling_once(
    tmp_path: Path,
) -> None:
    """N2, p6-review-r2 (BLOCKER): `provision()`'s PREPARING->READY promotion used to call
    `probe_runtime_health` exactly once, immediately, with no wait — nothing waits for the DAG's
    TERMINAL process (`spawn_world` only waits on `depends_on` edges), so a bundle whose
    readiness-bearing process is terminal was marked `UNHEALTHY` on every provision regardless of
    how quickly it actually came up. A prober that fails twice before passing proves the
    promotion actually POLLS.

    Q3, p6-review-r3: `_manifest_with_terminal_readiness`, not `_manifest_with_fast_readiness_
    poll` — the latter kept the `tools` readiness entry, backed by `tools-api`, which `agent`'s
    own `depends_on` wait also polls during `spawn_world` and silently ate the scripted failures
    BEFORE the promote-time poll ever ran, so `len(http_calls) >= 3` passed even with a reverted,
    single-sample `_poll_runtime_health` (verified empirically). The terminal-readiness manifest's
    only probe is backed by `agent` itself, which nothing `depends_on` — every HTTP call recorded
    is unambiguously the promote-site poll."""
    manifest = _manifest_with_terminal_readiness(timeout_seconds=0.3)
    source, bundle_dir = _provision_dirs(tmp_path)
    prober = _recording_selective_prober(protocol=CapabilityProtocol.HTTP, fail_times=2)
    provider = _sql_spy_provider(prober=prober, secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert runtimes[0].state is pr.RuntimeState.READY
    http_calls = [
        c for c in prober.calls if c.get("protocol") is CapabilityProtocol.HTTP
    ]
    assert len(http_calls) >= 3  # 2 failures + the passing call.


def test_reset_promotion_polls_the_declared_probe_rather_than_sampling_once(
    tmp_path: Path,
) -> None:
    """N2, p6-review-r2 (BLOCKER): same fix, the OTHER promote site — `reset()`'s post-sentinel
    probe. `template_database` postgres skips `_wait_for_store_ready` entirely on reset (the
    job-shared engine is already running).

    Q3, p6-review-r3: `_manifest_with_terminal_readiness`, same reasoning as the provision-side
    test above — its only readiness probe is backed by `agent`, the DAG's terminal process, which
    nothing `depends_on`, so nothing else on the reset path can consume the flaky prober's failure
    budget before the promotion poll itself runs."""
    manifest = _manifest_with_terminal_readiness(timeout_seconds=0.3)
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = runtimes[0]
    flaky = _recording_selective_prober(protocol=CapabilityProtocol.HTTP, fail_times=2)
    provider._context = dc_replace(provider._context, prober=flaky)

    asyncio.run(provider.reset(runtime, work_directory=tmp_path))
    assert runtime.state is pr.RuntimeState.READY
    http_calls = [
        c for c in flaky.calls if c.get("protocol") is CapabilityProtocol.HTTP
    ]
    assert len(http_calls) >= 3


def _manifest_with_terminal_readiness(
    *, timeout_seconds: float = 0.05
) -> EnvironmentBundleV2:
    """§2a's own documented shape: a readiness-bearing capability backed by the DAG's TERMINAL
    process (`agent` — nothing in `_manifest()`'s topology ever names it in a `depends_on`).
    `wait_for_dependency` never checks it during build; only `probe_runtime_health`'s promote-time
    sweep of `manifest.readiness` does. The pre-existing `tools` entry is dropped so `tools-api`'s
    own depends_on wait (from `agent`) has nothing to poll and returns immediately — build must
    succeed regardless of what the promote-time prober does."""
    return _manifest(
        lambda body: {
            **body,
            "readiness": [
                {
                    "capability": "control",
                    "path": "/health",
                    "timeout_seconds": timeout_seconds,
                    "interval_seconds": 0.01,
                },
            ],
            "capabilities": {
                **body["capabilities"],
                "control": {
                    "protocol": "http",
                    "service": "agent",
                    "configuration_name": None,
                },
            },
        }
    )


def test_provision_promotion_exhausts_the_poll_and_returns_unhealthy_never_raises(
    tmp_path: Path,
) -> None:
    """N2, p6-review-r2 (BLOCKER): exhaustion is a state DECISION, not a provisioning failure —
    `provision()` must return the world `UNHEALTHY`, never raise, when the declared probe never
    passes within its own timeout — pinned against the exact §2a "terminal process" shape N2
    itself names."""
    manifest = _manifest_with_terminal_readiness(timeout_seconds=0.05)
    source, bundle_dir = _provision_dirs(tmp_path)

    def always_fails_http(**kwargs: Any) -> bool:
        return kwargs.get("protocol") is not CapabilityProtocol.HTTP

    provider = _sql_spy_provider(
        prober=always_fails_http,
        secrets_path=tmp_path / "secrets.json",
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert runtimes[0].state is pr.RuntimeState.UNHEALTHY


# --- N3, p6-review-r2 (MAJOR): a failed first provision() must not poison the provider -----------


def test_provision_failed_first_freeze_resets_identity_so_a_retry_retries_the_build(
    tmp_path: Path,
) -> None:
    """N3, p6-review-r2: the job identity (`_manifest`/`_bundle_digest`) used to be committed
    BEFORE `freeze_baseline` ran — a failure there left the instance claiming an identity with no
    build output, so an in-process retry with the SAME bundle took the RECONCILE branch and
    raised `internal_invariant_violated` instead of retrying the build and surfacing the real,
    often-retryable cause. A `sql_runner` that raises only on its FIRST call (freeze's own `CREATE
    DATABASE` for the template) then behaves normally proves the retry re-attempts the build."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    call_count = [0]
    real_spy = SqlSpy()

    def flaky_sql_runner(**kwargs: Any) -> list[tuple[Any, ...]]:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("connection reset by peer")
        return real_spy(**kwargs)

    provider = _sql_spy_provider(
        sql_runner=flaky_sql_runner,
        secrets_path=tmp_path / "secrets.json",
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
                require_declared_user=False,
            )
        )
    assert (
        excinfo.value.code == "store_statement_failed"
    )  # the REAL cause, not a bogus invariant.
    assert (
        provider._manifest is None
    )  # identity reset, not left claiming a half-built job.

    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert runtimes[0].state is pr.RuntimeState.READY


# --- N4, p6-review-r2 (MAJOR): no spawn path may orphan a live engine on a partial failure --------


def _manifest_with_two_postgres_stores() -> EnvironmentBundleV2:
    return _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {
                    "name": "postgres2",
                    "kind": "managed",
                    "engine": "postgres",
                    "version": "16",
                    "user": "svc-data",
                    "depends_on": [],
                },
                *body["processes"][1:],
            ],
            "capabilities": {
                **body["capabilities"],
                "database2": {
                    "protocol": "postgres",
                    "service": "postgres2",
                    "configuration_name": "DATABASE2_URL",
                },
            },
            "seed": {
                "stores": [
                    body["seed"]["stores"][0],
                    {
                        "capability": "database2",
                        "migrations": [],
                        "seed_files": [],
                        "baseline": {
                            "strategy": "template_database",
                            "inputs_digest": "sha256:" + "e" * 64,
                        },
                        "sentinel": {"query": "SELECT 1", "expected": "1"},
                    },
                ]
            },
        }
    )


def test_freeze_baseline_terminates_an_earlier_stores_handle_on_a_later_stores_failure(
    tmp_path: Path,
) -> None:
    """N4, p6-review-r2 (MAJOR): a raise partway through `freeze_baseline`'s per-store loop used
    to leave every EARLIER store's own job-shared engine live and referenced NOWHERE — `self.
    _job_shared_handles` is only ever assigned once this function RETURNS. Store 1 (`postgres`)
    succeeds and is sealed `template_database`; store 2 (`postgres2`) fails its own seed — store
    1's handle must be terminated, not orphaned holding its formula port.
    """
    manifest = _manifest_with_two_postgres_stores()
    real_spy = SqlSpy()

    def flaky_sql_runner(**kwargs: Any) -> list[tuple[Any, ...]]:
        statement = kwargs["statement"]
        if (
            statement.startswith("CREATE DATABASE")
            and "alk_baseline_postgres2" in statement
        ):
            raise RuntimeError("simulated store-2 seed failure")
        return real_spy(**kwargs)

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        sql_runner=flaky_sql_runner,
        work_directory=tmp_path,
    )
    handles: list[Any] = []
    original_runner = ctx.runner

    def recording_runner(*args: Any, **kwargs: Any) -> Any:
        handle = original_runner(*args, **kwargs)
        handles.append(handle)
        return handle

    ctx = dc_replace(ctx, runner=recording_runner)

    with pytest.raises(pr.ProcessRuntimeError):
        pr.freeze_baseline(manifest, bundle_digest=manifest.digest, context=ctx)

    assert len(handles) == 2  # both postgres/postgres2 were actually spawned.
    assert all(handle.terminated for handle in handles)  # neither orphaned.


def test_ensure_world_publishes_partial_handles_so_close_can_terminate_them(
    tmp_path: Path,
) -> None:
    """N4, p6-review-r2 (MAJOR): a mid-clone failure (here: `agent`'s own `depends_on` wait on
    `tools-api` timing out) used to drop `tools-api`'s already-spawned handle on the floor — `self.
    _world_handles[world_index]` was only ever assigned on `_ensure_world`'s SUCCESS path.
    `close()` (or the next reconcile) must be able to terminate it instead of orphaning it.
    """
    manifest = _manifest_with_fast_readiness_poll(timeout_seconds=0.05)
    source, bundle_dir = _provision_dirs(tmp_path)

    def failing_http_prober(**kwargs: Any) -> bool:
        return kwargs.get("protocol") is not CapabilityProtocol.HTTP

    provider = _sql_spy_provider(
        prober=failing_http_prober,
        secrets_path=tmp_path / "secrets.json",
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=1,
                require_declared_user=False,
            )
        )
    assert excinfo.value.code == "depends_on_timeout"
    tools_handle = provider._world_handles[0]["tools-api"]
    assert tools_handle.handle.terminated is False  # not yet — close() has not run.

    asyncio.run(provider.close(work_directory=tmp_path))
    assert tools_handle.handle.terminated is True  # N4: reachable, and now cleaned up.


# --- N9, p6-review-r2 (MAJOR): filesystem failures must not escape untyped -----------------------


def test_spawn_managed_process_data_dir_setup_failure_is_typed_spawn_failed(
    tmp_path: Path,
) -> None:
    """N9, p6-review-r2: §4.6 — filesystem failures during provisioning are `infrastructure`.
    mkdir/chmod/chown for a (re)spawned engine's own data directory used to raise bare, giving
    `provision()`'s caller nothing to map."""
    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="pw")

    def failing_chown(path: Path, uid: int, gid: int) -> None:
        raise PermissionError("simulated: svc-control cannot chown this path")

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        pr.spawn_managed_process(
            postgres,
            port=14000,
            data_dir=tmp_path / "pg",
            credentials=creds,
            runner=lambda *a, **k: FakeHandle(),
            sync_run=_fake_sync_run,
            user_resolver=_fake_user_resolver({"svc-data": (2222, 3333)}),
            chown=failing_chown,
        )
    assert excinfo.value.code == "spawn_failed"
    assert excinfo.value.stage == "spawn"


def test_run_conformance_gate_never_raises_on_a_filesystem_fault_during_reset(
    tmp_path: Path,
) -> None:
    """N9, p6-review-r2: M3's own "TRULY never raises" promise used to hold only for
    `ProcessRuntimeError` — `reset_world` beneath the gate runs `rmtree`/`copytree` (`_seal_world_
    store`'s own `datadir_copy` work), and a filesystem fault there must still degrade the gate,
    never crash the job."""
    manifest = _manifest(
        lambda body: {
            **body,
            "seed": {
                "stores": [
                    {
                        **body["seed"]["stores"][0],
                        "baseline": {
                            "strategy": "datadir_copy",
                            "inputs_digest": "sha256:" + "a" * 64,
                        },
                    }
                ]
            },
        }
    )
    sql_spy = SqlSpy()
    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, sql_runner=sql_spy, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    world_handles = {
        index: pr._clone_or_reset_world(
            manifest,
            index,
            context=ctx,
            baseline=freeze_result.build_output,
            job_shared_handles=freeze_result.job_shared_handles,
            existing_handles={},
        ).handles
        for index in (0, 1)
    }

    def failing_copy(src: Path, dst: Path) -> None:
        raise OSError("simulated: disk full mid-copy")

    broken_ctx = dc_replace(ctx, copy=failing_copy)
    passed, reason = pr.run_conformance_gate(
        manifest,
        context=broken_ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        world_handles=world_handles,
    )
    assert (passed, reason) == (False, "conformance_gate_failed")


# --- N5/N6, p6-review-r2 (MAJOR): rabbitmq.conf carries credentials; env var matches on disk ------


def _rabbitmq_process() -> Any:
    return pr.ManagedProcess.model_validate(
        {
            "name": "queue",
            "kind": "managed",
            "engine": "rabbitmq",
            "version": "3.13",
            "user": "svc-data",
            "depends_on": [],
        }
    )


def test_spawn_managed_process_rabbitmq_conf_carries_credentials(
    tmp_path: Path,
) -> None:
    """N5, p6-review-r2 (MAJOR): a BARE `rabbitmq-server` (no Docker, §0) reads `default_user`/
    `default_pass` from the generated `rabbitmq.conf`, not from `RABBITMQ_DEFAULT_USER`/
    `RABBITMQ_DEFAULT_PASS` (a Docker-entrypoint-only convention) — without this the node
    initializes with the built-in `guest` account and every rabbitmq call this module makes
    (harness-authenticated) 401s. Q4, p6-review-r3: no `loopback_users` assertion — that line
    widened `guest`'s reach instead of narrowing it and was dropped from the generated conf."""
    creds = pr.EngineCredentials(username="harness", password="s3cr3t")
    data_dir = tmp_path / "queue"
    pr.spawn_managed_process(
        _rabbitmq_process(),
        port=15003,
        data_dir=data_dir,
        credentials=creds,
        runner=lambda *a, **k: FakeHandle(),
        sync_run=_fake_sync_run,
    )
    conf_path = data_dir / "rabbitmq.conf"
    conf_text = conf_path.read_text(encoding="utf-8")
    assert "default_user = harness" in conf_text
    assert "default_pass = s3cr3t" in conf_text
    assert "loopback_users" not in conf_text
    assert (
        conf_path.stat().st_mode & 0o777
    ) == 0o600  # carries the password in cleartext.


def test_spawn_managed_process_rabbitmq_config_file_env_matches_the_on_disk_filename_exactly(
    tmp_path: Path,
) -> None:
    """N6, p6-review-r2 (MAJOR): `RABBITMQ_CONFIG_FILE` must carry the FULL path, extension
    included — the pre-3.7 "the server appends .conf itself" behavior this used to depend on is
    not how the catalog's 3.13 works, and the failure is silent (falls back to the default
    management port, 15672 — inside §2b's own per-world port band)."""
    creds = pr.EngineCredentials(username="harness", password="pw")
    data_dir = tmp_path / "queue"
    captured_env: dict[str, str] = {}

    def recording_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured_env.update(env)
        return FakeHandle()

    pr.spawn_managed_process(
        _rabbitmq_process(),
        port=15003,
        data_dir=data_dir,
        credentials=creds,
        runner=recording_runner,
        sync_run=_fake_sync_run,
    )
    conf_path = data_dir / "rabbitmq.conf"
    assert captured_env["RABBITMQ_CONFIG_FILE"] == str(conf_path)
    assert Path(captured_env["RABBITMQ_CONFIG_FILE"]).name == "rabbitmq.conf"
    assert conf_path.exists()


def test_spawn_managed_process_rabbitmq_conf_overwrites_a_datadir_copy_style_pre_populated_file(
    tmp_path: Path,
) -> None:
    """N5, p6-review-r2: a `datadir_copy` restore legitimately copies a PRIOR `rabbitmq.conf` in
    before this function ever runs (M8's own note) — the write must overwrite it (fresh
    credentials/port every spawn), never fail `FileExistsError` the way an `O_EXCL` create would."""
    data_dir = tmp_path / "queue"
    data_dir.mkdir(parents=True)
    (data_dir / "rabbitmq.conf").write_text("stale content from a copied baseline\n")
    creds = pr.EngineCredentials(username="harness", password="fresh-pw")
    pr.spawn_managed_process(
        _rabbitmq_process(),
        port=15003,
        data_dir=data_dir,
        credentials=creds,
        runner=lambda *a, **k: FakeHandle(),
        sync_run=_fake_sync_run,
    )
    conf_text = (data_dir / "rabbitmq.conf").read_text(encoding="utf-8")
    assert "fresh-pw" in conf_text
    assert "stale content" not in conf_text


# --- Q2, p6-review-r3 (MAJOR): rabbitmq's mnesia data path must not depend on the node name -------


def test_rabbitmq_daemon_env_mnesia_dir_is_node_name_free(tmp_path: Path) -> None:
    """Q2, p6-review-r3 (MAJOR): the port-derived `RABBITMQ_NODENAME` must not leak into the mnesia
    DATA path — `RABBITMQ_MNESIA_DIR` is a fixed path under `data_dir`, independent of port/node
    name. R2, p6-review-r4: `RABBITMQ_MNESIA_BASE` stays set alongside it (rabbitmq derives OTHER
    paths — e.g. `RABBITMQ_PLUGINS_EXPAND_DIR` — from the base, not the dir), so every path-valued
    key the env names, `MNESIA_BASE` included, must still resolve under `data_dir`."""
    creds = pr.EngineCredentials(username="harness", password="pw")
    env = pr.rabbitmq_daemon_env(data_dir=tmp_path, port=15003, credentials=creds)
    assert env["RABBITMQ_MNESIA_DIR"] == str(tmp_path / "mnesia")
    assert (
        "15003" not in env["RABBITMQ_MNESIA_DIR"]
    )  # not derived from the port/node name.
    for key in ("RABBITMQ_MNESIA_DIR", "RABBITMQ_MNESIA_BASE", "RABBITMQ_LOG_BASE"):
        assert Path(env[key]).is_relative_to(tmp_path), (
            f"{key} escapes data_dir: {env[key]}"
        )


def test_rabbitmq_daemon_env_names_the_same_relative_mnesia_path_in_every_world(
    tmp_path: Path,
) -> None:
    """Q2, p6-review-r3 (MAJOR): world 0's bootstrap and world 1's restored instance boot on
    DIFFERENT ports (rabbitmq is never job-shared, §2b), yet both name `<their own data_dir>/
    mnesia` for `RABBITMQ_MNESIA_DIR` — the port/node name no longer changes which relative path
    the daemon looks under. This is necessary, not sufficient, for the copied baseline to actually
    load past world 0 — see p6-review-r4 R1 (mnesia's on-disk schema is still node-name-bound)."""
    manifest = _manifest_with_rabbitmq_store()
    envs: list[dict[str, str]] = []

    def runner(
        argv: list[str], *, cwd: Path, env: dict, log_path: Path, user=None, group=None
    ):
        envs.append(env)
        return FakeHandle()

    ctx = _spawn_context(
        manifest, bundle_dir=tmp_path, runner=runner, work_directory=tmp_path
    )
    freeze_result = pr.freeze_baseline(
        manifest, bundle_digest=manifest.digest, context=ctx
    )
    bootstrap_env = next(e for e in envs if "RABBITMQ_MNESIA_DIR" in e)

    envs.clear()
    pr._clone_or_reset_world(
        manifest,
        1,
        context=ctx,
        baseline=freeze_result.build_output,
        job_shared_handles=freeze_result.job_shared_handles,
        existing_handles={},
    )
    world1_env = next(e for e in envs if "RABBITMQ_MNESIA_DIR" in e)

    bootstrap_data_dir = tmp_path / "managed" / "queue"
    world1_data_dir = tmp_path / "worlds" / "w1" / "queue"
    assert bootstrap_env["RABBITMQ_MNESIA_DIR"] == str(bootstrap_data_dir / "mnesia")
    assert world1_env["RABBITMQ_MNESIA_DIR"] == str(world1_data_dir / "mnesia")
    # different ports still mean different node names (epmd registration) — only the DATA path,
    # not the identity, had to stop depending on it.
    assert world1_env["RABBITMQ_NODENAME"] != bootstrap_env["RABBITMQ_NODENAME"]


# --- N7, p6-review-r2 (MAJOR): rabbitmq readiness must probe BOTH listeners -----------------------


def test_wait_for_store_ready_probes_both_amqp_and_management_listeners_for_rabbitmq(
    tmp_path: Path,
) -> None:
    """N7, p6-review-r2: the AMQP listener comes up during CORE boot; the management plugin's HTTP
    listener comes up in the PLUGIN boot step that follows — the same B4 race, unfixed until now
    for the one engine whose seed/sentinel/canary statements all travel over the listener that was
    never probed."""
    manifest = _manifest_with_rabbitmq_store()
    process = next(p for p in manifest.processes if p.name == "queue")
    probed: list[tuple[Any, int, str | None]] = []

    def recording_prober(**kwargs: Any) -> bool:
        probed.append((kwargs["protocol"], kwargs["port"], kwargs.get("path")))
        return True

    ctx = _spawn_context(
        manifest,
        bundle_dir=tmp_path,
        prober=recording_prober,
        work_directory=tmp_path,
    )
    port = ctx.port_plan.port_for("queue", 0)
    pr._wait_for_store_ready(manifest, process, port=port, context=ctx)

    seen = {(protocol, p) for protocol, p, _ in probed}
    assert (pr.CapabilityProtocol.AMQP, port) in seen
    management_port = pr._rabbitmq_management_port(port)
    assert (CapabilityProtocol.HTTP, management_port) in seen
    http_paths = [
        path for protocol, _, path in probed if protocol is CapabilityProtocol.HTTP
    ]
    assert "/api/overview" in http_paths


# --- N11, p6-review-r2 (MAJOR): a dead job-shared engine must be respawned + every world resealed -


def test_reset_respawns_a_dead_job_shared_engine_and_reseals_every_tracked_world(
    tmp_path: Path,
) -> None:
    """N11, p6-review-r2 (MAJOR): a job-shared `template_database` engine (postgres, in
    `_manifest()`) that dies mid-job used to be carried forward as if healthy — every subsequent
    `reset()`/reconcile then raised `store_statement_failed` against a data directory sitting
    right there on disk. Respawned from that surviving data directory, then EVERY world this
    provider currently tracks (not just the one being reset) has its logical DB re-sealed from the
    template — `reset()` is only called for world 0 here; world 1's own reseal can only have come
    from the respawn's own multi-world loop.
    """
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    sql_spy = SqlSpy()
    provider = _sql_spy_provider(
        sql_runner=sql_spy, secrets_path=tmp_path / "secrets.json"
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    old_shared = provider._job_shared_handles["postgres"]
    old_shared.handle.terminated = (
        True  # simulate the shared postgres dying (e.g. OOM-killed).
    )
    sql_spy.calls.clear()

    asyncio.run(provider.reset(runtimes[0], work_directory=tmp_path))

    new_shared = provider._job_shared_handles["postgres"]
    assert new_shared is not old_shared  # respawned, not silently reused dead.
    statements = [statement for _, statement in sql_spy.calls]
    assert 'CREATE DATABASE "w0" TEMPLATE "alk_baseline_postgres"' in statements
    assert (
        'CREATE DATABASE "w1" TEMPLATE "alk_baseline_postgres"' in statements
    )  # never the
    # `reset()` TARGET — only reachable via the respawn's own reseal-every-tracked-world loop.
    assert runtimes[0].state is pr.RuntimeState.READY


def test_dead_job_shared_engine_respawn_failure_is_typed_naming_the_engine(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    old_shared = provider._job_shared_handles["postgres"]
    old_shared.handle.terminated = True

    def failing_runner(argv, *, cwd, env, log_path, user=None, group=None):
        raise OSError("no such device")

    provider._context = dc_replace(provider._context, runner=failing_runner)

    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(provider.reset(runtimes[0], work_directory=tmp_path))
    assert excinfo.value.code == "store_statement_failed"
    assert excinfo.value.process == "postgres"


# --- N12/N13, p6-review-r2 (MINOR): termination order, SIGINT for postgres, kill escalation -------


def test_close_terminates_in_reverse_topological_order(tmp_path: Path) -> None:
    """N12, p6-review-r2: dependency-first termination (`spawn_world`'s own insertion order) sends
    SIGTERM to a per-world engine while its own dependents may still hold open connections,
    guaranteeing the full escalation wait every time. Reversed so dependents (`agent`, `tools-api`)
    are torn down BEFORE the engine they depend on (`postgres`)."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    terminate_order: list[str] = []

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        handle = FakeHandle()
        name = Path(cwd).name

        def recording_terminate(
            _name: str = name, _handle: FakeHandle = handle
        ) -> None:
            terminate_order.append(_name)
            _handle.terminated = True

        handle.terminate = recording_terminate  # type: ignore[method-assign]
        return handle

    provider = _sql_spy_provider(runner=runner, secrets_path=tmp_path / "secrets.json")
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    terminate_order.clear()
    asyncio.run(provider.close(work_directory=tmp_path))
    per_world_order = [
        name for name in terminate_order if name in ("tools-api", "agent")
    ]
    assert per_world_order == ["agent", "tools-api"]


def test_close_prefers_sigint_over_sigterm_for_postgres(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    interrupted: list[str] = []

    def runner(argv, *, cwd, env, log_path, user=None, group=None):
        handle = FakeHandle()
        name = Path(cwd).name

        def recording_interrupt(
            _name: str = name, _handle: FakeHandle = handle
        ) -> None:
            interrupted.append(_name)
            _handle.interrupted = True
            _handle.terminated = True

        handle.interrupt = recording_interrupt  # type: ignore[method-assign]
        return handle

    provider = _sql_spy_provider(runner=runner, secrets_path=tmp_path / "secrets.json")
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    asyncio.run(provider.close(work_directory=tmp_path))
    assert (
        "postgres" in interrupted
    )  # SIGINT (fast shutdown), never SIGTERM, for postgres.


def test_terminate_and_wait_escalates_to_kill_when_the_process_ignores_terminate() -> (
    None
):
    """N13, p6-review-r2: `FakeHandle.wait` used to report the fake as exited the instant
    `terminate()` was called, so no test ever reached the `kill()` branch — M7's escalation half
    had zero coverage. A handle that ignores `terminate()`/`interrupt()` entirely (`wait()` only
    reports exited once `kill()` has actually run) forces the real escalation path."""

    class StubbornHandle:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def is_running(self) -> bool:
            return not self.killed

        def captured_output(self) -> str:
            return ""

        def terminate(self) -> None:
            self.terminated = (
                True  # acknowledges the signal but does NOT actually exit.
            )

        def interrupt(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> bool:
            return self.killed

        def kill(self) -> None:
            self.killed = True

    handle = StubbornHandle()
    pr._terminate_and_wait(handle, timeout=0.01)
    assert handle.terminated is True
    assert handle.killed is True  # escalation actually happened.
    assert handle.is_running() is False  # reaped.


def test_terminate_and_wait_logs_when_the_process_survives_even_kill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N13, p6-review-r2: the post-`kill()` `wait()` return used to be discarded — a child that
    STILL had not exited after SIGKILL (a wedged kernel wait, nothing left to escalate to) left no
    trace anywhere. Now logged."""

    class UnkillableHandle:
        def is_running(self) -> bool:
            return True

        def captured_output(self) -> str:
            return ""

        def terminate(self) -> None:
            pass

        def interrupt(self) -> None:
            pass

        def wait(self, timeout: float) -> bool:
            return False  # never reports exited, even after kill() below.

        def kill(self) -> None:
            pass

    with caplog.at_level("WARNING", logger="fi.alk.harness.process_runtime"):
        pr._terminate_and_wait(UnkillableHandle(), timeout=0.01)
    assert any(
        "did not exit even after kill" in record.message for record in caplog.records
    )


# --- N14, p6-review-r2 (MINOR): chmod before chown -------------------------------------------------


def test_spawn_managed_process_chmods_before_chowning_the_data_dir(
    tmp_path: Path,
) -> None:
    """N14, p6-review-r2: `chmod` after `chown` cannot succeed once ownership has moved — `svc-
    control` still owns `data_dir` at chmod time and it is free; after chown, a non-root `svc-
    control` that can chown but not re-chmod a path it no longer owns would raise
    `PermissionError`. Pinned by recording the directory's OWN mode at the moment `chown` fires."""
    manifest = _manifest()
    postgres = manifest.processes[0]
    creds = pr.EngineCredentials(username="harness", password="pw")
    data_dir = tmp_path / "pg"
    mode_at_chown_time: list[int] = []

    def recording_chown(path: Path, uid: int, gid: int) -> None:
        if path == data_dir:
            mode_at_chown_time.append(data_dir.stat().st_mode & 0o777)

    pr.spawn_managed_process(
        postgres,
        port=14000,
        data_dir=data_dir,
        credentials=creds,
        runner=lambda *a, **k: FakeHandle(),
        sync_run=_fake_sync_run,
        user_resolver=_fake_user_resolver({"svc-data": (2222, 3333)}),
        chown=recording_chown,
    )
    assert mode_at_chown_time == [0o700]


# --- N16, p6-review-r2 (MINOR): terminate backends before dropping a world's shared logical DB ----


def test_drop_world_shared_databases_terminates_backends_before_dropping(
    tmp_path: Path,
) -> None:
    """N16, p6-review-r2: mirrors `_reset_template_database`'s own sibling call — a lingering
    backend connection on the world's logical DB otherwise blocks this DROP exactly the way it
    would block a reuse-time one, silently re-opening the space leak m4 closed."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    sql_spy = SqlSpy()
    provider = _sql_spy_provider(
        sql_runner=sql_spy, secrets_path=tmp_path / "secrets.json"
    )
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    sql_spy.calls.clear()
    provider._drop_world_shared_databases(1)
    statements = [statement for _, statement in sql_spy.calls]
    assert len(statements) == 2
    assert statements[0].startswith("SELECT pg_terminate_backend")
    assert statements[1] == 'DROP DATABASE IF EXISTS "w1"'


# --- healthy() port method (task 2c) ---------------------------------------------------------------


def test_provider_healthy_port_never_promotes_from_preparing_or_unhealthy(
    tmp_path: Path,
) -> None:
    """Task 2c / p6-review-r2: unlike the module-level `healthy()`, the PORT method must never
    promote from ANY state — not even `preparing`, which only `provision()`'s own poll (N2)
    promotes out of."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = runtimes[0]

    runtime.state = pr.RuntimeState.PREPARING
    ok = asyncio.run(provider.healthy(runtime, work_directory=tmp_path))
    assert ok is True
    assert runtime.state is pr.RuntimeState.PREPARING  # never promoted, even on a pass.

    runtime.state = pr.RuntimeState.UNHEALTHY
    ok = asyncio.run(provider.healthy(runtime, work_directory=tmp_path))
    assert ok is True
    assert runtime.state is pr.RuntimeState.UNHEALTHY  # still never promoted.


def test_provider_healthy_port_demotes_ready_to_unhealthy_on_a_failing_probe(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = runtimes[0]
    assert runtime.state is pr.RuntimeState.READY

    provider._prober = lambda **kwargs: False
    ok = asyncio.run(provider.healthy(runtime, work_directory=tmp_path))
    assert ok is False
    assert runtime.state is pr.RuntimeState.UNHEALTHY


def test_provider_healthy_port_demotes_both_the_providers_record_and_the_callers_object(
    tmp_path: Path,
) -> None:
    """After N1 these are the SAME object, but the port method must not silently rely on that —
    pinned by checking both references independently."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    runtime = runtimes[0]
    provider._prober = lambda **kwargs: False
    asyncio.run(provider.healthy(runtime, work_directory=tmp_path))
    assert runtime.state is pr.RuntimeState.UNHEALTHY
    assert provider._runtimes[0].state is pr.RuntimeState.UNHEALTHY
    assert (
        provider._runtimes[0] is runtime
    )  # N1: one object per world for the provider's life.


def test_provider_healthy_port_raises_typed_before_provision(tmp_path: Path) -> None:
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    fake_runtime = pr.EnvironmentRuntime(
        runtime_id="x",
        world_index=0,
        bundle_digest="sha256:" + "0" * 64,
        state=pr.RuntimeState.PREPARING,
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(provider.healthy(fake_runtime, work_directory=tmp_path))
    assert excinfo.value.code == "internal_invariant_violated"


# --- voice dispatch identity on runtime metadata ---------------------------------------------


def test_spawn_source_process_carries_rendered_dispatch_agent_name(
    tmp_path: Path,
) -> None:
    """The dial identity exists only in the rendered per-world env; the handle must carry the
    RESOLVED value (world index substituted), and carry None when the process declares none."""
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    plan = _solo_port_plan("svc")

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        return FakeHandle()

    with_name = pr.spawn_source_process(
        _source_process(environment={"LIVEKIT_AGENT_NAME": "agent-w{{WORLD_INDEX}}"}),
        build_dir=build_dir,
        world_dir=tmp_path / "worlds" / "w2" / "svc",
        world_index=2,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert with_name.dispatch_agent_name == "agent-w2"

    without = pr.spawn_source_process(
        _source_process(environment={}),
        build_dir=build_dir,
        world_dir=tmp_path / "worlds" / "w0" / "svc",
        world_index=0,
        port_plan=plan,
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert without.dispatch_agent_name is None


def test_spawn_source_process_dispatches_to_effective_injected_agent_name(
    tmp_path: Path,
) -> None:
    """A target override must update both the worker env and the simulator dispatch identity."""
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured.update(env)
        return FakeHandle()

    process = _source_process(
        environment={"LIVEKIT_AGENT_NAME": "generated-w{{WORLD_INDEX}}"},
        secret_purposes=[SecretPurpose.TARGET_PROVIDER],
    )
    spawned = pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=tmp_path / "worlds" / "w0" / "svc",
        world_index=0,
        port_plan=_solo_port_plan("svc"),
        configuration_addresses={},
        secret_values={"LIVEKIT_AGENT_NAME": "customer-agent"},
        secret_purposes={"LIVEKIT_AGENT_NAME": "target_provider"},
        runner=fake_runner,
    )

    assert captured["LIVEKIT_AGENT_NAME"] == "customer-agent"
    assert spawned.dispatch_agent_name == "customer-agent"


def test_spawn_source_process_injects_child_safe_livekit_tool_trace(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build" / "svc"
    build_dir.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_runner(argv, *, cwd, env, log_path, user=None, group=None):
        captured.update(argv=argv, env=env)
        return FakeHandle()

    process = _source_process(
        environment={"HARNESS_TOOL_TRACE": "{{WORLD_DIR}}/agent-tool-calls.jsonl"}
    ).model_copy(update={"run_command": [".venv/bin/python", "agent.py", "start"]})
    world_dir = tmp_path / "worlds" / "w0" / "svc"
    pr.spawn_source_process(
        process,
        build_dir=build_dir,
        world_dir=world_dir,
        world_index=0,
        port_plan=_solo_port_plan("svc"),
        configuration_addresses={},
        secret_values={},
        secret_purposes={},
        runner=fake_runner,
    )
    assert captured["argv"] == [".venv/bin/python", "agent.py", "start"]
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(world_dir)
    assert "ALK_HARNESS_ORIGINAL_ARGV" not in captured["env"]
    assert (world_dir / "sitecustomize.py").is_file()

    # A world reset spawns the process again in the same agent-owned scratch directory. The
    # immutable hook must be reused rather than rewritten.
    (world_dir / "sitecustomize.py").chmod(0o444)
    world_dir.chmod(0o555)
    try:
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=world_dir,
            world_index=0,
            port_plan=_solo_port_plan("svc"),
            configuration_addresses={},
            secret_values={},
            secret_purposes={},
            runner=fake_runner,
        )
    finally:
        world_dir.chmod(0o755)


def test_dispatch_metadata_sets_the_key_only_for_exactly_one_distinct_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zero agents -> absent (the call runner's typed pre-dial failure owns that outcome);
    one distinct name (even from several handles) -> set; conflicting names -> absent, and
    loudly, because the pre-dial message cannot say WHY the key is missing."""

    def handle(name: str | None) -> pr.SpawnedWorldProcess:
        return pr.SpawnedWorldProcess(
            process_name="p",
            handle=FakeHandle(),
            port=1,
            world_index=0,
            dispatch_agent_name=name,
        )

    assert pr._dispatch_metadata({}) == {}
    assert pr._dispatch_metadata({"a": handle(None)}) == {}
    assert pr._dispatch_metadata({"a": handle("agent-w0")}) == {
        "livekit_agent_name": "agent-w0"
    }
    assert pr._dispatch_metadata(
        {"a": handle("agent-w0"), "b": handle("agent-w0")}
    ) == {"livekit_agent_name": "agent-w0"}
    with caplog.at_level("WARNING"):
        assert (
            pr._dispatch_metadata({"a": handle("agent-w0"), "b": handle("other")}) == {}
        )
    assert any(
        "distinct LIVEKIT_AGENT_NAME" in record.message for record in caplog.records
    )


def test_provision_surfaces_dispatch_identity_on_runtime_metadata(
    tmp_path: Path,
) -> None:
    """End to end through provision(): a bundle whose agent process declares LIVEKIT_AGENT_NAME
    lands the per-world RESOLVED value on runtime.metadata, which is exactly where the call
    runner reads its dial identity."""

    def add_dispatch_env(body: dict[str, Any]) -> dict[str, Any]:
        for process in body["processes"]:
            if process["name"] == "agent":
                process["environment"]["LIVEKIT_AGENT_NAME"] = "agent-w{{WORLD_INDEX}}"
        return body

    manifest = _manifest(add_dispatch_env)
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    assert [runtime.metadata for runtime in runtimes] == [
        {"livekit_agent_name": "agent-w0"},
        {"livekit_agent_name": "agent-w1"},
    ]


# ============================================================================================
# Track B-prov: C1 consumability + C2 admission / scan / partial-failure / freeze
# ============================================================================================


def _consumable_manifest(
    *, fixed_port: int = 8080, consumable: bool = True
) -> EnvironmentBundleV2:
    """The shared `_manifest()` with `tools-api` given a (by default consumable) fixed_port."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1] = {
            **body["processes"][1],
            "fixed_port": fixed_port,
            "fixed_port_consumable": consumable,
        }
        return body

    return _manifest(mutate)


# --- item 1: plan_ports consumability (pure) -------------------------------------------------


def test_plan_ports_consumable_at_w_gt_1_allocates_formula_ports_excluding_declared() -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    plan = pr.plan_ports(manifest, instances=4)
    assert plan.effective_instances == 4
    assert plan.degraded_reason is None
    for world_index in range(4):
        port = plan.port_for("tools-api", world_index)
        assert port != 8080  # declared value EXCLUDED at W>1.
        assert port in range(15000, 15800)  # a per-world formula band port.
    # every world, world 0 included, gets its OWN allocated port.
    assert len({plan.port_for("tools-api", w) for w in range(4)}) == 4


def test_plan_ports_consumable_at_w1_honors_the_declared_port() -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    plan = pr.plan_ports(manifest, instances=1)
    assert plan.effective_instances == 1
    assert plan.port_for("tools-api", 0) == 8080  # honored exactly, plan-time W=1 carve-out.


def test_plan_ports_non_consumable_fixed_port_still_forces_effective_one() -> None:
    manifest = _consumable_manifest(fixed_port=8080, consumable=False)
    plan = pr.plan_ports(manifest, instances=4)
    assert plan.effective_instances == 1
    assert plan.degraded_reason == "fixed_port"
    assert plan.port_for("tools-api", 0) == 8080


def test_plan_ports_mixed_bundle_lands_at_effective_one_and_honors_consumable() -> None:
    """A code-fixed sibling forces effective 1; the consumability iff-rule then keys on the
    EFFECTIVE count (=1) and the consumable declaration IS honored (D15)."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        # tools-api consumable @8080; agent code-fixed @9000 (forces effective 1).
        body["processes"][1] = {
            **body["processes"][1],
            "fixed_port": 8080,
            "fixed_port_consumable": True,
        }
        body["processes"][2] = {
            **body["processes"][2],
            "fixed_port": 9000,
            "fixed_port_consumable": False,
        }
        return body

    manifest = _manifest(mutate)
    plan = pr.plan_ports(manifest, instances=4)
    assert plan.effective_instances == 1
    assert plan.degraded_reason == "fixed_port"
    assert plan.port_for("tools-api", 0) == 8080  # consumable honored at effective 1.
    assert plan.port_for("agent", 0) == 9000


# --- item 3: admission math (pure) -----------------------------------------------------------


def test_admit_parallelism_clamps_on_observed_cpu() -> None:
    # cpu_fit = floor((2.1 - 0.5) / (0.6 + 0.2)) = floor(2.0) = 2
    assert pr.admit_parallelism(
        4, cpu_observed=2.1, mem_observed_gib=None, cpu_declared=None, mem_declared_gib=None
    ) == 2


def test_admit_parallelism_falls_back_to_declared_per_dimension() -> None:
    # observed cpu None -> declared 1.0 -> cpu_fit = floor((1-0.5)/0.8)=0 -> W'=max(1,0)=1
    assert pr.admit_parallelism(
        4, cpu_observed=None, mem_observed_gib=8.0, cpu_declared=1.0, mem_declared_gib=None
    ) == 1


def test_admit_parallelism_no_bound_does_not_clamp() -> None:
    assert pr.admit_parallelism(
        4, cpu_observed=None, mem_observed_gib=None, cpu_declared=None, mem_declared_gib=None
    ) == 4


def test_admit_parallelism_never_zero_or_above_requested() -> None:
    assert pr.admit_parallelism(
        4, cpu_observed=0.1, mem_observed_gib=0.1, cpu_declared=None, mem_declared_gib=None
    ) == 1
    assert pr.admit_parallelism(
        2, cpu_observed=64.0, mem_observed_gib=256.0, cpu_declared=None, mem_declared_gib=None
    ) == 2


# --- cgroup-v2 delegated-subtree observation (pure) ------------------------------------------


def _fake_cgroup_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rel: str | None,
    root_files: dict[str, str],
    subtree_files: dict[str, str],
) -> None:
    """Wire `pr`'s cgroup-v2 roots at a faked unified hierarchy: `/proc/self/cgroup` carries the
    `0::<rel>` line (or is absent when `rel is None`), `root_files` land at the hierarchy root and
    `subtree_files` in the delegated `<rel>` subtree."""
    root = tmp_path / "cgroup"
    root.mkdir(parents=True, exist_ok=True)
    for name, text in root_files.items():
        (root / name).write_text(text, encoding="utf-8")
    if rel is not None:
        subtree = root / rel.lstrip("/")
        subtree.mkdir(parents=True, exist_ok=True)
        for name, text in subtree_files.items():
            (subtree / name).write_text(text, encoding="utf-8")
        proc = tmp_path / "proc_self_cgroup"
        proc.write_text(f"0::/{rel.lstrip('/')}\n", encoding="utf-8")
    else:
        proc = tmp_path / "absent_proc_self_cgroup"  # never created.
    monkeypatch.setattr(pr, "_CGROUP_V2_ROOT", root)
    monkeypatch.setattr(pr, "_PROC_SELF_CGROUP", proc)


def test_cgroup_cpu_quota_reads_the_delegated_subtree_not_the_looser_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Root advertises 8 vCPU (host/parent); the sandbox's delegated subtree binds it to 2.
    _fake_cgroup_v2(
        tmp_path,
        monkeypatch,
        rel="sandbox/job",
        root_files={"cpu.max": "800000 100000"},
        subtree_files={"cpu.max": "200000 100000"},
    )
    assert pr._cgroup_cpu_quota() == 2.0  # delegated wins; never the looser 8.0.


def test_cgroup_memory_limit_reads_the_delegated_subtree_not_the_looser_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cgroup_v2(
        tmp_path,
        monkeypatch,
        rel="sandbox/job",
        root_files={"memory.max": str(8 * 1024**3)},  # 8 GiB at the root.
        subtree_files={"memory.max": str(2 * 1024**3)},  # 2 GiB delegated.
    )
    assert pr._cgroup_memory_limit_gib() == 2.0


def test_cgroup_cpu_quota_falls_back_to_root_when_proc_self_cgroup_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No readable /proc/self/cgroup -> the resolver falls back to the hierarchy-root file.
    _fake_cgroup_v2(
        tmp_path,
        monkeypatch,
        rel=None,
        root_files={"cpu.max": "400000 100000"},
        subtree_files={},
    )
    assert pr._cgroup_cpu_quota() == 4.0


def test_cgroup_cpu_quota_unbounded_delegated_max_is_unbounded_not_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A readable delegated `max` (unbounded) is authoritative — it must NOT re-read the root's
    # looser bound.
    _fake_cgroup_v2(
        tmp_path,
        monkeypatch,
        rel="sandbox/job",
        root_files={"cpu.max": "200000 100000"},
        subtree_files={"cpu.max": "max"},
    )
    assert pr._cgroup_cpu_quota() is None


# --- item 2: pre-plan scan (pure) ------------------------------------------------------------


def test_scan_loopback_ports_finds_all_three_forms() -> None:
    found = pr.scan_loopback_ports(
        {
            "A": "http://localhost:8080/x",
            "B": "127.0.0.1:9000",
            "C": "http://[::1]:7000",
            "D": "https://example.com:443",  # not loopback.
        }
    )
    assert found == {"A": {8080}, "B": {9000}, "C": {7000}}


# --- C1 §4 bind-error attribution (pure) -----------------------------------------------------


def test_parse_bind_error_port_reads_the_errored_port() -> None:
    assert pr.parse_bind_error_port("OSError: [Errno 48] Address already in use: 0.0.0.0:8081") == 8081
    assert pr.parse_bind_error_port("started_check timed out") is None


def test_attribute_declared_port_by_consumable_process_is_terminal() -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name="tools-api",
        log_tail="[Errno 48] Address already in use: 127.0.0.1:8080",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=False,
    )
    assert reason == "port_not_consumable"


def test_attribute_formula_port_collision_is_stale_squat_graceful() -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    plan = pr.plan_ports(manifest, instances=2)
    squatted = plan.port_for("tools-api", 1)  # a live formula port.
    reason = pr.attribute_world_start_failure(
        process_name="tools-api",
        log_tail=f"[Errno 48] Address already in use: 127.0.0.1:{squatted}",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=False,
    )
    assert reason == "world_start_failed"


def test_attribute_default_port_by_knob_bearing_worker_is_terminal() -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name="agent",
        log_tail="[Errno 48] Address already in use: 0.0.0.0:8081",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=True,
    )
    assert reason == "port_not_consumable"


def test_attribute_no_bind_evidence_knob_bearing_is_conformance_gate_failed() -> None:
    manifest = _manifest()
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name="agent",
        log_tail="worker crashed for an unrelated reason",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=True,
    )
    assert reason == "conformance_gate_failed"


# --- C1 §4 arm 4 ownership guard (the consumable-declared OWNER of the port) ------------------


def test_attribute_consumable_owner_colliding_on_its_own_port_is_terminal() -> None:
    """(a) the consumable owner binding its OWN declared port at world 1 is the arm-4 defect."""
    manifest = _consumable_manifest(fixed_port=8080)  # tools-api owns consumable 8080.
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name="tools-api",  # the owner.
        log_tail="[Errno 48] Address already in use: 127.0.0.1:8080",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=False,
    )
    assert reason == "port_not_consumable"


def test_attribute_non_owner_on_a_consumable_port_is_not_terminal() -> None:
    """(b) a NON-owner (agent) colliding on tools-api's declared consumable port is NOT the
    arm-4 defect — it falls through to the declared-port graceful arm (world_start_failed)."""
    manifest = _consumable_manifest(fixed_port=8080)  # tools-api owns 8080, agent does not.
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name="agent",  # NOT the owner of 8080.
        log_tail="[Errno 48] Address already in use: 127.0.0.1:8080",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=False,
    )
    assert reason == "world_start_failed"


def test_attribute_unrecoverable_none_process_on_consumable_port_is_not_terminal() -> None:
    """(c) an unrecoverable failure (process_name is None) on a declared consumable port never
    fires the terminal arm — `.get() == None` is False — and falls through gracefully."""
    manifest = _consumable_manifest(fixed_port=8080)
    plan = pr.plan_ports(manifest, instances=2)
    reason = pr.attribute_world_start_failure(
        process_name=None,  # unrecoverable — no owning process to blame.
        log_tail="[Errno 48] Address already in use: 127.0.0.1:8080",
        manifest=manifest,
        port_plan=plan,
        knob_bearing=False,
    )
    assert reason != "port_not_consumable"
    assert reason == "world_start_failed"


# --- integration: admission clamp (item 3) ---------------------------------------------------


def test_provision_admission_clamps_to_one_and_records_resource_limited(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: 1.0,  # cpu_fit = floor((1-0.5)/0.8) = 0 -> W'=1
        mem_observer=lambda: None,
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["requested_parallelism"] == 4
    assert build["effective_parallelism"] == 1
    assert build["degrade_reason"] == "resource_limited"
    assert build["degrade_events"] == [
        {"reason": "resource_limited", "from_w": 4, "to_w": 1}
    ]


def test_provision_admission_falls_back_to_declared_when_observed_read_raises(
    tmp_path: Path,
) -> None:
    (tmp_path / "job.json").write_text(
        json.dumps({"job_id": "j", "runtime": {"cpu_units": 1, "memory_mb": 64000}})
    )
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)

    def boom() -> float | None:
        raise OSError("cgroup read blew up")

    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=boom,  # falls back to declared cpu_units=1 -> cpu_fit 0 -> W'=1
        mem_observer=lambda: None,  # memory declared 64GiB is generous; cpu binds.
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_reason"] == "resource_limited"


def test_provision_no_observed_no_declared_does_not_clamp(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: None,
        mem_observer=lambda: None,
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0, 1, 2]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == []


# --- integration: pre-plan scan (item 2) -----------------------------------------------------


def _no_clamp(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        cpu_observer=lambda: None,
        mem_observer=lambda: None,
        listener_probe=lambda port: False,
    )
    base.update(overrides)
    return base


def test_provision_scan_degrade_tier_declared_port_degrades_to_one(tmp_path: Path) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "http://localhost:8080/tools"}))
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},
        **_no_clamp(),
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0]
    # plan-time W=1 carve-out: the declared port is honored again.
    assert runtimes[0].endpoints["tools"].address == "http://localhost:8080"
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == [
        {"reason": "literal_local_endpoint", "from_w": 4, "to_w": 1}
    ]


def test_provision_scan_warn_tier_non_declared_port_runs_at_requested_w(tmp_path: Path) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "http://localhost:18090"}))
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},
        **_no_clamp(),
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0, 1, 2, 3]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == []  # warn tier never degrades.


def test_provision_scan_skipped_at_w1(tmp_path: Path) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "http://localhost:8080"}))
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},
        **_no_clamp(),
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == []  # no scan at ceiling 1.


def test_provision_per_ref_resolution_gap_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "job.json").write_text(
        json.dumps(
            {
                "job_id": "j",
                "agent": {
                    "secret_refs": {
                        "LIVEKIT_API_KEY": {
                            "manager": "platform-vault",
                            "key": "k",
                            "version": "1",
                            "purpose": "target_provider",
                        }
                    }
                },
            }
        )
    )
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    # secrets.json absent -> the declared ref has no value -> typed job failure, fail closed.
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    with pytest.raises(pr.ProcessRuntimeError):
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=2,
                require_declared_user=False,
            )
        )


# --- integration: partial world-start failure (item 4) ---------------------------------------


def _inject_world_failure(
    provider: pr.ProcessRuntimeProvider,
    fail_index: int,
    *,
    process: str,
    log: str,
) -> None:
    """Wrap `_ensure_world` so a chosen world index raises a typed failure carrying a partial
    handle whose captured log is `log` (the bind-error evidence attribution reads)."""
    original = provider._ensure_world

    def wrapped(world_index: int) -> None:
        if world_index == fail_index:
            exc = pr.ProcessRuntimeError(
                "spawn", "spawn_failed", "world build failed", process=process
            )
            handle = FakeHandle(output=log)
            exc.partial_handles = {  # type: ignore[attr-defined]
                process: pr.SpawnedWorldProcess(
                    process_name=process,
                    handle=handle,
                    port=0,
                    world_index=world_index,
                )
            }
            raise exc
        return original(world_index)

    provider._ensure_world = wrapped  # type: ignore[method-assign]


def test_provision_main_loop_partial_failure_continues_on_prefix(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    _inject_world_failure(provider, 2, process="agent", log="boom (no bind error)")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0, 1]  # contiguous prefix, no renumbering.
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["effective_parallelism"] == 2
    assert build["degrade_events"] == [
        {"reason": "world_start_failed", "from_w": 4, "to_w": 2}
    ]


def test_provision_world_zero_failure_on_first_build_is_a_job_failure(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    _inject_world_failure(provider, 0, process="agent", log="boom")
    with pytest.raises(pr.ProcessRuntimeError):
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=2,
                require_declared_user=False,
            )
        )
    # E=0 is unrepresentable — no degrade event was written.
    build_path = tmp_path / "artifacts" / "build.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        assert build.get("degrade_events", []) == []


# --- integration: port_not_consumable terminal vs world_start_failed (item 5) ----------------


def test_provision_gate_branch_declared_port_bind_by_consumable_is_terminal(
    tmp_path: Path,
) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    _inject_world_failure(
        provider,
        1,
        process="tools-api",
        log="[Errno 48] Address already in use: 127.0.0.1:8080",
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=2,
                require_declared_user=False,
            )
        )
    assert excinfo.value.code == "port_not_consumable"
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    # NO degrade event, no synthetic conformance=False (terminal, not a degrade).
    assert build["degrade_events"] == []
    assert build["conformance"] is None


def test_provision_gate_branch_formula_squat_is_graceful_world_start_failed(
    tmp_path: Path,
) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    squatted = pr.plan_ports(manifest, instances=2).port_for("tools-api", 1)
    _inject_world_failure(
        provider,
        1,
        process="tools-api",
        log=f"[Errno 48] Address already in use: 127.0.0.1:{squatted}",
    )
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0]  # graceful degrade to 1.
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == [
        {"reason": "world_start_failed", "from_w": 2, "to_w": 1}
    ]


def test_provision_gate_listener_check_finds_declared_listener_terminal(
    tmp_path: Path,
) -> None:
    manifest = _consumable_manifest(fixed_port=8080)
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: None,
        mem_observer=lambda: None,
        listener_probe=lambda port: port == 8080,  # a lying consumable left 8080 bound.
    )
    with pytest.raises(pr.ProcessRuntimeError) as excinfo:
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=2,
                require_declared_user=False,
            )
        )
    assert excinfo.value.code == "port_not_consumable"


def test_port_not_consumable_is_in_the_section_2f_table_with_agent_domain() -> None:
    # D28: `port_not_consumable` must live inside §2f's closed code table so the outbound
    # `_section_2f_code` passthrough (hosted_entrypoint) ships the actionable terminal code intact
    # instead of clamping it to `spawn_failed`. Its fallback domain matches the AGENT that
    # `_raise_port_not_consumable` carries directly.
    assert pr.SECTION_2F_DOMAIN["port_not_consumable"] is FailureDomain.AGENT


# --- integration: per-build-identity freeze (item 6) -----------------------------------------


def test_provision_reconcile_carries_the_first_port_plan_forward(tmp_path: Path) -> None:
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=3,
            require_declared_user=False,
        )
    )
    frozen = provider._frozen_port_plan
    # A reconcile call passing a DIFFERENT instances must NOT re-plan.
    asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            require_declared_user=False,
        )
    )
    assert provider._context.port_plan is frozen
    assert provider._context.port_plan.effective_instances == 3


def test_provision_digest_rebuild_carries_ledger_forward_and_reclassifies_ports(
    tmp_path: Path,
) -> None:
    # First build (W=4): a warn-tier secret whose port is NOT declared -> retained, no degrade.
    manifest_a = _manifest()
    manifest_b = _consumable_manifest(fixed_port=18090)  # NEWLY declares the retained port.
    manifest_b = _manifest(
        lambda body: {
            **body,
            "processes": [
                body["processes"][0],
                {**body["processes"][1], "fixed_port": 18090, "fixed_port_consumable": True},
                body["processes"][2],
            ],
            "digest": "sha256:" + "9" * 64,
        }
    )
    source, bundle_dir = _provision_dirs(tmp_path)
    secrets_path = tmp_path / "secrets.json"
    secrets_path.write_text(json.dumps({"LIVEKIT_API_KEY": "http://localhost:18090"}))
    provider = _sql_spy_provider(
        secrets_path=secrets_path,
        secret_purpose_map={"LIVEKIT_API_KEY": "target_provider"},
        **_no_clamp(),
    )
    first = asyncio.run(
        provider.provision(
            manifest_a,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0, 1, 2, 3]  # warn tier: runs at 4.
    assert 18090 in provider._retained_scan_ports

    # Digest rebuild: the new manifest NEWLY declares 18090 -> reclassify -> degrade to 1.
    second = asyncio.run(
        provider.provision(
            manifest_b,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert {"reason": "literal_local_endpoint", "from_w": 4, "to_w": 1} in build[
        "degrade_events"
    ]


def test_provision_digest_rebuild_fresh_admission_above_ceiling_appends_no_resource_limited(
    tmp_path: Path,
) -> None:
    """C2 §6 strictly-decreasing `to_w` invariant across a rebuild: a fresh admission that only
    equals/exceeds the CARRIED ceiling must not append a `resource_limited` entry AFTER an
    existing lower `world_start_failed` entry (which would land a higher `to_w` later in the
    ledger). Stage 1 clamps to the carried ceiling BEFORE attributing resource_limited."""
    manifest_a = _manifest()
    manifest_b = _manifest(lambda body: {**body, "digest": "sha256:" + "9" * 64})
    source, bundle_dir = _provision_dirs(tmp_path)
    cpu_box: dict[str, float | None] = {"value": None}  # first build: no admission clamp.
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: cpu_box["value"],
        mem_observer=lambda: None,
        listener_probe=lambda port: False,
    )
    # First build (W=4): admission does not clamp; world 2 dies -> world_start_failed 4->2.
    _inject_world_failure(provider, 2, process="agent", log="boom (no bind error)")
    first = asyncio.run(
        provider.provision(
            manifest_a,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0, 1]
    assert provider._effective_ceiling == 2

    # Digest rebuild: fresh admission = 3 (cpu_fit floor((3.1-0.5)/0.8)=3) EXCEEDS the carried
    # ceiling of 2, so the min-clamp yields 2 (no further reduction) and NO resource_limited is
    # recorded. The carried world_start_failed entry stays the sole, ordered entry.
    cpu_box["value"] = 3.1
    second = asyncio.run(
        provider.provision(
            manifest_b,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0, 1]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == [
        {"reason": "world_start_failed", "from_w": 4, "to_w": 2}
    ]  # no resource_limited appended; order + strictly-decreasing to_w preserved.


# --- integration: spawn-time override warning (item 8) ---------------------------------------


def test_spawn_time_override_warning_fires_when_secret_overrides_a_guarded_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    process = _source_process(
        name="agent",
        environment={"FI_LOAD_THRESHOLD": "inf"},
        secret_purposes=["target_provider"],
    )
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level("WARNING"):
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=tmp_path / "w0",
            world_index=0,
            port_plan=_solo_port_plan("agent"),
            configuration_addresses={},
            secret_values={"FI_LOAD_THRESHOLD": "0.7"},
            secret_purposes={"FI_LOAD_THRESHOLD": "target_provider"},
            runner=lambda *a, **k: FakeHandle(),
            require_declared_user=False,
        )
    assert any("FI_LOAD_THRESHOLD" in r.message for r in caplog.records)
    # the values themselves must never be logged.
    assert all("0.7" not in r.message for r in caplog.records)


def test_spawn_time_override_warning_silent_when_no_rendered_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    process = _source_process(
        name="agent", environment={}, secret_purposes=["target_provider"]
    )  # no rendered guarded key.
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level("WARNING"):
        pr.spawn_source_process(
            process,
            build_dir=build_dir,
            world_dir=tmp_path / "w0",
            world_index=0,
            port_plan=_solo_port_plan("agent"),
            configuration_addresses={},
            secret_values={"FI_LOAD_THRESHOLD": "0.7"},
            secret_purposes={"FI_LOAD_THRESHOLD": "target_provider"},
            runner=lambda *a, **k: FakeHandle(),
            require_declared_user=False,
        )
    assert not any("FI_LOAD_THRESHOLD" in r.message for r in caplog.records)


def test_provision_two_stage_degrade_yields_two_ordered_ledger_entries(tmp_path: Path) -> None:
    """C2 checklist 15: resource_limited (4->3) then world_start_failed (3->2) → two entries,
    in causal order, requested constant."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: 3.1,  # cpu_fit = floor((3.1-0.5)/0.8) = floor(3.25) = 3 -> W'=3
        mem_observer=lambda: None,
        listener_probe=lambda port: False,
    )
    _inject_world_failure(provider, 2, process="agent", log="boom (no bind error)")
    runtimes = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in runtimes] == [0, 1]
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["degrade_events"] == [
        {"reason": "resource_limited", "from_w": 4, "to_w": 3},
        {"reason": "world_start_failed", "from_w": 3, "to_w": 2},
    ]
    assert build["requested_parallelism"] == 4
    assert build["degrade_reason"] == "world_start_failed"  # final state mirror.


def test_provision_reconcile_failure_re_raises_and_leaves_ledger_unchanged(
    tmp_path: Path,
) -> None:
    """C2 §5 rule 2R: the stage-5 catches are NOT armed on reconcile — a rebuild failure
    re-raises to WorldPool with NO ledger entry and NO ceiling change."""
    manifest = _manifest()
    source, bundle_dir = _provision_dirs(tmp_path)
    provider = _sql_spy_provider(secrets_path=tmp_path / "secrets.json", **_no_clamp())
    first = asyncio.run(
        provider.provision(
            manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=2,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0, 1]
    ledger_before = json.loads(
        (tmp_path / "artifacts" / "build.json").read_text()
    )["degrade_events"]
    # Mark world 1 sick so the reconcile call actually rebuilds it, and make that rebuild fail.
    provider._runtimes[1].state = pr.RuntimeState.UNHEALTHY
    _inject_world_failure(provider, 1, process="agent", log="rebuild failed")
    with pytest.raises(pr.ProcessRuntimeError):
        asyncio.run(
            provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=2,
                require_declared_user=False,
            )
        )
    assert ledger_before == []  # nothing was recorded on the first (clean) build.
    assert provider._effective_ceiling == 2  # unchanged by the reconcile failure.


# --- D40 (per-track review r2): ledger monotonicity + per-attempt state across a rebuild --------


def test_provision_digest_rebuild_normalizes_ledger_to_strictly_decreasing_chain(
    tmp_path: Path,
) -> None:
    """C2 §6 / C4 §2: an in-place `to_w` update on a digest rebuild can lower the always-first
    `resource_limited` entry BELOW a carried `world_start_failed`, leaving the ledger's `to_w`
    non-decreasing. After any mutation the ledger is normalized to a strictly-decreasing chain
    (ordered by `to_w` descending, `from_w` re-derived), regardless of firing order."""
    manifest_a = _manifest()
    manifest_b = _manifest(lambda body: {**body, "digest": "sha256:" + "9" * 64})
    source, bundle_dir = _provision_dirs(tmp_path)
    cpu_box: dict[str, float | None] = {"value": 3.1}  # cpu_fit floor((3.1-0.5)/0.8)=3 -> W'=3.
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: cpu_box["value"],
        mem_observer=lambda: None,
        listener_probe=lambda port: False,
    )
    # First build (W=4): admission clamps to 3 (resource_limited 4->3); world 2 dies afterward
    # (world_start_failed 3->2) -> [resource_limited 4->3, world_start_failed 3->2].
    _inject_world_failure(provider, 2, process="agent", log="boom (no bind error)")
    first = asyncio.run(
        provider.provision(
            manifest_a,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0, 1]
    build_a = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build_a["degrade_events"] == [
        {"reason": "resource_limited", "from_w": 4, "to_w": 3},
        {"reason": "world_start_failed", "from_w": 3, "to_w": 2},
    ]

    # Digest rebuild: fresh admission clamps to 1 (below the carried ceiling of 2). Updating the
    # existing resource_limited entry's to_w to 1 IN PLACE would leave [resource_limited 4->1,
    # world_start_failed 3->2] (to_w 1,2 -- NOT strictly decreasing); normalization reorders by
    # to_w descending and re-derives from_w into a monotone chain.
    cpu_box["value"] = 1.0  # cpu_fit floor((1-0.5)/0.8)=0 -> W'=1.
    second = asyncio.run(
        provider.provision(
            manifest_b,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0]
    events = json.loads((tmp_path / "artifacts" / "build.json").read_text())[
        "degrade_events"
    ]
    assert events == [
        {"reason": "world_start_failed", "from_w": 4, "to_w": 2},
        {"reason": "resource_limited", "from_w": 2, "to_w": 1},
    ]
    # Strictly-decreasing to_w, and each from_w == its predecessor's to_w (requested W leads).
    to_ws = [e["to_w"] for e in events]
    assert to_ws == sorted(to_ws, reverse=True) and len(set(to_ws)) == len(to_ws)
    assert events[0]["from_w"] == 4
    for previous, following in zip(events, events[1:]):
        assert following["from_w"] == previous["to_w"]


def test_provision_failed_rebuild_preserves_carried_ceiling_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 §1 rule 3 / N3: a FAILED digest rebuild must NOT drop the previous successful job
    identity. If it did, the next retry would take the fresh-first-build path -- wiping the ledger
    and letting the ceiling grow back on a fresh admission. Preserving `_manifest`/`_bundle_digest`
    (and the carried ceiling + ledger, never reset here) keeps the retry on the rebuild path."""
    manifest_a = _manifest()
    manifest_b = _manifest(lambda body: {**body, "digest": "sha256:" + "9" * 64})
    source, bundle_dir = _provision_dirs(tmp_path)
    cpu_box: dict[str, float | None] = {"value": 2.1}  # cpu_fit floor((2.1-0.5)/0.8)=2 -> W'=2.
    provider = _sql_spy_provider(
        secrets_path=tmp_path / "secrets.json",
        cpu_observer=lambda: cpu_box["value"],
        mem_observer=lambda: None,
        listener_probe=lambda port: False,
    )
    # First build (W=4): admission clamps to 2 -> resource_limited 4->2, effective ceiling 2.
    first = asyncio.run(
        provider.provision(
            manifest_a,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in first] == [0, 1]
    assert provider._effective_ceiling == 2

    # A digest rebuild whose build fails. cpu is now generous, so a fresh-first-build retry WOULD
    # re-admit to 4 (ceiling grows back) on a wiped ledger -- the exact regression this guards.
    cpu_box["value"] = 100.0
    real_build = pr.build_process_trees
    fail = {"on": True}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if fail["on"]:
            raise OSError("disk full during rebuild")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pr, "build_process_trees", flaky)
    with pytest.raises(pr.ProcessRuntimeError):
        asyncio.run(
            provider.provision(
                manifest_b,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=tmp_path,
                instances=4,
                require_declared_user=False,
            )
        )

    # Retry the SAME rebuild, now succeeding: it must re-enter the rebuild path and carry the
    # ceiling (2) + ledger forward, NOT re-admit up to 4 on a wiped ledger.
    fail["on"] = False
    second = asyncio.run(
        provider.provision(
            manifest_b,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=4,
            require_declared_user=False,
        )
    )
    assert [r.world_index for r in second] == [0, 1]
    assert provider._effective_ceiling == 2
    build = json.loads((tmp_path / "artifacts" / "build.json").read_text())
    assert build["effective_parallelism"] == 2
    assert build["degrade_events"] == [
        {"reason": "resource_limited", "from_w": 4, "to_w": 2}
    ]
