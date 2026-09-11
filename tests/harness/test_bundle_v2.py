"""`futureagi.environment-bundle.v2` model validation, per `hosted-execution-seams.md` v1.7 §2.

Two lanes: the spec's own §2a/§2b/§2c example structures, transcribed here and proven to parse
(the model-layer accept side), against the rejections the model is responsible for on its own —
wrong schema version, an unknown process kind, a store with no sentinel, a capability whose
protocol disagrees with the engine actually backing it, a sentinel shape that disagrees with its
capability's protocol, a strategy the capability's protocol-implied engine does not support, a
store on a capability protocol this module cannot seed at all, unresolved
`service`/`control_service`/capability references (scoped to `kind: process`, per p3-round2's B3),
a duplicate process name, and a resolved secret value anywhere in the manifest.
`compute_inputs_digest` is checked against a hand-computed vector, not by calling back into itself.

Rules that need a repo checkout, the job the bundle will run under, or the §2e checklist
(`compose_not_hosted`, `engine_unsupported`, `no_sql_store`, `depends_on` cycles, placeholder
vocabulary, reserved-name scanning) belong to Phase 4's preflight and are out of scope here.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from fi.alk.harness.bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    BundleRuntimeV2,
    BundleV2Error,
    EnvironmentBundleV2,
    ManagedEngine,
    ManagedProcess,
    SourceProcess,
    StoreEntry,
    compute_inputs_digest,
    load_bundle_v2,
    seal_bundle_v2,
)

# --- §2a/§2b/§2c: the spec's own examples, transcribed verbatim -----------------------------

RUNTIME_EXAMPLE = {"kind": "process", "control_service": "agent", "evidence_seam": "http_tool"}

POSTGRES_PROCESS_EXAMPLE = {
    "name": "postgres",
    "kind": "managed",
    "engine": "postgres",
    "version": "16",
    "user": "svc-data",
    "depends_on": [],
}

TOOLS_API_PROCESS_EXAMPLE = {
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
}

AGENT_PROCESS_EXAMPLE = {
    "name": "agent",
    "kind": "source",
    "working_directory": ".",
    "build_commands": [["pip", "install", "-r", "requirements.txt"]],
    "run_command": ["python", "agent/agent.py", "start"],
    "environment": {
        "DATABASE_URL": "{{DATABASE_URL}}",
        "TOOLS_API_URL": "{{TOOLS_API_URL}}",
        "LIVEKIT_AGENT_NAME": "agent-w{{WORLD_INDEX}}",
    },
    "secret_purposes": ["target_provider"],
    "user": "svc-agent",
    "depends_on": ["postgres", "tools-api"],
}

# §2b catalog-table engines beyond postgres, for the capability-protocol/sentinel/strategy pairing
# tests (F1) — a store's engine now comes from its capability's protocol, so these tests need a
# real managed process of the matching engine for the capability's `service` to resolve to.
REDIS_PROCESS_EXAMPLE = {
    "name": "cache",
    "kind": "managed",
    "engine": "redis",
    "version": "7",
    "user": "svc-data",
    "depends_on": [],
}

RABBITMQ_PROCESS_EXAMPLE = {
    "name": "queue",
    "kind": "managed",
    "engine": "rabbitmq",
    "version": "3.13",
    "user": "svc-data",
    "depends_on": [],
}

# The spec's own `<64-hex>` placeholder, filled with a real hex string — the example is only
# illustrating the digest's shape, not a value to reproduce.
SEED_STORE_EXAMPLE = {
    "capability": "database",
    "migrations": ["db/schema.sql"],
    "seed_files": ["db/seed.sql"],
    "baseline": {
        "strategy": "template_database",
        "inputs_digest": "sha256:" + "a" * 64,
    },
    "sentinel": {"query": "SELECT count(*) FROM riders", "expected": "12"},
}

FULL_MANIFEST_EXAMPLE = {
    "schema_version": BUNDLE_V2_SCHEMA_VERSION,
    "digest": "sha256:" + "0" * 64,
    "name": "demo",
    "runtime": RUNTIME_EXAMPLE,
    "processes": [POSTGRES_PROCESS_EXAMPLE, TOOLS_API_PROCESS_EXAMPLE, AGENT_PROCESS_EXAMPLE],
    "seed": {"stores": [SEED_STORE_EXAMPLE]},
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
    "readiness": [{"capability": "tools", "path": "/healthz"}],
    "files": [{"path": "db/schema.sql", "sha256": "b" * 64, "size": 10}],
    "provenance": {
        "source_kind": "repository",
        "repository": "org/repo",
        "commit": "a" * 40,
        # Bare 64-hex — what v1's `source_fingerprint` producer actually emits, unlike the
        # `sha256:`-prefixed `digest`/`inputs_digest`.
        "source_digest": "c" * 64,
    },
}


def test_the_runtime_example_from_2a_is_accepted() -> None:
    runtime = BundleRuntimeV2.model_validate(RUNTIME_EXAMPLE)
    assert runtime.kind.value == "process"
    assert runtime.evidence_seam.value == "http_tool"


def test_the_managed_process_example_from_2b_is_accepted() -> None:
    process = ManagedProcess.model_validate(POSTGRES_PROCESS_EXAMPLE)
    assert process.engine.value == "postgres"
    assert process.version == "16"


@pytest.mark.parametrize(
    "example", [TOOLS_API_PROCESS_EXAMPLE, AGENT_PROCESS_EXAMPLE], ids=["tools-api", "agent"]
)
def test_the_source_process_examples_from_2b_are_accepted(example: dict) -> None:
    process = SourceProcess.model_validate(example)
    assert process.run_command
    assert process.working_directory == example["working_directory"]


def test_the_seed_store_example_from_2c_is_accepted() -> None:
    store = StoreEntry.model_validate(SEED_STORE_EXAMPLE)
    assert store.baseline.strategy.value == "template_database"
    assert store.sentinel.implied_engine is ManagedEngine.POSTGRES


def test_the_full_manifest_assembled_from_the_spec_examples_is_accepted() -> None:
    bundle = EnvironmentBundleV2.model_validate(FULL_MANIFEST_EXAMPLE)
    assert bundle.name == "demo"
    assert [process.name for process in bundle.processes] == ["postgres", "tools-api", "agent"]
    assert bundle.seed is not None and bundle.seed.stores[0].capability == "database"


# --- rejections the model is responsible for on its own -------------------------------------


def test_wrong_schema_version_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "schema_version": "futureagi.environment-bundle.v1"}
    with pytest.raises(ValidationError, match="bundle_schema_unsupported"):
        EnvironmentBundleV2.model_validate(manifest)


def test_unknown_process_kind_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [{**POSTGRES_PROCESS_EXAMPLE, "kind": "container"}],
    }
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_store_with_no_sentinel_is_rejected() -> None:
    incomplete = {key: value for key, value in SEED_STORE_EXAMPLE.items() if key != "sentinel"}
    with pytest.raises(ValidationError, match="sentinel"):
        StoreEntry.model_validate(incomplete)


# The engine a store answers to is now the engine behind its *capability's protocol* (F1), decided
# in the root validator — so these cases are built as full manifests, not bare `StoreEntry`s: the
# capability, its protocol, and a real process for `service` to resolve to (F5) all have to be in
# scope together for the rule to fire at all.
@pytest.mark.parametrize(
    "process,protocol,sentinel,strategy",
    [
        # redis's sentinel shape only pairs with datadir_copy or empty (§2b's catalog table).
        (REDIS_PROCESS_EXAMPLE, "redis", {"key": "warm", "expected": "1"}, "template_database"),
        # rabbitmq's sentinel shape only pairs with datadir_copy.
        (RABBITMQ_PROCESS_EXAMPLE, "amqp", {"queue": "jobs", "expected_depth": 0}, "empty"),
    ],
    ids=["redis", "rabbitmq"],
)
def test_a_strategy_the_capabilitys_engine_does_not_support_is_rejected(
    process: dict, protocol: str, sentinel: dict, strategy: str
) -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [*FULL_MANIFEST_EXAMPLE["processes"], process],
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "store": {
                "protocol": protocol,
                "service": process["name"],
                "configuration_name": "STORE_URL",
            },
        },
        "seed": {
            "stores": [
                SEED_STORE_EXAMPLE,
                {
                    "capability": "store",
                    "baseline": {"strategy": strategy, "inputs_digest": "sha256:" + "a" * 64},
                    "sentinel": sentinel,
                },
            ]
        },
    }
    with pytest.raises(ValidationError, match="seed_strategy_unsupported"):
        EnvironmentBundleV2.model_validate(manifest)


def test_postgres_excludes_the_empty_strategy() -> None:
    """The one row of §2b's catalog table with no dedicated param above: postgres supports only
    `template_database`/`datadir_copy`, unlike redis and rabbitmq which both accept a store with
    no baseline state at all. Reuses the base manifest's own postgres/database pairing rather than
    adding a process, since postgres already backs its only seed store."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "seed": {
            "stores": [
                {
                    **SEED_STORE_EXAMPLE,
                    "baseline": {"strategy": "empty", "inputs_digest": "sha256:" + "a" * 64},
                }
            ]
        },
    }
    with pytest.raises(ValidationError, match="seed_strategy_unsupported"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_redis_capability_with_a_postgres_shaped_sentinel_is_rejected() -> None:
    """The capability's protocol decides the engine, not the sentinel's own shape — a redis
    capability paired with a postgres-shaped sentinel is a shape mismatch even though the sentinel
    is internally well-formed and the strategy is one postgres would have accepted. The backing
    process is genuinely redis (B1, p3-round2-review, requires protocol/engine agreement before
    this check runs), so this stays isolated to the sentinel-shape question alone."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [*FULL_MANIFEST_EXAMPLE["processes"], REDIS_PROCESS_EXAMPLE],
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "database": {
                **FULL_MANIFEST_EXAMPLE["capabilities"]["database"],
                "protocol": "redis",
                "service": "cache",
            },
        },
    }
    with pytest.raises(ValidationError, match="sentinel_shape_mismatch"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_capability_protocol_that_disagrees_with_its_backing_process_engine_is_rejected() -> None:
    """B1 (p3-round2-review): the sentinel and strategy below are both self-consistent with the
    *declared* protocol (redis) — which is exactly what let this accept an invalid bundle before
    the fix. The capability's `service` actually names the postgres process; only comparing the
    protocol against `ManagedProcess.engine` catches it, since neither the sentinel shape nor the
    strategy pairing is wrong on its own terms."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "database": {**FULL_MANIFEST_EXAMPLE["capabilities"]["database"], "protocol": "redis"},
        },
        "seed": {
            "stores": [
                {
                    "capability": "database",
                    "baseline": {"strategy": "datadir_copy", "inputs_digest": "sha256:" + "a" * 64},
                    "sentinel": {"key": "warm", "expected": "1"},
                }
            ]
        },
    }
    with pytest.raises(ValidationError, match="capability_engine_mismatch"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_store_on_an_unmapped_protocol_capability_is_rejected() -> None:
    """B2 (p3-round2-review): before the fix, a store on a capability outside the
    postgres/redis/amqp map made the engine lookup return `None` and the loop `continue`d,
    skipping the sentinel and strategy checks entirely — a store on the `http` `tools` capability
    validated with any sentinel and any strategy. It must now be rejected explicitly."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "seed": {
            "stores": [
                SEED_STORE_EXAMPLE,
                {
                    "capability": "tools",
                    "baseline": {"strategy": "empty", "inputs_digest": "sha256:" + "a" * 64},
                    "sentinel": {"key": "anything", "expected": "1"},
                },
            ]
        },
    }
    with pytest.raises(ValidationError, match="store_protocol_unsupported"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_capability_engine_mismatch_is_caught_even_with_no_seed_store_at_all() -> None:
    """F19 (p4-round1-review): before the fix, `capability_engine_mismatch` only ran inside the
    `seed.stores` loop — a capability with no store entry at all (used only for a `{{...}}`
    address, never seeded) was never compared, and could silently point `service` at the wrong
    engine."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "cache": {
                "protocol": "redis", "service": "postgres", "configuration_name": "CACHE_URL"
            },
        },
    }
    with pytest.raises(ValidationError, match="capability_engine_mismatch"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_store_whose_capability_service_is_a_source_process_is_rejected() -> None:
    """F19 (p4-round1-review): a postgres-protocol store backed by a source process has no
    managed engine to migrate or seed at all — previously accepted, since only a *wrong* managed
    engine was checked, never a missing one."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "database": {
                **FULL_MANIFEST_EXAMPLE["capabilities"]["database"], "service": "tools-api"
            },
        },
    }
    with pytest.raises(ValidationError, match="store_service_not_managed"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_postgres_capability_with_a_typod_redis_shaped_sentinel_is_rejected() -> None:
    """A postgres capability whose sentinel was typo'd to redis's `{key, expected}` shape is
    rejected naming the sentinel mismatch, not `seed_strategy_unsupported` against an engine
    (`redis`) the bundle never declared — the bug this replaces would have named the wrong one."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "seed": {"stores": [{**SEED_STORE_EXAMPLE, "sentinel": {"key": "warm", "expected": "1"}}]},
    }
    with pytest.raises(ValidationError, match="sentinel_shape_mismatch") as exc_info:
        EnvironmentBundleV2.model_validate(manifest)
    assert "seed_strategy_unsupported" not in str(exc_info.value)


def test_a_sentinel_mixing_two_protocol_shapes_is_rejected() -> None:
    store = {
        "capability": "database",
        "baseline": {"strategy": "empty", "inputs_digest": "sha256:" + "a" * 64},
        "sentinel": {"query": "SELECT 1", "expected": "1", "key": "also-set"},
    }
    with pytest.raises(ValidationError, match="sentinel_shape_invalid"):
        StoreEntry.model_validate(store)


def test_unknown_field_on_a_process_entry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ManagedProcess.model_validate({**POSTGRES_PROCESS_EXAMPLE, "mounts": ["/data"]})


@pytest.mark.parametrize(
    "bad_name", ["/etc", "../../etc", "Tools-Api", "tools_api!", "-leading-dash"],
    ids=["absolute", "traversal", "uppercase", "punctuation", "leading-dash"],
)
def test_a_process_name_outside_the_closed_pattern_is_rejected(bad_name: str) -> None:
    """F3, p5-round1-review — BLOCKER, model layer. `name` is path-joined into
    `/work/build/<name>/` and `/work/worlds/w<N>/<name>/` verbatim (§2b) — an unvalidated name
    used to make `Path("/work/build") / "/etc"` collapse to `Path("/etc")`, which the provisioner
    then `rmtree`'d as svc-control. Both process classes carry the same
    `^[a-z0-9][a-z0-9_-]*$` pattern (§0 v1.8); `process_runtime.py`'s own `_ensure_within` is the
    defense-in-depth backstop for a caller that bypasses this model layer entirely (see
    `test_process_runtime.py`'s `test_build_tree_dir_rejects_a_name_that_escapes_the_work_directory`
    and siblings — "tests both layers," per the worklist)."""
    with pytest.raises(ValidationError, match="process_name_invalid"):
        ManagedProcess.model_validate({**POSTGRES_PROCESS_EXAMPLE, "name": bad_name})
    with pytest.raises(ValidationError, match="process_name_invalid"):
        SourceProcess.model_validate({**AGENT_PROCESS_EXAMPLE, "name": bad_name})


def test_process_names_matching_every_2b_example_are_accepted() -> None:
    """Every example name in §2b's own JSON (`postgres`, `tools-api`, `agent`) must keep working —
    the pattern is closed, not merely restrictive."""
    for name in ("postgres", "tools-api", "agent"):
        assert ManagedProcess.model_validate({**POSTGRES_PROCESS_EXAMPLE, "name": name}).name == name


# --- §2a/§2b/§2d rules the model owns on its own, exercised at manifest scope ----------------


def test_a_process_runtime_without_evidence_seam_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "runtime": {"kind": "process", "control_service": "agent"}}
    with pytest.raises(ValidationError, match="evidence_seam_required"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_process_runtime_with_no_processes_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "processes": []}
    with pytest.raises(ValidationError, match="processes_required"):
        EnvironmentBundleV2.model_validate(manifest)


def test_an_external_runtime_carrying_processes_or_seed_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "runtime": {"kind": "external"}}
    with pytest.raises(ValidationError, match="processes_and_seed_forbidden"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_readiness_probe_naming_an_unresolved_capability_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "readiness": [{"capability": "does-not-exist"}]}
    with pytest.raises(ValidationError, match="capability_unresolved"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_duplicate_process_name_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [*FULL_MANIFEST_EXAMPLE["processes"], dict(POSTGRES_PROCESS_EXAMPLE)],
    }
    with pytest.raises(ValidationError, match="process_name_duplicate"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_capability_service_naming_an_unknown_process_is_rejected() -> None:
    """The underscore/hyphen slip finding 5 names directly: a `service` of `tools_api` against a
    process actually named `tools-api` must be caught here, not surface as an infrastructure
    failure when the provisioner finds nothing to attach the capability to."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "tools": {**FULL_MANIFEST_EXAMPLE["capabilities"]["tools"], "service": "tools_api"},
        },
    }
    with pytest.raises(ValidationError, match="service_unresolved"):
        EnvironmentBundleV2.model_validate(manifest)


def test_an_unresolved_control_service_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "runtime": {**RUNTIME_EXAMPLE, "control_service": "not-a-process"},
    }
    with pytest.raises(ValidationError, match="control_service_unresolved"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_control_service_resolving_to_a_managed_engine_is_rejected() -> None:
    """N9 (p4-round2-review): `control_service` names the agent-side service the world handle and
    evidence seam attach to (§2a) — a datastore in that role is incoherent. For THIS fixture,
    pre-fix, the user-assignment loop below still caught it, just under the wrong code: `postgres`
    got its expected `svc-data` (fine), but `agent` — carrying `user: "svc-agent"` from
    `AGENT_PROCESS_EXAMPLE` — is no longer `control_service`, so the loop demands `svc-tools` for
    it instead and raises `user_assignment_invalid: agent must be svc-tools, got svc-agent` (B1,
    p4-round3-review). The genuine silent-load case needs a bundle where no source process claims
    `svc-agent` at all (e.g. only `postgres` + `tools-api`) — every expectation is then satisfied
    and the bundle loads with no agent-side process in it. This test still discriminates the fix
    either way: the pre-fix code (`user_assignment_invalid`) does not match the post-fix code
    (`control_service_unresolved`), so it goes red on a revert."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "runtime": {**RUNTIME_EXAMPLE, "control_service": "postgres"},
    }
    with pytest.raises(ValidationError, match="control_service_unresolved"):
        EnvironmentBundleV2.model_validate(manifest)


# --- §2b/§0 user assignment (F5, p4-round1-review): control service -> svc-agent, other source ->
# svc-tools, managed engine -> svc-data. Decidable from the manifest's own fields once
# `control_service` is resolved. --------------------------------------------------------------


def test_the_control_service_process_with_the_wrong_user_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [
            POSTGRES_PROCESS_EXAMPLE,
            TOOLS_API_PROCESS_EXAMPLE,
            {**AGENT_PROCESS_EXAMPLE, "user": "svc-tools"},
        ],
    }
    with pytest.raises(ValidationError, match="user_assignment_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_non_control_source_process_claiming_svc_agent_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [
            POSTGRES_PROCESS_EXAMPLE,
            {**TOOLS_API_PROCESS_EXAMPLE, "user": "svc-agent"},
            AGENT_PROCESS_EXAMPLE,
        ],
    }
    with pytest.raises(ValidationError, match="user_assignment_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_managed_engine_with_a_non_svc_data_user_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [
            {**POSTGRES_PROCESS_EXAMPLE, "user": "svc-tools"},
            TOOLS_API_PROCESS_EXAMPLE,
            AGENT_PROCESS_EXAMPLE,
        ],
    }
    with pytest.raises(ValidationError, match="user_assignment_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


# --- §2d configuration_name must not collide with the fixed placeholder vocabulary (F8,
# p4-round1-review) — a collision would render the builtin token instead of the capability's
# address, with no error and no way to spell the intended value. -------------------------------


def test_a_configuration_name_matching_a_fixed_placeholder_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "database": {
                **FULL_MANIFEST_EXAMPLE["capabilities"]["database"],
                "configuration_name": "WORLD_DIR",
            },
        },
    }
    with pytest.raises(ValidationError, match="configuration_name_reserved"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_configuration_name_matching_the_port_host_prefix_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "capabilities": {
            **FULL_MANIFEST_EXAMPLE["capabilities"],
            "database": {
                **FULL_MANIFEST_EXAMPLE["capabilities"]["database"],
                "configuration_name": "PORT_DB",
            },
        },
    }
    with pytest.raises(ValidationError, match="configuration_name_reserved"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_resolved_secret_value_in_process_environment_is_rejected() -> None:
    """v1's manifest-level guard, reapplied: `environment`/`build_environment` are new in v2 and
    are exactly where a resolved credential lands if an authoring stage inlines one instead of
    routing it through `secret_purposes`."""
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "processes": [
            POSTGRES_PROCESS_EXAMPLE,
            TOOLS_API_PROCESS_EXAMPLE,
            {
                **AGENT_PROCESS_EXAMPLE,
                "environment": {
                    **AGENT_PROCESS_EXAMPLE["environment"],
                    "STRIPE_API_KEY": "sk_live_x",
                },
            },
        ],
    }
    with pytest.raises(ValidationError, match="resolved_secret_forbidden"):
        EnvironmentBundleV2.model_validate(manifest)


def test_an_external_runtime_with_a_capability_is_accepted() -> None:
    """B3 (p3-round2-review): `kind: external` has no `processes` array to resolve capability
    `service`/`control_service` against (§2a omits `processes` for it entirely) — the F5
    resolution checks must not run for it, or every external bundle carrying any capability at all
    would be unloadable, which §2a never intended."""
    manifest = {
        "schema_version": BUNDLE_V2_SCHEMA_VERSION,
        "digest": "sha256:" + "0" * 64,
        "name": "external-demo",
        "runtime": {"kind": "external"},
        "capabilities": {
            "target": {
                "protocol": "http",
                "service": "customer-endpoint",
                "configuration_name": "TARGET_URL",
            },
        },
        "provenance": {"source_kind": "remote", "source_digest": "c" * 64},
    }
    bundle = EnvironmentBundleV2.model_validate(manifest)
    assert bundle.runtime.kind.value == "external"
    assert bundle.capabilities["target"].service == "customer-endpoint"


# --- F7 / B6: the digest-shape regexes, exercised on their rejection side too ----------------


def test_a_non_hex_file_sha256_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "files": [{"path": "db/schema.sql", "sha256": "not-hex", "size": 10}],
    }
    with pytest.raises(ValidationError, match="file_sha256_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_non_hex_source_digest_is_rejected() -> None:
    manifest = {
        **FULL_MANIFEST_EXAMPLE,
        "provenance": {**FULL_MANIFEST_EXAMPLE["provenance"], "source_digest": "not-hex"},
    }
    with pytest.raises(ValidationError, match="source_digest_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_malformed_bundle_digest_is_rejected() -> None:
    manifest = {**FULL_MANIFEST_EXAMPLE, "digest": "not-a-digest"}
    with pytest.raises(ValidationError, match="bundle_digest_invalid"):
        EnvironmentBundleV2.model_validate(manifest)


def test_a_malformed_inputs_digest_is_rejected() -> None:
    store = {
        **SEED_STORE_EXAMPLE,
        "baseline": {"strategy": "template_database", "inputs_digest": "not-a-digest"},
    }
    with pytest.raises(ValidationError, match="inputs_digest_invalid"):
        StoreEntry.model_validate(store)


# --- §2c inputs_digest: byte-exact construction, checked against a hand-computed vector -----


def test_compute_inputs_digest_matches_a_hand_computed_vector(tmp_path) -> None:
    (tmp_path / "db").mkdir()
    (tmp_path / "café").mkdir()
    schema = b"CREATE TABLE riders (id int);"
    # Multi-byte content pins content_length as a byte count, not a character count — reading via
    # read_text() + len(text) would compute 8 here instead of 9 and still hash to something, just
    # not this. Path with a non-ASCII segment pins the path encoding as utf-8, not latin-1.
    seed = "-- café\n".encode("utf-8")
    notes = b"-- see docs"
    (tmp_path / "db" / "schema.sql").write_bytes(schema)
    (tmp_path / "db" / "seed.sql").write_bytes(seed)
    (tmp_path / "café" / "notes.sql").write_bytes(notes)

    # Built from §2c's own words, not by calling the helper: for each file in migrations then
    # seed_files, in listed order, <relative_path>\n<content_length>\n<content_bytes>, then
    # <engine>:<version>\n.
    expected = hashlib.sha256()
    expected.update(b"db/schema.sql\n" + str(len(schema)).encode() + b"\n" + schema)
    expected.update(b"db/seed.sql\n" + str(len(seed)).encode() + b"\n" + seed)
    notes_path = "café/notes.sql".encode("utf-8")
    expected.update(notes_path + b"\n" + str(len(notes)).encode() + b"\n" + notes)
    expected.update(b"postgres:16\n")

    got = compute_inputs_digest(
        tmp_path,
        ["db/schema.sql"],
        ["db/seed.sql", "café/notes.sql"],
        engine=ManagedEngine.POSTGRES,
        version="16",
    )
    assert got == "sha256:" + expected.hexdigest()


def test_compute_inputs_digest_is_order_sensitive_not_sorted(tmp_path) -> None:
    """Two migrations reversed must hash differently — order is part of the identity, since
    migrations apply in sequence, not in whatever order sorting would put them in."""
    (tmp_path / "a.sql").write_bytes(b"A")
    (tmp_path / "b.sql").write_bytes(b"B")

    forward = compute_inputs_digest(
        tmp_path, ["a.sql", "b.sql"], [], engine=ManagedEngine.POSTGRES, version="16"
    )
    backward = compute_inputs_digest(
        tmp_path, ["b.sql", "a.sql"], [], engine=ManagedEngine.POSTGRES, version="16"
    )
    assert forward != backward


# --- §2d bundle digest: byte-exact construction, checked against a hand-computed vector (F4,
# p4-round1-review) — the single normative implementation, `seal_bundle_v2`, over `BundleFileV2`
# directly, with no v1 model conversion. -------------------------------------------------------


def test_seal_bundle_v2_matches_a_hand_computed_vector() -> None:
    """Built from §2d's own words (v1.7), not by calling back into `seal_bundle_v2` or into
    `json.dumps` with the same settings: sha256 over the canonical dump of the manifest minus
    `digest`/`files`, then for each `files[]` record, in listed order, the canonical dump of
    `{path, sha256, size}` prefixed by its byte length as 8 bytes big-endian. "Canonical" =
    `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. The literal core
    JSON below is transcribed from a minimal, otherwise-empty manifest's own normalized field set
    (a hand-verified constant, not a derived one) — a real, if trivial, `external`-kind bundle
    with no processes, no seed, no capabilities, and two files. A non-ASCII `name` pins
    `ensure_ascii=False` (an `ensure_ascii=True` implementation would diverge here), and listing
    `b.txt` before `a.txt` pins IN LISTED ORDER against an implementation that silently sorts
    (N6, p4-round2-review)."""
    file_sha_a = "b" * 64
    file_sha_b = "c" * 64
    manifest = EnvironmentBundleV2.model_validate(
        {
            "schema_version": BUNDLE_V2_SCHEMA_VERSION,
            "digest": "sha256:" + "0" * 64,
            "name": "café",
            "runtime": {"kind": "external"},
            "provenance": {"source_kind": "remote", "source_digest": "a" * 64},
            "files": [
                {"path": "b.txt", "sha256": file_sha_b, "size": 5},
                {"path": "a.txt", "sha256": file_sha_a, "size": 3},
            ],
        }
    )

    core_json = (
        '{"capabilities":{},"metadata":{},"name":"café","processes":[],"provenance":'
        '{"adopted_files":[],"commit":null,"generated_files":[],"generator":'
        '"fi.alk.harness","generator_version":"1","repository":null,"source_digest":"'
        + "a" * 64
        + '","source_kind":"remote"},"readiness":[],"runtime":'
        '{"control_service":null,"document":null,"evidence_seam":null,"kind":"external"},'
        '"schema_version":"futureagi.environment-bundle.v2","seed":null}'
    )
    record_b_json = '{"path":"b.txt","sha256":"' + file_sha_b + '","size":5}'
    record_a_json = '{"path":"a.txt","sha256":"' + file_sha_a + '","size":3}'

    expected = hashlib.sha256(core_json.encode("utf-8"))
    for record_json in (record_b_json, record_a_json):
        encoded = record_json.encode("utf-8")
        expected.update(len(encoded).to_bytes(8, "big"))
        expected.update(encoded)

    assert seal_bundle_v2(manifest) == "sha256:" + expected.hexdigest()


def test_seal_bundle_v2_ignores_the_manifests_own_digest_field() -> None:
    """§2d: the digest is computed over the manifest minus `digest` and `files` — a manifest
    whose only difference is its own (placeholder or stale) `digest` value must seal identically."""
    body = {
        **FULL_MANIFEST_EXAMPLE,
        "runtime": {"kind": "external"},
        "processes": [],
        "seed": None,
        "capabilities": {},
        "readiness": [],
    }
    a = EnvironmentBundleV2.model_validate({**body, "digest": "sha256:" + "0" * 64})
    b = EnvironmentBundleV2.model_validate({**body, "digest": "sha256:" + "1" * 64})
    assert seal_bundle_v2(a) == seal_bundle_v2(b)


# --- load_bundle_v2 --------------------------------------------------------------------------


def test_load_bundle_v2_rejects_a_v1_manifest_with_a_typed_error(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "futureagi.environment-bundle.v1"})
    )
    with pytest.raises(BundleV2Error, match="bundle_schema_unsupported"):
        load_bundle_v2(tmp_path)


def test_load_bundle_v2_parses_a_valid_manifest_from_its_directory(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(FULL_MANIFEST_EXAMPLE))
    bundle = load_bundle_v2(tmp_path)
    assert bundle.name == "demo"
