"""The §2e pre-provision checklist (`process_preflight.py`), per `hosted-execution-seams.md` v1.9.

Every checklist item gets at least one rejection test carrying its named code, plus one clean
accept-lane run of the full checklist. Bundles are built as real directories under `tmp_path` with
real file bytes — the digest and per-file checks need actual content to hash, so a manifest built
purely in memory (as `test_bundle_v2.py` does) cannot exercise this module. No docker; every
managed-engine/process concept here is a manifest fact, never a running service.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from fi.alk.harness import bundle as bundle_module
from fi.alk.harness import bundle_v2 as bundle_v2_module
from fi.alk.harness import process_preflight as process_preflight_module
from fi.alk.harness import process_runtime as process_runtime_module
from fi.alk.harness.bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    EnvironmentBundleV2,
    ManagedEngine,
    compute_inputs_digest,
    seal_bundle_v2,
)
from fi.alk.harness.process_preflight import PreflightError, preflight_bundle

SCHEMA_SQL = b"CREATE TABLE riders (id int);\n"
SEED_SQL = b"INSERT INTO riders VALUES (1);\n"

TARGET_PROVIDER_REFS = {"LIVEKIT_API_KEY": "target_provider"}


def _base_manifest_body() -> dict[str, Any]:
    """A minimal, otherwise-clean `kind: process` manifest: one postgres store behind a real
    seeded capability, one source process that claims the job's only `target_provider` secret."""
    return {
        "schema_version": BUNDLE_V2_SCHEMA_VERSION,
        "name": "demo",
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
                "name": "agent",
                "kind": "source",
                "working_directory": ".",
                "build_commands": [["pip", "install", "-r", "requirements.txt"]],
                "run_command": ["python", "agent.py"],
                "environment": {
                    "DATABASE_URL": "{{DATABASE_URL}}",
                    "LIVEKIT_AGENT_NAME": "agent-w{{WORLD_INDEX}}",
                },
                "secret_purposes": ["target_provider"],
                "user": "svc-agent",
                "depends_on": ["postgres"],
            },
        ],
        "capabilities": {
            "database": {
                "protocol": "postgres",
                "service": "postgres",
                "configuration_name": "DATABASE_URL",
            },
        },
        "readiness": [],
        "provenance": {
            "source_kind": "repository",
            "repository": "org/repo",
            "source_digest": "c" * 64,
        },
        "metadata": {},
    }


def _build_bundle(
    root: Path,
    *,
    body_overrides: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    extra_files: dict[str, bytes] | None = None,
    unlisted_files: dict[str, bytes] | None = None,
    include_seed: bool = True,
) -> EnvironmentBundleV2:
    """Write a real, digest-consistent bundle directory and return its parsed manifest, sealed
    through `seal_bundle_v2` — the module under test's own producer, not a second reimplementation
    of it (the accept lane must prove this module agrees with a real producer, not just with
    itself). ``unlisted_files`` writes real bytes to disk without ever hashing them into
    ``files[]`` — the shape a producer bug or a stray leftover file takes, as opposed to
    ``extra_files``, which is hashed in and fully listed."""
    root.mkdir(parents=True, exist_ok=True)
    body = _base_manifest_body()
    file_contents = {"db/schema.sql": SCHEMA_SQL, "db/seed.sql": SEED_SQL}
    if extra_files:
        file_contents.update(extra_files)

    files: list[dict[str, Any]] = []
    for relative, content in file_contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    body["files"] = files

    if include_seed:
        digest = compute_inputs_digest(
            root,
            ["db/schema.sql"],
            ["db/seed.sql"],
            engine=ManagedEngine.POSTGRES,
            version="16",
        )
        body["seed"] = {
            "stores": [
                {
                    "capability": "database",
                    "migrations": ["db/schema.sql"],
                    "seed_files": ["db/seed.sql"],
                    "baseline": {
                        "strategy": "template_database",
                        "inputs_digest": digest,
                    },
                    "sentinel": {
                        "query": "SELECT count(*) FROM riders",
                        "expected": "1",
                    },
                }
            ]
        }

    if body_overrides is not None:
        body = body_overrides(body)

    body["digest"] = "sha256:" + "0" * 64
    normalized = EnvironmentBundleV2.model_validate(body)
    body["digest"] = seal_bundle_v2(normalized)

    if unlisted_files:
        for relative, content in unlisted_files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    (root / "manifest.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return EnvironmentBundleV2.model_validate(body)


# --- accept lane --------------------------------------------------------------------------------


def test_a_clean_bundle_passes_the_whole_checklist(tmp_path: Path) -> None:
    manifest = _build_bundle(tmp_path)
    result = preflight_bundle(tmp_path, manifest, parallelism=2, secret_refs=TARGET_PROVIDER_REFS)
    assert result is None


# --- item 1: digest verification ----------------------------------------------------------------


def test_a_file_whose_bytes_changed_after_sealing_is_rejected(tmp_path: Path) -> None:
    manifest = _build_bundle(tmp_path)
    (tmp_path / "db" / "schema.sql").write_bytes(b"MUTATED")
    with pytest.raises(PreflightError, match="bundle_file_changed") as excinfo:
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert excinfo.value.code == "bundle_file_changed"


def test_a_files_entry_missing_from_disk_is_rejected(tmp_path: Path) -> None:
    manifest = _build_bundle(tmp_path)
    (tmp_path / "db" / "schema.sql").unlink()
    with pytest.raises(PreflightError, match="bundle_file_missing"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_bundle_digest_that_does_not_match_the_recomputed_value_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _build_bundle(tmp_path)
    tampered = manifest.model_copy(update={"digest": "sha256:" + "9" * 64})
    with pytest.raises(PreflightError, match="bundle_digest_mismatch"):
        preflight_bundle(tmp_path, tampered, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


# --- item 2: path safety on the filesystem --------------------------------------------------------


def test_a_symlink_anywhere_under_the_bundle_is_rejected(tmp_path: Path) -> None:
    """The model already forbids unsafe strings in `files[].path`; this is the filesystem-level
    complement — a symlink is rejected even when it sits outside `files[]` entirely."""
    manifest = _build_bundle(tmp_path)
    (tmp_path / "evil-link").symlink_to(tmp_path / "db" / "schema.sql")
    with pytest.raises(PreflightError, match="bundle_symlink_forbidden"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_the_root_manifest_json_itself_being_a_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    """N3 (p4-round2-review): the root `manifest.json` is exempt from the listing check (a
    manifest cannot list itself), but that exemption must not extend to its symlink status — a
    symlinked root manifest was previously read straight through by `_verify_digest` and
    `_verify_unknown_fields`, the very item whose job is to stop path escapes."""
    manifest = _build_bundle(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-manifest-target.json"
    outside.write_bytes((tmp_path / "manifest.json").read_bytes())
    (tmp_path / "manifest.json").unlink()
    (tmp_path / "manifest.json").symlink_to(outside)
    with pytest.raises(PreflightError, match="bundle_symlink_forbidden"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_file_present_on_disk_but_not_listed_in_files_is_rejected(
    tmp_path: Path,
) -> None:
    """F1 (p4-round1-review): a bundle directory containing a file the producer never hashed into
    `files[]` was invisible to both the digest check and the secret scan — this is the case that
    previously slipped a `.env` through undetected. `extra_files` (used by the item-3 tests below)
    would have hashed it in; `unlisted_files` writes the same bytes without listing them at all."""
    manifest = _build_bundle(tmp_path, unlisted_files={".env": b"SECRET=1\n"})
    with pytest.raises(PreflightError, match="bundle_file_unlisted"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


# --- item 3: secret material in the bundle's own files -------------------------------------------


def test_a_dotenv_file_in_the_bundle_is_rejected(tmp_path: Path) -> None:
    manifest = _build_bundle(tmp_path, extra_files={".env": b"SECRET=1\n"})
    with pytest.raises(PreflightError, match="secret_in_bundle"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_high_entropy_secret_content_in_a_bundle_file_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _build_bundle(
        tmp_path,
        extra_files={"db/notes.sql": b"-----BEGIN RSA PRIVATE KEY-----\nabc\n"},
    )
    with pytest.raises(PreflightError, match="secret_in_bundle"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


# --- item 4: unknown-field translation ------------------------------------------------------------


def test_an_unknown_field_added_to_the_manifest_on_disk_is_rejected(
    tmp_path: Path,
) -> None:
    """The `manifest` argument is already-parsed and therefore already clean; this proves the
    translation fires against the bytes on disk, which is what a drifted or hand-edited
    `manifest.json` would look like to a fresh `EnvironmentBundleV2.model_validate` call."""
    manifest = _build_bundle(tmp_path)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw["mounts"] = ["/data"]
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PreflightError, match="unknown_field") as excinfo:
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert excinfo.value.code == "unknown_field"


def test_a_valid_but_drifted_manifest_on_disk_is_rejected(tmp_path: Path) -> None:
    """F12 (p4-round1-review): re-validating the bytes on disk only catches drift that makes the
    file *invalid* — this proves drift that leaves it valid (a changed `run_command`) is caught
    too, by actually comparing the two dumps rather than discarding the re-validated one."""
    manifest = _build_bundle(tmp_path)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw["processes"][1]["run_command"] = ["python", "other.py"]
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PreflightError, match="bundle_manifest_drifted") as excinfo:
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert excinfo.value.code == "bundle_manifest_drifted"


def test_an_unreadable_manifest_json_on_disk_is_rejected(tmp_path: Path) -> None:
    """F18 (p4-round1-review): `bundle_manifest_invalid`'s parse-failure branch, untested before."""
    manifest = _build_bundle(tmp_path)
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(PreflightError, match="bundle_manifest_invalid"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_non_extra_forbidden_model_rejection_surfaces_its_own_code(
    tmp_path: Path,
) -> None:
    """F13/F18 (p4-round1-review): `_translate_validation_error`'s fallback path was untested. A
    malformed on-disk `digest` (item 1 never reads it — only the `manifest` argument's own valid
    digest and file hashes, which are untouched here) reaches item 4's re-validation and raises
    `bundle_digest_invalid`, a bare code with no trailing colon at all — exactly the case the old
    `:`-only regex flattened to the generic `bundle_manifest_invalid`."""
    manifest = _build_bundle(tmp_path)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw["digest"] = "not-a-valid-digest"
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PreflightError, match="bundle_digest_invalid") as excinfo:
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert excinfo.value.code == "bundle_digest_invalid"


# --- §2a: compose is not a legal hosted runtime kind ----------------------------------------------


def test_kind_compose_is_rejected(tmp_path: Path) -> None:
    """N1 (p4-round2-review): the gate sits above item 1, so this fires regardless of whether
    `compose.yaml` exists on disk or is listed in `files[]` — a compose bundle need carry neither
    to be rejected, which is why this fixture writes no such file at all."""
    manifest = _build_bundle(
        tmp_path,
        body_overrides=lambda b: {
            **b,
            "runtime": {"kind": "compose", "document": "compose.yaml"},
            "processes": [],
            "seed": None,
        },
        include_seed=False,
    )
    with pytest.raises(PreflightError, match="compose_not_hosted"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs={})


# --- item 5: placeholder vocabulary, secret purposes, depends_on, engine catalog, interpreter,
# seed_missing, reserved names, build_requires_root, seed files on disk+listed ---------------------


def test_a_placeholder_outside_the_closed_vocabulary_is_rejected(
    tmp_path: Path,
) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["FOO"] = "{{NOT_A_REAL_TOKEN}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="unknown_placeholder"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_port_placeholder_naming_an_unknown_process_is_rejected(
    tmp_path: Path,
) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["PEER_PORT"] = "{{PORT_ghost-service}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="unknown_placeholder"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_port_placeholder_with_an_empty_name_falls_through_to_unknown_placeholder(
    tmp_path: Path,
) -> None:
    """F18 (p4-round1-review): `(.+)` in `_NAMED_PLACEHOLDER` requires at least one character after
    `PORT_`/`HOST_`, so `{{PORT_}}` does not match the named-placeholder pattern at all — it falls
    through to the configuration-name set, misses, and is rejected as `unknown_placeholder`. A
    plausible regression target if `(.+)` is ever relaxed to `(.*)`."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["EMPTY"] = "{{PORT_}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="unknown_placeholder"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_token_naming_a_capability_with_no_configuration_name_is_capability_unresolved(
    tmp_path: Path,
) -> None:
    """F15 (p4-round1-review): a null `configuration_name` is structurally unspellable by any
    placeholder — but when the unmatched token happens to be exactly a declared capability's own
    slug, the real problem is the capability's missing name, not an unrecognized token, so this is
    reported `capability_unresolved` naming the capability rather than the generic
    `unknown_placeholder`."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["capabilities"]["cache"] = {"protocol": "http", "service": "postgres"}
        body["processes"][1]["environment"]["FOO"] = "{{cache}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="capability_unresolved"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_build_environment_rejects_any_placeholder_even_a_legal_one(
    tmp_path: Path,
) -> None:
    """F6 (p4-round1-review): §2b's `build_environment` takes NO placeholders at all — this uses
    `{{WORLD_DIR}}`, a perfectly legal token in `environment`, specifically because that is the
    case a naive "scan against the same vocabulary" implementation would wave through."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["build_environment"] = {"TMPDIR": "{{WORLD_DIR}}"}
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="unknown_placeholder"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_the_fixed_and_named_placeholders_are_accepted(tmp_path: Path) -> None:
    """`{{WORLD_INDEX}}`/`{{WORLD_DIR}}`/`{{DB_NAME}}` need no lookup; `{{PORT_<name>}}` and
    `{{HOST_<name>}}` need only a real process name — neither needs a capability."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["SCRATCH"] = "{{WORLD_DIR}}"
        body["processes"][1]["environment"]["DB"] = "{{DB_NAME}}"
        body["processes"][1]["environment"]["PEER"] = "{{HOST_postgres}}:{{PORT_postgres}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    result = preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert result is None


def test_a_build_command_requiring_root_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["build_commands"] = [["apt-get", "install", "-y", "ffmpeg"]]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="build_requires_root"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_build_command_with_sudo_anywhere_in_the_step_is_rejected(
    tmp_path: Path,
) -> None:
    """F18 (p4-round1-review): only `apt-get` as argv[0] was exercised before — `"sudo" in step`
    is exact-token list membership, not a substring match, so `sudo` appearing anywhere in the
    argv list (not just as argv[0]) must trip it too."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["build_commands"] = [["scripts/setup.sh", "--with-sudo", "sudo"]]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="build_requires_root"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_target_provider_ref_no_process_lists_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["secret_purposes"] = []
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="secret_unclaimed"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_customer_process_cannot_claim_simulator_provider_secrets(
    tmp_path: Path,
) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["secret_purposes"].append("simulator_provider")
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    refs = {**TARGET_PROVIDER_REFS, "SIMULATOR_DEEPGRAM_API_KEY": "simulator_provider"}
    with pytest.raises(PreflightError, match="secret_purpose_forbidden"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=refs)


def test_a_listed_secret_purpose_with_no_supplying_ref_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _build_bundle(tmp_path)
    with pytest.raises(PreflightError, match="secret_missing"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs={})


def test_omitting_secret_refs_is_a_typeerror(tmp_path: Path) -> None:
    """F2 (p4-round1-review): `secret_refs` is required, not optional with an empty-dict default —
    an optional default meant `secret_unclaimed` was structurally unreachable for any caller that
    relied on it, while `secret_missing` fired on bundles the default should have let alone. This
    pins the signature itself: omitting the argument must fail loudly, at the call site, rather
    than silently reintroducing the empty-dict default."""
    manifest = _build_bundle(tmp_path)
    with pytest.raises(TypeError):
        preflight_bundle(tmp_path, manifest, parallelism=1)  # type: ignore[call-arg]


def test_an_unrecognized_secret_purpose_value_is_rejected(tmp_path: Path) -> None:
    """F2 (p4-round1-review), the signature's second defect: `secret_refs` values are only ever
    compared by `==` against `SecretPurpose.TARGET_PROVIDER.value` — a typo'd purpose string (or
    §1's raw per-alias dict shape, whose `purpose` values are dicts, not strings) silently never
    matches, producing the same wrong verdict as the missing-validation default did. Validating
    eagerly turns that into a loud, immediate `ValueError`."""
    manifest = _build_bundle(tmp_path)
    with pytest.raises(ValueError, match="not a SecretPurpose"):
        preflight_bundle(
            tmp_path,
            manifest,
            parallelism=1,
            secret_refs={"LIVEKIT_API_KEY": "target-provider"},
        )


def test_a_depends_on_naming_an_unknown_process_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["depends_on"] = ["postgres", "ghost"]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="depends_on_unresolved"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_depends_on_cycle_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][0]["depends_on"] = ["agent"]
        body["processes"][1]["depends_on"] = ["postgres"]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="depends_on_cycle"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_an_engine_version_outside_the_catalog_pin_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][0]["version"] = "15"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="engine_unsupported"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


@pytest.mark.parametrize("colliding_port", [14000, 14099, 15000, 15799])
def test_a_fixed_port_colliding_with_a_port_formula_band_is_rejected(tmp_path: Path, colliding_port: int) -> None:
    """F11, p5-round1-review: `fixed_port` forces effective parallelism to 1, but the literal
    value was never checked against the provisioner's own port-formula bands — a bundle declaring
    `fixed_port: 14000` collides with a job-shared engine at ordinal 0, and the failure mode is an
    opaque bind error inside a customer process, not a bundle rejection. `fixed_port_reserved` is
    in §2e's closed failure-code table as of v1.9."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["fixed_port"] = colliding_port
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="fixed_port_reserved"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


@pytest.mark.parametrize("colliding_port", [24000, 24099, 25000, 25799])
def test_a_fixed_port_colliding_with_the_rabbitmq_management_band_is_rejected(
    tmp_path: Path, colliding_port: int
) -> None:
    """M4: `process_runtime.py`'s own rabbitmq management listener binds at `amqp_port + 10000`
    (`_rabbitmq_management_port`) -- a `fixed_port` landing there is not covered by either base
    port-formula band, so a bundle could claim a port the harness itself is about to bind. Bands:
    24000-24099 (job-shared amqp shifted) and 25000-25799 (per-world amqp shifted, the one
    reachable in practice since rabbitmq's catalog entry is `datadir_copy`-only)."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["fixed_port"] = colliding_port
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="fixed_port_reserved"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_fixed_port_outside_both_bands_is_accepted(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["fixed_port"] = 9000
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    assert preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS) is None


def test_a_postgres_capability_with_no_store_entry_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["capabilities"]["other_db"] = {
            "protocol": "postgres",
            "service": "postgres",
            "configuration_name": "OTHER_DB_URL",
        }
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="seed_missing"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def _reseal_schema_sql(tmp_path: Path, content: bytes) -> EnvironmentBundleV2:
    """Rewrites `db/schema.sql` on disk and reseals through `seal_bundle_v2` — so digest
    verification (item 1, which runs before item 5's content-scanning checks) passes on the
    mutated content, isolating a test to the check the mutation is actually meant to exercise."""
    (tmp_path / "db" / "schema.sql").write_bytes(content)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for record in raw["files"]:
        if record["path"] == "db/schema.sql":
            record["sha256"] = hashlib.sha256(content).hexdigest()
            record["size"] = len(content)
    raw["seed"]["stores"][0]["baseline"]["inputs_digest"] = compute_inputs_digest(
        tmp_path,
        ["db/schema.sql"],
        ["db/seed.sql"],
        engine=ManagedEngine.POSTGRES,
        version="16",
    )
    raw["digest"] = "sha256:" + "0" * 64
    normalized = EnvironmentBundleV2.model_validate(raw)
    raw["digest"] = seal_bundle_v2(normalized)
    (tmp_path / "manifest.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return EnvironmentBundleV2.model_validate(raw)


def test_the_reserved_conformance_name_in_migration_content_is_rejected(
    tmp_path: Path,
) -> None:
    _build_bundle(tmp_path)
    manifest = _reseal_schema_sql(tmp_path, b"CREATE TABLE _alk_conformance (id int);\n")
    with pytest.raises(PreflightError, match="reserved_name"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_the_reserved_conformance_name_is_matched_case_insensitively(
    tmp_path: Path,
) -> None:
    """F9 (p4-round1-review): postgres folds an unquoted identifier to lower case, so
    `CREATE TABLE _ALK_CONFORMANCE` creates the reserved table under its lower-case name — a
    case-sensitive scan would miss exactly the evasion this exists to catch."""
    _build_bundle(tmp_path)
    manifest = _reseal_schema_sql(tmp_path, b"CREATE TABLE _ALK_CONFORMANCE (id int);\n")
    with pytest.raises(PreflightError, match="reserved_name"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_similarly_named_table_and_a_comment_mentioning_the_reserved_name_are_accepted(
    tmp_path: Path,
) -> None:
    """F9 (p4-round1-review) regression: `alk_conformance_backup` (no leading underscore, and a
    different table) must not trip the scan — this is exactly what the lookarounds exist to keep
    open. A `--` comment mentioning the reserved name as prose, not as an identifier it defines,
    must not trip it either, now that comments are stripped before scanning."""
    _build_bundle(tmp_path)
    manifest = _reseal_schema_sql(
        tmp_path,
        b"CREATE TABLE alk_conformance_backup (id int);\n-- never create _alk_conformance here\n",
    )
    result = preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)
    assert result is None


def test_a_migration_path_not_listed_in_files_is_rejected(tmp_path: Path) -> None:
    """`seed_file_unlisted` (item 5) and `bundle_file_unlisted` (item 2, F1 p4-round1-review) share
    the same "on disk but not in files[]" condition — item 2's bundle-wide walk now runs first and
    always wins this exact scenario, which is why this asserts the earlier-numbered item's code
    rather than the one this test used to name before F1 closed item 2's gap."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["seed"]["stores"][0]["migrations"] = ["db/schema.sql", "db/extra.sql"]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    # On disk, but never hashed into files[].
    (tmp_path / "db" / "extra.sql").write_bytes(b"-- extra\n")
    with pytest.raises(PreflightError, match="bundle_file_unlisted"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_seed_file_path_that_does_not_exist_on_disk_is_rejected(
    tmp_path: Path,
) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["seed"]["stores"][0]["seed_files"] = ["db/seed.sql", "db/ghost.sql"]
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="seed_file_missing"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_recorded_inputs_digest_that_does_not_match_the_seed_files_is_rejected(
    tmp_path: Path,
) -> None:
    """F14 (p4-round1-review): §2c makes `inputs_digest` the baseline identity attempt-retry reuse
    trusts absolutely — nothing on either side of the seam validated it before. `engine`/`version`
    come from the store's capability's own backing `ManagedProcess` (postgres/16 here)."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["seed"]["stores"][0]["baseline"]["inputs_digest"] = "sha256:" + "f" * 64
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="inputs_digest_mismatch"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


# --- item 6: no_sql_store ------------------------------------------------------------------------


def test_a_process_bundle_with_no_postgres_capability_is_rejected(
    tmp_path: Path,
) -> None:
    """Keeps the `database` capability (so the `{{DATABASE_URL}}` placeholder in `agent`'s
    environment still resolves) and only changes its protocol away from postgres, isolating this
    from the placeholder-vocabulary check that would otherwise fire first."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["capabilities"]["database"]["protocol"] = "http"
        body["seed"] = None
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate, include_seed=False)
    with pytest.raises(PreflightError, match="no_sql_store"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


# --- item 7: resource sanity ---------------------------------------------------------------------


def test_more_than_100_processes_is_rejected(tmp_path: Path) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        extra = [
            {
                "name": f"extra-{i}",
                "kind": "managed",
                "engine": "redis",
                "version": "7",
                "user": "svc-data",
                "depends_on": [],
            }
            for i in range(100)
        ]
        body["processes"] = body["processes"] + extra
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="process_count_exceeded"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


@pytest.mark.parametrize("parallelism", [0, 9], ids=["below-range", "above-range"])
def test_parallelism_outside_1_to_8_is_rejected(tmp_path: Path, parallelism: int) -> None:
    manifest = _build_bundle(tmp_path)
    with pytest.raises(PreflightError, match="parallelism_out_of_range"):
        preflight_bundle(
            tmp_path,
            manifest,
            parallelism=parallelism,
            secret_refs=TARGET_PROVIDER_REFS,
        )


# --- ordering: the checklist runs in the contract's numbered order, and a bundle violating two
# rules at once must report the earlier-numbered one (F17, p4-round1-review) — every rejection
# fixture above proves presence, none of them alone proves position. --------------------------


def test_a_secret_file_and_an_unknown_placeholder_together_report_the_earlier_numbered_item(
    tmp_path: Path,
) -> None:
    """Item 3 (secret scan) before item 5 (placeholder vocabulary)."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["FOO"] = "{{NOT_A_REAL_TOKEN}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate, extra_files={".env": b"SECRET=1\n"})
    with pytest.raises(PreflightError, match="secret_in_bundle"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_changed_file_and_a_reserved_name_together_report_the_earlier_numbered_item(
    tmp_path: Path,
) -> None:
    """Item 1 (digest verification) before item 5's reserved-name scan — the mutated file carries
    both a byte-level tamper the digest catches and a reserved name the content scan would catch,
    but the manifest is never resealed, so item 1 sees it first."""
    manifest = _build_bundle(tmp_path)
    (tmp_path / "db" / "schema.sql").write_bytes(b"CREATE TABLE _alk_conformance (id int);\n")
    with pytest.raises(PreflightError, match="bundle_file_changed"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs=TARGET_PROVIDER_REFS)


def test_a_placeholder_and_bad_parallelism_together_report_the_earlier_numbered_item(
    tmp_path: Path,
) -> None:
    """Item 5 (placeholder vocabulary) before item 7 (resource sanity)."""

    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        body["processes"][1]["environment"]["FOO"] = "{{NOT_A_REAL_TOKEN}}"
        return body

    manifest = _build_bundle(tmp_path, body_overrides=mutate)
    with pytest.raises(PreflightError, match="unknown_placeholder"):
        preflight_bundle(tmp_path, manifest, parallelism=99, secret_refs=TARGET_PROVIDER_REFS)


def test_a_compose_bundle_with_an_unlisted_file_reports_compose_not_hosted(
    tmp_path: Path,
) -> None:
    """N1 (p4-round2-review): the compose gate before item 1, before item 2 (`test_kind_compose_
    is_rejected` alone cannot prove position, since it now carries no file to trip item 2 at all).
    An unlisted `compose.yaml` would report `bundle_file_unlisted` if item 2 ran first — this pins
    that the gate wins instead."""
    manifest = _build_bundle(
        tmp_path,
        body_overrides=lambda b: {
            **b,
            "runtime": {"kind": "compose", "document": "compose.yaml"},
            "processes": [],
            "seed": None,
        },
        include_seed=False,
    )
    (tmp_path / "compose.yaml").write_bytes(b"services: {}\n")
    with pytest.raises(PreflightError, match="compose_not_hosted"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs={})


def test_a_compose_bundle_with_a_changed_listed_file_reports_compose_not_hosted(
    tmp_path: Path,
) -> None:
    """B2 (p4-round3-review): the fixture above only pins the gate above item 2 — its sole
    competing violation (`compose.yaml` unlisted) never reaches item 1, since the bundle is
    otherwise digest-consistent. This mutates a LISTED file's bytes after sealing, without
    resealing, so item 1 would otherwise raise `bundle_file_changed` — proving the gate wins
    above item 1 too, not just item 2."""
    manifest = _build_bundle(
        tmp_path,
        body_overrides=lambda b: {
            **b,
            "runtime": {"kind": "compose", "document": "compose.yaml"},
            "processes": [],
            "seed": None,
        },
        include_seed=False,
    )
    (tmp_path / "compose.yaml").write_bytes(b"services: {}\n")
    (tmp_path / "db" / "schema.sql").write_bytes(b"MUTATED")  # still listed in files[]; unsealed.
    with pytest.raises(PreflightError, match="compose_not_hosted"):
        preflight_bundle(tmp_path, manifest, parallelism=1, secret_refs={})


# --- kind: external ------------------------------------------------------------------------------


def test_kind_external_skips_the_process_block_but_still_enforces_resource_sanity(
    tmp_path: Path,
) -> None:
    """F18 (p4-round1-review): the only prior `external` coverage was model-layer
    (`test_bundle_v2.py`) — nothing proved that `preflight_bundle` itself skips item 5's process
    block and item 6 (`no_sql_store`) for `kind: external`, and still applies item 7."""
    manifest = _build_bundle(
        tmp_path,
        body_overrides=lambda b: {
            **b,
            "runtime": {"kind": "external"},
            "processes": [],
            "seed": None,
            "capabilities": {
                "target": {
                    "protocol": "http",
                    "service": "customer-endpoint",
                    "configuration_name": "TARGET_URL",
                },
            },
        },
        include_seed=False,
    )
    # Item 5 (placeholder/secret-purpose/depends_on/engine-catalog/seed checks) and item 6
    # (no_sql_store) never run — nothing in this manifest could satisfy either, since it has no
    # processes at all.
    assert preflight_bundle(tmp_path, manifest, parallelism=2, secret_refs={}) is None
    # Item 7 still runs regardless of runtime kind.
    with pytest.raises(PreflightError, match="parallelism_out_of_range"):
        preflight_bundle(tmp_path, manifest, parallelism=99, secret_refs={})


# --- §2e-table containment (B3/S3, p4-round3-review) ---------------------------------------------
#
# `PreflightError`'s own docstring claims "every code this module raises is in that [§2e] table" —
# true today, per p4-round3-review's part-B finding, but mechanically unenforced: a future
# `raise ValueError("some_new_code: ...")` anywhere in the model layer would silently leak an
# unlisted code across a seam the contract calls closed, and nothing short of re-deriving the claim
# by hand (as that review did) would catch it. This makes the claim a running test instead.
#
# Approach (grep-free, as the worklist asked): walk each module's SOURCE TEXT with `ast`, not
# `grep` — a plain substring/regex scan over raw text can't distinguish an actual `raise
# PreflightError(...)`/`raise ValueError(...)` call from a code fragment inside a docstring or a
# comment quoting one. Two extraction shapes:
#   - `PreflightError(...)`: the whole first positional argument, when it is a literal string, IS
#     the code (every call in this codebase spells it that way) — `_translate_validation_error`'s
#     `extra_forbidden` branch RETURNS its `PreflightError(...)` rather than raising it directly
#     (the caller does `raise _translate_validation_error(exc) from exc`), so the walk matches
#     every `Call` node by callee name, not only ones sitting directly under a `Raise`.
#   - `ValueError(...)` (the model layer): the code is only the LEADING token of the message, up
#     to `:` or end-of-string — recovered by taking the static leading text of the first argument
#     (a plain string constant, an f-string's first literal segment, or the left operand of a `+`
#     concatenation — every shape actually used in `bundle_v2.py`/`bundle.py`) and applying the
#     same `[a-z][a-z0-9_]*` leading-token regex `_translate_validation_error`'s own runtime
#     fallback uses. A `ValueError` whose message has no static leading text at all (fully
#     dynamic) would be skipped rather than mis-coded — none exist in these three modules today.
#
# Scope judgement call: `bundle.py` also carries `EnvironmentBundle`/`BundleRuntime` (v1's OWN,
# separate model classes) with their own, unrelated `ValueError` raises (e.g. plain prose with no
# code prefix at all, `bundle_digest_invalid`/`bundle_schema_unsupported` reused as v1's own
# names) — none of that is reachable from `EnvironmentBundleV2.model_validate(...)`'s validation
# tree, since v1 and v2 are disjoint model hierarchies (`bundle_v2.py`'s own docstring: "v1 stays
# untouched"). The only `bundle.py` surface v2 actually calls into is the two helper functions
# `bundle_v2.py` imports and its own validators invoke — `_safe_relative`, `_reject_secret_values`
# — so this test walks exactly those two functions' source (`inspect.getsource`), not the whole
# file; walking the whole file would both over-include unreachable v1 codes and risk masking a
# real gap behind noise from a hierarchy this claim was never about.


_CODE_LEADING_TOKEN = re.compile(r"^([a-z][a-z0-9_]*)(?::|$)")


def _static_leading_text(node: ast.expr) -> str | None:
    """The static leading text of a raise-argument expression, or `None` if it has none at all
    (a fully dynamic value, e.g. a bare name or an f-string starting with a `{...}`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return str(node.values[0].value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _static_leading_text(node.left)
    return None


def _raised_codes(source_text: str, callee_name: str, *, whole_first_argument: bool) -> set[str]:
    codes: set[str] = set()
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == callee_name):
            continue
        if not node.args:
            continue
        if whole_first_argument:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                codes.add(first.value)
            continue
        leading = _static_leading_text(node.args[0])
        if leading is None:
            continue
        match = _CODE_LEADING_TOKEN.match(leading)
        if match:
            codes.add(match.group(1))
    return codes


# §2e's closed failure-code table (v1.9), transcribed verbatim — the single source of truth every
# raised code is checked against. Split exactly as the contract text splits it, purely for
# reviewability against the spec; the test below treats it as one flat set.
_SECTION_2E_CONTRACT_RULE_CODES = frozenset(
    {
        "compose_not_hosted",
        "engine_unsupported",
        "no_sql_store",
        "seed_missing",
        "seed_strategy_unsupported",
        "sentinel_shape_mismatch",
        "store_protocol_unsupported",
        "capability_engine_mismatch",
        "store_service_not_managed",
        "reserved_name",
        "unknown_placeholder",
        "unknown_field",
        "secret_in_bundle",
        "secret_unclaimed",
        "secret_missing",
        "secret_purpose_forbidden",
        "build_requires_root",
        "user_assignment_invalid",
        "configuration_name_duplicate",
        "configuration_name_required",
        "configuration_name_reserved",
        "sentinel_shape_invalid",
        "capability_unresolved",
        "service_unresolved",
        "control_service_unresolved",
        "process_name_duplicate",
        "inputs_digest_mismatch",
        "fixed_port_reserved",  # §2e, v1.9.
    }
)
_SECTION_2E_MECHANICAL_CODES = frozenset(
    {
        "bundle_schema_unsupported",
        "bundle_manifest_invalid",
        "bundle_manifest_drifted",
        "bundle_digest_mismatch",
        "bundle_digest_invalid",
        "inputs_digest_invalid",
        "file_sha256_invalid",
        "source_digest_invalid",
        "bundle_file_missing",
        "bundle_file_changed",
        "bundle_file_unlisted",
        "bundle_symlink_forbidden",
        "bundle_path_unsafe",
        "depends_on_unresolved",
        "depends_on_cycle",
        "seed_file_missing",
        "seed_file_unlisted",
        "process_count_exceeded",
        "parallelism_out_of_range",
        "evidence_seam_required",
        "processes_required",
        "processes_and_seed_forbidden",
        "document_only_for_compose",
        "compose_runtime_requires_document",
        "build_command_step_empty",
        "started_check_requires_exactly_one_of_port_or_log_marker",
        "resolved_secret_forbidden",
        "capability_slug_invalid",
        "process_name_invalid",  # §0/§2b v1.8: the process-`name` pattern rule.
    }
)
_SECTION_2E_CODES = _SECTION_2E_CONTRACT_RULE_CODES | _SECTION_2E_MECHANICAL_CODES


def test_every_code_these_modules_can_raise_is_in_the_closed_section_2e_table() -> None:
    raised: set[str] = set()
    raised |= _raised_codes(
        Path(inspect.getfile(process_preflight_module)).read_text(encoding="utf-8"),
        "PreflightError",
        whole_first_argument=True,
    )
    raised |= _raised_codes(
        Path(inspect.getfile(bundle_v2_module)).read_text(encoding="utf-8"),
        "ValueError",
        whole_first_argument=False,
    )
    raised |= _raised_codes(
        inspect.getsource(bundle_module._safe_relative),
        "ValueError",
        whole_first_argument=False,
    )
    raised |= _raised_codes(
        inspect.getsource(bundle_module._reject_secret_values),
        "ValueError",
        whole_first_argument=False,
    )
    unlisted = raised - _SECTION_2E_CODES
    assert not unlisted, f"raised but not in §2e's table: {sorted(unlisted)}"


def test_the_extraction_itself_finds_a_nonempty_set_in_every_source() -> None:
    """Guards the test above against a false pass from a broken extractor (e.g. an import that
    silently resolves to the wrong file, or a callee-name typo) — each source individually must
    contribute at least one code, not just the union as a whole."""
    preflight_codes = _raised_codes(
        Path(inspect.getfile(process_preflight_module)).read_text(encoding="utf-8"),
        "PreflightError",
        whole_first_argument=True,
    )
    bundle_v2_codes = _raised_codes(
        Path(inspect.getfile(bundle_v2_module)).read_text(encoding="utf-8"),
        "ValueError",
        whole_first_argument=False,
    )
    assert "unknown_field" in preflight_codes  # the `return`-not-`raise` case (see module note).
    assert "compose_not_hosted" in preflight_codes
    assert len(bundle_v2_codes) > 10
    assert "bundle_path_unsafe" in _raised_codes(
        inspect.getsource(bundle_module._safe_relative),
        "ValueError",
        whole_first_argument=False,
    )
    assert "resolved_secret_forbidden" in _raised_codes(
        inspect.getsource(bundle_module._reject_secret_values),
        "ValueError",
        whole_first_argument=False,
    )


# --- §2f-table containment (v1.8 addition) --------------------------------------------------------
#
# v1.8 gave `process_runtime.py` its own closed table (§2f) for the subset of `ProcessRuntimeError`
# codes that cross the outbound seam. Same containment argument as §2e above, extended to the
# module the new table covers — worklist item: `unsupported_capability_protocol` and
# `source_tree_unavailable` are new in this fix pass and must show up here or the test (correctly)
# fails. `ProcessRuntimeError(stage, code, message, ...)` carries its code at positional index 1,
# not index 0 (`PreflightError`'s shape), so this is a distinct extraction, not a reuse of
# `_raised_codes`.


def _raised_codes_at_index(source_text: str, callee_name: str, *, index: int) -> set[str]:
    codes: set[str] = set()
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == callee_name):
            continue
        if len(node.args) <= index:
            continue
        argument = node.args[index]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            codes.add(argument.value)
    return codes


# §2f's closed table (v1.8), transcribed verbatim, plus v1.10's two additions below.
_SECTION_2F_CODES = frozenset(
    {
        "source_tree_unavailable",
        "build_failed",
        "runtime_unsupported",
        "spawn_failed",
        "depends_on_timeout",
        "unsupported_capability_protocol",
        # `seed_failed` (v1.10, §2f): a §2c migration/seed step exited nonzero against the freshly
        # started store — customer-authored content, deterministic, `environment` domain (never
        # retried). Landed in the frozen table this version; no longer an out-of-vocabulary flag.
        "seed_failed",
        # `store_statement_failed` (v1.10, §2f): a managed store errored or rejected a provisioner-
        # ISSUED statement (CREATE/DROP/ALTER DATABASE, sentinel or canary probe) after passing
        # readiness — the harness's own statements, so a deterministic failure here is a harness/
        # engine fault, `infrastructure` domain (retryable), never `seed_failed` (that code is
        # reserved for the customer's own migration/seed content).
        "store_statement_failed",
    }
)
# `ProcessRuntimeError` also raises codes that are deliberately INTERNAL-only — each marks a
# precondition `preflight_bundle` should already have made impossible (a placeholder token or a
# missing credential preflight itself should have caught), so by the module's own docstring these
# "never cross the outbound seam directly" and have no §2f entry to begin with. Excluded from
# CONTAINMENT, not from extraction — a genuinely new internal code still surfaces in the raised
# set for a human to classify, since only these documented names are exempted.
_INTERNAL_ONLY_RUNTIME_CODES = frozenset(
    {
        "internal_unknown_placeholder",
        "internal_missing_credentials",
        # Phase 6: marks a bundle-shape/state invariant an earlier layer (the model layer, or this
        # module's own baseline-freeze-before-clone ordering) should already guarantee — e.g.
        # `reset()` called before `provision()`, or a store's backing service turning out not to be a
        # `ManagedProcess` despite `bundle_v2`'s `store_service_not_managed` check. A bug to fix here,
        # never a bundle defect the outbound seam needs a name for — same status as the other two.
        "internal_invariant_violated",
    }
)


def test_every_section_2f_code_process_runtime_raises_is_in_the_closed_table() -> None:
    raised = _raised_codes_at_index(
        Path(inspect.getfile(process_runtime_module)).read_text(encoding="utf-8"),
        "ProcessRuntimeError",
        index=1,
    )
    # `process_name_invalid` (F3's defense-in-depth path-containment check, `_ensure_within`) is
    # not a NEW §2f code — it deliberately reuses the EXISTING §2e model-layer code as a backstop
    # for when the model layer is bypassed, so `_SECTION_2E_CODES` is a legitimate source too, not
    # just §2f's own six entries.
    unlisted = raised - _SECTION_2F_CODES - _SECTION_2E_CODES - _INTERNAL_ONLY_RUNTIME_CODES
    assert not unlisted, (
        f"raised but not in §2f's table (nor §2e's, nor a documented internal-only code): {sorted(unlisted)}"
    )


def test_the_section_2f_extraction_itself_finds_a_nonempty_set() -> None:
    raised = _raised_codes_at_index(
        Path(inspect.getfile(process_runtime_module)).read_text(encoding="utf-8"),
        "ProcessRuntimeError",
        index=1,
    )
    assert "build_failed" in raised
    assert "spawn_failed" in raised
    assert "source_tree_unavailable" in raised
    assert "unsupported_capability_protocol" in raised
