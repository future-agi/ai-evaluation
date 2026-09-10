"""The §2e pre-provision checklist — `hosted-execution-seams.md` v1.9 — as a single gate the
in-sandbox provisioner runs before starting anything.

`bundle_v2.py` validates everything decidable from the manifest's own field values alone; this
module covers what its docstring names as deferred: the bundle's actual files on disk (digest and
per-file hashes, symlinks, path escapes, secret content), the pydantic `extra="forbid"` ->
`unknown_field` translation, and every rule that needs the job the bundle will run under
(placeholder vocabulary, secret purposes against the job's `secret_refs`, the `depends_on` graph,
the engine catalog, `seed_missing`, `inputs_digest` verification, reserved-name content scanning,
`no_sql_store`, and resource sanity). `seed_strategy_unsupported`, `sentinel_shape_mismatch`,
`capability_unresolved`, `configuration_name_duplicate`, `user_assignment_invalid`, and
`capability_engine_mismatch` are already enforced by the model layer and are not repeated here.

A missing interpreter (§0, v1.7) is a BUILD-time failure, not a preflight one — no manifest field
carries an interpreter demand, so this module has nothing to check and does not attempt to.

`preflight_bundle` runs the checklist in the contract's own order and raises on the first
violation, never a crash — every failure is a `PreflightError` carrying a code from §2e's
failure-code table (v1.7). The caller (the provisioner) is responsible for mapping that into a
FAILED terminal state with `FailureDomain.ENVIRONMENT` in `HarnessStage.VALIDATING_ENVIRONMENT`,
per §2e's closing rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .artifacts import _SECRET_CONTENT, _SECRET_FILES
from .bundle import CapabilityProtocol
from .bundle_v2 import (
    BUNDLE_V2_MANIFEST,
    BUNDLE_V2_SCHEMA_VERSION,
    BundleFileV2,
    EnvironmentBundleV2,
    ManagedEngine,
    ManagedProcess,
    RuntimeKindV2,
    SecretPurpose,
    SourceProcess,
    compute_inputs_digest,
    seal_bundle_v2,
)
from .provider_import import ProviderImportSpec
from .provider_lifecycle import ProviderLifecycleSpec, ProviderScope

_SECRET_PURPOSE_VALUES = {member.value for member in SecretPurpose}


class PreflightError(RuntimeError):
    """A §2e checklist rule rejected the bundle.

    ``code`` is one of §2e's failure-code table (v1.9): "contract-rule" codes, each named by a
    numbered checklist item's prose, and "mechanical" codes for plumbing failures the contract
    describes but does not formalize as a rule (a missing bundle file, an out-of-range
    ``parallelism``). Every code this module raises is in that table — including
    ``fixed_port_reserved`` (F11, p5-round1-review; added to the table by v1.9), which guards
    against a `fixed_port` aliasing the provisioner's own port-formula bands.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

# §2b's catalog table. `ManagedEngine` already closes which *engines* exist; this closes which
# *version* of each is the one the snapshot actually ships.
_ENGINE_CATALOG_VERSION: dict[ManagedEngine, str] = {
    ManagedEngine.POSTGRES: "16",
    ManagedEngine.REDIS: "7",
    ManagedEngine.RABBITMQ: "3.13",
}

# §0 (v1.7): a repo needing an interpreter the snapshot lacks fails at BUILD time, reported
# `runtime_unsupported` there — not here. The manifest carries no interpreter-demand field (the
# source tree isn't embedded in the bundle, so preflight can't see `.python-version`/`engines`
# even if it wanted to), so this module has no interpreter check to run.

# §2c: migrations/seeds must not create these — checked as a source-content scan, not a manifest
# field, since the identifier lives inside SQL/scripts the model layer never parses.
# `re.IGNORECASE`: postgres folds an unquoted identifier to lower case, so `CREATE TABLE
# _ALK_CONFORMANCE` creates the reserved table under its lower-case name — case-insensitive
# matching is the only way to catch that (F9, p4-round1-review). This is slightly over-broad for
# redis/rabbitmq, whose names are case-sensitive, but over-broad on a reserved-name check is the
# safe direction. Known false-positive surface, left as-is (documented rather than fixed): the scan
# reads whole file bytes with no lexical awareness beyond stripped `--`/`/* */` comments below, so
# a quoted string literal containing the reserved name (e.g. as inserted *data*) still trips it.
# The stripping below is a false-NEGATIVE surface in the opposite direction, equally lexer-free and
# equally left as-is: a `--` or `/*` inside a string literal (not a comment) deletes real content
# up to the next line-end or `*/`, which can delete a reserved-name definition that follows it on
# the same statement (N7, p4-round2-review).
_RESERVED_NAME = "_alk_conformance"
_RESERVED_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])" + re.escape(_RESERVED_NAME) + r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# §2b closed placeholder vocabulary.
_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
_FIXED_PLACEHOLDERS = {"JOB_ID", "WORLD_INDEX", "WORLD_DIR", "DB_NAME"}
_NAMED_PLACEHOLDER = re.compile(r"^(PORT|HOST)_(.+)$")

# §2a: Dockerfile-style install lines requiring root privileges have no process-copy equivalent —
# the provisioner never runs as root and never will (§0's guest is unprivileged throughout).
_ROOT_BUILD_COMMANDS = {
    "apt-get",
    "apt",
    "apt-cache",
    "dpkg",
    "yum",
    "dnf",
    "apk",
    "pacman",
    "sudo",
}

_MAX_PROCESSES = 100
_MIN_PARALLELISM = 1
_MAX_PARALLELISM = 8

# §2b's own port formulas (`process_runtime.plan_ports`): job-shared `14000 + ordinal`
# (ordinal <= 99, §2e item 7's process cap) and per-world `15000 + 100*world_index + ordinal`
# (world_index <= 7, §1's parallelism cap). A `fixed_port` landing inside either band can alias a
# formula port the provisioner is about to hand to a *different* process — F11, p5-round1-review.
# `fixed_port` forces W=1, so the collision surface is small, but the failure mode is a bind
# error inside a customer process, not a bundle rejection, which is strictly worse. Mirrored here
# rather than imported from `process_runtime.py`: preflight has no business depending on the
# execution module, and both bands are fixed by the contract, not by any runtime state.
_JOB_SHARED_PORT_BAND = range(14000, 14100)
_PER_WORLD_PORT_BAND = range(15000, 15800)

# `process_runtime.py`'s own `_rabbitmq_management_port` formula (`amqp_port + 10000`) —
# mirrored here for the same reason as the two bands above: preflight has no business depending
# on the execution module. The rabbitmq catalog entry supports `datadir_copy` only (no
# `template_database`), so its amqp port is always drawn from the PER-WORLD band in practice
# today; the job-shared shift is reserved too, defensively, since the formula itself is generic
# and nothing about this band's math depends on which base band it is applied to.
_RABBITMQ_MANAGEMENT_PORT_OFFSET = 10000
_JOB_SHARED_RABBITMQ_MANAGEMENT_BAND = range(
    _JOB_SHARED_PORT_BAND.start + _RABBITMQ_MANAGEMENT_PORT_OFFSET,
    _JOB_SHARED_PORT_BAND.stop + _RABBITMQ_MANAGEMENT_PORT_OFFSET,
)
_PER_WORLD_RABBITMQ_MANAGEMENT_BAND = range(
    _PER_WORLD_PORT_BAND.start + _RABBITMQ_MANAGEMENT_PORT_OFFSET,
    _PER_WORLD_PORT_BAND.stop + _RABBITMQ_MANAGEMENT_PORT_OFFSET,
)


def preflight_bundle(
    bundle_dir: Path,
    manifest: EnvironmentBundleV2,
    *,
    parallelism: int,
    secret_refs: dict[str, str],
) -> None:
    """Run the complete §2e checklist against a sealed v2 bundle directory, in the contract's own
    numbered order. Raises ``PreflightError`` on the first violation; returns ``None`` when clean.

    ``manifest`` is the already-parsed model the caller obtained from ``load_bundle_v2`` — item 4
    (the pydantic ``extra_forbidden`` -> ``unknown_field`` translation bundle_v2's own docstring
    defers here) is implemented by re-validating the bytes on disk, which is also where this
    function's own read of ``manifest.json`` for step 1 comes from; a caller that already trusts
    ``manifest`` still gets a genuine check that the file backing it hasn't drifted since.

    ``secret_refs`` maps each job secret alias to its ``SecretRef.purpose`` value (§1) — item 5's
    ``secret_unclaimed``/``secret_missing`` pair needs it and the contract's own entrypoint
    signature (§2e's charter) does not carry it. Required, not optional: §4's provider port hands
    the provisioner ``work_directory``, and `/work/job.json` is readable from it, so every real
    caller has the job's resolved refs — there is no legitimate caller that cannot supply this.
    Pass ``{}`` explicitly for a job with no secret refs at all, rather than omitting the argument:
    an optional default silently both under- and over-enforced the check it exists for (F2,
    p4-round1-review), which required-and-explicit closes. Every value must be a ``SecretPurpose``
    value; anything else raises ``ValueError`` immediately, before any bundle content is checked.
    """
    bundle_dir = Path(bundle_dir)
    for alias, purpose in secret_refs.items():
        # `isinstance` first: §1's raw `agent.secret_refs` shape is `{alias: {manager, key,
        # version, purpose}}`, a dict — an unhashable value would otherwise raise TypeError against
        # the `in` check below instead of the ValueError this docstring promises (N8, p4-round2-
        # review).
        if not isinstance(purpose, str) or purpose not in _SECRET_PURPOSE_VALUES:
            raise ValueError(
                f"secret_refs[{alias!r}] = {purpose!r} is not a SecretPurpose value"
            )

    if manifest.runtime.kind is RuntimeKindV2.COMPOSE:
        # §2a: "a hosted job with kind: compose fails preflight" — not one of §2e's seven numbered
        # items, so ahead of item 1 rather than slotted between them: every item below assumes
        # v2's processes/seed shape, which a compose bundle need not carry, and a compose bundle's
        # own files (its document, e.g.) carry no obligation to be exhaustively listed in files[]
        # the way a hosted bundle's do — checking file-listing first mis-reported that case as
        # bundle_file_unlisted instead of compose_not_hosted (N1, p4-round2-review).
        raise PreflightError(
            "compose_not_hosted", "kind: compose is not a legal hosted runtime"
        )

    files = _verify_digest(bundle_dir, manifest)  # 1
    walked_files = _verify_path_safety(bundle_dir, files)  # 2
    _scan_bundle_files_for_secrets(bundle_dir, walked_files)  # 3
    _verify_unknown_fields(bundle_dir, manifest)  # 4

    if manifest.runtime.kind is RuntimeKindV2.PROCESS:
        _verify_placeholder_vocabulary(manifest)  # 5
        _verify_no_root_build_commands(manifest)  # 5 (§2a)
        _verify_secret_purposes(manifest, secret_refs)  # 5
        _verify_provider_lifecycle(manifest)  # 5
        _verify_provider_import(manifest)  # 5
        _verify_depends_on(manifest)  # 5
        _verify_engine_catalog(manifest)  # 5
        _verify_fixed_port_not_reserved(manifest)  # 5 / §2b
        _verify_seed_missing(manifest)  # 5 / §2c
        _verify_reserved_names(bundle_dir, manifest)  # 5
        _verify_seed_files_on_disk_and_listed(bundle_dir, manifest, files)  # 5

    _verify_no_sql_store(manifest)  # 6
    _verify_resource_sanity(manifest, parallelism=parallelism)  # 7


# --- item 1: digest verification -------------------------------------------------------------


def _verify_digest(
    bundle_dir: Path, manifest: EnvironmentBundleV2
) -> list[BundleFileV2]:
    root = bundle_dir.resolve()
    try:
        raw = json.loads((root / BUNDLE_V2_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError("bundle_manifest_invalid", str(exc)) from exc
    on_disk_schema_version = (
        raw.get("schema_version") if isinstance(raw, dict) else None
    )
    if on_disk_schema_version != BUNDLE_V2_SCHEMA_VERSION:
        # §2e item 1 opens with "schema_version is …bundle.v2" — checked here, at item 1,
        # rather than left to surface three items late through item 4's re-validation fallback
        # (F11, p4-round1-review).
        raise PreflightError("bundle_schema_unsupported", str(on_disk_schema_version))
    for record in manifest.files:
        path = root / record.path
        if not path.is_file():
            raise PreflightError("bundle_file_missing", record.path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        if digest.hexdigest() != record.sha256 or size != record.size:
            raise PreflightError("bundle_file_changed", record.path)
    recomputed = seal_bundle_v2(manifest)
    if recomputed != manifest.digest:
        raise PreflightError(
            "bundle_digest_mismatch",
            f"expected {manifest.digest}, computed {recomputed}",
        )
    return manifest.files


# --- item 2: path safety on the filesystem itself ----------------------------------------------


def _verify_path_safety(bundle_dir: Path, files: list[BundleFileV2]) -> list[Path]:
    """The model already rejects unsafe strings in `files[].path` (`_safe_relative`); this walks
    the actual filesystem, which a string check cannot: a symlinked directory can make an
    innocent-looking relative path resolve outside the bundle root.

    Every non-directory entry except the bundle root's own `manifest.json` must be recorded in
    `files[]` (`bundle_file_unlisted`) — a file physically present but never listed was invisible
    to both the digest check above and the secret scan that follows, which is exactly what let an
    unlisted `.env` through undetected (F1, p4-round1-review). The `manifest.json` exemption is by
    exact root path, not by basename (F10, p4-round1-review): a nested `db/manifest.json` gets no
    special treatment, only `bundle_dir/manifest.json` itself. The exemption covers only the
    listing check, not the symlink check — a symlinked root `manifest.json` would otherwise be
    waved through here and then read straight through by `_verify_digest`/`_verify_unknown_fields`,
    the very item whose job is to stop path escapes (N3, p4-round2-review).

    Returns the walked file paths so the secret scan (item 3) can run against what the filesystem
    actually contains rather than against `files[]` again.
    """
    root = bundle_dir.resolve()
    manifest_path = root / BUNDLE_V2_MANIFEST
    listed = {record.path for record in files}
    walked: list[Path] = []
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise PreflightError(
                "bundle_symlink_forbidden", str(entry.relative_to(root))
            )
        if entry == manifest_path:
            continue
        if entry.is_dir():
            continue
        relative = entry.relative_to(root).as_posix()
        if relative not in listed:
            raise PreflightError("bundle_file_unlisted", relative)
        walked.append(entry)
    return walked


# --- item 3: secret material in the bundle's own files ------------------------------------------


def _scan_bundle_files_for_secrets(bundle_dir: Path, walked_files: list[Path]) -> None:
    """Reuses `artifacts.py`'s own file-name and content secret scan unchanged — the same
    high-entropy-token regexes and credential-file-name set this codebase already applies to
    sealed run artifacts, applied here to a sealed bundle's files instead.

    Scoped to every file item 2's filesystem walk actually found, not to `files[]` (F1,
    p4-round1-review) — an unlisted secret file is already rejected by item 2's own
    `bundle_file_unlisted` check, but this scan must not depend on that running first to be
    correct on its own terms.
    """
    root = bundle_dir.resolve()
    for path in walked_files:
        relative = path.relative_to(root).as_posix()
        posix_path = PurePosixPath(relative)
        if (
            posix_path.name in _SECRET_FILES
            or posix_path.suffix.lower() in _SECRET_SUFFIXES
        ):
            raise PreflightError(
                "secret_in_bundle", f"{relative}: forbidden secret-shaped file"
            )
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if any(pattern.search(chunk) for pattern in _SECRET_CONTENT):
                    raise PreflightError(
                        "secret_in_bundle", f"{relative}: high-entropy secret-scan hit"
                    )


# --- item 4: unknown-field translation ----------------------------------------------------------


def _verify_unknown_fields(bundle_dir: Path, manifest: EnvironmentBundleV2) -> None:
    target = bundle_dir / BUNDLE_V2_MANIFEST
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError("bundle_manifest_invalid", str(exc)) from exc
    try:
        revalidated = EnvironmentBundleV2.model_validate(raw)
    except ValidationError as exc:
        raise _translate_validation_error(exc) from exc
    if revalidated.model_dump(mode="json") != manifest.model_dump(mode="json"):
        # Re-validating catches drift that makes the file *invalid*; it says nothing about drift
        # that leaves it valid (a changed `run_command`, a flipped `user`) unless the two dumps are
        # actually compared (F12, p4-round1-review).
        raise PreflightError(
            "bundle_manifest_drifted",
            "manifest.json on disk no longer matches manifest argument",
        )


def _translate_validation_error(exc: ValidationError) -> PreflightError:
    """§2b: "unknown keys in a process entry are a preflight error (`unknown_field`)" — the model
    layer's docstring defers this exact translation here, since pydantic's own `extra_forbidden`
    carries no contract vocabulary of its own. Every other model-layer rejection already embeds
    its own snake_case code as the leading token of its message (see `bundle_v2.py`'s
    `model_validator`s); that code is preserved rather than collapsed into a generic one.
    """
    for error in exc.errors():
        if error.get("type") == "extra_forbidden":
            location = ".".join(str(part) for part in error["loc"])
            return PreflightError("unknown_field", f"{location}: unknown field")
    # `(?::|$)`, not just `:` (F13, p4-round1-review): a bare code with no trailing detail (e.g.
    # `bundle_digest_invalid`) is the entire message, with nothing after it to require a colon
    # before. Scans every error, not just the first, since pydantic's own ordering is not the
    # contract's priority — the first message that yields a recognizable code wins.
    for error in exc.errors():
        message = str(error.get("msg", ""))
        matched = re.match(r"(?:Value error, )?([a-z][a-z0-9_]*)(?::|$)", message)
        if matched:
            return PreflightError(matched.group(1), message)
    return PreflightError("bundle_manifest_invalid", str(exc))


# --- item 5: everything the model layer needs the job or the files for ------------------------


def _verify_placeholder_vocabulary(manifest: EnvironmentBundleV2) -> None:
    """§2b's closed `{{...}}` vocabulary, checked in `environment`. `build_environment` takes NO
    placeholders at all (§2b) — any `{{...}}` match there is rejected outright, never resolved
    against the vocabulary below (F6, p4-round1-review).

    `{{<CONFIGURATION_NAME>}}` can only ever resolve to a capability whose `configuration_name`
    is non-null — a capability left null is therefore structurally unreachable by any placeholder,
    which is what makes this scan also enforce §2d's "non-null whenever referenced by any process
    `environment`... entry" without a second pass. When the unmatched token is exactly a declared
    capability's slug and that capability's `configuration_name` is null, the real problem is the
    missing name, not the token — reported `capability_unresolved` naming the capability, rather
    than the generic `unknown_placeholder` every other unmatched token gets (F15, p4-round1-review;
    a deliberate resolution — §2d names no other string a producer could have meant).
    """
    known_names = {process.name for process in manifest.processes}
    known_configuration_names = {
        capability.configuration_name
        for capability in manifest.capabilities.values()
        if capability.configuration_name
    }
    unresolved_capability_slugs = {
        slug
        for slug, capability in manifest.capabilities.items()
        if not capability.configuration_name
    }
    for process in manifest.processes:
        if not isinstance(process, SourceProcess):
            continue
        for key, value in (process.build_environment or {}).items():
            match = _PLACEHOLDER.search(value)
            if match:
                raise PreflightError(
                    "unknown_placeholder",
                    f"{process.name}.build_environment.{key}: {{{{{match.group(1)}}}}} — "
                    "build_environment takes no placeholders",
                )
        for key, value in process.environment.items():
            for match in _PLACEHOLDER.finditer(value):
                token = match.group(1)
                if token in _FIXED_PLACEHOLDERS:
                    continue
                named = _NAMED_PLACEHOLDER.match(token)
                if named:
                    _, name = named.groups()
                    if name in known_names:
                        continue
                    raise PreflightError(
                        "unknown_placeholder",
                        f"{process.name}.environment.{key}: {{{{{token}}}}} names an unknown "
                        "process",
                    )
                if token in known_configuration_names:
                    continue
                if token in unresolved_capability_slugs:
                    raise PreflightError(
                        "capability_unresolved",
                        f"{process.name}.environment.{key}: {{{{{token}}}}} names capability "
                        f"{token!r}, which has no configuration_name",
                    )
                raise PreflightError(
                    "unknown_placeholder",
                    f"{process.name}.environment.{key}: {{{{{token}}}}} is not in the closed "
                    "placeholder vocabulary",
                )


def _verify_no_root_build_commands(manifest: EnvironmentBundleV2) -> None:
    for process in manifest.processes:
        if not isinstance(process, SourceProcess):
            continue
        for step in process.build_commands:
            if step[0] in _ROOT_BUILD_COMMANDS or "sudo" in step:
                raise PreflightError(
                    "build_requires_root",
                    f"{process.name}: build step {step!r} requires root",
                )


def _verify_secret_purposes(
    manifest: EnvironmentBundleV2, secret_refs: dict[str, str]
) -> None:
    """§2b: both directions, scoped to `target_provider` only — `source_checkout` and any other
    gateway-only purpose never crosses into the guest (§0 step 3) and has nothing to claim here."""
    simulator_claimants = [
        process.name
        for process in manifest.processes
        if isinstance(process, SourceProcess)
        and SecretPurpose.SIMULATOR_PROVIDER in process.secret_purposes
    ]
    if simulator_claimants:
        raise PreflightError(
            "secret_purpose_forbidden",
            "customer processes cannot claim simulator_provider credentials: "
            + ", ".join(sorted(simulator_claimants)),
        )

    ref_has_target_provider = any(
        purpose == SecretPurpose.TARGET_PROVIDER.value
        for purpose in secret_refs.values()
    )
    process_claims_target_provider = any(
        SecretPurpose.TARGET_PROVIDER in process.secret_purposes
        for process in manifest.processes
        if isinstance(process, SourceProcess)
    )
    lifecycle = manifest.metadata.get("provider_lifecycle")
    lifecycle_claims_target_provider = bool(
        isinstance(lifecycle, dict) and lifecycle.get("required_secrets")
    )
    provider_import_claims_target_provider = isinstance(
        manifest.metadata.get("provider_import"), dict
    )
    connect_only_claims_target_provider = isinstance(
        manifest.metadata.get("provider_connect_only"), dict
    )
    guest_claims_target_provider = (
        process_claims_target_provider
        or lifecycle_claims_target_provider
        or provider_import_claims_target_provider
        or connect_only_claims_target_provider
    )
    if ref_has_target_provider and not guest_claims_target_provider:
        raise PreflightError(
            "secret_unclaimed",
            "a target_provider secret ref is not listed by any process",
        )
    if guest_claims_target_provider and not ref_has_target_provider:
        raise PreflightError(
            "secret_missing",
            "a process or provider lifecycle requires target_provider secrets but the job "
            "supplies no such ref",
        )


def _verify_provider_lifecycle(manifest: EnvironmentBundleV2) -> None:
    raw = manifest.metadata.get("provider_lifecycle")
    if raw is None:
        return
    try:
        spec = ProviderLifecycleSpec.model_validate(raw)
    except ValueError as exc:
        raise PreflightError("bundle_manifest_invalid", str(exc)) from exc
    if spec.scope is ProviderScope.ATTEMPT:
        raise PreflightError(
            "bundle_manifest_invalid",
            "attempt-scoped targets require a routing service; use scope: world for now",
        )
    capability = manifest.capabilities.get(spec.public_capability)
    if capability is None or capability.protocol is not CapabilityProtocol.HTTP:
        raise PreflightError(
            "capability_unresolved",
            f"{spec.public_capability!r} must name an HTTP capability",
        )
    process_name = spec.process or manifest.runtime.control_service
    if not any(
        isinstance(process, SourceProcess) and process.name == process_name
        for process in manifest.processes
    ):
        raise PreflightError(
            "service_unresolved",
            f"{process_name!r} must name a source process",
        )


def _verify_provider_import(manifest: EnvironmentBundleV2) -> None:
    raw = manifest.metadata.get("provider_import")
    if raw is None:
        return
    try:
        spec = ProviderImportSpec.model_validate(raw)
    except ValueError as exc:
        raise PreflightError("bundle_manifest_invalid", str(exc)) from exc
    capability = manifest.capabilities.get(spec.public_capability)
    if capability is None or capability.protocol is not CapabilityProtocol.HTTP:
        raise PreflightError(
            "capability_unresolved",
            f"{spec.public_capability!r} must name an HTTP capability",
        )


def _verify_depends_on(manifest: EnvironmentBundleV2) -> None:
    graph = {process.name: list(process.depends_on) for process in manifest.processes}
    for name, deps in graph.items():
        unknown = sorted(dep for dep in deps if dep not in graph)
        if unknown:
            raise PreflightError(
                "depends_on_unresolved",
                f"{name} depends_on unknown process(es): {', '.join(unknown)}",
            )

    unvisited, in_progress, done = 0, 1, 2
    state = {name: unvisited for name in graph}

    def visit(name: str, stack: list[str]) -> None:
        state[name] = in_progress
        stack.append(name)
        for dep in graph[name]:
            if state[dep] == in_progress:
                cycle = stack[stack.index(dep) :] + [dep]
                raise PreflightError("depends_on_cycle", " -> ".join(cycle))
            if state[dep] == unvisited:
                visit(dep, stack)
        stack.pop()
        state[name] = done

    for name in sorted(graph):
        if state[name] == unvisited:
            visit(name, [])


def _verify_engine_catalog(manifest: EnvironmentBundleV2) -> None:
    for process in manifest.processes:
        if not isinstance(process, ManagedProcess):
            continue
        pinned = _ENGINE_CATALOG_VERSION[process.engine]
        if process.version != pinned:
            raise PreflightError(
                "engine_unsupported",
                f"{process.name}: {process.engine.value} {process.version} is not supported; "
                f"the snapshot ships {process.engine.value} {pinned}",
            )


def _verify_fixed_port_not_reserved(manifest: EnvironmentBundleV2) -> None:
    for process in manifest.processes:
        if not isinstance(process, SourceProcess) or process.fixed_port is None:
            continue
        if (
            process.fixed_port in _JOB_SHARED_PORT_BAND
            or process.fixed_port in _PER_WORLD_PORT_BAND
            or process.fixed_port in _JOB_SHARED_RABBITMQ_MANAGEMENT_BAND
            or process.fixed_port in _PER_WORLD_RABBITMQ_MANAGEMENT_BAND
        ):
            raise PreflightError(
                "fixed_port_reserved",
                f"{process.name}: fixed_port {process.fixed_port} falls inside the provisioner's "
                "own port-formula bands (14000-14099 job-shared, 15000-15799 per-world, "
                "24000-24099/25000-25799 rabbitmq management)",
            )


def _verify_seed_missing(manifest: EnvironmentBundleV2) -> None:
    covered = {
        store.capability for store in (manifest.seed.stores if manifest.seed else [])
    }
    missing = sorted(
        slug
        for slug, capability in manifest.capabilities.items()
        if capability.protocol is CapabilityProtocol.POSTGRES and slug not in covered
    )
    if missing:
        raise PreflightError(
            "seed_missing",
            "postgres-protocol capability with no store entry: " + ", ".join(missing),
        )


def _verify_reserved_names(bundle_dir: Path, manifest: EnvironmentBundleV2) -> None:
    if manifest.seed is None:
        return
    root = bundle_dir.resolve()
    for store in manifest.seed.stores:
        for relative_path in (*store.migrations, *store.seed_files):
            path = root / relative_path
            if not path.is_file():
                continue  # reported by `_verify_seed_files_on_disk_and_listed`
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip `--`-to-EOL and `/* ... */` comments before scanning (F9, p4-round1-review) —
            # a generated seed file's own note about the reservation ("-- never create
            # _alk_conformance here") would otherwise trip the scan on prose, not on an identifier
            # it defines. Quoted string literals containing the name as *data* remain a known
            # false-positive surface: the scan has no lexer, only comment-stripping. The stripping
            # is a false-NEGATIVE surface in the opposite direction, equally lexer-free and equally
            # left as-is (N7, p4-round2-review; B4, p4-round3-review): a `--` or `/*` inside a
            # string literal (not a comment) deletes real content up to the next line-end or `*/`,
            # which can delete a reserved-name definition that follows it on the same statement.
            code = _SQL_BLOCK_COMMENT.sub("", _SQL_LINE_COMMENT.sub("", text))
            if _RESERVED_NAME_PATTERN.search(code):
                raise PreflightError(
                    "reserved_name",
                    f"{relative_path} defines the reserved conformance-canary identifier "
                    f"{_RESERVED_NAME!r}",
                )


def _verify_seed_files_on_disk_and_listed(
    bundle_dir: Path, manifest: EnvironmentBundleV2, files: list[BundleFileV2]
) -> None:
    """Digest verification (item 1) already guarantees every ``files[]``-listed path exists, so a
    path missing from disk entirely is ``seed_file_missing`` regardless of whether it was ever
    listed. ``seed_file_unlisted`` stays here as a second, store-scoped statement of the same
    "listed" rule item 2's own walk now enforces bundle-wide (F1, p4-round1-review) — through
    `preflight_bundle`'s full sequence item 2's ``bundle_file_unlisted`` always fires first for any
    file the walk visits; the root ``manifest.json`` is exempt from that walk, so a store path
    naming it still reaches here (N2, p4-round2-review).

    Once every migration/seed file for a store is confirmed present and listed, its recorded
    ``inputs_digest`` is recomputed and compared (F14, p4-round1-review; §2c makes it the baseline
    identity attempt-retry reuse trusts absolutely, and nothing else on either side of the seam
    ever validated it). ``engine``/``version`` come from the store's capability's own backing
    ``ManagedProcess`` — guaranteed to exist by `bundle_v2`'s ``store_service_not_managed`` check;
    a non-``ManagedProcess`` backing here would mean that guarantee broke, raised as a typed
    ``PreflightError`` rather than asserted, since this module's charter is a rejection on every
    path, never a crash (N8, p4-round2-review).
    """
    if manifest.seed is None:
        return
    listed = {record.path for record in files}
    root = bundle_dir.resolve()
    processes_by_name = {process.name: process for process in manifest.processes}
    for store in manifest.seed.stores:
        for relative_path in (*store.migrations, *store.seed_files):
            if not (root / relative_path).is_file():
                raise PreflightError(
                    "seed_file_missing", f"{relative_path} does not exist on disk"
                )
            if relative_path not in listed:
                raise PreflightError(
                    "seed_file_unlisted", f"{relative_path} is not listed in files[]"
                )
        capability = manifest.capabilities[store.capability]
        engine_process = processes_by_name[capability.service]
        if not isinstance(engine_process, ManagedProcess):
            raise PreflightError(
                "store_service_not_managed",
                f"{store.capability}: service {capability.service!r} is not a managed engine",
            )
        recomputed = compute_inputs_digest(
            root,
            store.migrations,
            store.seed_files,
            engine=engine_process.engine,
            version=engine_process.version,
        )
        if recomputed != store.baseline.inputs_digest:
            raise PreflightError(
                "inputs_digest_mismatch",
                f"{store.capability}: expected {store.baseline.inputs_digest}, computed "
                f"{recomputed}",
            )


# --- item 6: no_sql_store ------------------------------------------------------------------------


def _verify_no_sql_store(manifest: EnvironmentBundleV2) -> None:
    if manifest.runtime.kind is not RuntimeKindV2.PROCESS:
        return
    if not any(
        capability.protocol is CapabilityProtocol.POSTGRES
        for capability in manifest.capabilities.values()
    ):
        raise PreflightError(
            "no_sql_store",
            "kind: process requires at least one postgres-protocol capability",
        )


# --- item 7: resource sanity ----------------------------------------------------------------------


def _verify_resource_sanity(manifest: EnvironmentBundleV2, *, parallelism: int) -> None:
    if len(manifest.processes) > _MAX_PROCESSES:
        raise PreflightError(
            "process_count_exceeded",
            f"{len(manifest.processes)} processes exceeds the {_MAX_PROCESSES} cap",
        )
    if not (_MIN_PARALLELISM <= parallelism <= _MAX_PARALLELISM):
        raise PreflightError(
            "parallelism_out_of_range",
            f"parallelism={parallelism} is outside {_MIN_PARALLELISM}..{_MAX_PARALLELISM}",
        )


__all__ = ["PreflightError", "preflight_bundle"]
