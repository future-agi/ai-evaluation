"""Source-bound published image identities sealed before Compose certification."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contract import Runtime, runtime_public_environment
from .runtime_selection import ComposeSelection, select_runtime


_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
_PLATFORM = re.compile(
    r"linux/(?:amd64|arm64|arm|386|ppc64le|s390x|riscv64)(?:/v[0-9]+)?"
)
_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*")


def image_repository(reference: str) -> str:
    if not isinstance(reference, str) or not 1 <= len(reference) <= 512:
        raise ValueError("compose_image_reference_invalid")
    name, separator, digest = reference.partition("@")
    if separator and not _DIGEST.fullmatch(digest):
        raise ValueError("compose_image_digest_invalid")
    last = name.rsplit("/", 1)[-1]
    if ":" in last:
        tag = last.rsplit(":", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
            raise ValueError("compose_image_reference_invalid")
        name = name[: -(len(tag) + 1)]
    parts = name.split("/")
    registry = "docker.io"
    if len(parts) > 1 and (
        "." in parts[0] or ":" in parts[0] or parts[0] == "localhost"
    ):
        registry = parts.pop(0)
        if not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]{1,5})?", registry
        ):
            raise ValueError("compose_image_reference_invalid")
    if not parts or any(not _COMPONENT.fullmatch(part) for part in parts):
        raise ValueError("compose_image_reference_invalid")
    if registry in {"index.docker.io", "registry-1.docker.io"}:
        registry = "docker.io"
    if registry == "docker.io" and len(parts) == 1:
        parts.insert(0, "library")
    return registry + "/" + "/".join(parts)


def validate_image_platform(value: str) -> str:
    if not isinstance(value, str) or not _PLATFORM.fullmatch(value):
        raise ValueError("compose_image_platform_invalid")
    return value


class ResolvedComposeImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    service: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", max_length=128)
    declared_reference: str = Field(max_length=512)
    immutable_reference: str = Field(max_length=512)
    platform: str
    source_url: str | None = Field(default=None, max_length=2048)
    source_revision: str | None = Field(default=None, max_length=256)
    version: str | None = Field(default=None, max_length=256)
    correspondence: Literal["source_revision", "source_declared_digest", "dependency"]

    @field_validator("declared_reference", "immutable_reference")
    @classmethod
    def valid_reference(cls, value, info):
        image_repository(value)
        if info.field_name == "immutable_reference" and "@" not in value:
            raise ValueError("compose_image_digest_required")
        return value

    @field_validator("platform")
    @classmethod
    def valid_platform(cls, value):
        return validate_image_platform(value)

    @field_validator("source_url", "source_revision", "version")
    @classmethod
    def valid_label(cls, value):
        if value is not None and any(ord(char) < 32 for char in value):
            raise ValueError("compose_image_label_invalid")
        return value


def selected_images(
    source: Path, runtime: Runtime | dict, document: dict, *, control_service: str
) -> dict[str, dict]:
    raw = runtime.model_dump() if isinstance(runtime, Runtime) else runtime
    selection = select_runtime(
        source,
        raw,
        public_environment=runtime_public_environment(raw),
        validate_environment_dependencies=False,
    )
    services = document.get("services", {})
    if (
        not isinstance(selection, ComposeSelection)
        or selection.control_service != control_service
        or not isinstance(services, dict)
        or not 1 <= len(services) <= 64
        or set(services) != set(selection.services)
    ):
        raise ValueError("compose_image_source_selection_mismatch")
    images = {}
    for name, expected in selection.services.items():
        actual = services[name]
        if (
            not isinstance(actual, dict)
            or bool(actual.get("build")) != bool(expected.get("build"))
            or actual.get("image") != expected.get("image")
            or actual.get("platform") != expected.get("platform")
        ):
            raise ValueError("compose_image_source_selection_mismatch")
        if expected.get("build"):
            continue
        image_repository(expected.get("image"))
        if expected.get("platform"):
            validate_image_platform(expected["platform"])
        images[name] = expected
    return images


def source_repository(value: str) -> str:
    if "://" not in value:
        value = "https://github.com/" + value
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password:
        raise ValueError("compose_image_source_invalid")
    path = parsed.path.rstrip("/").removesuffix(".git")
    if parsed.hostname == "github.com":
        path = path.lower()
    return parsed.netloc.lower() + path


def image_identity(
    *,
    service,
    reference,
    platform,
    inspected,
    repository,
    revision,
    application,
    source_declared_digest=False,
):
    expected_repository = image_repository(reference)
    validate_image_platform(platform)
    actual_platform = inspected.get("Os", "") + "/" + inspected.get("Architecture", "")
    variant = inspected.get("Variant")
    if variant:
        actual_platform += "/" + variant
    if actual_platform != platform and not (
        len(platform.split("/")) == 2 and actual_platform.startswith(platform + "/")
    ):
        raise ValueError("compose_image_platform_mismatch")
    identities = set()
    for value in inspected.get("RepoDigests") or []:
        if image_repository(value) == expected_repository and "@" in value:
            identities.add(expected_repository + "@" + value.split("@", 1)[1])
    if "@" in reference:
        identities &= {expected_repository + "@" + reference.split("@", 1)[1]}
    if len(identities) != 1:
        raise ValueError("compose_image_digest_unresolved")
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    source_url = labels.get("org.opencontainers.image.source") or None
    source_revision = labels.get("org.opencontainers.image.revision") or None
    correspondence = "dependency"
    if application:
        if (
            source_url
            and repository
            and source_repository(source_url) != source_repository(repository)
        ):
            raise ValueError("compose_image_source_mismatch")
        if source_revision and revision and source_revision != revision:
            raise ValueError("compose_image_revision_mismatch")
        if (
            source_url
            and repository
            and source_revision
            and revision
            and re.fullmatch(r"[a-f0-9]{40,64}", revision)
        ):
            correspondence = "source_revision"
        elif "@" in reference and source_declared_digest:
            correspondence = "source_declared_digest"
        else:
            raise ValueError("compose_image_source_correspondence_missing")
    return ResolvedComposeImage(
        service=service,
        declared_reference=reference,
        immutable_reference=identities.pop(),
        platform=actual_platform,
        source_url=source_url,
        source_revision=source_revision,
        version=labels.get("org.opencontainers.image.version") or None,
        correspondence=correspondence,
    )


def lock_document(document: dict, identities: list[ResolvedComposeImage]) -> None:
    services = document["services"]
    expected = {name for name, value in services.items() if not value.get("build")}
    if (
        len(identities) != len(expected)
        or {item.service for item in identities} != expected
    ):
        raise ValueError("compose_image_lock_selection_mismatch")
    for item in identities:
        service = services[item.service]
        if (
            service.get("image") != item.declared_reference
            or image_repository(item.immutable_reference)
            != image_repository(item.declared_reference)
            or (
                service.get("platform")
                and service["platform"] != item.platform
                and not item.platform.startswith(service["platform"] + "/")
            )
        ):
            raise ValueError("compose_image_lock_selection_mismatch")
    for item in identities:
        services[item.service]["image"] = item.immutable_reference
        services[item.service]["platform"] = item.platform
