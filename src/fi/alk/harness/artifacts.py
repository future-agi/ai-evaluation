"""Content-addressed artifact manifests and call-evidence integrity checks."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path, PurePosixPath
from pydantic import BaseModel, Field, model_validator

ARTIFACT_MANIFEST_VERSION = "futureagi.harness-artifacts.v1"
ARTIFACT_MANIFEST_NAME = "artifact-manifest.json"


class ArtifactIntegrityError(RuntimeError):
    pass


class ArtifactRecord(BaseModel):
    path: str
    sha256: str
    size: int = Field(ge=0)
    kind: str
    media_type: str
    scenario: str | None = None

    @model_validator(mode="after")
    def _safe_path(self) -> "ArtifactRecord":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or not self.path:
            raise ValueError(f"artifact_path_unsafe: {self.path}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(f"artifact_digest_invalid: {self.path}")
        return self


class ScenarioEvidence(BaseModel):
    scenario: str
    result_path: str
    canonical_status: str
    transcript_path: str | None = None
    recording_paths: list[str] = Field(default_factory=list)
    tool_trace_path: str | None = None


class ArtifactManifest(BaseModel):
    schema_version: str = ARTIFACT_MANIFEST_VERSION
    run_id: str
    digest: str
    files: list[ArtifactRecord] = Field(default_factory=list)
    scenarios: list[ScenarioEvidence] = Field(default_factory=list)
    total_bytes: int = Field(ge=0)
    complete: bool = True

    @model_validator(mode="after")
    def _valid(self) -> "ArtifactManifest":
        if self.schema_version != ARTIFACT_MANIFEST_VERSION:
            raise ValueError("artifact_manifest_version_unsupported")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest):
            raise ValueError("artifact_manifest_digest_invalid")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact_manifest_duplicate_path")
        return self


_SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
_SECRET_FILES = {".env", ".env.local", ".npmrc", ".pypirc", "id_rsa"}
_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "canceled",
    # Chat conversations use domain-specific endings. All three mean execution
    # stopped and produced durable evidence; only ``finished`` implies that the
    # simulated user achieved its conversational goal. Artifact sealing must
    # preserve the other two as findings rather than turn them into an
    # infrastructure failure.
    "finished",
    "gave-up",
    "ran-out-of-turns",
}


def seal_artifacts(
    root: str | Path,
    *,
    run_id: str,
    max_bytes: int = 1_073_741_824,
    expected_scenarios: int | None = None,
) -> ArtifactManifest:
    """Validate evidence, hash every retained file, and atomically seal the directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactIntegrityError(f"artifact_root_missing: {root}")
    records: list[ArtifactRecord] = []
    total = 0
    for path in sorted(root.rglob("*")):
        # The outbox remains append-only until the platform acknowledges the terminal event.
        # It is transport state, not immutable run evidence, and is therefore not sealed.
        if (
            path.name in {ARTIFACT_MANIFEST_NAME, "harness-events.jsonl"}
            or path.is_dir()
        ):
            continue
        if path.is_symlink():
            raise ArtifactIntegrityError(
                f"artifact_symlink_forbidden: {path.relative_to(root)}"
            )
        relative = path.relative_to(root).as_posix()
        if path.name in _SECRET_FILES or path.suffix.lower() in {
            ".pem",
            ".key",
            ".p12",
            ".pfx",
        }:
            raise ArtifactIntegrityError(f"artifact_secret_file_forbidden: {relative}")
        digest, size = _hash_file(path, relative)
        total += size
        if total > max_bytes:
            raise ArtifactIntegrityError(
                f"artifact_size_limit_exceeded: {total} > {max_bytes}"
            )
        records.append(
            ArtifactRecord(
                path=relative,
                sha256=digest,
                size=size,
                kind=_kind(path),
                media_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                scenario=_scenario(relative),
            )
        )

    scenarios = _scenario_evidence(root)
    if expected_scenarios is not None and len(scenarios) != expected_scenarios:
        raise ArtifactIntegrityError(
            "artifact_scenario_count_mismatch: "
            f"expected {expected_scenarios}, found {len(scenarios)}"
        )
    core = {
        "schema_version": ARTIFACT_MANIFEST_VERSION,
        "run_id": run_id,
        "files": [record.model_dump(mode="json") for record in records],
        "scenarios": [item.model_dump(mode="json") for item in scenarios],
        "total_bytes": total,
        "complete": True,
    }
    digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    manifest = ArtifactManifest(**core, digest=digest)
    target = root / ARTIFACT_MANIFEST_NAME
    temporary = root / f".{ARTIFACT_MANIFEST_NAME}.tmp-{os.getpid()}"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return manifest


def load_artifact_manifest(
    root: str | Path, *, verify: bool = True
) -> ArtifactManifest:
    root = Path(root).expanduser().resolve()
    path = root / ARTIFACT_MANIFEST_NAME
    try:
        manifest = ArtifactManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ArtifactIntegrityError(f"artifact_manifest_invalid: {exc}") from exc
    if verify:
        for record in manifest.files:
            candidate = (root / record.path).resolve()
            if root not in candidate.parents or not candidate.is_file():
                raise ArtifactIntegrityError(f"artifact_missing: {record.path}")
            digest, size = _hash_file(candidate, record.path)
            if digest != record.sha256 or size != record.size:
                raise ArtifactIntegrityError(f"artifact_changed: {record.path}")
        expected = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    manifest.model_dump(mode="json", exclude={"digest"}),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        if expected != manifest.digest:
            raise ArtifactIntegrityError("artifact_manifest_digest_mismatch")
    return manifest


def _scenario_evidence(root: Path) -> list[ScenarioEvidence]:
    evidence: list[ScenarioEvidence] = []
    for result_path in sorted((root / "runs").glob("*/*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"result_invalid: {result_path.relative_to(root)}"
            ) from exc
        relative = result_path.relative_to(root).as_posix()
        scenario = result_path.parent.name
        status = str(
            result.get("ended") or (result.get("measured") or {}).get("status") or ""
        ).lower()
        if status not in _TERMINAL_STATUSES:
            raise ArtifactIntegrityError(
                f"result_not_terminal: {scenario}: {status or 'missing'}"
            )
        transcript_path = result_path.parent / "transcript.txt"
        transcript = str(result.get("transcript") or "").strip()
        if not transcript and (
            not transcript_path.is_file()
            or not transcript_path.read_text(encoding="utf-8").strip()
        ):
            raise ArtifactIntegrityError(f"transcript_missing: {scenario}")

        recordings: list[str] = []
        for track in result.get("tracks") or []:
            raw = str(track.get("path") or "") if isinstance(track, dict) else ""
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = result_path.parent / candidate
            candidate = candidate.resolve()
            if (
                root not in candidate.parents
                or not candidate.is_file()
                or candidate.stat().st_size == 0
            ):
                raise ArtifactIntegrityError(f"recording_missing: {scenario}: {raw}")
            recordings.append(candidate.relative_to(root).as_posix())
        if (
            result.get("recording") or (result.get("measured") or {}).get("room")
        ) and not recordings:
            raise ArtifactIntegrityError(f"recording_evidence_missing: {scenario}")

        tool_trace = result_path.parent / "agent-tool-calls.jsonl"
        evidence.append(
            ScenarioEvidence(
                scenario=scenario,
                result_path=relative,
                canonical_status=status,
                transcript_path=(
                    transcript_path.relative_to(root).as_posix()
                    if transcript_path.is_file()
                    else None
                ),
                recording_paths=sorted(recordings),
                tool_trace_path=(
                    tool_trace.relative_to(root).as_posix()
                    if tool_trace.is_file()
                    else None
                ),
            )
        )
    return evidence


def _hash_file(path: Path, relative: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            if any(pattern.search(chunk) for pattern in _SECRET_CONTENT):
                raise ArtifactIntegrityError(
                    f"artifact_secret_material_detected: {relative}"
                )
    return digest.hexdigest(), size


def _kind(path: Path) -> str:
    if path.name == "result.json":
        return "result"
    if path.name == "transcript.txt":
        return "transcript"
    if path.name == "agent-tool-calls.jsonl":
        return "tool_trace"
    if path.suffix.lower() in {".wav", ".mp3", ".ogg", ".webm", ".m4a"}:
        return "recording"
    if path.name == "harness-events.jsonl":
        return "event_stream"
    return "artifact"


def _scenario(relative: str) -> str | None:
    parts = PurePosixPath(relative).parts
    if len(parts) >= 4 and parts[0] == "runs":
        return parts[2]
    return None


__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "ARTIFACT_MANIFEST_VERSION",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ArtifactRecord",
    "ScenarioEvidence",
    "load_artifact_manifest",
    "seal_artifacts",
]
