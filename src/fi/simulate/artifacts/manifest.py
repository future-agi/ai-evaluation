from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue, model_validator

from fi.simulate.agent.wrapper import ArtifactType
from fi.simulate.evidence import EvidenceClass
from fi.simulate._hashing import content_hash

ARTIFACT_MANIFEST_SCHEMA_VERSION = "futureagi.artifact-manifest.v1"


class ArtifactManifestEntry(BaseModel):
    artifact_id: str
    test_case_id: str | None = None
    type: ArtifactType
    path: str | None = None
    uri: str | None = None
    checksum: str
    size_bytes: int = Field(ge=0)
    mime_type: str | None = None
    codec: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    participant_id: str | None = None
    track_id: str | None = None
    leg_id: str | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    evidence_class: EvidenceClass
    evidence_source_id: str
    redacted: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_location(self) -> "ArtifactManifestEntry":
        if not self.path and not self.uri:
            raise ValueError("artifact_location_missing: path or uri is required")
        if self.path and self.uri:
            raise ValueError("artifact_location_ambiguous: provide path or uri, not both")
        if not self.checksum.startswith("sha256:"):
            raise ValueError("artifact_checksum_invalid: checksum must use sha256:<digest>")
        return self


class ArtifactManifest(BaseModel):
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION
    run_id: str
    entries: list[ArtifactManifestEntry] = Field(default_factory=list)
    manifest_hash: str | None = None

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"manifest_hash"}, exclude_none=True)
        payload["entries"] = sorted(payload["entries"], key=lambda item: item["artifact_id"])
        return content_hash(payload)

    @model_validator(mode="after")
    def _stamp_hash(self) -> "ArtifactManifest":
        expected = self.content_hash()
        if self.manifest_hash is not None and self.manifest_hash != expected:
            raise ValueError("artifact_manifest_hash_mismatch")
        object.__setattr__(self, "manifest_hash", expected)
        return self
