import hashlib

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import ValidationError

from fi.alk.harness.bundle import BundleError
from fi.alk.harness.compose_integration import validate_source
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.provision import source_fingerprint
from fi.alk.harness.public_material import PublicTestFixtureAdmission
from fi.alk.harness.artifacts import seal_artifacts, ArtifactIntegrityError
from fi.alk.harness.generated_runtime import (
    _copy_sanitized_context,
    GeneratedRuntimeError,
)


def fixture_review(source, relative):
    return PublicTestFixtureAdmission(
        source_digest=source_fingerprint(source),
        path=relative,
        file_sha256=hashlib.sha256((source / relative).read_bytes()).hexdigest(),
        reason="Entire public fixture reviewed for fixed local fake-server use.",
        evidence="https://example.org/public-source/pinned-revision",
    )


def private_key():
    return ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.mark.parametrize(
    "relative", ["chat/tests/server.ts", "tests/network/material.py"]
)
def test_whole_file_review_covers_multiple_fixed_keys_and_invalidates_edits(
    tmp_path, relative
):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    data = private_key() + private_key()
    path.write_bytes(data)
    with pytest.raises(BundleError, match="secret_material"):
        validate_source(tmp_path)
    approval = fixture_review(tmp_path, relative)
    validate_source(tmp_path, fixtures=[approval])
    assert approval.purpose == "public_test_fixture"
    assert b"BEGIN PRIVATE KEY" not in approval.model_dump_json().encode()
    path.write_bytes(data + private_key())
    with pytest.raises(ValueError, match="source_changed"):
        validate_source(tmp_path, fixtures=[approval])
    source_updated = approval.model_copy(
        update={"source_digest": source_fingerprint(tmp_path)}
    )
    with pytest.raises(BundleError, match="file_changed"):
        validate_source(tmp_path, fixtures=[source_updated])


def test_binary_key_fixture_and_synthetic_marker_need_explicit_review(tmp_path):
    (tmp_path / "corpus.key").write_bytes(bytes(range(32)))
    (tmp_path / "negative-test.ts").write_text(
        "-----BEGIN PRIVATE KEY----- invalid test input"
    )
    with pytest.raises(BundleError):
        validate_source(tmp_path)
    approvals = [
        fixture_review(tmp_path, name) for name in ["corpus.key", "negative-test.ts"]
    ]
    validate_source(tmp_path, fixtures=approvals)
    with pytest.raises(ValueError, match="duplicate_path"):
        validate_source(tmp_path, fixtures=[approvals[0], *approvals])
    (tmp_path / "unrelated.txt").write_text("source changed")
    with pytest.raises(ValueError, match="source_changed"):
        validate_source(tmp_path, fixtures=approvals)


def test_fixture_review_does_not_authorize_other_token_findings_or_exports(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "test.ts"
    path.write_bytes(private_key())
    review = fixture_review(source, "test.ts")
    validate_source(source, fixtures=[review])
    with pytest.raises(ArtifactIntegrityError):
        seal_artifacts(source, run_id="test")
    with pytest.raises(GeneratedRuntimeError):
        _copy_sanitized_context(source, tmp_path / "generated")
    path.write_bytes(path.read_bytes() + b"sk-unapprovedExampleValue123456789")
    review = fixture_review(source, "test.ts")
    with pytest.raises(BundleError, match="unreviewed_match"):
        validate_source(source, fixtures=[review])


@pytest.mark.parametrize(
    "name", [".env", ".pypirc", "credentials", "id_rsa", "private.pem"]
)
def test_fixture_review_cannot_authorize_other_credential_filenames(tmp_path, name):
    (tmp_path / name).write_bytes(private_key())
    with pytest.raises(BundleError, match="secret_file_forbidden"):
        validate_source(tmp_path, fixtures=[fixture_review(tmp_path, name)])


def test_fixture_job_is_local_only_and_paths_cannot_escape(tmp_path):
    (tmp_path / "test.ts").write_bytes(private_key())
    review = fixture_review(tmp_path, "test.ts")
    raw = dict(
        job_id="test",
        run_id="test",
        execution="local",
        source={"kind": "local_repository", "local_path": str(tmp_path)},
        agent={"connector": "auto"},
        public_test_fixture_admissions=[review.model_dump()],
    )
    assert HarnessJob.model_validate(raw).public_test_fixture_admissions == [review]
    with pytest.raises(ValidationError, match="local_operator_required"):
        HarnessJob.model_validate({**raw, "execution": "hosted"})
    for path in ["../outside", "/absolute", "dir/../test", "a\\b"]:
        with pytest.raises(ValidationError, match="path_unsafe"):
            PublicTestFixtureAdmission.model_validate(
                {**review.model_dump(), "path": path}
            )


def test_fixture_and_token_reviews_must_both_match(tmp_path):
    from fi.alk.harness.public_material import PublicMaterialAdmission

    token = b"sk-fixedPublicExampleValue123456789"
    path = tmp_path / "test.ts"
    path.write_bytes(private_key() + token)
    fixture = fixture_review(tmp_path, "test.ts")
    token_review = PublicMaterialAdmission(
        **fixture.model_dump(exclude={"purpose"}),
        pattern="sk_token",
        match_sha256=hashlib.sha256(token).hexdigest(),
    )
    validate_source(tmp_path, [token_review], fixtures=[fixture])
    path.rename(tmp_path / "moved.ts")
    changed = fixture.model_copy(update={"source_digest": source_fingerprint(tmp_path)})
    with pytest.raises((ValueError, BundleError)):
        validate_source(tmp_path, fixtures=[changed])


def test_fixture_provenance_is_bound_and_non_compose_is_rejected(tmp_path):
    from fi.alk.harness.bundle_v2 import (
        BundleProvenanceV2,
        EnvironmentBundleV2,
        BUNDLE_V2_SCHEMA_VERSION,
    )

    (tmp_path / "test.ts").write_bytes(private_key())
    review = fixture_review(tmp_path, "test.ts")
    with pytest.raises(ValidationError, match="source_changed"):
        BundleProvenanceV2(
            source_kind="local",
            source_digest="0" * 64,
            public_test_fixture_admissions=[review],
        )
    provenance = BundleProvenanceV2(
        source_kind="local",
        source_digest=review.source_digest,
        public_test_fixture_admissions=[review],
    )
    with pytest.raises(ValidationError, match="requires_native_compose"):
        EnvironmentBundleV2(
            schema_version=BUNDLE_V2_SCHEMA_VERSION,
            digest="sha256:" + "0" * 64,
            name="test",
            runtime={"kind": "external"},
            provenance=provenance,
        )
