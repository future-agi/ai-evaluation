import base64
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from fi.alk.harness.artifacts import (
    ArtifactIntegrityError,
    load_artifact_manifest,
    seal_artifacts,
)
from fi.alk.harness.bundle import BundleError, _inspect_file
from fi.alk.harness.generated_runtime import _copy_sanitized_context
from fi.alk.harness.process_preflight import (
    PreflightError,
    _scan_bundle_files_for_secrets,
)

BOUNDARIES = ["bundle", "artifact", "preflight", "generated_context"]


@pytest.fixture(scope="module")
def pem_material():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fixture.invalid")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return {
        "public": key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
        "rsa_public": key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.PKCS1
        ),
        "certificate": cert.public_bytes(serialization.Encoding.PEM),
        "chain": cert.public_bytes(serialization.Encoding.PEM) * 3,
        "private": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        "encrypted": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"test-only"),
        ),
    }


def assert_boundary(source, key, boundary, allowed):
    if boundary == "generated_context":
        copied = source.parent / "context"
        _copy_sanitized_context(source, copied)
        assert (copied / key.name).exists() is allowed
        if allowed:
            assert (copied / key.name).read_bytes() == key.read_bytes()
        return

    def check():
        if boundary == "bundle":
            assert _inspect_file(source, key).size == key.stat().st_size
        elif boundary == "artifact":
            assert [
                item.path
                for item in seal_artifacts(source, run_id="public-material").files
            ] == [key.name]
            assert [item.path for item in load_artifact_manifest(source).files] == [
                key.name
            ]
        else:
            _scan_bundle_files_for_secrets(source, [key])

    if allowed:
        check()
    else:
        with pytest.raises(
            (BundleError, ArtifactIntegrityError, PreflightError),
            match="forbidden|secret_in_bundle",
        ):
            check()


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("kind", ["public", "rsa_public", "certificate", "chain"])
def test_public_material_survives_all_boundaries(
    tmp_path, pem_material, boundary, kind
):
    source = tmp_path / "source"
    source.mkdir()
    key = source / "verification.pem"
    key.write_bytes(pem_material[kind])
    assert_boundary(source, key, boundary, True)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize(
    "kind",
    [
        "private",
        "encrypted",
        "mixed",
        "duplicate",
        "malformed",
        "disguised_private",
        "oversized",
        "trailing",
        "hidden_der",
    ],
)
def test_nonpublic_pem_remains_blocked(tmp_path, pem_material, boundary, kind):
    public_der = base64.b64decode(b"".join(pem_material["public"].splitlines()[1:-1]))
    contents = {
        **pem_material,
        "mixed": pem_material["public"] + pem_material["private"],
        "duplicate": pem_material["public"] * 2,
        "malformed": b"-----BEGIN PUBLIC KEY-----\nYWJj\n-----END PUBLIC KEY-----\n",
        "disguised_private": pem_material["private"].replace(
            b"PRIVATE KEY", b"PUBLIC KEY"
        ),
        "oversized": pem_material["public"] + b" " * 65536,
        "trailing": pem_material["public"] + b"API_KEY=not-public",
        "hidden_der": b"-----BEGIN PUBLIC KEY-----\n"
        + base64.b64encode(public_der + b"hidden secret")
        + b"\n-----END PUBLIC KEY-----\n",
    }
    source = tmp_path / "source"
    source.mkdir()
    key = source / "verification.pem"
    key.write_bytes(contents[kind])
    assert_boundary(source, key, boundary, False)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize(
    "filename",
    [".env", "id_rsa", "verification.key", "verification.p12", "verification.pfx"],
)
def test_public_contents_do_not_override_credential_filenames(
    tmp_path, pem_material, boundary, filename
):
    source = tmp_path / "source"
    source.mkdir()
    key = source / filename
    key.write_bytes(pem_material["public"])
    assert_boundary(source, key, boundary, False)


@pytest.mark.parametrize("name", [".env", "credential.key", "credential.p12", "credential.pfx"])
def test_generated_copy_keeps_credential_name_policy_for_public_symlink(
    tmp_path, pem_material, name
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "public.pem").write_bytes(pem_material["public"])
    (source / name).symlink_to("public.pem")
    copied = tmp_path / "context"
    _copy_sanitized_context(source, copied)
    assert (copied / "public.pem").read_bytes() == pem_material["public"]
    assert not (copied / name).exists()
