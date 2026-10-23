from pathlib import Path

import pytest

from fi.alk.harness.artifacts import ArtifactIntegrityError, _hash_file
from fi.alk.harness.bundle import BundleError, _inspect_file
from fi.alk.harness.generated_runtime import (
    GeneratedRuntimeError,
    _copy_sanitized_context,
)
from fi.alk.harness.process_preflight import (
    PreflightError,
    _scan_bundle_files_for_secrets,
)


def scan(kind: str, root: Path, path: Path) -> None:
    if kind == "bundle":
        _inspect_file(root, path)
    elif kind == "artifacts":
        _hash_file(path, path.name)
    elif kind == "preflight":
        _scan_bundle_files_for_secrets(root, [path])
    else:
        destination = root.parent / "copy"
        _copy_sanitized_context(root, destination)
        assert (destination / path.name).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("kind", ["bundle", "artifacts", "preflight", "generated"])
@pytest.mark.parametrize(
    "key_type",
    ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"],
)
@pytest.mark.parametrize("separator", ["", "\n", r"\n"])
def test_complete_ellipsis_example_is_not_secret(tmp_path, kind, key_type, separator):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "documentation.txt"
    path.write_text(
        f'# EXAMPLE="-----BEGIN {key_type}-----{separator}...{separator}-----END {key_type}-----"'
    )
    scan(kind, root, path)


@pytest.mark.parametrize("kind", ["bundle", "artifacts", "preflight", "generated"])
@pytest.mark.parametrize(
    "content",
    [
        b"-----BEGIN PRIVATE KEY-----",
        b"-----BEGIN PRIVATE KEY-----...-----END RSA PRIVATE KEY-----",
        b"-----BEGIN PRIVATE KEY-----...not-a-key-----END PRIVATE KEY-----",
        b"-----BEGIN PRIVATE KEY-----AAAA-----END PRIVATE KEY-----",
        b"-----BEGIN PRIVATE KEY-----....-----END PRIVATE KEY-----",
        b"-----BEGIN PRIVATE KEY-----...-----END PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----real-material",
        b"-----BEGIN PRIVATE KEY-----...-----END PRIVATE KEY-----\n"
        + b"ghp_"
        + b"A" * 25,
    ],
)
def test_nonplaceholder_or_other_secret_still_rejected(tmp_path, kind, content):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "documentation.txt"
    path.write_bytes(content)
    with pytest.raises(
        (BundleError, ArtifactIntegrityError, PreflightError, GeneratedRuntimeError)
    ):
        scan(kind, root, path)
