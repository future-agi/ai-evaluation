from __future__ import annotations

import base64
import binascii
import re
import hashlib
import mmap
import os
from pathlib import Path, PurePosixPath
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)

# A complete ellipsis-only example contains no key; malformed key bodies still fail closed.
SECRET_CONTENT_PATTERNS = (
    re.compile(
        rb"-----BEGIN (?P<kind>(?:RSA |EC |OPENSSH )?PRIVATE KEY)-----"
        rb"(?!(?:[ \t\r\n]|\\n)*\.\.\.(?:[ \t\r\n]|\\n)*-----END (?P=kind)-----)"
    ),
    re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
)

_MAX_PUBLIC_PEM_BYTES = 64 * 1024
_PUBLIC_PEM = re.compile(
    rb"\s*-----BEGIN (PUBLIC KEY|RSA PUBLIC KEY|CERTIFICATE)-----\r?\n"
    rb"([A-Za-z0-9+/=\r\n]+)-----END \1-----\s*"
)


def is_public_pem(path: Path) -> bool:
    if path.suffix.lower() != ".pem" or path.is_symlink():
        return False
    with path.open("rb") as stream:
        data = stream.read(_MAX_PUBLIC_PEM_BYTES + 1)
    if len(data) > _MAX_PUBLIC_PEM_BYTES:
        return False
    matches = list(_PUBLIC_PEM.finditer(data))
    if not matches or b"".join(match[0] for match in matches) != data:
        return False
    if len(matches) > 1 and any(match[1] != b"CERTIFICATE" for match in matches):
        return False
    try:
        for match in matches:
            der = base64.b64decode(re.sub(rb"\s", b"", match[2]), validate=True)
            if match[1] == b"CERTIFICATE":
                parsed = x509.load_pem_x509_certificate(match[0]).public_bytes(
                    Encoding.DER
                )
            else:
                format = (
                    PublicFormat.PKCS1
                    if match[1] == b"RSA PUBLIC KEY"
                    else PublicFormat.SubjectPublicKeyInfo
                )
                parsed = load_pem_public_key(match[0]).public_bytes(
                    Encoding.DER, format
                )
            if der != parsed:
                return False
    except (ValueError, UnsupportedAlgorithm, binascii.Error):
        return False
    return True


class ReviewedPublicSourceFile(BaseModel):
    """Operator evidence bound to the complete immutable source file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str = Field(min_length=1, max_length=4096)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1, max_length=2000)
    evidence: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _safe_path(self):
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or not path.parts
            or any(ord(char) < 32 for char in self.path)
            or path.as_posix() != self.path
            or ".." in path.parts
            or "\\" in self.path
            or ":" in self.path
        ):
            raise ValueError("public_material_admission_path_unsafe")
        for value in (self.reason, self.evidence):
            if any(
                pattern.search(value.encode()) for pattern in SECRET_CONTENT_PATTERNS
            ):
                raise ValueError("public_material_admission_raw_match_forbidden")
        return self


class PublicMaterialAdmission(ReviewedPublicSourceFile):
    pattern: Literal["github_token", "aws_access_key", "sk_token"]
    match_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicTestFixtureAdmission(ReviewedPublicSourceFile):
    """Whole-file review of publicly committed fixed cryptographic test material."""

    purpose: Literal["public_test_fixture"] = "public_test_fixture"


_PATTERN_NAMES = ("private_key", "github_token", "aws_access_key", "sk_token")


def inspect_public_file(
    path: Path,
    admissions: Sequence[PublicMaterialAdmission] = (),
    *,
    fixture: PublicTestFixtureAdmission | None = None,
) -> tuple[str, int]:
    """Scan complete files so tokens crossing read boundaries cannot escape review."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                digest = hashlib.sha256(data).hexdigest()
                allowed = {(item.pattern, item.match_sha256) for item in admissions}
                if any(item.file_sha256 != digest for item in admissions):
                    raise ValueError("public_material_admission_file_changed")
                if fixture is not None and fixture.file_sha256 != digest:
                    raise ValueError("public_test_fixture_file_changed")
                observed = set()
                for name, pattern in zip(_PATTERN_NAMES, SECRET_CONTENT_PATTERNS):
                    for match in pattern.finditer(data):
                        if name == "private_key" and fixture is not None:
                            continue
                        finding = (name, hashlib.sha256(match[0]).hexdigest())
                        if finding not in allowed or name == "private_key":
                            raise ValueError("public_material_unreviewed_match")
                        observed.add(finding)
                if observed != allowed:
                    raise ValueError("public_material_admission_match_missing")
        else:
            digest = hashlib.sha256(b"").hexdigest()
            if admissions:
                raise ValueError("public_material_admission_match_missing")
            if fixture is not None and fixture.file_sha256 != digest:
                raise ValueError("public_test_fixture_file_changed")
    return digest, size


def is_public_npm_config(path: Path) -> bool:
    """Recognize a bounded public release-policy subset, never npm authentication."""
    if path.name.lower() != ".npmrc" or path.is_symlink():
        return False
    with path.open("rb") as stream:
        data = stream.read(64 * 1024 + 1)
    if len(data) > 64 * 1024:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        return False
    if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
        return False
    seen_age = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator:
            return False
        if key == "min-release-age":
            if seen_age or not re.fullmatch(r"[0-9]{1,6}", value):
                return False
            seen_age = True
        elif key == "min-release-age-exclude[]":
            if len(value) > 214 or not re.fullmatch(
                r"(?:@[a-z0-9][a-z0-9._~-]*/)?[a-z0-9][a-z0-9._~-]*\*?", value
            ):
                return False
        else:
            return False
    try:
        inspect_public_file(path)
    except ValueError:
        return False
    return True
