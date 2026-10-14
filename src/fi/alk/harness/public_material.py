from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

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
