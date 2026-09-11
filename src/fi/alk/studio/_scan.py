"""Content scan for pulled artifacts (Phase 7, ARCH §2g).

Downloaded personas/scenarios are persistent content entering agent context —
a stored-injection channel (R§1 2606.04425). Every string field is scanned
before admission against rule classes aligned with the gate-enforced red-team
corpus taxonomy (``V1_REDTEAM_RESEARCH_ATTACK_TYPES`` × ``..._SURFACES``),
plus secret-marker and executable-content smells.

Two-level encoding (ARCH §4 canon): scan result tokens are ``clean`` |
``flagged``; a flagged artifact's ENVELOPE disposition is ``quarantined``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

CONTENT_SCAN_RESULTS = ("clean", "flagged")
SCAN_RULES_VERSION = "redteam_corpus_taxonomy"

# (substring-pattern, attack_type, surface) — attack types/surfaces are
# members of the gate-enforced 10x6 taxonomy (trinity.py:1897-1917), reused
# as rule classes, never duplicated as a new vocabulary.
STORED_INJECTION_MARKERS: Tuple[Tuple[str, str, str], ...] = (
    ("ignore previous instructions", "prompt_injection", "instruction"),
    ("ignore earlier instructions", "prompt_injection", "instruction"),
    ("ignore the scenario constraints", "indirect_prompt_injection", "instruction"),
    ("disregard the system prompt", "prompt_injection", "instruction"),
    ("reveal the system prompt", "prompt_injection", "instruction"),
    ("reveal any hidden policy", "prompt_injection", "instruction"),
    ("when you read this later", "memory_poisoning", "memory"),
    ("store this instruction for future", "memory_poisoning", "memory"),
    ("remember this for all future conversations", "sleeper_memory_poisoning", "memory"),
    ("fetch the url and execute", "tool_chaining", "tool"),
    ("run this command", "tool_chaining", "tool"),
    ("exfiltrate", "credential_exfiltration", "environment"),
    ("send the credentials", "credential_exfiltration", "environment"),
)

# Secret markers: NAMES only, never values (Phase-3 redaction doctrine).
_SECRET_MARKERS = (
    "api_key=", "secret_key=", "password:", "-----begin private key-----",
    "x-api-key:", "bearer ",
)

_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/=]{120,}")
_CREDENTIAL_URL_RE = re.compile(r"https?://\S*(?:key|token|secret)=\S+", re.IGNORECASE)
_EXCERPT_LIMIT = 80


class DownloadRejected(RuntimeError):
    """A pulled payload was scan-flagged — envelope disposition quarantined."""

    def __init__(self, message: str, *, findings: List[Dict[str, Any]], pin: Any = None):
        super().__init__(message)
        self.findings = findings
        self.pin = pin
        self.disposition = "quarantined"


def _walk_strings(value: Any, path: str = "") -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    if isinstance(value, str):
        found.append((path or "$", value))
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda kv: str(kv[0])):
            child = f"{path}.{key}" if path else str(key)
            found.extend(_walk_strings(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, f"{path}[{index}]"))
    return found


def _excerpt(text: str, needle: str) -> str:
    lowered = text.lower()
    start = max(0, lowered.find(needle.lower()) - 10)
    return text[start:start + _EXCERPT_LIMIT]


def scan_content(payload: Any, *, rules: str = SCAN_RULES_VERSION) -> Dict[str, Any]:
    """Walk every string field; return ``{"rules", "status", "findings"}``
    with status ``clean`` or ``flagged`` (ARCH §4 two-level encoding)."""
    findings: List[Dict[str, Any]] = []
    for field, text in _walk_strings(payload):
        lowered = text.lower()
        for pattern, attack_type, surface in STORED_INJECTION_MARKERS:
            if pattern in lowered:
                findings.append({
                    "field": field,
                    "attack_type": attack_type,
                    "surface": surface,
                    "excerpt": _excerpt(text, pattern),
                    "rule_source": rules,
                })
        for marker in _SECRET_MARKERS:
            if marker in lowered:
                findings.append({
                    "field": field,
                    "attack_type": "credential_exfiltration",
                    "surface": "environment",
                    "excerpt": marker,  # marker NAME only, never the value
                    "rule_source": "secret_markers",
                })
        if _BASE64_BLOB_RE.search(text):
            findings.append({
                "field": field,
                "attack_type": "tool_chaining",
                "surface": "long_context",
                "excerpt": "base64-like blob over length threshold",
                "rule_source": "executable_content_smells",
            })
        if _CREDENTIAL_URL_RE.search(text):
            findings.append({
                "field": field,
                "attack_type": "credential_exfiltration",
                "surface": "retrieval",
                "excerpt": "credential-bearing URL",
                "rule_source": "executable_content_smells",
            })
    return {
        "rules": rules,
        "status": "flagged" if findings else "clean",
        "findings": findings,
    }


__all__ = [
    "CONTENT_SCAN_RESULTS",
    "DownloadRejected",
    "SCAN_RULES_VERSION",
    "STORED_INJECTION_MARKERS",
    "scan_content",
]
