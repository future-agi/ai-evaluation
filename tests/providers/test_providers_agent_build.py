from __future__ import annotations

from typing import Any

import pytest

from fi.alk.harness.provider_import import _safe_profile_value
from fi.alk.providers.agent import _redact

PLAIN_DOCUMENTS: list[Any] = [
    {"name": "Support line", "voice": {"voiceId": "marissa", "speed": 1.0}},
    {
        "api_key": "sk-live-1",
        "APIKey": "x",
        "Authorization": "Bearer y",
        "client_secret": 3,
    },
    {"credentials": {"user": "u"}, "tokens": ["a", "b"], "refresh_token": None},
    {"server": {"url": "https://user:pw@hooks.example.com:8443/call?sig=abc#frag"}},
    {"links": ["http://example.com/a?b=c", "not a url", "mailto:someone@example.com"]},
    {"nested": [{"deep": {"secretValue": "s"}}, 1, True, None, 2.5]},
    ["https://example.com/x?y=1", {"api_key": "k"}],
    "plain string",
    42,
]


@pytest.mark.parametrize("document", PLAIN_DOCUMENTS)
def test_redaction_equals_safe_profile_value_on_plain_documents(document: Any) -> None:
    assert _redact(document) == _safe_profile_value(document)
