"""A run clones the provider assistant the agent was adapted as, or refuses before creating one."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fi.alk.harness.provider_import import (
    ProviderImportError,
    ProviderImportSpec,
    clone_provider_target,
    inspect_provider_target,
    provider_profile_digest,
)
from fi.alk.harness.provider_lifecycle import ProviderContext

CONTEXT = ProviderContext(
    attempt_id="attempt-1",
    world_id="0",
    provider="vapi",
    public_base_url="https://world.example",
    event_url="https://world.example/provider/events",
    tool_base_url="https://world.example/provider/tools",
    provider_resource_prefix="alk-attempt-1-w0",
    idempotency_key="attempt-1:0:vapi",
    expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


def _provider(prompt: str, tool_parameters: dict):
    calls: list[tuple[str, str]] = []

    def request(method, url, api_key, body):
        calls.append((method, url))
        if method == "GET" and "/assistant/" in url:
            return {
                "id": "source-assistant",
                "name": "Front desk",
                "model": {
                    "provider": "openai",
                    "messages": [{"role": "system", "content": prompt}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "book_appointment",
                                "parameters": tool_parameters,
                            },
                        }
                    ],
                },
            }
        if method == "POST" and url.endswith("/assistant"):
            return {"id": "cloned-assistant"}
        return {}

    return request, calls


BOOK = {
    "type": "object",
    "properties": {"patient_id": {"type": "string"}, "slot_id": {"type": "string"}},
}


def _adapted_digest() -> str:
    request, _ = _provider("Book appointments.", BOOK)
    return provider_profile_digest(
        inspect_provider_target(
            "vapi", source_target_id="source-assistant", api_key="key", request=request
        )
    )


def _spec(digest: str) -> ProviderImportSpec:
    return ProviderImportSpec(
        type="vapi",
        source_target_id="source-assistant",
        public_capability="tools",
        environment_tools=["book_appointment"],
        profile_digest=digest,
    )


def test_the_adapted_assistant_is_cloned():
    request, _ = _provider("Book appointments.", BOOK)
    receipt = clone_provider_target(
        _spec(_adapted_digest()), context=CONTEXT, api_key="key", request=request
    )
    assert receipt.target.id == "cloned-assistant"


@pytest.mark.parametrize(
    ("prompt", "parameters"),
    [
        ("Book appointments, and upsell dermatology.", BOOK),
        ("Book appointments.", {**BOOK, "properties": {"patient": {"type": "string"}}}),
    ],
    ids=["prompt-changed", "tool-changed"],
)
def test_an_assistant_changed_since_adaptation_is_refused_before_anything_is_created(
    prompt, parameters
):
    request, calls = _provider(prompt, parameters)
    with pytest.raises(ProviderImportError) as raised:
        clone_provider_target(
            _spec(_adapted_digest()), context=CONTEXT, api_key="key", request=request
        )
    assert str(raised.value).startswith("provider_configuration_changed")
    assert not [call for call in calls if call[0] != "GET"]
