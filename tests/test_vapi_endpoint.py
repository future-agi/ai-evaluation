from __future__ import annotations

import asyncio

import httpx
import pytest

from fi.simulate.endpoints.vapi import VapiCallOriginator


def test_vapi_originator_posts_existing_resource_ids() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = request.content
        return httpx.Response(201, json={"id": "call_123", "status": "queued"})

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            transport=httpx.MockTransport(handler),
        )
        originator = VapiCallOriginator(
            api_key="test-key",
            assistant_id="assistant_123",
            phone_number_id="phone_123",
            destination="+12065550100",
            client=client,
        )
        call = await originator.start()
        await client.aclose()
        assert call.call_id == "call_123"
        assert call.status == "queued"

    asyncio.run(run())

    assert captured["path"] == "/call"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == (
        b'{"assistantId":"assistant_123","phoneNumberId":"phone_123",'
        b'"customer":{"number":"+12065550100"}}'
    )


def test_vapi_originator_rejects_missing_response_id() -> None:
    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            transport=httpx.MockTransport(lambda _: httpx.Response(201, json={})),
        )
        originator = VapiCallOriginator(
            api_key="test-key",
            assistant_id="assistant_123",
            phone_number_id="phone_123",
            destination="+12065550100",
            client=client,
        )
        with pytest.raises(ValueError, match="vapi_call_response_missing_id"):
            await originator.start()
        await client.aclose()

    asyncio.run(run())
