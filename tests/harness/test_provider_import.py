from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError

import pytest

import fi.alk.harness.provider_import as provider_import

from fi.alk.harness.provider_import import (
    ProviderImportError,
    ProviderImportSpec,
    clone_provider_target,
    destroy_imported_target,
    inspect_provider_target,
)
from fi.alk.harness.provider_lifecycle import ProviderContext


def _context(provider: str) -> ProviderContext:
    return ProviderContext(
        attempt_id="attempt-1",
        world_id="0",
        provider=provider,
        public_base_url="https://world.example",
        event_url="https://world.example/provider/events",
        tool_base_url="https://world.example/provider/tools",
        provider_resource_prefix="alk-attempt-1-w0",
        idempotency_key=f"attempt-1:0:{provider}",
        expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_provider_http_error_surfaces_bounded_response_without_credential(
    monkeypatch,
) -> None:
    secret = "do-not-leak"
    response = BytesIO(b'{"message":"source IP is not permitted"}')

    def fail(_request, *, timeout):
        assert timeout == 30
        raise HTTPError(
            "https://api.vapi.ai/assistant/source",
            403,
            "Forbidden",
            {},
            response,
        )

    monkeypatch.setattr(provider_import, "urlopen", fail)
    with pytest.raises(ProviderImportError) as raised:
        provider_import._request_json(
            "GET", "https://api.vapi.ai/assistant/source", secret, None
        )

    message = str(raised.value)
    assert "HTTP 403" in message
    assert "source IP is not permitted" in message
    assert secret not in message


def test_vapi_import_profile_includes_prompt_and_reusable_tools_without_secrets() -> (
    None
):
    def request(method, url, api_key, body):
        assert method == "GET"
        assert body is None
        assert api_key == "vapi-secret"
        if "/assistant/" in url:
            return {
                "name": "Bookings",
                "firstMessage": "How can I help?",
                "server": {"url": "https://user:pass@example.test/hook?token=hidden"},
                "model": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "system", "content": "Book a table."}],
                    "toolIds": ["tool-1"],
                },
                "credentialId": "credential-1",
            }
        return {
            "id": "tool-1",
            "type": "function",
            "function": {
                "name": "reserve",
                "parameters": {
                    "type": "object",
                    "required": ["party_size"],
                },
            },
            "server": {"url": "https://example.test/reserve?signature=hidden"},
        }

    profile = inspect_provider_target(
        "vapi",
        source_target_id="assistant-1",
        api_key="vapi-secret",
        request=request,
    )

    assert profile["first_message"] == "How can I help?"
    assert profile["model"]["messages"][0]["content"] == "Book a table."
    assert profile["reusable_tools"][0]["function"]["name"] == "reserve"
    assert (
        profile["reusable_tools"][0]["server"]["url"] == "https://example.test/reserve"
    )
    assert "vapi-secret" not in str(profile)
    assert "credential-1" not in str(profile)


def test_retell_import_profile_includes_prompt_and_exact_tool_schema() -> None:
    def request(method, url, api_key, body):
        assert method == "GET"
        assert body is None
        assert api_key == "retell-secret"
        if "/get-agent/" in url:
            return {
                "agent_name": "Preferences",
                "voice_id": "voice-1",
                "response_engine": {"type": "retell-llm", "llm_id": "llm-1"},
            }
        return {
            "general_prompt": "Always record a stated preference before ending.",
            "begin_message": "What preference should I save?",
            "general_tools": [
                {
                    "type": "custom",
                    "name": "record_preference",
                    "url": "https://example.test/provider/tools/record_preference?key=nope",
                    "parameters": {
                        "type": "object",
                        "required": ["preference"],
                        "properties": {"preference": {"type": "string"}},
                    },
                }
            ],
        }

    profile = inspect_provider_target(
        "retell",
        source_target_id="agent-1",
        api_key="retell-secret",
        request=request,
    )

    assert profile["general_prompt"].startswith("Always record")
    tool = profile["general_tools"][0]
    assert tool["parameters"]["required"] == ["preference"]
    assert tool["url"] == "https://example.test/provider/tools/record_preference"


def test_vapi_import_clones_tools_rewires_urls_and_cleans_up() -> None:
    calls: list[tuple[str, str, object]] = []

    def request(method, url, api_key, body):
        assert api_key == "vapi-secret"
        calls.append((method, url, body))
        if method == "GET" and "/assistant/" in url:
            return {
                "id": "source-assistant",
                "orgId": "org",
                "latestVersion": 3,
                "isServerUrlSecretSet": False,
                "name": "Production",
                "server": {"url": "https://prod.example/events"},
                "model": {
                    "provider": "openai",
                    "toolIds": ["source-tool"],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "lookup"},
                            "server": {"url": "https://prod.example/tools/lookup"},
                        }
                    ],
                },
            }
        if method == "GET" and "/tool/" in url:
            return {
                "id": "source-tool",
                "type": "apiRequest",
                "name": "lookup",
                "url": "https://prod.example/api/orders",
            }
        if method == "POST" and url.endswith("/tool"):
            return {"id": "cloned-tool"}
        if method == "POST" and url.endswith("/assistant"):
            return {"id": "cloned-assistant"}
        return {}

    spec = ProviderImportSpec(
        type="vapi",
        source_target_id="source-assistant",
        public_capability="tools",
        environment_tools=["lookup"],
    )
    receipt = clone_provider_target(
        spec, context=_context("vapi"), api_key="vapi-secret", request=request
    )

    assistant_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/assistant")
    )
    assert assistant_body["server"]["url"] == "https://world.example/provider/events"
    assert "latestVersion" not in assistant_body
    assert "isServerUrlSecretSet" not in assistant_body
    assert assistant_body["model"]["toolIds"] == ["cloned-tool"]
    assert (
        assistant_body["model"]["tools"][0]["server"]["url"]
        == "https://world.example/provider/tools/tools/lookup"
    )
    tool_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/tool")
    )
    assert tool_body["url"] == "https://world.example/provider/tools/api/orders"
    assert receipt.target.id == "cloned-assistant"
    assert [resource.id for resource in receipt.resources] == [
        "cloned-tool",
        "cloned-assistant",
    ]

    destroy_imported_target(
        spec, receipt=receipt, api_key="vapi-secret", request=request
    )
    deletes = [(method, url) for method, url, _body in calls if method == "DELETE"]
    assert deletes == [
        ("DELETE", "https://api.vapi.ai/assistant/cloned-assistant"),
        ("DELETE", "https://api.vapi.ai/tool/cloned-tool"),
    ]


def test_vapi_import_cleans_up_tools_when_later_tool_copy_fails() -> None:
    calls: list[tuple[str, str]] = []

    def request(method, url, _api_key, _body):
        calls.append((method, url))
        if url.endswith("/assistant/source"):
            return {"id": "source", "model": {"toolIds": ["one", "two"]}}
        if url.endswith("/tool/one"):
            return {"id": "one", "type": "function", "function": {"name": "one"}}
        if method == "POST" and url.endswith("/tool"):
            return {"id": "cloned-one"}
        if url.endswith("/tool/two"):
            raise ProviderImportError("provider_api_error: GET returned HTTP 500")
        if method == "DELETE" and url.endswith("/tool/cloned-one"):
            return {}
        raise AssertionError((method, url))

    with pytest.raises(ProviderImportError, match="HTTP 500"):
        clone_provider_target(
            ProviderImportSpec(
                type="vapi",
                source_target_id="source",
                public_capability="api",
                environment_tools=["one", "two"],
            ),
            context=_context("vapi"),
            api_key="secret",
            request=request,
        )

    assert calls[-1] == ("DELETE", "https://api.vapi.ai/tool/cloned-one")


def test_retell_import_clones_llm_rewires_custom_tools_then_agent() -> None:
    calls: list[tuple[str, str, object]] = []

    def request(method, url, api_key, body):
        assert api_key == "retell-secret"
        calls.append((method, url, body))
        if method == "GET" and "/get-agent/" in url:
            return {
                "agent_id": "source-agent",
                "version": 7,
                "voice_id": "retell-Cimo",
                "response_engine": {"type": "retell-llm", "llm_id": "source-llm"},
                "webhook_url": "https://prod.example/events",
            }
        if method == "GET" and "/get-retell-llm/" in url:
            return {
                "llm_id": "source-llm",
                "version": 3,
                "general_prompt": "Help the caller.",
                "general_tools": [
                    {
                        "type": "custom",
                        "name": "lookup",
                        "url": "https://prod.example/provider/tools/lookup",
                    },
                    {"type": "end_call", "name": "end_call"},
                ],
            }
        if method == "POST" and url.endswith("/create-retell-llm"):
            return {"llm_id": "cloned-llm"}
        if method == "POST" and url.endswith("/create-agent"):
            return {"agent_id": "cloned-agent"}
        return {}

    spec = ProviderImportSpec(
        type="retell",
        source_target_id="source-agent",
        public_capability="tools",
        environment_tools=["lookup"],
    )
    receipt = clone_provider_target(
        spec, context=_context("retell"), api_key="retell-secret", request=request
    )

    llm_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/create-retell-llm")
    )
    assert (
        llm_body["general_tools"][0]["url"]
        == "https://world.example/provider/tools/lookup"
    )
    assert "url" not in llm_body["general_tools"][1]
    agent_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/create-agent")
    )
    assert agent_body["response_engine"] == {
        "type": "retell-llm",
        "llm_id": "cloned-llm",
    }
    assert agent_body["webhook_url"] == "https://world.example/provider/events"
    assert receipt.target.id == "cloned-agent"

    destroy_imported_target(
        spec, receipt=receipt, api_key="retell-secret", request=request
    )
    deletes = [(method, url) for method, url, _body in calls if method == "DELETE"]
    assert deletes == [
        ("DELETE", "https://api.retellai.com/delete-agent/cloned-agent"),
        ("DELETE", "https://api.retellai.com/delete-retell-llm/cloned-llm"),
    ]


def test_retell_import_rejects_an_opaque_response_engine() -> None:
    def request(method, url, api_key, body):
        return {
            "agent_id": "source-agent",
            "response_engine": {
                "type": "custom-llm",
                "llm_websocket_url": "wss://opaque",
            },
        }

    with pytest.raises(ProviderImportError, match="response_engine_unsupported"):
        clone_provider_target(
            ProviderImportSpec(
                type="retell",
                source_target_id="source-agent",
                public_capability="tools",
            ),
            context=_context("retell"),
            api_key="retell-secret",
            request=request,
        )


def test_retell_import_rejects_tool_without_environment_implementation() -> None:
    def request(method, url, api_key, body):
        if "/get-agent/" in url:
            return {
                "agent_id": "source-agent",
                "response_engine": {"type": "retell-llm", "llm_id": "source-llm"},
            }
        return {
            "llm_id": "source-llm",
            "general_tools": [
                {"type": "custom", "name": "refund_order", "url": "https://prod"}
            ],
        }

    with pytest.raises(ProviderImportError, match="tool_implementation_missing"):
        clone_provider_target(
            ProviderImportSpec(
                type="retell",
                source_target_id="source-agent",
                public_capability="tools",
                environment_tools=["lookup_order"],
            ),
            context=_context("retell"),
            api_key="retell-secret",
            request=request,
        )


def test_retell_conversation_flow_profile_includes_graph_and_tools() -> None:
    def request(method, url, api_key, body):
        assert method == "GET"
        assert body is None
        assert api_key == "retell-secret"
        if "/get-agent/" in url:
            return {
                "agent_name": "Appointments",
                "voice_id": "retell-Grace",
                "response_engine": {
                    "type": "conversation-flow",
                    "conversation_flow_id": "flow-source",
                },
            }
        return {
            "conversation_flow_id": "flow-source",
            "start_speaker": "agent",
            "model_choice": {"type": "cascading", "model": "gpt-4.1"},
            "nodes": [{"id": "start", "instruction": "Help the rider."}],
            "tools": [
                {
                    "type": "custom",
                    "name": "fetch_appointment_details",
                    "url": "https://user:pass@example.test/fetch?token=hidden",
                }
            ],
        }

    profile = inspect_provider_target(
        "retell",
        source_target_id="agent-1",
        api_key="retell-secret",
        request=request,
    )

    assert profile["response_engine_type"] == "conversation-flow"
    assert profile["conversation_flow"]["nodes"][0]["instruction"] == "Help the rider."
    assert (
        profile["conversation_flow"]["tools"][0]["url"] == "https://example.test/fetch"
    )


def test_retell_chat_profile_uses_chat_agent_endpoint() -> None:
    requested: list[str] = []

    def request(method, url, api_key, body):
        assert method == "GET"
        assert api_key == "retell-secret"
        assert body is None
        requested.append(url)
        if "/get-chat-agent/" in url:
            return {
                "agent_name": "Collections chat",
                "response_engine": {"type": "retell-llm", "llm_id": "llm-1"},
            }
        return {"llm_id": "llm-1", "general_prompt": "Verify identity first."}

    profile = inspect_provider_target(
        "retell",
        source_target_id="chat-agent-1",
        api_key="retell-secret",
        target_modality="chat",
        request=request,
    )

    assert profile["modality"] == "chat"
    assert profile["general_prompt"] == "Verify identity first."
    assert requested[0].endswith("/get-chat-agent/chat-agent-1")


def test_retell_chat_import_clones_and_deletes_chat_agent() -> None:
    calls: list[tuple[str, str, object]] = []

    def request(method, url, api_key, body):
        assert api_key == "retell-secret"
        calls.append((method, url, body))
        if method == "GET" and "/get-chat-agent/" in url:
            return {
                "agent_id": "source-chat",
                "version": 2,
                "agent_name": "Collections",
                "response_engine": {"type": "retell-llm", "llm_id": "source-llm"},
                "webhook_url": "https://prod.example/events",
            }
        if method == "GET" and "/get-retell-llm/" in url:
            return {"llm_id": "source-llm", "general_prompt": "Collect safely."}
        if method == "POST" and url.endswith("/create-retell-llm"):
            return {"llm_id": "cloned-llm"}
        if method == "POST" and url.endswith("/create-chat-agent"):
            return {"agent_id": "cloned-chat"}
        return {}

    spec = ProviderImportSpec(
        type="retell",
        target_modality="chat",
        source_target_id="source-chat",
        public_capability="tools",
        environment_tools=[],
    )
    receipt = clone_provider_target(
        spec, context=_context("retell"), api_key="retell-secret", request=request
    )

    assert receipt.target.kind == "chat_agent"
    assert receipt.target.id == "cloned-chat"
    assert [resource.kind for resource in receipt.resources] == [
        "retell_llm",
        "chat_agent",
    ]
    assert any(
        url.endswith("/create-chat-agent")
        for method, url, _ in calls
        if method == "POST"
    )

    destroy_imported_target(
        spec, receipt=receipt, api_key="retell-secret", request=request
    )
    deletes = [(method, url) for method, url, _ in calls if method == "DELETE"]
    assert deletes == [
        ("DELETE", "https://api.retellai.com/delete-chat-agent/cloned-chat"),
        ("DELETE", "https://api.retellai.com/delete-retell-llm/cloned-llm"),
    ]


def test_retell_import_clones_conversation_flow_rewires_all_nested_tools() -> None:
    calls: list[tuple[str, str, object]] = []

    def request(method, url, api_key, body):
        assert api_key == "retell-secret"
        calls.append((method, url, body))
        if method == "GET" and "/get-agent/" in url:
            return {
                "agent_id": "source-agent",
                "version": 7,
                "agent_name": "Appointments",
                "voice_id": "retell-Grace",
                "response_engine": {
                    "type": "conversation-flow",
                    "conversation_flow_id": "source-flow",
                    "version": 1,
                },
                "webhook_url": "https://prod.example/events",
            }
        if method == "GET" and "/get-conversation-flow/" in url:
            return {
                "conversation_flow_id": "source-flow",
                "version": 3,
                "is_published": True,
                "start_speaker": "agent",
                "model_choice": {"type": "cascading", "model": "gpt-4.1"},
                "tools": [
                    {
                        "type": "custom",
                        "name": "fetch_appointment_details",
                        "url": "https://prod.example/api/fetch-rider-appointment",
                    }
                ],
                "nodes": [
                    {
                        "id": "book",
                        "tool": {
                            "type": "custom",
                            "name": "create_booking",
                            "url": "https://prod.example/api/create-rider-booking",
                        },
                    }
                ],
            }
        if method == "POST" and url.endswith("/create-conversation-flow"):
            return {"conversation_flow_id": "cloned-flow"}
        if method == "POST" and url.endswith("/create-agent"):
            return {"agent_id": "cloned-agent"}
        return {}

    spec = ProviderImportSpec(
        type="retell",
        source_target_id="source-agent",
        public_capability="tools",
        environment_tools=["fetch_appointment_details", "create_booking"],
    )
    receipt = clone_provider_target(
        spec, context=_context("retell"), api_key="retell-secret", request=request
    )

    flow_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/create-conversation-flow")
    )
    assert "conversation_flow_id" not in flow_body
    assert "version" not in flow_body
    assert "is_published" not in flow_body
    assert (
        flow_body["tools"][0]["url"]
        == "https://world.example/provider/tools/api/fetch-rider-appointment"
    )
    assert (
        flow_body["nodes"][0]["tool"]["url"]
        == "https://world.example/provider/tools/api/create-rider-booking"
    )
    agent_body = next(
        body
        for method, url, body in calls
        if method == "POST" and url.endswith("/create-agent")
    )
    assert agent_body["response_engine"] == {
        "type": "conversation-flow",
        "conversation_flow_id": "cloned-flow",
    }
    assert agent_body["webhook_url"] == "https://world.example/provider/events"
    assert receipt.metadata["source_response_engine_type"] == "conversation-flow"
    assert [resource.kind for resource in receipt.resources] == [
        "conversation_flow",
        "voice_agent",
    ]

    destroy_imported_target(
        spec, receipt=receipt, api_key="retell-secret", request=request
    )
    deletes = [(method, url) for method, url, _body in calls if method == "DELETE"]
    assert deletes == [
        ("DELETE", "https://api.retellai.com/delete-agent/cloned-agent"),
        ("DELETE", "https://api.retellai.com/delete-conversation-flow/cloned-flow"),
    ]


def test_retell_conversation_flow_rejects_nested_missing_tool_before_create() -> None:
    calls: list[tuple[str, str]] = []

    def request(method, url, _api_key, _body):
        calls.append((method, url))
        if "/get-agent/" in url:
            return {
                "response_engine": {
                    "type": "conversation-flow",
                    "conversation_flow_id": "source-flow",
                }
            }
        return {
            "conversation_flow_id": "source-flow",
            "nodes": [
                {
                    "tool": {
                        "type": "custom",
                        "name": "cancel_appointment",
                        "url": "https://prod.example/cancel",
                    }
                }
            ],
        }

    with pytest.raises(ProviderImportError, match="cancel_appointment"):
        clone_provider_target(
            ProviderImportSpec(
                type="retell",
                source_target_id="source-agent",
                public_capability="tools",
                environment_tools=["fetch_appointment_details"],
            ),
            context=_context("retell"),
            api_key="retell-secret",
            request=request,
        )

    assert not any(method == "POST" for method, _url in calls)


def test_retell_conversation_flow_is_deleted_if_agent_creation_fails() -> None:
    calls: list[tuple[str, str]] = []

    def request(method, url, _api_key, _body):
        calls.append((method, url))
        if "/get-agent/" in url:
            return {
                "response_engine": {
                    "type": "conversation-flow",
                    "conversation_flow_id": "source-flow",
                }
            }
        if method == "GET":
            return {
                "conversation_flow_id": "source-flow",
                "start_speaker": "agent",
                "nodes": [],
            }
        if url.endswith("/create-conversation-flow"):
            return {"conversation_flow_id": "cloned-flow"}
        if url.endswith("/create-agent"):
            raise ProviderImportError("provider_api_error: POST returned HTTP 400")
        return {}

    with pytest.raises(ProviderImportError, match="HTTP 400"):
        clone_provider_target(
            ProviderImportSpec(
                type="retell",
                source_target_id="source-agent",
                public_capability="tools",
            ),
            context=_context("retell"),
            api_key="retell-secret",
            request=request,
        )

    assert calls[-1] == (
        "DELETE",
        "https://api.retellai.com/delete-conversation-flow/cloned-flow",
    )
