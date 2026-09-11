from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from fi.alk._paths import project_root
from fi.alk.studio import _generate
from fi.simulate.agent.definition import (
    AgentDefinition,
    LLMConfig,
    ProviderEvidenceConfig,
)
from fi.simulate.endpoints.vapi import VapiCallOriginator
from fi.simulate.evidence.providers.base import EvidenceContext
from fi.simulate.evidence.providers.vapi import (
    VapiEvidenceSource,
    _extract_vapi_recording_urls,
)
from fi.simulate.simulation import generator, livekit_models
from fi.simulate.simulation.engines import livekit


def _agent(**updates: object) -> AgentDefinition:
    values = {
        "name": "support-agent",
        "url": "wss://livekit.example.com",
        "room_name": "support-room",
        "system_prompt": "Help the caller.",
    }
    values.update(updates)
    return AgentDefinition(**values)


@pytest.mark.parametrize(
    "config",
    [
        LLMConfig(provider="openai", model="gpt-4.1-mini", temperature=0.2),
        LLMConfig(provider="google", model="gemini-2.5-flash", temperature=0.4),
    ],
)
def test_scenario_generator_uses_configured_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
    config: LLMConfig,
) -> None:
    captured: list[LLMConfig] = []

    class FakeStream:
        async def to_str_iterable(self):
            yield '{"personas":[{"persona":{"name":"Priya"},"situation":"Needs help.","outcome":"Resolved."}]}'

    class FakeLLM:
        def chat(self, *, chat_ctx):
            assert chat_ctx.items
            return FakeStream()

    def build_llm(value: LLMConfig) -> FakeLLM:
        captured.append(value)
        return FakeLLM()

    # The generator imports the builder lazily inside __init__, so the name
    # resolves in livekit_models at call time, never on the generator module.
    monkeypatch.setattr(livekit_models, "build_livekit_llm", build_llm)

    personas = asyncio.run(
        generator.ScenarioGenerator(_agent(), llm_config=config).generate(
            "support",
            1,
        )
    )

    assert captured == [config]
    assert personas[0].persona["name"] == "Priya"


def test_managed_room_uses_rendered_values_and_invocation_uniqueness() -> None:
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="sdk {run_id} {index}",
        room_mode="managed",
    )

    first = livekit._resolve_room_name(
        runtime,
        run_id="run-a",
        test_case_id="case-aaaaaaaaaaaa",
        index=4,
        invocation_id="first-invocation",
    )
    second = livekit._resolve_room_name(
        runtime,
        run_id="run-a",
        test_case_id="case-aaaaaaaaaaaa",
        index=4,
        invocation_id="second-invocation",
    )

    assert first.startswith("sdk-run-a-4-first-invocation-")
    assert first != second


def test_external_room_preserves_caller_controlled_name() -> None:
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="operator/{run_id}/{invocation_id}",
        room_mode="external",
    )

    room_name = livekit._resolve_room_name(
        runtime,
        run_id="run-a",
        test_case_id="case-a",
        index=0,
        invocation_id="invocation-a",
    )

    assert room_name == "operator/run-a/invocation-a"


def test_agent_first_silence_requires_bidirectional_messages() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hello"),
                SimpleNamespace(type="message", role="user", text_content="Thanks"),
            ]
        )
    )

    asyncio.run(
        asyncio.wait_for(
            livekit._wait_for_agent_first_silence(session, timeout_seconds=0.01),
            timeout=1,
        )
    )


def test_vapi_polling_window_matches_caller_and_callee_numbers(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "calls": [
                    {
                        "id": "wrong-call",
                        "customer": {"number": "+14155550101"},
                        "phoneNumber": {"number": "+14155550202"},
                    },
                    {
                        "id": "matched-call",
                        "customer": {"number": "+14155550100"},
                        "phoneNumber": {"number": "+14155550200"},
                        "createdAt": "2026-07-30T10:00:00Z",
                    },
                ]
            },
        )

    async def run() -> str | None:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            transport=httpx.MockTransport(handler),
        )
        source = VapiEvidenceSource(
            ProviderEvidenceConfig(
                provider="vapi",
                call_id_source="polling_window",
                polling_window_seconds=60,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run-vapi",
                test_case_id="case-vapi",
                case_directory=tmp_path,
                started_at=datetime(2026, 7, 30, 10, 0, tzinfo=timezone.utc),
                caller_phone="+14155550100",
                callee_phone="+14155550200",
            )
        )
        call_id = await source._locate_call_id()
        await client.aclose()
        return call_id

    assert asyncio.run(run()) == "matched-call"
    assert requests[0].url.path == "/call"
    assert "createdAtGt" in str(requests[0].url)
    assert "createdAtLt" in str(requests[0].url)


def test_vapi_teardown_evidence_keeps_raw_reason_and_marks_sdk_source(
    tmp_path: Path,
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    source = VapiEvidenceSource(
        ProviderEvidenceConfig(provider="vapi", call_id_source="originator_response"),
        api_key="test-key",
        client=client,
    )

    async def connect() -> None:
        await source.connect(
            EvidenceContext(
                run_id="run-vapi",
                test_case_id="case-vapi",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                termination_source="vapi_originator_cleanup",
            )
        )

    asyncio.run(connect())
    summary = source._summarize(
        {"status": "ended", "endedReason": "call-deleted"},
        "call-123",
        [],
    )
    asyncio.run(client.aclose())

    assert summary.metadata["ended_reason"] == "call-deleted"
    assert summary.metadata["ended_reason_interpretation"] == "sdk_originator_teardown"


def test_vapi_recording_download_uses_authenticated_artifact_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.vapi.ai":
            assert request.url.path == "/call/call-123/stereo-recording"
            assert request.headers["Authorization"] == "Bearer test-key"
            return httpx.Response(
                302,
                headers={"Location": "https://storage.example/signed-recording"},
            )
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=b"recording-bytes")

    async def run() -> bytes:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx.MockTransport(handler),
        )
        source = VapiEvidenceSource(
            ProviderEvidenceConfig(
                provider="vapi",
                call_id_source="originator_response",
            ),
            api_key="test-key",
            client=client,
        )
        try:
            return await source._get_recording("call-123", "stereo")
        finally:
            await client.aclose()

    assert asyncio.run(run()) == b"recording-bytes"
    assert len(requests) == 2


def test_vapi_recording_discovery_supports_current_and_legacy_payloads() -> None:
    current = _extract_vapi_recording_urls(
        {
            "artifact": {
                "recording": {
                    "mono": {"combinedUrl": "private-mono"},
                    "stereoUrl": "private-stereo",
                }
            }
        }
    )
    legacy = _extract_vapi_recording_urls(
        {
            "artifact": {
                "recordingUrl": "legacy-mono",
                "stereoRecordingUrl": "legacy-stereo",
            }
        }
    )

    assert current["combined"] == "private-mono"
    assert current["stereo"] == "private-stereo"
    assert legacy["combined"] == "legacy-mono"
    assert legacy["stereo"] == "legacy-stereo"


def test_vapi_evidence_keeps_tool_call_identities(tmp_path: Path) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    source = VapiEvidenceSource(
        ProviderEvidenceConfig(provider="vapi", call_id_source="originator_response"),
        api_key="test-key",
        client=client,
    )

    async def run() -> dict:
        await source.connect(
            EvidenceContext(
                run_id="run-vapi",
                test_case_id="case-vapi",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
            )
        )
        summary = source._summarize(
            {
                "status": "ended",
                "messages": [
                    {
                        "toolCalls": [
                            {
                                "id": "call-end",
                                "function": {
                                    "name": "endCall",
                                    "arguments": {"reason": "resolved"},
                                },
                            }
                        ],
                    },
                    {
                        "toolCallResultList": [
                            {
                                "toolCallId": "call-end",
                                "name": "endCall",
                                "result": "ok",
                            }
                        ],
                    },
                ],
            },
            "call-123",
            [],
        )
        await client.aclose()
        return summary.metadata

    metadata = asyncio.run(run())

    assert metadata["tool_calls"] == [
        {
            "id": "call-end",
            "name": "endCall",
            "arguments": {"reason": "resolved"},
        }
    ]
    assert metadata["tool_results"] == [
        {"tool_call_id": "call-end", "name": "endCall", "result": "ok"}
    ]


def test_vapi_originator_supports_provider_managed_phone_number() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/phone-number/phone-123":
            return httpx.Response(200, json={"provider": "vapi"})
        return httpx.Response(
            201,
            json={"id": "call-123", "status": "queued"},
        )

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            transport=httpx.MockTransport(handler),
        )
        originator = VapiCallOriginator(
            api_key="test-key",
            assistant_id="assistant-123",
            phone_number_id="phone-123",
            destination="+14155550100",
            client=client,
        )
        call = await originator.start()
        await client.aclose()
        assert call.call_id == "call-123"
        assert call.status == "queued"

    asyncio.run(run())
    assert [request.url.path for request in requests] == [
        "/phone-number/phone-123",
        "/call",
    ]


def test_platform_agent_updates_version_instead_of_creating_new_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        api_key="api-key",
        secret_key="secret-key",
        api_url="https://platform.example",
    )
    payload, _ = _generate._agent_payload(_agent(system_prompt="Updated prompt."))
    calls: list[tuple[str, str]] = []

    def request_json(url, _headers, *, method="GET", payload=None, timeout=30.0):
        del payload, timeout
        calls.append((method, url))
        if url.startswith("https://platform.example/simulate/agent-definitions/?"):
            return {"results": [{"id": "agent-123", "agent_name": payload_name}]}
        if url.endswith("/agent-123/"):
            return {
                "active_version": {
                    "id": "version-1",
                    "commit_message": "Agent Learning Kit configuration stale",
                }
            }
        if url.endswith("/agent-123/versions/create/"):
            return {"version": {"id": "version-2"}}
        raise AssertionError(url)

    payload_name = payload["agent_name"]
    monkeypatch.setattr(_generate, "_request_json", request_json)

    reference = _generate.ensure_platform_agent(
        _agent(system_prompt="Updated prompt."),
        config=config,
    )

    assert reference.agent_definition_id == "agent-123"
    assert reference.agent_version_id == "version-2"
    assert (
        "POST",
        "https://platform.example/simulate/agent-definitions/agent-123/versions/create/",
    ) in calls
    assert not any(url.endswith("/agent-definitions/create/") for _, url in calls)


def test_platform_scenario_reuses_existing_name_for_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        api_key="api-key",
        secret_key="secret-key",
        api_url="https://platform.example",
    )
    fetched: dict[str, object] = {}

    def request_json(url, _headers, *, method="GET", payload=None, timeout=30.0):
        del method, payload, timeout
        assert url.startswith("https://platform.example/simulate/scenarios/?")
        return {"results": [{"id": "scenario-123", "name": "repeatable"}]}

    def fetch(scenario_id, **kwargs):
        fetched["scenario_id"] = scenario_id
        fetched.update(kwargs)
        return "reused-scenario"

    monkeypatch.setattr(_generate, "_request_json", request_json)
    monkeypatch.setattr(_generate, "fetch_scenario", fetch)

    result = _generate.generate_scenario(
        _generate.PlatformScenarioRequest(
            name="repeatable",
            platform_agent_definition_id="agent-123",
        ),
        config=config,
    )

    assert result == "reused-scenario"
    assert fetched["scenario_id"] == "scenario-123"
    assert fetched["platform_agent_definition_id"] == "agent-123"


def test_project_root_discovers_source_tree_and_rejects_unrelated_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "kit"
    source = root / "src" / "fi" / "alk" / "nested"
    source.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'kit'\n")
    module = source / "module.py"
    module.write_text("pass\n")

    assert project_root(module) == root
    with pytest.raises(RuntimeError, match="agent_learning_kit_project_root_not_found"):
        project_root(tmp_path / "unrelated")
