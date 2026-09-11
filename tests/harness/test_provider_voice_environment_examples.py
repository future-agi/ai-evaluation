from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient


ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "harness"
    / "provider_voice_environment"
)


class _Response:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self._raw = json.dumps(body or {}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._raw


def _module(provider: str) -> ModuleType:
    path = ROOT / provider / "provider_target.py"
    spec = importlib.util.spec_from_file_location(f"fixture_{provider}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> Path:
    context = {
        "schema_version": "1",
        "attempt_id": "attempt-1",
        "world_id": "0",
        "provider": provider,
        "public_base_url": "https://world.example",
        "event_url": "https://world.example/provider/events",
        "tool_base_url": "https://world.example/provider/tools",
        "provider_resource_prefix": "alk-attempt-1-w0",
        "idempotency_key": "attempt-1:world-0",
        "expires_at": "2026-09-02T00:00:00Z",
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    output_path = tmp_path / "provider-target.json"
    monkeypatch.setenv("ALK_PROVIDER_CONTEXT", str(context_path))
    monkeypatch.setenv("ALK_PROVIDER_OUTPUT", str(output_path))
    monkeypatch.setenv("ALK_EVENT_URL", context["event_url"])
    monkeypatch.setenv("ALK_TOOL_BASE_URL", context["tool_base_url"])
    return output_path


def test_vapi_code_fixture_creates_a_tool_wired_assistant_and_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module("vapi")
    output = _environment(monkeypatch, tmp_path, "vapi")
    monkeypatch.setenv("VAPI_API_KEY", "not-persisted")
    calls: list[Any] = []

    def urlopen(request, timeout):
        calls.append(request)
        if request.method == "POST":
            return _Response({"id": "assistant-copy"})
        return _Response()

    monkeypatch.setattr(module, "urlopen", urlopen)
    module.provision()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["target"] == {"kind": "assistant", "id": "assistant-copy"}
    create = json.loads(calls[0].data)
    tool = create["model"]["tools"][0]
    assert tool["function"]["name"] == "record_preference"
    assert tool["server"]["url"].endswith("/provider/tools/record_preference")
    assert "not-persisted" not in output.read_text(encoding="utf-8")

    monkeypatch.setenv("ALK_PROVIDER_RECEIPT", str(output))
    module.destroy()
    assert calls[-1].method == "DELETE"
    assert calls[-1].full_url.endswith("/assistant/assistant-copy")


def test_retell_code_fixture_creates_llm_then_agent_and_cleans_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module("retell")
    output = _environment(monkeypatch, tmp_path, "retell")
    monkeypatch.setenv("RETELL_API_KEY", "not-persisted")
    monkeypatch.setenv("RETELL_VOICE_ID", "voice-known")
    calls: list[Any] = []

    def urlopen(request, timeout):
        calls.append(request)
        if request.full_url.endswith("/create-retell-llm"):
            return _Response({"llm_id": "llm-copy"})
        if request.full_url.endswith("/create-agent"):
            return _Response({"agent_id": "agent-copy"})
        return _Response()

    monkeypatch.setattr(module, "urlopen", urlopen)
    module.provision()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["target"] == {"kind": "voice_agent", "id": "agent-copy"}
    llm = json.loads(calls[0].data)
    assert llm["general_tools"][0]["name"] == "record_preference"
    assert llm["general_tools"][0]["url"].endswith("/provider/tools/record_preference")
    agent = json.loads(calls[1].data)
    assert agent["voice_id"] == "voice-known"
    assert agent["response_engine"]["llm_id"] == "llm-copy"
    assert "not-persisted" not in output.read_text(encoding="utf-8")

    monkeypatch.setenv("ALK_PROVIDER_RECEIPT", str(output))
    module.destroy()
    assert [request.full_url.rsplit("/", 1)[-1] for request in calls[-2:]] == [
        "agent-copy",
        "llm-copy",
    ]


def test_conversation_flow_import_backend_implements_all_declared_tool_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROVIDER_TRACE_PATH", str(tmp_path / "provider-trace.jsonl"))
    path = ROOT / "conversation_flow_import_backend" / "agent.py"
    spec = importlib.util.spec_from_file_location("conversation_flow_backend", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)

    fetched = client.post(
        "/provider/tools/api/fetch-rider-appointment",
        json={"args": {"rider_name": "Alex Morgan", "date_of_birth": "1985/06/15"}},
    ).json()
    assert fetched["booking_found"] is True
    booking_id = fetched["booking_id"]

    updated = client.post(
        "/provider/tools/api/update-rider-appointment",
        json={
            "arguments": json.dumps(
                {"booking_id": booking_id, "new_appointment_time": "2:00 PM"}
            )
        },
    ).json()
    assert updated["update_success"] is True
    assert updated["updated_fields"] == ["appointment_time"]

    canceled = client.post(
        "/provider/tools/api/cancel-rider-appointment",
        json={"booking_id": booking_id, "cancellation_reason": "schedule changed"},
    ).json()
    assert canceled["cancellation_success"] is True

    created = client.post(
        "/provider/tools/api/create-rider-booking",
        json={
            "rider_name": "Morgan Lee",
            "pickup_location": "1 Main Street",
            "dropoff_location": "General Hospital",
            "appointment_date": "September 12, 2026",
            "appointment_time": "10:30 AM",
            "vehicle_type": "sedan",
        },
    ).json()
    assert created["booking_success"] is True

    trace = (tmp_path / "provider-trace.jsonl").read_text(encoding="utf-8")
    for name in (
        "fetch_appointment_details",
        "update_appointment",
        "cancel_appointment",
        "create_booking",
    ):
        assert f'"kind": "tool.{name}"' in trace
