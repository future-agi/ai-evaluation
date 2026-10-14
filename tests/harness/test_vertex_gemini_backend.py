import asyncio
from typing import Any

from fi.alk.harness.backends.vertex_gemini import (
    _flattened,
    _forget_old_reads,
    _successful_terminal_save,
)


def test_session_replies_with_installed_adk(monkeypatch) -> None:
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from fi.alk.harness.backends import ModelReply, SessionSpec, StageDone
    from fi.alk.harness.backends.vertex_gemini import VertexGeminiSession

    async def generate(self, llm_request, stream=False):
        assert llm_request.contents[-1].parts[0].text == "Hello"
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Hello back")])
        )

    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

    async def run():
        session = VertexGeminiSession(
            SessionSpec(system_prompt="Reply briefly", max_turns=1),
            model="gemini-3.7-flash",
        )
        await session.start()
        try:
            await session.send("Hello")
            return [event async for event in session.replies()]
        finally:
            await session.stop()

    events = asyncio.run(run())
    replies = [event for event in events if isinstance(event, ModelReply)]
    assert replies[0].parts[0].text == "Hello back"
    assert isinstance(events[-1], StageDone)
    assert events[-1].outcome == "success"


def test_successful_authoring_save_is_terminal() -> None:
    assert _successful_terminal_save(
        "mcp__world__save_world", {"content": "Saved to /work/authoring."}
    )
    assert _successful_terminal_save(
        "mcp__provision__save_environment", {"content": "Saved."}
    )
    assert _successful_terminal_save(
        "mcp__scenarios__save_scenarios", {"content": "Saved 10 scenarios."}
    )
    assert _successful_terminal_save(
        "mcp__source_data__finish_review", {"content": "Saved 12 invariants."}
    )
    assert _successful_terminal_save(
        "mcp__repair__submit_world_ir_patch", {"content": "Patch accepted."}
    )


def test_provider_timeout_reaches_sdk_and_fails_without_retry(monkeypatch) -> None:
    import httpx
    from google.genai import _api_client

    from fi.alk.harness.backends import SessionSpec, StageDone
    from fi.alk.harness.backends.vertex_gemini import VertexGeminiSession

    timeouts = []

    async def timeout(self, http_request, stream=False):
        timeouts.append(http_request.timeout)
        raise httpx.ReadTimeout("Provider request timed out")

    monkeypatch.setattr(_api_client.BaseApiClient, "_async_request_once", timeout)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def run():
        session = VertexGeminiSession(
            SessionSpec(system_prompt="Reply briefly", max_turns=2),
            model="gemini-3.7-flash",
        )
        await session.start()
        try:
            await session.send("Hello")
            return [event async for event in session.replies()]
        finally:
            await session.stop()

    events = asyncio.run(run())
    assert timeouts == [300.0]
    assert isinstance(events[-1], StageDone)
    assert events[-1].outcome == "failed"
    assert events[-1].is_error
    assert events[-1].errors == ["Provider request timed out"]


def test_terminal_save_closes_adk_invocation_in_its_original_context(monkeypatch, caplog):
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_response import LlmResponse
    from google.adk.telemetry import _instrumentation, tracing
    from google.genai import types
    from opentelemetry.sdk.trace import TracerProvider

    from fi.alk.harness.backends import SessionSpec, ToolServer, ToolSpec
    from fi.alk.harness.backends.vertex_gemini import VertexGeminiBackend
    from fi.alk.harness.session import Stage

    async def generate(self, llm_request, stream=False):
        if llm_request.contents[-1].parts[0].function_response is not None:
            await asyncio.Event().wait()
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(
                    id="save", name="mcp__world__save_world", args={}
                ))],
            )
        )

    async def save(args):
        return {"content": "Saved."}

    provider = TracerProvider()
    monkeypatch.setattr(tracing, "tracer", provider.get_tracer("regression"))
    monkeypatch.setattr(_instrumentation, "resolve_schema_version", lambda: 1)
    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

    async def run():
        async with Stage(
            SessionSpec(
                system_prompt="Save the world",
                servers={"world": ToolServer(
                    name="world", tools=[ToolSpec("save_world", "Save", {}, save)]
                )},
                max_turns=2,
                model="gemini-3.7-flash",
            ),
            backend=VertexGeminiBackend(),
            overheard=False,
        ) as stage:
            events = [event async for event in stage.stream("Save")]
            assert events[-1].detail["outcome"] == "success"
            assert stage.history[-1].outcome == "success"

    try:
        asyncio.run(run())
        assert "Failed to detach context" not in caplog.text
    finally:
        provider.shutdown()


def test_rejected_save_and_non_save_tools_are_not_terminal() -> None:
    assert not _successful_terminal_save(
        "mcp__world__save_world", {"is_error": True, "content": "Not saved."}
    )
    assert not _successful_terminal_save(
        "mcp__world__check_world", {"content": "All checks pass."}
    )
    assert not _successful_terminal_save("mcp__world__save_world", "Saved")


def _read(call_id: str, name: str, text: str) -> Any:
    from google.genai import types

    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id, name=name, response={"content": text}
                )
            )
        ],
    )


def _said(part: Any) -> str:
    return _flattened(part.parts[0].function_response.response)


def test_small_sessions_keep_every_read() -> None:
    contents = [
        _read(f"c{n}", "mcp__scenarios__inspect_world", "x" * 100) for n in range(10)
    ]
    _forget_old_reads(contents)
    assert all(_said(one) == "x" * 100 for one in contents)


def test_a_long_session_forgets_all_but_the_newest_reads() -> None:
    contents = [
        _read(f"c{n}", "mcp__scenarios__inspect_world", "x" * 10_000) for n in range(10)
    ]
    _forget_old_reads(contents)
    kept = [one for one in contents if _said(one) == "x" * 10_000]
    assert len(kept) == 2
    assert kept == contents[-2:]
    assert "inspect_world" in _said(contents[0])


def test_forgetting_is_per_tool_and_leaves_writes_alone() -> None:
    contents = [
        _read("c0", "mcp__scenarios__inspect_world", "w" * 30_000),
        _read("c1", "mcp__scenarios__inspect_world", "w" * 30_000),
        _read("c2", "mcp__scenarios__inspect_world", "w" * 30_000),
        _read("c3", "mcp__scenarios__inspect_scenario", "s" * 100),
        _read("c4", "mcp__scenarios__submit_scenario", "gate said no" * 500),
    ]
    _forget_old_reads(contents)
    assert _said(contents[0]).startswith("[dropped")
    assert _said(contents[1]) == "w" * 30_000
    assert _said(contents[3]) == "s" * 100
    assert _said(contents[4]) == "gate said no" * 500


def test_forgetting_twice_changes_nothing_more() -> None:
    contents = [
        _read(f"c{n}", "mcp__scenarios__inspect_world", "x" * 10_000) for n in range(10)
    ]
    _forget_old_reads(contents)
    once = [_said(one) for one in contents]
    _forget_old_reads(contents)
    assert [_said(one) for one in contents] == once


def test_forgetting_leaves_the_session_event_alone() -> None:
    from google.genai import types

    session = [
        _read(f"c{n}", "mcp__scenarios__inspect_world", "x" * 10_000) for n in range(10)
    ]
    request = [
        types.Content(role=one.role, parts=[part.model_copy() for part in one.parts])
        for one in session
    ]
    _forget_old_reads(request)
    assert _said(request[0]).startswith("[dropped")
    assert all(_said(one) == "x" * 10_000 for one in session)


def test_compaction_is_configured_and_can_be_switched_off(monkeypatch) -> None:
    from fi.alk.harness.backends import vertex_gemini

    monkeypatch.setattr(vertex_gemini, "COMPACT_ABOVE_TOKENS", 90_000)
    monkeypatch.setattr(vertex_gemini, "EVENTS_KEPT_RAW", 12)
    config = vertex_gemini._compaction()
    assert config.token_threshold == 90_000
    assert config.event_retention_size == 12

    monkeypatch.setattr(vertex_gemini, "COMPACT_ABOVE_TOKENS", 0)
    assert vertex_gemini._compaction() is None


def test_a_tool_call_carries_the_agent_that_made_it() -> None:
    from fi.alk.harness.backends import Call

    assert Call(id="c1", name="x").by == ""
    assert Call(id="c1", name="x", by="scenario_writer").by == "scenario_writer"


def test_a_cache_read_is_not_charged_at_the_full_input_rate() -> None:
    from fi.alk.harness.backends.vertex_gemini import CACHE_READ_SHARE, priced

    full = priced("gemini-3.7-flash", 1_000_000, 0, 0)
    all_cached = priced("gemini-3.7-flash", 1_000_000, 0, 1_000_000)
    assert full == 0.75
    assert all_cached == round(0.75 * CACHE_READ_SHARE, 6) or abs(all_cached - 0.075) < 1e-9
    assert priced("gemini-3.7-flash", 1_000, 0, 999_999) >= 0


def test_the_claude_sdk_drives_gemini_only_behind_the_gateway(monkeypatch):
    from fi.alk.harness.backends import resolve

    backend = resolve("claude-gemini")
    assert backend.name == "claude"

    monkeypatch.delenv("AGENTCC_API_KEY", raising=False)
    assert not backend.can_drive("gemini-3.7-flash")
    assert not backend.can_drive("vertex_ai/gemini-3.7-flash")

    monkeypatch.setenv("AGENTCC_API_KEY", "not-a-real-key")
    assert backend.can_drive("vertex_ai/gemini-3.7-flash")


def test_the_gateway_route_never_carries_the_vertex_claude_flag(monkeypatch):
    from fi.alk.harness.config import provider_env

    monkeypatch.setenv("AGENTCC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("AGENTCC_BASE_URL", "https://gateway.example.test/")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")
    env = provider_env("vertex_ai/gemini-3.7-flash")
    assert env["CLAUDE_CODE_USE_VERTEX"] == "0"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example.test"
    for pinned in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert env[pinned] == "vertex_ai/gemini-3.7-flash"
    # A model the SDK does not know has no window to look up, so the run declares one.
    assert env["CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"] == "1"

    monkeypatch.setenv("AGENTCC_CLAUDE_MODEL_ALIAS", "claude-sonnet-4-6")
    aliased = provider_env("vertex_ai/gemini-3.7-flash")
    assert aliased["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"
    assert "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT" not in aliased
    monkeypatch.delenv("AGENTCC_CLAUDE_MODEL_ALIAS")

    monkeypatch.delenv("AGENTCC_API_KEY")
    straight = provider_env("gemini-3.7-flash")
    assert straight["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert "ANTHROPIC_BASE_URL" not in straight


def test_schema_preserves_mixed_scalar_union_in_installed_sdk():
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _gemini_schema

    converted = _gemini_schema({
        "type": "object",
        "properties": {"values": {"type": "array", "items": {
            "type": ["string", "integer", "number", "boolean", "null"]}}},
    })
    schema = types.Schema.model_validate(converted)
    items = schema.properties["values"].items
    assert items.type is None
    assert items.nullable is True
    assert [branch.type for branch in items.any_of] == [
        types.Type.STRING, types.Type.INTEGER, types.Type.NUMBER, types.Type.BOOLEAN,
    ]


def test_schema_preserves_existing_nullable_string_shape():
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _gemini_schema

    converted = _gemini_schema({"type": ["string", "null"], "description": "value"})
    assert converted == {"type": "string", "nullable": True, "description": "value"}
    assert types.Schema.model_validate(converted).type == types.Type.STRING


def test_schema_preserves_nonnullable_scalar_union():
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _gemini_schema

    converted = _gemini_schema({"type": ["integer", "boolean"]})
    schema = types.Schema.model_validate(converted)
    assert schema.nullable is None
    assert [branch.type for branch in schema.any_of] == [types.Type.INTEGER, types.Type.BOOLEAN]


def test_submit_contract_success_values_keep_numeric_and_null_sdk_types(tmp_path):
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _json_schema
    from fi.alk.harness.tools import contract_tools

    submit = next(
        tool
        for tool in contract_tools(tmp_path).tools
        if tool.name == "submit_contract"
    )
    declaration = types.FunctionDeclaration(
        name=submit.name, parameters=_json_schema(submit.input_schema)
    )
    interface = declaration.parameters.properties["runtime"].properties["interface"]
    predicate = interface.properties["failure_rule"].properties["success_values"]
    assert "[null] for JSON null" in predicate.description
    assert "[0] for numeric zero" in predicate.description
    assert "[false] for boolean false" in predicate.description
    assert "empty list is invalid" in predicate.description
    items = predicate.items
    assert items.nullable is True
    assert [branch.type for branch in items.any_of] == [
        types.Type.STRING,
        types.Type.NUMBER,
        types.Type.BOOLEAN,
    ]


def test_submit_contract_map_shapes_survive_vertex_conversion(tmp_path):
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _json_schema
    from fi.alk.harness.tools import contract_tools

    submit = next(
        tool
        for tool in contract_tools(tmp_path).tools
        if tool.name == "submit_contract"
    )
    declaration = types.FunctionDeclaration(
        name=submit.name, parameters=_json_schema(submit.input_schema)
    )
    runtime = declaration.parameters.properties["runtime"]
    config = types.Schema.model_validate(
        runtime.properties["config_files"].additional_properties
    )
    assert config.type == types.Type.OBJECT
    assert config.additional_properties is False
    assert set(config.properties) == {"format", "values", "secret_aliases"}
    for name in ("values", "secret_aliases"):
        entry = types.Schema.model_validate(
            config.properties[name].additional_properties
        )
        assert entry.type == types.Type.STRING
    secrets = types.Schema.model_validate(
        runtime.properties["secret_files"].additional_properties
    )
    assert secrets.type == types.Type.ARRAY
    assert secrets.items.type == types.Type.OBJECT
    assert {"alias", "destination"} <= set(secrets.items.required)
