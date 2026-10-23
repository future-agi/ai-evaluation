import json

import pytest

from fi.alk.harness.tools import accept_contract


def payload(**fields):
    return {
        "agent": "chat",
        "modality": "chat",
        "real_use_cases": ["Answer questions"],
        **fields,
    }


@pytest.mark.parametrize("store", [None, {}, {"kind": " "}])
def test_autonomous_discovery_requires_explicit_storage_decision(tmp_path, store):
    result = accept_contract(
        payload(data_store=store), tmp_path, require_setup_contract=True
    )
    assert result["is_error"]
    message = result["content"][0]["text"]
    assert "missing-storage-decision" in message
    assert "none" in message and "inspect the source" in message
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("kind", ["none", "filesystem", "in_process", "sqlite"])
def test_explicit_storage_decision_is_saved(tmp_path, kind):
    result = accept_contract(
        payload(data_store={"kind": kind}, runtime={"interface": {"kind": "callable"}}),
        tmp_path,
        require_setup_contract=True,
    )
    assert not result.get("is_error")
    assert (
        json.loads((tmp_path / "contract.json").read_text())["data_store"]["kind"]
        == kind
    )


def test_legacy_contracts_keep_optional_storage(tmp_path):
    assert not accept_contract(payload(), tmp_path).get("is_error")


def test_voice_contracts_keep_optional_storage(tmp_path):
    assert not accept_contract(
        payload(modality="voice"), tmp_path, require_setup_contract=True
    ).get("is_error")


@pytest.mark.parametrize("source_kind,required", [("repo", True), ("provider", False)])
def test_stage_scopes_setup_admission_to_repository_sources(
    tmp_path, monkeypatch, source_kind, required
):
    from types import SimpleNamespace
    from importlib import import_module

    understand = import_module("fi.alk.harness.understand")

    captured = {}

    def contract_tools(destination, available_evals, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(understand, "contract_tools", contract_tools)
    monkeypatch.setattr(
        understand, "read_only_session", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(understand, "Stage", lambda *args, **kwargs: object())
    monkeypatch.setattr(understand, "load_skill", lambda _: "")
    source = SimpleNamespace(
        name="fixture",
        kind=source_kind,
        briefing=lambda: "",
        workdir=lambda: tmp_path,
        servers=lambda: {},
        builtin_tools=lambda: (),
    )
    understand.open_stage(source, out=tmp_path, require_setup_contract=True)
    assert captured["require_setup_contract"] is required


@pytest.mark.parametrize(
    "generic,interactive,required",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_cli_requires_setup_only_for_autonomous_generic_discovery(
    tmp_path, monkeypatch, generic, interactive, required
):
    import asyncio
    from types import SimpleNamespace
    from fi.alk.harness import cli

    captured = {}

    def stage(source, **kwargs):
        captured.update(kwargs)
        return object(), tmp_path

    async def converse(*args, **kwargs):
        pass

    source = SimpleNamespace(name="fixture", kind="repo")
    monkeypatch.setattr(cli, "resolve", lambda *args, **kwargs: source)
    monkeypatch.setattr(cli, "open_stage", stage)
    monkeypatch.setattr(cli, "_converse", converse)
    monkeypatch.setattr(cli, "opening", lambda _: "")
    monkeypatch.setattr(cli, "_guidance", lambda _: "")
    monkeypatch.setattr(cli, "load", lambda _: None)
    args = SimpleNamespace(
        kind="repo",
        name="fixture",
        path=tmp_path,
        out=tmp_path,
        interactive=interactive,
        job=SimpleNamespace(metadata={"generic_harness_v1": generic}),
    )
    assert asyncio.run(cli._understand(args)) == 1
    assert captured["require_setup_contract"] is required
    assert captured["max_turns"] == (140 if required else 70)


@pytest.mark.parametrize(
    "runtime",
    [
        None,
        {},
        {"command": ["python", "-m", "agent_cli"]},
        {"interface": {}},
        {"interface": {"kind": " "}},
    ],
)
def test_autonomous_chat_requires_explicit_interface_before_saving(tmp_path, runtime):
    result = accept_contract(
        payload(data_store={"kind": "none"}, runtime=runtime),
        tmp_path,
        require_setup_contract=True,
    )
    assert result["is_error"]
    message = result["content"][0]["text"]
    assert "missing-chat-interface" in message
    assert "ALK_USER_MESSAGE" in message
    assert "interactive CLI" in message
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("kind", ["http", "callable", "command"])
def test_explicit_chat_interface_is_saved(tmp_path, kind):
    result = accept_contract(
        payload(
            data_store={"kind": "none"},
            runtime={"interface": {"kind": kind, "port": 8000, "path": "/chat"}},
        ),
        tmp_path,
        require_setup_contract=True,
    )
    assert not result.get("is_error")
    saved = json.loads((tmp_path / "contract.json").read_text())
    assert saved["runtime"]["interface"]["kind"] == kind


def test_setup_admission_returns_both_missing_decisions(tmp_path):
    result = accept_contract(payload(), tmp_path, require_setup_contract=True)
    assert result["is_error"]
    message = result["content"][0]["text"]
    assert "missing-storage-decision" in message
    assert "missing-chat-interface" in message
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("interface", [
    {"kind": "", "protocol": "acp"},
    {"kind": "acp", "protocol": "acp"},
    {"kind": "websocket", "port": 8000, "path": "/ws"},
    {"kind": "command", "protocol": "acp"},
    {"kind": "callable", "protocol": "openai_chat"},
])
def test_generated_chat_rejects_unsupported_transport(tmp_path, interface):
    result = accept_contract(
        payload(data_store={"kind": "none"}, runtime={"interface": interface}),
        tmp_path,
        require_setup_contract=True,
    )
    assert result["is_error"]
    message = result["content"][0]["text"]
    assert "http" in message and "callable" in message and "command" in message
    assert "fi.alk" in message and "json_template" in message
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("modality,required", [("voice", True), ("chat", False)])
def test_legacy_and_voice_websocket_admission_unchanged(tmp_path, modality, required):
    result = accept_contract(
        payload(modality=modality, runtime={"interface": {
            "kind": "websocket", "port": 8000, "path": "/ws",
        }}),
        tmp_path,
        require_setup_contract=required,
    )
    assert not result.get("is_error")


@pytest.mark.parametrize("required", [False, True])
def test_chat_interface_schema_teaches_the_supported_bridge(tmp_path, required):
    from fi.alk.harness.contract import RuntimeInterface
    from fi.alk.harness.tools import contract_tools

    server = contract_tools(tmp_path, require_setup_contract=required)
    schema = server.tools[0].input_schema
    fields = schema["properties"]["runtime"]["properties"]["interface"]["properties"]
    assert "command" in fields["kind"]["enum"]
    assert "websocket" in fields["kind"]["enum"]
    assert "Generated chat supports http, callable, or command" in fields["kind"]["description"]
    assert "JSON stdin" in fields["kind"]["description"]
    assert "require fi.alk" in fields["protocol"]["description"]
    canonical = RuntimeInterface.model_json_schema()["properties"]
    assert canonical["kind"]["description"] == fields["kind"]["description"]
    assert canonical["protocol"]["description"] == fields["protocol"]["description"]
