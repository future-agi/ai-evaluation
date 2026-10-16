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
        payload(data_store=store), tmp_path, require_storage_decision=True
    )
    assert result["is_error"]
    message = result["content"][0]["text"]
    assert "missing-storage-decision" in message
    assert "none" in message and "inspect the source" in message
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("kind", ["none", "filesystem", "in_process", "sqlite"])
def test_explicit_storage_decision_is_saved(tmp_path, kind):
    result = accept_contract(
        payload(data_store={"kind": kind}), tmp_path, require_storage_decision=True
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
        payload(modality="voice"), tmp_path, require_storage_decision=True
    ).get("is_error")


@pytest.mark.parametrize("source_kind,required", [("repo", True), ("provider", False)])
def test_stage_scopes_storage_admission_to_repository_sources(
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
    understand.open_stage(source, out=tmp_path, require_storage_decision=True)
    assert captured["require_storage_decision"] is required


@pytest.mark.parametrize(
    "generic,interactive,required",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_cli_requires_storage_only_for_autonomous_generic_discovery(
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
    assert captured["require_storage_decision"] is required
