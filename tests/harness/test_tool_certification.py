from __future__ import annotations

from pathlib import Path

from fi.alk.harness.contract import AgentContract
from fi.alk.harness.tool_certification import (
    ToolAvailability,
    certify_tool_inventory,
)


def _contract(*, mode: str = "import") -> AgentContract:
    return AgentContract.model_validate(
        {
            "agent": "test",
            "real_use_cases": ["inspect status"],
            "tools": [{"name": "check_status", "args": ["id"]}],
            "tool_entrypoints": [
                {
                    "tool": "check_status",
                    "mode": mode,
                    "module": "agent",
                    "callable": "check_status",
                    "endpoint": "/status" if mode == "service" else "",
                }
            ],
        }
    )


def test_compiled_handler_is_certified_and_report_is_stable(tmp_path: Path) -> None:
    handlers = tmp_path / "handlers"
    handlers.mkdir()
    (handlers / "check_status.py").write_text("pass\n", encoding="utf-8")

    first = certify_tool_inventory(_contract(), tmp_path, external_provider=False)
    second = certify_tool_inventory(_contract(), tmp_path, external_provider=False)

    assert first == second
    assert first.certified_or_runtime_only == first.total == 1
    assert first.tools[0].availability is ToolAvailability.CERTIFIED


def test_missing_implementation_and_orphan_entrypoint_are_rejected(tmp_path: Path) -> None:
    payload = _contract().model_dump(mode="python")
    payload["tool_entrypoints"].append(
        {"tool": "invented", "mode": "service", "endpoint": "/x"}
    )
    contract = AgentContract.model_validate(payload)
    report = certify_tool_inventory(contract, tmp_path, external_provider=False)

    assert report.certified_or_runtime_only == 0
    assert {item.availability for item in report.tools} == {ToolAvailability.REJECTED}


def test_external_provider_tools_are_explicitly_runtime_only(tmp_path: Path) -> None:
    report = certify_tool_inventory(_contract(), tmp_path, external_provider=True)

    assert report.certified_or_runtime_only == report.total == 1
    assert report.tools[0].availability is ToolAvailability.RUNTIME_ONLY
