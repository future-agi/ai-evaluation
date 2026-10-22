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


def _voice(tools: list[str], entries: list[dict], *, modality: str = "voice"):
    """A nudge-rebuilt contract: the first build's entrypoints, tools added since without one."""
    return AgentContract.model_validate(
        {
            "agent": "rides",
            "modality": modality,
            "real_use_cases": ["book a ride"],
            "runtime": {"language": "python", "command": ["python", "agent.py"]},
            "tools": [{"name": name} for name in tools],
            "tool_entrypoints": entries,
        }
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "agent").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "agent" / "ride_agent.py").write_text(
        "class RideAgent:\n"
        "    @function_tool()\n"
        "    async def search_places(self, text): ...\n"
        "    @function_tool(name='send_otp')\n"
        "    async def _send(self): ...\n",
        encoding="utf-8",
    )
    (source / "tests" / "test_rides.py").write_text(
        "def cancel_ride():\n    pass\n", encoding="utf-8"
    )
    return source


def test_a_voice_worker_runs_rebuilt_tools_its_source_defines(tmp_path: Path) -> None:
    contract = _voice(
        ["book_ride", "search_places", "send_otp"],
        [{"tool": "book_ride", "mode": "service", "endpoint": "book_ride"}],
    )

    report = certify_tool_inventory(
        contract, tmp_path, external_provider=False, source=_source(tmp_path)
    )

    assert report.certified_or_runtime_only == report.total == 3
    runtime = {item.tool: item.entrypoint_mode for item in report.tools}
    assert runtime == {
        "book_ride": "service",
        "search_places": "runtime",
        "send_otp": "runtime",
    }


def test_a_tool_the_runs_source_does_not_define_is_rejected_by_name(
    tmp_path: Path,
) -> None:
    contract = _voice(["search_places", "cancel_ride"], [])

    report = certify_tool_inventory(
        contract, tmp_path, external_provider=False, source=_source(tmp_path)
    )

    rejected = {
        item.tool: item.reason
        for item in report.tools
        if item.availability is ToolAvailability.REJECTED
    }
    assert rejected == {"cancel_ride": "the submitted source defines no cancel_ride"}


def test_tools_without_an_entrypoint_stay_rejected_outside_a_voice_worker(
    tmp_path: Path,
) -> None:
    chat = _voice(["search_places"], [], modality="chat")
    source = _source(tmp_path)

    for report in (
        certify_tool_inventory(chat, tmp_path, external_provider=False, source=source),
        certify_tool_inventory(
            _voice(["search_places"], []), tmp_path, external_provider=False
        ),
    ):
        assert report.tools[0].availability is ToolAvailability.REJECTED
        assert report.tools[0].entrypoint_mode == "missing"


def test_a_contract_with_every_entrypoint_certifies_the_same_with_its_source(
    tmp_path: Path,
) -> None:
    contract = _voice(
        ["book_ride"], [{"tool": "book_ride", "mode": "service", "endpoint": "book_ride"}]
    )

    assert certify_tool_inventory(
        contract, tmp_path, external_provider=False, source=_source(tmp_path)
    ) == certify_tool_inventory(contract, tmp_path, external_provider=False)
