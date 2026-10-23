from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fi.alk.harness import runtime_repair
from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.diagnostics import HarnessDiagnostic
from fi.alk.harness.job import HarnessStage


@pytest.mark.parametrize("submit_during_investigation", [True, False])
def test_repair_decides_from_actual_bounded_source_evidence(
    tmp_path, monkeypatch, capsys, submit_during_investigation
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("# padding\n" * 100 + "def main(): pass\n")
    (source / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    contract = AgentContract(agent="test", runtime=Runtime(language="python"))
    specs = []
    briefings = []
    submissions = []

    class FakeStage:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            specs.append(spec)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            briefings.append(briefing)
            if "investigation" in self.spec.servers:
                readers = self.spec.servers["investigation"].tools
                reader = next(item for item in readers if item.name == "Read")
                await reader.handler({"file_path": "app.py"})
                if not submit_during_investigation:
                    return SimpleNamespace(
                        outcome="failed", turns=12, error="Maximum turns reached"
                    )
            else:
                assert "def main(): pass" in briefing
                assert "untrusted_investigation_evidence" in briefing
            submit = self.spec.servers["runtime_repair"].tools[0]
            submissions.append(
                await submit.handler(
                    {
                        "runtime": Runtime(
                            language="python", command=["python", "app.py"]
                        ).model_dump(),
                        "evidence_paths": ["app.py", "pyproject.toml"],
                        "summary": "Use the source entrypoint.",
                    }
                )
            )

    monkeypatch.setattr(runtime_repair, "Stage", FakeStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="runtime",
        code="generated_runtime_plan_invalid",
        message="wrong entrypoint",
    )
    patch = asyncio.run(
        runtime_repair.request_runtime_plan_patch(source, contract, (diagnostic,))
    )
    assert patch.runtime.command == ["python", "app.py"]
    assert all(not item.get("is_error") for item in submissions)
    assert [item.max_turns for item in specs] == (
        [80] if submit_during_investigation else [80, 4]
    )
    assert all(item.builtins == () for item in specs)
    if not submit_during_investigation:
        assert list(specs[1].servers) == ["runtime_repair"]
        assert [t.name for t in specs[1].servers["runtime_repair"].tools] == [
            "submit_runtime_plan_patch"
        ]
        assert len(briefings[1]) - len(briefings[0]) < 25000
    assert "def main" not in capsys.readouterr().out


def test_repair_decision_rejects_invalid_patch_and_source_escape(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private-secret-content")
    (source / "link").symlink_to(outside)
    (source / "large.py").write_text("# bounded evidence\n" * 1000)
    contract = AgentContract(agent="test", runtime=Runtime(language="python"))
    specs = []

    class FakeStage:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            specs.append(spec)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            if "investigation" in self.spec.servers:
                readers = {
                    tool.name: tool for tool in self.spec.servers["investigation"].tools
                }
                for name, args in (
                    ("Read", {"file_path": str(outside)}),
                    ("Read", {"file_path": "../private.txt"}),
                    ("Read", {"file_path": "link"}),
                    (
                        "Grep",
                        {
                            "file_path": "large.py",
                            "path": "../private.txt",
                            "pattern": "secret",
                        },
                    ),
                    ("Glob", {"file_path": "large.py", "path": "..", "pattern": "*"}),
                ):
                    result = await readers[name].handler(args)
                    assert result["is_error"]
                    assert "private-secret-content" not in str(result)
                grep = await readers["Grep"].handler({"pattern": "bounded"})
                glob = await readers["Glob"].handler({"pattern": "*"})
                assert not grep.get("is_error") and "large.py" in str(grep)
                assert not glob.get("is_error") and "large.py" in str(glob)
                assert "link" not in str(grep) and "link" not in str(glob)
                for _ in range(12):
                    await readers["Read"].handler({"file_path": "large.py"})
            else:
                evidence = briefing.split("<untrusted_investigation_evidence>\n")[
                    1
                ].split("\n</untrusted_investigation_evidence>")[0]
                assert len(evidence) <= 24012
                assert "private-secret-content" not in briefing
                result = (
                    await self.spec.servers["runtime_repair"]
                    .tools[0]
                    .handler(
                        {
                            "runtime": Runtime(
                                language="python", command=["python", "missing.py"]
                            ).model_dump(),
                            "evidence_paths": ["missing.py"],
                            "summary": "Invalid candidate.",
                        }
                    )
                )
                assert result["is_error"]

    monkeypatch.setattr(runtime_repair, "Stage", FakeStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="runtime",
        code="generated_runtime_plan_invalid",
        message="wrong entrypoint",
    )
    with pytest.raises(RuntimeError, match="typed_runtime_plan_patch_not_submitted"):
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(source, contract, (diagnostic,))
        )
    assert len(specs) == 2
