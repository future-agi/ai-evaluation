from pathlib import Path

from fi.alk.harness.livekit_source import infer_livekit_agent_name_from_source


def test_infers_environment_controlled_dispatch_name(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        'agent_name = os.getenv("LIVEKIT_AGENT_NAME", "voice-agent")\n',
        encoding="utf-8",
    )
    assert infer_livekit_agent_name_from_source(tmp_path) == "voice-agent"


def test_infers_modern_static_rtc_session_dispatch_name(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        '@server.rtc_session(\n    agent_name="my-agent",\n)\nasync def entry(ctx): pass\n',
        encoding="utf-8",
    )
    assert infer_livekit_agent_name_from_source(tmp_path) == "my-agent"


def test_dynamic_rtc_session_dispatch_name_is_not_guessed(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "@server.rtc_session(agent_name=settings.agent_name)\nasync def entry(ctx): pass\n",
        encoding="utf-8",
    )
    assert infer_livekit_agent_name_from_source(tmp_path) == ""
