from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = (
        Path(__file__).parents[1]
        / "oss/simulation-acceptance/trigger_livekit_outbound.py"
    )
    spec = importlib.util.spec_from_file_location("trigger_livekit_outbound", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_target_client_falls_back_to_simulator(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    for name in (
        "LIVEKIT_TARGET_URL",
        "LIVEKIT_TARGET_API_KEY",
        "LIVEKIT_TARGET_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    simulator = object()

    target = module._client(
        url_env="LIVEKIT_TARGET_URL",
        api_key_env="LIVEKIT_TARGET_API_KEY",
        api_secret_env="LIVEKIT_TARGET_API_SECRET",
        fallback=simulator,
    )

    assert target is simulator


def test_target_client_requires_complete_credentials(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    monkeypatch.setenv("LIVEKIT_TARGET_URL", "wss://target.example.com")
    monkeypatch.delenv("LIVEKIT_TARGET_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_TARGET_API_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="credentials_incomplete"):
        module._client(
            url_env="LIVEKIT_TARGET_URL",
            api_key_env="LIVEKIT_TARGET_API_KEY",
            api_secret_env="LIVEKIT_TARGET_API_SECRET",
            fallback=object(),
        )


def test_target_client_uses_separate_project(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    captured = {}
    monkeypatch.setenv("LIVEKIT_TARGET_URL", "wss://target.example.com")
    monkeypatch.setenv("LIVEKIT_TARGET_API_KEY", "target-key")
    monkeypatch.setenv("LIVEKIT_TARGET_API_SECRET", "target-secret")
    monkeypatch.setattr(
        module.api,
        "LiveKitAPI",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    module._client(
        url_env="LIVEKIT_TARGET_URL",
        api_key_env="LIVEKIT_TARGET_API_KEY",
        api_secret_env="LIVEKIT_TARGET_API_SECRET",
        fallback=object(),
    )

    assert captured == {
        "url": "https://target.example.com",
        "api_key": "target-key",
        "api_secret": "target-secret",
    }
