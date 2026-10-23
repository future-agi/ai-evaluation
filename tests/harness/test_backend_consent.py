from types import SimpleNamespace

import pytest


@pytest.fixture
def backend_registry(monkeypatch):
    from fi.alk.harness import backends

    fake = SimpleNamespace(
        name="vertex-gemini",
        default_model="gemini-test",
        can_drive=lambda model: True,
    )
    monkeypatch.setitem(backends._LOADERS, "vertex-gemini", lambda: fake)
    monkeypatch.delenv("ALK_HARNESS_MODEL", raising=False)
    monkeypatch.delenv("ALK_GOOGLE_ADK_EXPLICITLY_APPROVED_BY_USER", raising=False)
    backends._LIVE.clear()
    yield backends
    backends._LIVE.clear()


@pytest.mark.parametrize(
    "name", ["vertex-gemini", "vertex_gemini", "vertexai-gemini", "gemini"]
)
def test_google_adk_backend_requires_explicit_user_approval(backend_registry, name):
    with pytest.raises(ValueError, match="explicit user approval"):
        backend_registry.resolve(name)


@pytest.mark.parametrize("value", ["", "true", "yes", "approved", "0"])
def test_google_adk_backend_rejects_ambiguous_approval_values(
    backend_registry, monkeypatch, value
):
    monkeypatch.setenv("ALK_GOOGLE_ADK_EXPLICITLY_APPROVED_BY_USER", value)

    with pytest.raises(ValueError, match="explicit user approval"):
        backend_registry.resolve("vertex-gemini")


def test_google_adk_backend_accepts_the_exact_user_approval(
    backend_registry, monkeypatch
):
    monkeypatch.setenv("ALK_GOOGLE_ADK_EXPLICITLY_APPROVED_BY_USER", "1")

    assert backend_registry.resolve("vertex-gemini").name == "vertex-gemini"


def test_default_backend_does_not_require_google_adk_approval(
    backend_registry, monkeypatch
):
    fake = SimpleNamespace(
        name="claude",
        default_model="claude-test",
        can_drive=lambda model: True,
    )
    monkeypatch.setitem(backend_registry._LOADERS, "claude", lambda: fake)

    assert backend_registry.resolve("claude").name == "claude"
