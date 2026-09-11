"""Actor-source resolvers (canonical plan §4.1) — the "drop in any agent" surface.

A turn-based target agent can be declared as any of several *kinds*, resolved to
a runnable ``AgentWrapper`` / callable the environment drives. The kinds and their
config keys are the pydantic-schema twin of the manifest ``agent:`` block
(``target`` / ``factory`` / ``args`` / ``kwargs`` / ``method`` / ``input_mode`` /
``system_prompt`` / …), so a manifest agent and a spec ActorSource are the same
declaration in two encodings — no third vocabulary.

Each resolver is attached to an ``EndpointProfile`` in ``profiles.py`` and reached
through the one ``endpoint_registry``. Heavy imports (``wrap_agent``, LLM clients,
``httpx``) are deferred into the resolver bodies so the planner/registry stay light.

Security (hosted runs execute *customer-supplied* config on our infra):

* Kinds that import + call caller-named Python (``python_callable`` /
  ``import_object`` / ``factory`` / ``framework``) are **rejected in hosted runs**
  — the gate lives in ``resolve_chat_target`` and reads ``EndpointProfile.
  runs_caller_code`` (deny-by-default), not a denylist. In-process execution of
  customer code belongs in the sandboxed runtime (``RuntimeIsolation`` above
  ``shared_runner_process``), not the runner process. A single explicit,
  scarily-named escape (``ALK_UNSAFE_INPROCESS_CODE_ACTORS``) exists only for a
  trusted operator-configured default target and tests — never set it in prod.
* Env reads by ``system_prompt`` / ``http`` are restricted in hosted runs to the
  keys the job itself provisioned (``secret_refs``), so a job cannot name an
  arbitrary env var (e.g. another tenant's secret) to exfiltrate.
* Local (developer) runs resolve with ``hosted=False`` and stay permissive — the
  developer is running their own code on their own machine.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Mapping, Optional

_UNSAFE_INPROCESS_ENV = "ALK_UNSAFE_INPROCESS_CODE_ACTORS"

_WRAP_KEYS = (
    "method",
    "input_mode",
    "input_key",
    "input_kwargs",
    "output_key",
    "system_prompt",
    "metadata",
)


class ActorSourceError(ValueError):
    """Raised when an actor-source config cannot be resolved to a runnable agent."""


def inprocess_code_allowed() -> bool:
    """Whether in-process execution of caller code is explicitly permitted (the
    trusted operator-default target / tests). Deny by default."""
    return os.environ.get(_UNSAFE_INPROCESS_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _allowed_env_keys(secret_refs: Optional[Mapping[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for name, ref in (secret_refs or {}).items():
        ref_key = getattr(ref, "key", None)
        keys.add(str(ref_key if ref_key is not None else name))
    return keys


def _require_env_allowed(
    env_key: str, secret_refs: Optional[Mapping[str, Any]], *, hosted: bool
) -> None:
    if not hosted:
        return
    if env_key not in _allowed_env_keys(secret_refs):
        raise ActorSourceError(
            f"env_not_provisioned: hosted actor may only read job-provisioned "
            f"secrets; {env_key!r} is not in the job's secret_env"
        )


def _load_attr(ref: Any) -> Any:
    if not isinstance(ref, str) or ":" not in ref:
        raise ActorSourceError("actor target requires 'module:attribute'")
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise ActorSourceError(f"actor target malformed: {ref!r}")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr, None)
    if obj is None:
        raise ActorSourceError(f"actor target not found: {ref!r}")
    return obj


def _wrap_opts(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in _WRAP_KEYS if config.get(key) is not None}


def _instantiate_if_factory(loaded: Any, config: Mapping[str, Any]) -> Any:
    if not (config.get("factory") or config.get("instantiate")):
        return loaded
    args = config.get("args") or config.get("factory_args") or []
    kwargs = config.get("kwargs") or config.get("factory_kwargs") or {}
    return loaded(*list(args), **dict(kwargs))


# --------------------------------------------------------------------------- #
# resolvers — (config, secret_refs, *, hosted) -> AgentWrapper | Callable
# The code-loading kinds are rejected in hosted runs by resolve_chat_target
# (profile.runs_caller_code) before they are ever called; ``hosted`` is accepted
# here for a uniform signature and defensive checks.
# --------------------------------------------------------------------------- #
def resolve_python_callable(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Callable[..., Any]:
    ref = config.get("target") or config.get("callable")
    target = _load_attr(ref)
    if not callable(target):
        raise ActorSourceError(f"actor target is not callable: {ref!r}")
    return target


def resolve_import_object(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Any:
    from fi.simulate.agent.generic import wrap_agent

    obj = _load_attr(config.get("target"))
    return wrap_agent(obj, **_wrap_opts(config))


def resolve_factory(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Any:
    from fi.simulate.agent.generic import wrap_agent

    cls = _load_attr(config.get("target"))
    instance = _instantiate_if_factory(cls, {**config, "factory": True})
    return wrap_agent(instance, **_wrap_opts(config))


def resolve_framework(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Any:
    from fi.simulate.agent.generic import wrap_agent

    loaded = _instantiate_if_factory(_load_attr(config.get("target")), config)
    return wrap_agent(loaded, **_wrap_opts(config))


def resolve_system_prompt(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Any:
    from fi.simulate.agent.wrappers import OpenAIAgentWrapper

    prompt = config.get("system_prompt") or config.get("prompt")
    if not prompt:
        raise ActorSourceError("system_prompt actor requires 'system_prompt'")
    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    _require_env_allowed(api_key_env, secret_refs, hosted=hosted)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ActorSourceError(f"system_prompt actor needs {api_key_env} in the env")
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ActorSourceError("system_prompt actor requires the openai package") from exc
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = config.get("base_url")
    if base_url:
        client_kwargs["base_url"] = str(base_url)
    client = openai.AsyncOpenAI(**client_kwargs)
    return OpenAIAgentWrapper(
        client, model=str(config.get("model", "gpt-4-turbo")), system_prompt=str(prompt)
    )


def resolve_http(
    config: Mapping[str, Any],
    secret_refs: Optional[Mapping[str, Any]] = None,
    *,
    hosted: bool = False,
) -> Any:
    from fi.simulate.endpoints._http_actor import HttpChatAgent

    url = config.get("url")
    if not isinstance(url, str) or not url:
        raise ActorSourceError("http actor requires config.url")
    # auth_env is derived from the job's own secret_refs, so it is provisioned by
    # construction — no arbitrary env read.
    return HttpChatAgent(
        url=url,
        auth_header=str(config.get("auth_header") or "Authorization"),
        auth_env=_auth_env_from_refs(secret_refs),
        extra_headers=_string_map(config.get("headers")),
    )


def _auth_env_from_refs(secret_refs: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not secret_refs:
        return None
    for purpose in ("api_key", "authorization", "token"):
        for name, ref in secret_refs.items():
            ref_purpose = getattr(ref, "purpose", None)
            ref_key = getattr(ref, "key", None)
            if ref_purpose == purpose or name == purpose:
                return ref_key
    return None


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


__all__ = [
    "ActorSourceError",
    "inprocess_code_allowed",
    "resolve_factory",
    "resolve_framework",
    "resolve_http",
    "resolve_import_object",
    "resolve_python_callable",
    "resolve_system_prompt",
]
