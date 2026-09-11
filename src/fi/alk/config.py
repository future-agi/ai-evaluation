from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping, Optional


DEFAULT_API_URL = "https://api.futureagi.com"
API_KEY_ENV_NAMES = (
    "FI_API_KEY",
    "FUTURE_AGI_API_KEY",
    "AGENT_LEARNING_API_KEY",
)
SECRET_KEY_ENV_NAMES = (
    "FI_SECRET_KEY",
    "FUTURE_AGI_SECRET_KEY",
    "AGENT_LEARNING_SECRET_KEY",
)


@dataclass(frozen=True)
class AgentLearningConfig:
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    api_url: str = DEFAULT_API_URL
    project_id: Optional[str] = None
    workspace_id: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "AgentLearningConfig":
        source = environ or os.environ
        api_key = next(
            (
                source[name]
                for name in API_KEY_ENV_NAMES
                if source.get(name)
            ),
            None,
        )
        secret_key = next(
            (
                source[name]
                for name in SECRET_KEY_ENV_NAMES
                if source.get(name)
            ),
            None,
        )
        return cls(
            api_key=api_key,
            secret_key=secret_key or api_key,
            api_url=source.get("FI_BASE_URL")
            or source.get("FUTURE_AGI_API_URL")
            or source.get("AGENT_LEARNING_API_URL")
            or DEFAULT_API_URL,
            project_id=source.get("AGENT_LEARNING_PROJECT_ID")
            or source.get("FUTURE_AGI_PROJECT_ID"),
            workspace_id=source.get("AGENT_LEARNING_WORKSPACE_ID")
            or source.get("FUTURE_AGI_WORKSPACE_ID"),
        )


_CONFIG = AgentLearningConfig.from_env()


def configure(
    *,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    api_url: Optional[str] = None,
    project_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> AgentLearningConfig:
    """Configure the shared SDK context for all agent-learning modules."""

    global _CONFIG
    updates = {}
    if api_key is not None:
        updates["api_key"] = api_key
        updates.setdefault("secret_key", api_key)
    if secret_key is not None:
        updates["secret_key"] = secret_key
    if api_url is not None:
        updates["api_url"] = api_url
    if project_id is not None:
        updates["project_id"] = project_id
    if workspace_id is not None:
        updates["workspace_id"] = workspace_id
    _CONFIG = replace(_CONFIG, **updates)
    _sync_env(_CONFIG)
    return _CONFIG


def current_config() -> AgentLearningConfig:
    return _CONFIG


def get_api_key(required: bool = False) -> Optional[str]:
    key = _CONFIG.api_key or AgentLearningConfig.from_env().api_key
    if required and not key:
        names = ", ".join(API_KEY_ENV_NAMES)
        raise RuntimeError(f"Missing Future AGI API key. Set one of: {names}.")
    return key


def _sync_env(config: AgentLearningConfig) -> None:
    if config.api_key:
        os.environ["AGENT_LEARNING_API_KEY"] = config.api_key
        os.environ["FUTURE_AGI_API_KEY"] = config.api_key
        os.environ["FI_API_KEY"] = config.api_key
    secret_key = config.secret_key or config.api_key
    if secret_key:
        os.environ["AGENT_LEARNING_SECRET_KEY"] = secret_key
        os.environ["FUTURE_AGI_SECRET_KEY"] = secret_key
        os.environ["FI_SECRET_KEY"] = secret_key
    if config.api_url:
        os.environ["FI_BASE_URL"] = config.api_url
        os.environ["FUTURE_AGI_API_URL"] = config.api_url
        os.environ["AGENT_LEARNING_API_URL"] = config.api_url
    if config.project_id:
        os.environ["AGENT_LEARNING_PROJECT_ID"] = config.project_id
        os.environ["FUTURE_AGI_PROJECT_ID"] = config.project_id
    if config.workspace_id:
        os.environ["AGENT_LEARNING_WORKSPACE_ID"] = config.workspace_id
        os.environ["FUTURE_AGI_WORKSPACE_ID"] = config.workspace_id


_sync_env(_CONFIG)
