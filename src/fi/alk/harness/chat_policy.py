from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .backends.base import ASK_TOOL, SessionSpec

KNOWN_CHAT_STAGES = frozenset(
    {"reception", "understand", "build", "scenarios", "run", "builder"}
)


class ChatPolicyError(RuntimeError):
    pass


class StagePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class LifecyclePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_user_input: bool = False
    interrupt_response: bool = False


class HarnessChatPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    stages: dict[str, StagePolicy] = Field(default_factory=dict)
    lifecycle: LifecyclePolicy = Field(default_factory=LifecyclePolicy)


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    harness_chat: HarnessChatPolicy


class EffectiveChatPolicy:
    def __init__(self, document: PolicyDocument, *, digest: str) -> None:
        self.document = document
        self.digest = digest

    def stage_enabled(self, stage: str) -> bool:
        if not self.document.harness_chat.enabled:
            return False
        configured = self.document.harness_chat.stages.get(stage)
        return bool(configured and configured.enabled)

    @property
    def can_request_user_input(self) -> bool:
        return bool(
            self.document.harness_chat.enabled
            and self.document.harness_chat.lifecycle.request_user_input
        )


def load_chat_policy(path: str | Path) -> EffectiveChatPolicy:
    target = Path(path)
    try:
        body = target.read_bytes()
        raw = yaml.safe_load(body)
        document = PolicyDocument.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ChatPolicyError(
            f"invalid hosted chat policy: {type(exc).__name__}"
        ) from exc
    if document.schema_version != 1:
        raise ChatPolicyError(
            f"unsupported hosted chat policy schema: {document.schema_version}"
        )
    unknown_stages = set(document.harness_chat.stages) - KNOWN_CHAT_STAGES
    if unknown_stages:
        raise ChatPolicyError(
            "unknown hosted chat stages: " + ", ".join(sorted(unknown_stages))
        )
    return EffectiveChatPolicy(
        document,
        digest=hashlib.sha256(body).hexdigest(),
    )


def apply_chat_policy(
    spec: SessionSpec,
    *,
    stage: str,
    policy: EffectiveChatPolicy,
) -> SessionSpec:
    if not policy.stage_enabled(stage):
        return SessionSpec(
            system_prompt=spec.system_prompt,
            cwd=spec.cwd,
            max_turns=spec.max_turns,
            model=spec.model,
            gated=spec.gated,
            thinking=spec.thinking,
            permission_override=spec.permission_override,
            idle_timeout_seconds=spec.idle_timeout_seconds,
            conversation=spec.conversation,
            workers=dict(spec.workers),
        )
    builtins = tuple(
        name
        for name in spec.builtins
        if name != ASK_TOOL or policy.can_request_user_input
    )
    return SessionSpec(
        system_prompt=spec.system_prompt,
        servers=dict(spec.servers),
        builtins=builtins,
        cwd=spec.cwd,
        max_turns=spec.max_turns,
        model=spec.model,
        ask=spec.ask if policy.can_request_user_input else None,
        gated=spec.gated,
        thinking=spec.thinking,
        permission_override=spec.permission_override,
        idle_timeout_seconds=spec.idle_timeout_seconds,
        conversation=spec.conversation,
        workers=dict(spec.workers),
    )
