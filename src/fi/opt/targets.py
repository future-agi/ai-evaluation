from __future__ import annotations

import copy
import hashlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field


OptimizationLayer = Literal[
    "objective",
    "harness",
    "integration",
    "framework",
    "streaming",
    "world",
    "security",
    "perception",
    "prompt",
    "planner",
    "autonomy",
    "policy",
    "tools",
    "memory",
    "router",
    "graph",
    "retrieval",
    "retriever",
    "model",
    "voice",
    "browser",
    "cua",
    "multi_agent",
    "orchestration",
    "action",
    "environment",
    "implementation",
    "evaluator",
    "custom",
]


class AgentCandidate(BaseModel):
    """
    A concrete agent/workflow configuration to evaluate.

    `config` is intentionally framework-neutral. It can represent a LangGraph
    graph config, CrewAI crew inputs, LiveKit voice settings, Pipecat pipeline
    parameters, browser/CUA policy, tool schemas, memory settings, or a plain
    prompt template.
    """

    id: str
    config: Dict[str, Any]
    target_name: Optional[str] = None
    layers: List[OptimizationLayer] = Field(default_factory=list)
    parent_id: Optional[str] = None
    patch: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        target_name: Optional[str] = None,
        layers: Optional[List[OptimizationLayer]] = None,
        parent_id: Optional[str] = None,
        patch: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentCandidate":
        payload = {
            "target_name": target_name,
            "config": config,
            "patch": patch or {},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            id=f"candidate_{digest}",
            config=copy.deepcopy(config),
            target_name=target_name,
            layers=list(layers or []),
            parent_id=parent_id,
            patch=copy.deepcopy(patch or {}),
            metadata=copy.deepcopy(metadata or {}),
        )

    def get_path(self, path: str, default: Any = None) -> Any:
        current: Any = self.config
        for part in _split_path(path):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return default
        return current

    def with_patch(
        self,
        patch: Dict[str, Any],
        *,
        layers: Optional[List[OptimizationLayer]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentCandidate":
        new_config = copy.deepcopy(self.config)
        for path, value in patch.items():
            set_path(new_config, path, value)
        merged_metadata = {**self.metadata, **(metadata or {})}
        return AgentCandidate.from_config(
            new_config,
            target_name=self.target_name,
            layers=layers or self.layers,
            parent_id=self.id,
            patch=patch,
            metadata=merged_metadata,
        )


class OptimizationTarget(BaseModel):
    """
    Framework-neutral optimization target.

    `search_space` maps dot paths to candidate values. Examples:
    - `prompt.system`: ["Be concise", "Ask one clarifying question first"]
    - `tools.0.description`: ["Search orders by id", "Search orders by id and email"]
    - `memory.strategy`: ["buffer", "summary", "vector"]
    - `router.default_model`: ["gpt-4o-mini", "claude-haiku"]
    - `voice.vad.min_silence_duration`: [0.1, 0.3, 0.5]
    - `browser.policy.allow_cross_origin`: [False, True]
    """

    name: str
    base_config: Dict[str, Any]
    layers: List[OptimizationLayer] = Field(default_factory=lambda: ["prompt"])
    search_space: Dict[str, List[Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def seed_candidate(self) -> AgentCandidate:
        return AgentCandidate.from_config(
            self.base_config,
            target_name=self.name,
            layers=self.layers,
            metadata={"kind": "seed", **self.metadata},
        )

    def iter_candidates(
        self,
        *,
        include_seed: bool = True,
        max_candidates: Optional[int] = None,
    ) -> Iterable[AgentCandidate]:
        count = 0
        if include_seed:
            yield self.seed_candidate()
            count += 1
            if max_candidates is not None and count >= max_candidates:
                return

        if not self.search_space:
            return

        paths = list(self.search_space.keys())
        value_lists = [self.search_space[path] for path in paths]
        for values in itertools.product(*value_lists):
            patch = dict(zip(paths, values))
            # Avoid duplicating the seed when every patch value equals base config.
            if all(self.seed_candidate().get_path(path) == value for path, value in patch.items()):
                continue
            yield self.seed_candidate().with_patch(
                patch,
                metadata={"kind": "search", "search_paths": paths, **self.metadata},
            )
            count += 1
            if max_candidates is not None and count >= max_candidates:
                return


class CandidateEvaluation(BaseModel):
    """Score and evidence for one evaluated candidate."""

    candidate: AgentCandidate
    score: float
    reason: str = ""
    individual_results: List[Any] = Field(default_factory=list)
    report: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = _split_path(path)
    if not parts:
        raise ValueError("Path cannot be empty.")

    current: Any = config
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(current, list):
            if not part.isdigit():
                raise ValueError(f"Expected numeric list index in path '{path}'.")
            list_index = int(part)
            _ensure_list_size(current, list_index)
            if current[list_index] is None:
                current[list_index] = [] if next_part.isdigit() else {}
            current = current[list_index]
        else:
            if part not in current or current[part] is None:
                current[part] = [] if next_part.isdigit() else {}
            current = current[part]

    final = parts[-1]
    if isinstance(current, list):
        if not final.isdigit():
            raise ValueError(f"Expected numeric list index in path '{path}'.")
        list_index = int(final)
        _ensure_list_size(current, list_index)
        current[list_index] = value
    else:
        current[final] = value


def _split_path(path: str) -> List[str]:
    return [part for part in path.split(".") if part]


def _ensure_list_size(items: List[Any], index: int) -> None:
    while len(items) <= index:
        items.append(None)
