from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .components import ComponentDiagnosis
from .targets import OptimizationTarget


@dataclass(frozen=True)
class AgentMutationBundle:
    """A coherent framework-aware config patch for agent optimization."""

    name: str
    framework: str
    component: str
    patch: dict[str, Any]
    reason: str = ""
    priority: float = 1.0
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameworkMutationRule:
    name: str
    framework: str
    components: tuple[str, ...]
    path_groups: tuple[tuple[str, ...], ...]
    reason: str
    priority: float = 1.0
    tags: tuple[str, ...] = ()


DEFAULT_FRAMEWORK_MUTATION_RULES: tuple[FrameworkMutationRule, ...] = (
    FrameworkMutationRule(
        name="langgraph_event_stream_checkpoint_bundle",
        framework="langgraph",
        components=("framework", "orchestration", "planner", "memory", "multi_agent"),
        path_groups=(
            ("langgraph", "stream_events", "framework.events.source", "framework.stream_events"),
            ("langgraph.nodes", "policy_node", "node"),
            ("planner.tool_sequence", "tool_sequence", "lookup", "tool"),
            ("memory.state_persistence", "checkpoint", "checkpointer", "case", "state"),
            ("langgraph.checkpointer", "framework.checkpoints", "checkpoint", "checkpoint_state"),
            ("framework.sessions", "thread_id", "session", "configurable"),
            ("framework.trace", "collector", "span", "event"),
        ),
        reason="Enable typed LangGraph event replay, graph nodes, tool order, and checkpoint/state capture together.",
        priority=4.0,
        tags=("stream_events", "checkpoint", "graph"),
    ),
    FrameworkMutationRule(
        name="langchain_event_stream_tool_bundle",
        framework="langchain",
        components=("framework", "tools", "retrieval", "memory", "orchestration"),
        path_groups=(
            ("langchain", "stream_events", "framework.events.source", "framework.stream_events"),
            ("callbacks", "tracing", "trace", "collector"),
            ("tool", "tool_calls", "tool_sequence", "schema"),
            ("retrieval", "retriever", "documents", "source"),
            ("memory", "state", "checkpoint", "persistence"),
        ),
        reason="Enable LangChain event streams plus tool, retrieval, and memory evidence.",
        priority=3.5,
        tags=("stream_events", "tools", "retrieval"),
    ),
    FrameworkMutationRule(
        name="openai_agents_trace_session_bundle",
        framework="openai_agents",
        components=("framework", "multi_agent", "tools", "policy", "memory"),
        path_groups=(
            ("openai_agents", "trace", "tracing", "span", "processor"),
            ("openai_agents.sessions", "session", "memory", "checkpoint"),
            ("handoff", "handoffs", "multi_agent"),
            ("guardrail", "guardrails", "policy", "safety"),
            ("tool", "tool_owner", "tool_output", "function_call"),
        ),
        reason="Capture OpenAI Agents traces, sessions, handoffs, guardrails, and tool ownership as one candidate.",
        priority=3.5,
        tags=("tracing", "sessions", "handoffs"),
    ),
    FrameworkMutationRule(
        name="otel_genai_trace_bundle",
        framework="opentelemetry",
        components=("framework", "model", "retrieval", "tools", "implementation"),
        path_groups=(
            ("otel", "opentelemetry", "otlp", "gen_ai", "semantic"),
            ("framework.trace", "span", "collector", "export"),
            ("model", "llm", "chat", "inference"),
            ("retrieval", "retriever", "data_source", "documents"),
            ("tool", "execute_tool", "function_call"),
        ),
        reason="Enable OpenTelemetry GenAI-style spans for inference, retrieval, and tool execution.",
        priority=3.2,
        tags=("otel", "gen_ai", "spans"),
    ),
    FrameworkMutationRule(
        name="mcp_tool_session_replay_bundle",
        framework="mcp",
        components=("framework", "tools", "implementation", "harness"),
        path_groups=(
            ("mcp", "tool_session", "session_export", "server"),
            ("tools.list", "tools_list", "mcp.tools.list", "tool_discovery"),
            ("schema", "input_schema", "tool_schema", "mcp.tools.schema"),
            ("tools.call", "tool_call", "mcp.tools.call", "arguments"),
            ("result", "tool_result", "mcp.tools.result", "output"),
            ("error", "tool_error", "mcp.tools.error", "exception"),
            ("resources", "resource_templates", "mcp.resources"),
        ),
        reason="Capture MCP tool discovery, schemas, calls, results, errors, and resources as one replayable session export.",
        priority=3.3,
        tags=("mcp", "tools", "schemas", "results"),
    ),
    FrameworkMutationRule(
        name="livekit_session_voice_pipeline_bundle",
        framework="livekit",
        components=("voice", "streaming", "framework", "perception", "tools"),
        path_groups=(
            ("livekit", "session_events", "agent_session", "room_io"),
            ("voice.trace", "trace", "timeline"),
            ("voice.webrtc", "webrtc", "get_stats", "getstats", "stats_source", "rtc_stats"),
            ("rtp", "rtp_counters", "packet", "packets", "packet_loss"),
            ("track", "track_identifier", "track_stats"),
            ("codec", "codec_id", "codec_stats"),
            ("audio_level", "audiolevel", "level_stats"),
            ("transport", "jitter", "packet_loss", "stable"),
            ("transcript", "transcription", "stt", "llm", "tts"),
            ("audio", "recording", "track", "webrtc", "media"),
            ("vad", "turn", "endpointing", "interruption"),
            ("routing", "route", "intent", "handoff"),
            ("tool", "handoff", "function"),
        ),
        reason="Capture LiveKit session events, WebRTC getStats/RTP/track/codec evidence, voice pipeline nodes, transcripts, media, turn handling, and tools together.",
        priority=3.4,
        tags=("voice", "session_events", "media", "webrtc"),
    ),
    FrameworkMutationRule(
        name="pipecat_frame_pipeline_bundle",
        framework="pipecat",
        components=("voice", "streaming", "framework", "perception"),
        path_groups=(
            ("pipecat", "frames", "frame_pipeline", "frame_source"),
            ("transcription", "stt", "transcript"),
            ("tts", "audio", "output_audio"),
            ("raw_pcm", "audio_decode", "waveform", "media"),
            ("interruption", "overlap", "barge"),
            ("timing", "latency", "turn", "endpointing"),
        ),
        reason="Enable Pipecat frame, transcript, TTS/audio, interruption, and timing capture as a bundle.",
        priority=3.4,
        tags=("voice", "frames", "timing"),
    ),
    FrameworkMutationRule(
        name="browser_cua_replay_bundle",
        framework="browser_cua",
        components=("browser", "cua", "perception", "action", "harness"),
        path_groups=(
            ("browser.trace", "playwright", "browser_use", "openai_cua", "trace"),
            ("screenshot", "video", "artifact", "visual"),
            ("har", "network", "resource_body"),
            ("actionability", "selector", "fallback"),
            ("storage", "cookies", "local_storage", "runtime"),
            ("allow_cross_origin", "domain"),
        ),
        reason="Capture browser/CUA traces, screenshots/video, network, actionability, storage/runtime, and safe domain policy.",
        priority=3.2,
        tags=("browser", "cua", "replay"),
    ),
    FrameworkMutationRule(
        name="redteam_campaign_matrix_evidence_bundle",
        framework="red_team",
        components=("security", "environment", "harness", "evaluator", "integration"),
        path_groups=(
            ("red_team.matrix_evidence", "coverage_matrix", "matrix_cells"),
            ("red_team.scenarios.matrix_cell_ids", "red_team.scenarios", "scenario_matrix"),
            ("red_team.runs.matrix_cell_ids", "red_team.runs", "run_artifact", "passed_run"),
            ("red_team.artifacts.matrix_cell_ids", "red_team.artifacts", "artifact_matrix"),
            ("red_team.artifacts.execution_evidence", "red_team.artifacts", "executed_evidence"),
            ("red_team.findings.matrix_cell_ids", "red_team.findings", "finding_mapping"),
            ("red_team.mitigations.matrix_cell_ids", "red_team.mitigations", "mitigation_mapping"),
            (
                "evaluation.red_team_campaign_quality.require_attack_surface_matrix",
                "require_attack_surface_matrix",
            ),
            (
                "evaluation.red_team_campaign_quality.require_run_artifacts",
                "require_run_artifacts",
            ),
            (
                "evaluation.red_team_campaign_quality.require_executed_run_evidence",
                "require_executed_run_evidence",
            ),
            (
                "evaluation.red_team_campaign_quality.require_finding_mapping",
                "require_finding_mapping",
            ),
            (
                "evaluation.red_team_campaign_quality.require_mitigation_mapping",
                "require_mitigation_mapping",
            ),
        ),
        reason=(
            "Bind red-team campaign matrix cells to scenario, passed-run, "
            "artifact, and mitigation evidence before relaxing quality gates."
        ),
        priority=3.8,
        tags=("red_team", "campaign", "matrix", "artifact", "mitigation"),
    ),
    FrameworkMutationRule(
        name="manifest_optimizer_trace_governance_bundle",
        framework="optimizer",
        components=("harness", "evaluator", "multi_agent", "planner", "integration"),
        path_groups=(
            ("optimizer.trace", "optimizer_trace", "trace", "trace_capture"),
            ("optimizer.society_trace", "optimizer_society_trace", "society_trace"),
            ("optimizer.roles", "optimizer.role", "roles", "role"),
            ("optimizer.role_graph", "role_graph", "role_graph.strategy"),
            ("optimizer.proposals", "optimizer.proposal", "proposals", "proposal"),
            ("optimizer.credit", "credit", "role_credit", "proposal_credit"),
            (
                "optimizer.governance.checks",
                "governance.checks",
                "governance_checks",
            ),
            (
                "optimizer.governance.contract_gate",
                "contract_gate",
                "contract",
            ),
            ("optimizer.governance.rollback", "rollback", "rollback_window"),
            (
                "optimizer.governance.search_locality",
                "search_locality",
                "locality",
            ),
            (
                "evaluation.optimizer_trace_quality.required_governance_signals",
                "required_governance_signals",
                "governance_signals",
            ),
            (
                "evaluation.optimizer_trace_quality.required_roles",
                "required_roles",
                "manifest_seed",
                "deterministic_search",
                "selection_steward",
            ),
            (
                "evaluation.optimizer_trace_quality.required_search_paths",
                "required_search_paths",
                "search_paths",
            ),
            (
                "evaluation.optimizer_trace_quality.min_role_count",
                "min_role_count",
                "role_count",
                "threshold",
            ),
            (
                "evaluation.optimizer_trace_quality.min_proposal_count",
                "min_proposal_count",
                "proposal_count",
                "threshold",
            ),
            (
                "evaluation.optimizer_trace_quality.min_governance_checks",
                "min_governance_checks",
                "governance_check_count",
                "threshold",
            ),
            (
                "evaluation.optimizer_trace_quality.min_credit_entries",
                "min_credit_entries",
                "role_credit",
            ),
            (
                "evaluation.optimizer_trace_quality.min_governance_pass_rate",
                "min_governance_pass_rate",
                "pass_rate",
            ),
            (
                "evaluation.optimizer_trace_quality.require_contract_gate",
                "require_contract_gate",
                "contract_gate",
            ),
            (
                "evaluation.optimizer_trace_quality.require_rollback",
                "require_rollback",
                "rollback",
            ),
            (
                "evaluation.optimizer_trace_quality.require_locality",
                "require_locality",
                "search_locality",
            ),
            (
                "optimization.target.search_space",
                "target.search_space",
                "manifest_search_space",
            ),
            (
                "optimization.optimizer.max_candidates",
                "optimizer.max_candidates",
                "max_candidates",
            ),
            ("optimization.threshold", "optimization.target_score", "threshold"),
            (
                "evaluation.manifest_optimization_quality.required_search_paths",
                "manifest_optimization_quality.required_search_paths",
                "required_search_paths",
            ),
            (
                "evaluation.manifest_optimization_quality.required_metrics",
                "manifest_optimization_quality.required_metrics",
                "required_metrics",
            ),
            (
                "evaluation.manifest_optimization_quality.min_history_count",
                "min_history_count",
                "history_count",
            ),
            (
                "evaluation.manifest_optimization_quality.min_candidate_count",
                "min_candidate_count",
                "candidate_count",
            ),
            (
                "evaluation.manifest_optimization_quality.min_patch_count",
                "min_patch_count",
                "patch_count",
            ),
            (
                "evaluation.manifest_optimization_quality.min_metric_count",
                "min_metric_count",
                "metric_count",
            ),
            (
                "evaluation.manifest_optimization_quality.min_final_score",
                "min_final_score",
                "manifest_quality_threshold",
            ),
            (
                "evaluation.manifest_optimization_quality.require_passed",
                "require_passed",
                "passed",
            ),
            (
                "evaluation.manifest_optimization_quality.require_best_candidate",
                "require_best_candidate",
                "best_candidate",
            ),
            (
                "evaluation.manifest_optimization_quality.require_best_config",
                "require_best_config",
                "best_config",
            ),
            (
                "evaluation.manifest_optimization_quality.require_history",
                "require_history",
                "history",
            ),
            (
                "evaluation.manifest_optimization_quality.require_candidate_patches",
                "require_candidate_patches",
                "candidate_patches",
            ),
            (
                "evaluation.manifest_optimization_quality.require_metrics",
                "require_metrics",
                "metrics",
            ),
            (
                "evaluation.manifest_optimization_quality.require_search_paths",
                "require_search_paths",
                "search_paths",
            ),
        ),
        reason=(
            "Bind manifest optimization search space and optimizer society-trace "
            "governance evidence before tightening trace and manifest quality gates."
        ),
        priority=3.9,
        tags=("manifest_optimization", "trace", "governance", "society_trace"),
    ),
    FrameworkMutationRule(
        name="rag_grounding_memory_bundle",
        framework="rag",
        components=("retrieval", "memory", "policy", "evaluator"),
        path_groups=(
            ("retrieval", "retriever", "source", "document"),
            ("citation", "citations", "attribution"),
            ("grounded", "grounded_only", "source_consistent"),
            ("stale", "fresh", "current"),
            ("memory", "memory_write", "write_resolution", "recall"),
        ),
        reason="Bundle current retrieval, citation/attribution, grounded generation, freshness, and memory-write settings.",
        priority=3.0,
        tags=("retrieval", "grounding", "memory"),
    ),
    FrameworkMutationRule(
        name="multi_agent_handoff_review_bundle",
        framework="multi_agent",
        components=("multi_agent", "planner", "memory", "policy"),
        path_groups=(
            ("multi_agent", "role", "specialist"),
            ("handoff", "contract", "context"),
            ("review", "qa", "critic"),
            ("memory", "shared", "case_summary"),
            ("reconciliation", "accepted_source", "evidence"),
        ),
        reason="Bundle specialist routing, handoff contracts, review, shared memory, and evidence-weighted reconciliation.",
        priority=3.0,
        tags=("multi_agent", "handoff", "review"),
    ),
)


class AgentMutationLibrary:
    """
    Proposes coherent config patches from framework and diagnosis evidence.

    The library is intentionally value-aware: each proposed patch uses only
    paths and values that already exist in `OptimizationTarget.search_space`.
    """

    def __init__(
        self,
        *,
        rules: Optional[Iterable[FrameworkMutationRule]] = None,
        bundles: Optional[Iterable[AgentMutationBundle]] = None,
        name: str = "default_framework_mutations",
    ) -> None:
        self.rules = (
            DEFAULT_FRAMEWORK_MUTATION_RULES
            if rules is None
            else tuple(rules)
        )
        self.bundles = tuple(bundles or ())
        self.name = name

    def propose(
        self,
        target: OptimizationTarget,
        *,
        diagnoses: Sequence[ComponentDiagnosis] = (),
        search_paths: Optional[Sequence[str]] = None,
        max_bundles: Optional[int] = None,
    ) -> list[AgentMutationBundle]:
        allowed_paths = [path for path in (search_paths or target.search_space) if path in target.search_space]
        if not allowed_paths:
            return []

        proposed: list[AgentMutationBundle] = []
        for bundle in self.bundles:
            filtered = _filter_patch(bundle.patch, target=target, allowed_paths=allowed_paths)
            if filtered:
                proposed.append(
                    AgentMutationBundle(
                        name=bundle.name,
                        framework=bundle.framework,
                        component=bundle.component,
                        patch=filtered,
                        reason=bundle.reason,
                        priority=bundle.priority,
                        tags=bundle.tags,
                    )
                )

        hints = _target_hints(target)
        for rule in self.rules:
            rule_score = _rule_score(rule, target=target, diagnoses=diagnoses, hints=hints)
            if rule_score <= 0:
                continue
            patch = _rule_patch(rule, target=target, allowed_paths=allowed_paths)
            if not patch:
                continue
            proposed.append(
                AgentMutationBundle(
                    name=rule.name,
                    framework=rule.framework,
                    component=_dominant_component(rule, diagnoses),
                    patch=patch,
                    reason=rule.reason,
                    priority=round(rule.priority + rule_score + len(patch) * 0.1, 4),
                    tags=rule.tags,
                )
            )

        proposed = _dedupe_bundles(proposed)
        proposed.sort(key=lambda item: (item.priority, len(item.patch), item.name), reverse=True)
        if max_bundles is not None:
            return proposed[:max(0, max_bundles)]
        return proposed


DEFAULT_AGENT_MUTATION_LIBRARY = AgentMutationLibrary()


def resolve_agent_mutation_library(
    value: Optional[AgentMutationLibrary | Iterable[AgentMutationBundle] | bool],
) -> Optional[AgentMutationLibrary]:
    if value is False:
        return None
    if value is None or value is True:
        return DEFAULT_AGENT_MUTATION_LIBRARY
    if isinstance(value, AgentMutationLibrary):
        return value
    return AgentMutationLibrary(bundles=value, rules=(), name="custom_mutation_bundles")


def dump_mutation_bundle(bundle: AgentMutationBundle) -> dict[str, Any]:
    return {
        "name": bundle.name,
        "framework": bundle.framework,
        "component": bundle.component,
        "patch": dict(bundle.patch),
        "reason": bundle.reason,
        "priority": bundle.priority,
        "tags": list(bundle.tags),
    }


def _rule_score(
    rule: FrameworkMutationRule,
    *,
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
    hints: str,
) -> float:
    score = 0.0
    framework_token = _normalize_token(rule.framework)
    if framework_token and framework_token in hints:
        score += 2.0
    for path in target.search_space:
        if _path_matches(path, (rule.framework, *rule.tags)):
            score += 0.4

    if not diagnoses:
        return score

    if score <= 0 and rule.framework not in {"rag", "multi_agent"}:
        return 0.0

    diagnosis_score = 0.0
    for diagnosis in diagnoses:
        if str(diagnosis.component) in rule.components:
            diagnosis_score += 2.0 * diagnosis.confidence
        for suggested_path in diagnosis.suggested_paths:
            if any(_path_matches(suggested_path, group) for group in rule.path_groups):
                diagnosis_score += 1.0 * diagnosis.confidence
    if diagnosis_score <= 0:
        return 0.0
    return score + diagnosis_score


def _rule_patch(
    rule: FrameworkMutationRule,
    *,
    target: OptimizationTarget,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    base = target.seed_candidate()
    for group in rule.path_groups:
        path = _best_matching_path(group, target=target, allowed_paths=allowed_paths, used_paths=patch)
        if path is None:
            continue
        value = _preferred_value(
            path,
            target.search_space[path],
            tokens=(*group, rule.framework, *rule.tags),
        )
        if value != base.get_path(path):
            patch[path] = value
    return patch


def _best_matching_path(
    tokens: Sequence[str],
    *,
    target: OptimizationTarget,
    allowed_paths: Sequence[str],
    used_paths: Mapping[str, Any],
) -> Optional[str]:
    candidates = [
        path
        for path in allowed_paths
        if path not in used_paths and target.search_space.get(path) and _path_matches(path, tokens)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (_path_match_score(path, tokens), -len(path), path), reverse=True)
    return candidates[0]


def _preferred_value(path: str, values: Sequence[Any], *, tokens: Sequence[str]) -> Any:
    if not values:
        return None
    ranked = sorted(
        values,
        key=lambda value: (_value_score(path, value, tokens), json.dumps(value, sort_keys=True, default=str)),
        reverse=True,
    )
    return ranked[0]


def _value_score(path: str, value: Any, tokens: Sequence[str]) -> float:
    normalized_path = _normalize_token(path)
    if isinstance(value, bool):
        if any(token in normalized_path for token in ("allow_cross_origin", "unsafe", "disable", "ignore")):
            return 3.0 if value is False else 0.0
        return 3.0 if value is True else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, (list, tuple, set)):
        return len(value) + sum(_value_score(path, item, tokens) * 0.1 for item in value)
    if isinstance(value, Mapping):
        return len(value) + sum(1.0 for item in value.values() if item)

    text = _normalize_token(value)
    score = 0.0
    for token in tokens:
        normalized = _normalize_token(token)
        if normalized and normalized in text:
            score += 2.0
    if "search_locality" in normalized_path:
        for preferred in ("local", "bounded", "neighborhood", "nearby"):
            if preferred in text:
                score += 2.0
        for avoided in ("global", "unbounded"):
            if avoided in text:
                score -= 1.0
    for preferred in (
        "enabled",
        "enable",
        "true",
        "capture",
        "captured",
        "full",
        "complete",
        "stream_events",
        "langgraph_stream_events",
        "trace",
        "tracing",
        "otel",
        "opentelemetry",
        "checkpoint",
        "checkpointer",
        "persistent",
        "thread_id",
        "configurable",
        "sqlite",
        "store",
        "case_status",
        "lookup_then_refund",
        "summary",
        "current",
        "fresh",
        "grounded",
        "citation",
        "evidence",
        "red_team",
        "explicit",
        "society_trace",
        "role_graph",
        "role_weighted",
        "proposal",
        "proposals",
        "credit",
        "governance",
        "contract_gate",
        "rollback",
        "search_locality",
        "bounded",
        "steward",
        "manifest",
        "manifest_optimization",
        "required",
        "requirement",
        "quality_gate",
    ):
        if preferred in text:
            score += 1.0
    for avoided in ("none", "disabled", "manual", "buffer", "refund_only", "stale", "unsafe"):
        if avoided in text:
            score -= 1.0
    return score


def _filter_patch(
    patch: Mapping[str, Any],
    *,
    target: OptimizationTarget,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    base = target.seed_candidate()
    allowed = set(allowed_paths)
    for path, value in patch.items():
        values = target.search_space.get(path)
        if path not in allowed or not values:
            continue
        if not any(_json_equal(value, candidate_value) for candidate_value in values):
            continue
        if not _json_equal(value, base.get_path(path)):
            filtered[path] = value
    return filtered


def _dedupe_bundles(bundles: Sequence[AgentMutationBundle]) -> list[AgentMutationBundle]:
    seen: set[str] = set()
    deduped: list[AgentMutationBundle] = []
    for bundle in bundles:
        key = json.dumps(bundle.patch, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bundle)
    return deduped


def _dominant_component(
    rule: FrameworkMutationRule,
    diagnoses: Sequence[ComponentDiagnosis],
) -> str:
    for diagnosis in diagnoses:
        if str(diagnosis.component) in rule.components:
            return str(diagnosis.component)
    return rule.components[0] if rule.components else "framework"


def _target_hints(target: OptimizationTarget) -> str:
    payload = {
        "name": target.name,
        "layers": target.layers,
        "metadata": target.metadata,
        "paths": list(target.search_space),
    }
    return _normalize_token(payload)


def _path_matches(path: str, tokens: Sequence[str]) -> bool:
    normalized_path = _normalize_token(path)
    return any(_normalize_token(token) in normalized_path for token in tokens if _normalize_token(token))


def _path_match_score(path: str, tokens: Sequence[str]) -> float:
    normalized_path = _normalize_token(path)
    score = 0.0
    for token in tokens:
        normalized = _normalize_token(token)
        if not normalized:
            continue
        if normalized_path == normalized:
            score += 4.0
        elif normalized_path.startswith(f"{normalized}.") or normalized_path.startswith(f"{normalized}_"):
            score += 3.0
        elif normalized in normalized_path:
            score += 1.0
    return score


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _normalize_token(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    return value.lower().replace("-", "_").replace(" ", "_")
