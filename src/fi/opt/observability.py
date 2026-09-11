from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from .targets import AgentCandidate, CandidateEvaluation


OBSERVABILITY_SCHEMA_VERSION = "agent-opt.observability.v1"
REGRESSION_DATASET_SCHEMA_VERSION = "agent-opt.regression-dataset.v1"
REGRESSION_DATASET_COVERAGE_SCHEMA_VERSION = "agent-opt.regression-dataset-coverage.v1"
DATASET_SINK_SCHEMA_VERSION = "agent-opt.futureagi-dataset-sink.v1"
FUTUREAGI_EXPERIMENT_HISTORY_SCHEMA_VERSION = "agent-opt.futureagi-experiment-history.v1"
REGISTRY_REPLAY_PACK_MANIFEST_SCHEMA_VERSION = "agent-opt.registry-replay-pack.v1"
REGISTRY_REPLAY_PACK_PROMOTION_SCHEMA_VERSION = "agent-opt.registry-replay-pack-promotion.v1"
REGISTRY_REPLAY_PACK_LINEAGE_SCHEMA_VERSION = "agent-opt.registry-replay-pack-lineage.v1"
REGISTRY_REPLAY_PACK_TRIAGE_SCHEMA_VERSION = "agent-opt.registry-replay-pack-triage.v1"
FUTUREAGI_REGRESSION_DATASET_COLUMNS = (
    {"name": "case_id", "data_type": "text"},
    {"name": "query", "data_type": "text"},
    {"name": "response", "data_type": "text"},
    {"name": "expected_response", "data_type": "json"},
    {"name": "observability", "data_type": "json"},
    {"name": "tags", "data_type": "array"},
    {"name": "metadata", "data_type": "json"},
)


class AgentObservabilityRecord(BaseModel):
    """One normalized production trace/evaluation record."""

    index: int
    source: str
    framework: str
    run_id: Optional[str] = None
    candidate_id: Optional[str] = None
    score: float
    passed: bool
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    trace_signals: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentObservabilityWindow(BaseModel):
    """A live or exported observability window ready for rollback monitoring."""

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    source: str
    framework: str
    candidate: Optional[AgentCandidate] = None
    records: list[AgentObservabilityRecord] = Field(default_factory=list)
    required_metrics: dict[str, float] = Field(default_factory=dict)
    required_trace_signals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def failures(self) -> list[str]:
        failures: list[str] = []
        for record in self.records:
            failures.extend(record.failures)
        return failures

    @property
    def average_score(self) -> Optional[float]:
        if not self.records:
            return None
        return sum(record.score for record in self.records) / len(self.records)

    def to_live_evaluations(
        self,
        *,
        candidate: Optional[AgentCandidate] = None,
    ) -> list[CandidateEvaluation]:
        active_candidate = candidate or self.candidate
        return [
            _evaluation_from_observability_record(record, candidate=active_candidate)
            for record in self.records
        ]

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentRegressionCase(BaseModel):
    """One replayable regression case derived from production observability."""

    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return self.model_dump()

    def to_futureagi_row(self) -> dict[str, Any]:
        observability = copy.deepcopy(self.input.get("observability", self.input))
        expected = copy.deepcopy(self.expected)
        metadata = {
            **copy.deepcopy(self.metadata),
            "dataset_case_id": self.id,
            "tags": list(self.tags),
        }
        run_id = observability.get("run_id") if isinstance(observability, Mapping) else None
        source = observability.get("source") if isinstance(observability, Mapping) else None
        framework = observability.get("framework") if isinstance(observability, Mapping) else None
        failures = observability.get("failures", []) if isinstance(observability, Mapping) else []
        query = "Replay observability regression case"
        if run_id:
            query += f" {run_id}"
        if source or framework:
            query += f" from {source or 'unknown'}/{framework or 'unknown'}"
        response = "; ".join(str(item) for item in failures) if failures else "passed"
        return {
            "case_id": self.id,
            "query": query,
            "response": response,
            "expected_response": expected,
            "observability": observability,
            "tags": list(self.tags),
            "metadata": metadata,
        }


class AgentRegressionDataset(BaseModel):
    """A durable regression/replay dataset built from observability windows."""

    schema_version: str = REGRESSION_DATASET_SCHEMA_VERSION
    name: str
    source: str
    framework: str
    cases: list[AgentRegressionCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [case.to_record() for case in self.cases]

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(record, sort_keys=True, default=str)
            for record in self.to_records()
        )

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        text = self.to_jsonl()
        target.write_text(text + ("\n" if text else ""))
        return target

    def to_futureagi_rows(self) -> list[dict[str, Any]]:
        return [case.to_futureagi_row() for case in self.cases]

    def coverage_report(
        self,
        *,
        target: Any = None,
        metric_path_hints: Optional[Mapping[str, Sequence[str]]] = None,
        tag_path_hints: Optional[Mapping[str, Sequence[str]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AgentRegressionDatasetCoverageReport":
        return build_agent_regression_dataset_coverage_report(
            self,
            target=target,
            metric_path_hints=metric_path_hints,
            tag_path_hints=tag_path_hints,
            metadata=metadata,
        )

    def to_observability_window(
        self,
        *,
        candidate: Optional[AgentCandidate] = None,
        source: Optional[str] = None,
        framework: Optional[str] = None,
        required_metrics: Optional[Mapping[str, float]] = None,
        required_trace_signals: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AgentObservabilityWindow:
        """
        Reconstruct an observability window from replay cases.

        This is the bridge for Future AGI dataset-driven re-optimization: rows
        pulled from a Future AGI regression dataset become the same live
        evaluation evidence consumed by AgentFeedbackOptimizer.
        """

        thresholds = _regression_dataset_required_metrics(
            self.cases,
            override=required_metrics,
        )
        signals = _regression_dataset_required_trace_signals(
            self.cases,
            override=required_trace_signals,
        )
        records = [
            _observability_record_from_regression_case(
                case,
                index=index,
                candidate=candidate,
                source=source or self.source,
                framework=framework or self.framework,
            )
            for index, case in enumerate(self.cases, start=1)
        ]
        window_metadata = {
            "kind": "regression_dataset_replay",
            "regression_dataset_name": self.name,
            "regression_dataset_schema_version": self.schema_version,
            "regression_case_count": len(self.cases),
            "regression_dataset_metadata": copy.deepcopy(self.metadata),
            **dict(metadata or {}),
        }
        return AgentObservabilityWindow(
            source=_resolve_window_source(records, fallback=source or self.source),
            framework=_resolve_window_framework(records, fallback=framework or self.framework),
            candidate=candidate,
            records=records,
            required_metrics=thresholds,
            required_trace_signals=signals,
            metadata=window_metadata,
        )

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentRegressionDatasetCoverageReport(BaseModel):
    """Coverage summary for a regression dataset before optimizer replay."""

    schema_version: str = REGRESSION_DATASET_COVERAGE_SCHEMA_VERSION
    dataset_name: str
    source: str
    framework: str
    case_count: int
    failed_case_count: int = 0
    passed_case_count: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    framework_counts: dict[str, int] = Field(default_factory=dict)
    tag_counts: dict[str, int] = Field(default_factory=dict)
    observed_metric_case_counts: dict[str, int] = Field(default_factory=dict)
    required_metric_case_counts: dict[str, int] = Field(default_factory=dict)
    failed_metric_case_counts: dict[str, int] = Field(default_factory=dict)
    required_metrics: dict[str, float] = Field(default_factory=dict)
    required_trace_signals: list[str] = Field(default_factory=list)
    trace_signal_case_counts: dict[str, int] = Field(default_factory=dict)
    missing_trace_signal_case_counts: dict[str, int] = Field(default_factory=dict)
    search_path_case_counts: dict[str, int] = Field(default_factory=dict)
    uncovered_required_metrics: list[str] = Field(default_factory=list)
    uncovered_search_paths: list[str] = Field(default_factory=list)
    failure_examples: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentDatasetSinkResult(BaseModel):
    """Result from exporting a regression dataset to Future AGI."""

    schema_version: str = DATASET_SINK_SCHEMA_VERSION
    provider: str
    dataset_name: str
    dataset_id: Optional[str] = None
    case_count: int
    endpoint: Optional[str] = None
    dry_run: bool = False
    status: str
    failures: list[str] = Field(default_factory=list)
    response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures and self.status in {"planned", "created", "inserted"}


class AgentRegistryReplayPackManifest(BaseModel):
    """Version-pinned manifest for a Future AGI registry replay pack."""

    schema_version: str = REGISTRY_REPLAY_PACK_MANIFEST_SCHEMA_VERSION
    name: str
    provider: str = "futureagi"
    registry_version: str
    dataset_name: str
    dataset_id: Optional[str] = None
    case_count: int
    case_ids: list[str] = Field(default_factory=list)
    case_signature: str
    retention_key: str
    selection_complete: bool = False
    coverage_score: float = 0.0
    selected_positive_count: int = 0
    selected_negative_count: int = 0
    required_presets: list[str] = Field(default_factory=list)
    required_invariant_families: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentRegistryReplayPackPromotionCheck(BaseModel):
    """Promotion gate for a pinned Future AGI registry replay pack."""

    schema_version: str = REGISTRY_REPLAY_PACK_PROMOTION_SCHEMA_VERSION
    promotable: bool
    dataset_name: str
    dataset_id: Optional[str] = None
    registry_version: str
    expected_registry_version: str
    expected_case_count: int
    loaded_case_count: int
    expected_case_signature: str
    loaded_case_signature: str
    coverage_score: float
    min_coverage_score: float
    selection_complete: bool
    replay_record_count: int = 0
    optimizer_score: Optional[float] = None
    min_optimizer_score: float
    failures: list[str] = Field(default_factory=list)
    manifest: AgentRegistryReplayPackManifest
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentRegistryReplayPackLineageEntry(BaseModel):
    """One versioned Future AGI registry replay-pack lineage row."""

    registry_version: str
    dataset_name: str
    dataset_id: Optional[str] = None
    retention_key: str
    case_count: int
    case_signature: str
    coverage_score: float
    selection_complete: bool
    required_presets: list[str] = Field(default_factory=list)
    required_invariant_families: list[str] = Field(default_factory=list)
    promotion_promotable: Optional[bool] = None
    loaded_case_count: Optional[int] = None
    replay_record_count: Optional[int] = None
    readback_signature_matches: Optional[bool] = None
    optimizer_score: Optional[float] = None
    min_optimizer_score: Optional[float] = None
    optimizer_backend: Optional[str] = None
    selected_patch_signature: Optional[str] = None
    failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRegistryReplayPackLineageTransition(BaseModel):
    """Delta from one registry replay-pack version to the next."""

    from_registry_version: str
    to_registry_version: str
    from_dataset_id: Optional[str] = None
    to_dataset_id: Optional[str] = None
    case_count_delta: int = 0
    coverage_delta: float = 0.0
    optimizer_score_delta: Optional[float] = None
    case_signature_changed: bool = False
    retention_key_changed: bool = False
    selected_patch_changed: Optional[bool] = None
    optimizer_backend_changed: Optional[bool] = None
    promotion_status_changed: Optional[bool] = None
    added_required_presets: list[str] = Field(default_factory=list)
    removed_required_presets: list[str] = Field(default_factory=list)
    added_invariant_families: list[str] = Field(default_factory=list)
    removed_invariant_families: list[str] = Field(default_factory=list)
    drift_reasons: list[str] = Field(default_factory=list)


class AgentRegistryReplayPackLineageReport(BaseModel):
    """Compare Future AGI registry replay packs across dataset versions."""

    schema_version: str = REGISTRY_REPLAY_PACK_LINEAGE_SCHEMA_VERSION
    provider: str = "futureagi"
    entry_count: int
    entries: list[AgentRegistryReplayPackLineageEntry] = Field(default_factory=list)
    transitions: list[AgentRegistryReplayPackLineageTransition] = Field(default_factory=list)
    latest_registry_version: Optional[str] = None
    latest_dataset_id: Optional[str] = None
    latest_promotable: Optional[bool] = None
    best_registry_version: Optional[str] = None
    best_dataset_id: Optional[str] = None
    best_optimizer_score: Optional[float] = None
    drift_detected: bool = False
    drift_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentRegistryReplayPackTriageReport(BaseModel):
    """Rollout triage for Future AGI registry replay-pack lineage drift."""

    schema_version: str = REGISTRY_REPLAY_PACK_TRIAGE_SCHEMA_VERSION
    provider: str = "futureagi"
    decision: str
    severity: str
    block_rollout: bool
    latest_registry_version: Optional[str] = None
    latest_dataset_id: Optional[str] = None
    baseline_registry_version: Optional[str] = None
    baseline_dataset_id: Optional[str] = None
    best_registry_version: Optional[str] = None
    best_dataset_id: Optional[str] = None
    latest_promotable: Optional[bool] = None
    coverage_delta: Optional[float] = None
    optimizer_score_delta: Optional[float] = None
    best_optimizer_score_gap: Optional[float] = None
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    drift_reasons: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


def load_agent_observability_feedback(
    payload: Any,
    *,
    candidate: Optional[AgentCandidate] = None,
    source: str = "auto",
    framework: str = "auto",
    required_metrics: Optional[Mapping[str, float]] = None,
    required_trace_signals: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentObservabilityWindow:
    """
    Normalize production observability exports into agent optimization feedback.

    Accepted payloads include generic run/feedback exports, OpenAI Agents trace
    processor exports, OpenTelemetry/TraceAI OTLP JSON, LiveKit session reports,
    plain JSON/JSONL strings, file paths, or lists of those records.
    """

    loaded = _load_payload(payload)
    raw_records = _observation_records(loaded)
    records: list[AgentObservabilityRecord] = []
    thresholds = {
        str(key): float(value)
        for key, value in dict(required_metrics or {}).items()
    }
    required_signals = [_normalize_signal(item) for item in required_trace_signals or []]
    required_signals = [item for item in required_signals if item]
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, Mapping):
            raw_record = {"value": raw_record}
        records.append(
            _normalize_observability_record(
                dict(raw_record),
                index=index,
                candidate=candidate,
                source=source,
                framework=framework,
                required_metrics=thresholds,
                required_trace_signals=required_signals,
            )
        )

    return AgentObservabilityWindow(
        source=_resolve_window_source(records, fallback=source),
        framework=_resolve_window_framework(records, fallback=framework),
        candidate=candidate,
        records=records,
        required_metrics=thresholds,
        required_trace_signals=required_signals,
        metadata=dict(metadata or {}),
    )


def load_agent_report_replay_cases(
    payload: Any,
    *,
    candidate: Optional[AgentCandidate] = None,
    source: str = "futureagi",
    framework: str = "agent_report",
    required_metrics: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentObservabilityWindow:
    """
    Normalize replay cases with attached agent-report evaluations.

    This bridges deterministic replay packs, including domain-package registry
    mutation cases, into the same observability window consumed by regression
    dataset export and feedback optimizers. No hosted service is called here.
    """

    loaded = _load_payload(payload)
    raw_cases = _observation_records(loaded)
    thresholds = {str(key): float(value) for key, value in dict(required_metrics or {}).items()}
    records: list[AgentObservabilityRecord] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        case = _ensure_mapping(raw_case)
        expected = _ensure_mapping(case.get("expected") or case.get("expected_response"))
        case_thresholds = _float_mapping(expected.get("required_metrics"))
        for name, threshold in case_thresholds.items():
            thresholds.setdefault(name, threshold)
        raw_evidence = _agent_report_case_raw_evidence(case)
        evaluation = (
            raw_evidence.get("agent_report_evaluation")
            or raw_evidence.get("evaluation")
            or case.get("agent_report_evaluation")
        )
        metrics = _agent_report_evaluation_metrics(evaluation)
        if not metrics:
            metrics = _float_mapping(raw_evidence.get("metrics") or case.get("metrics"))
        active_thresholds = dict(thresholds or case_thresholds)
        failures = _agent_report_evaluation_failures(evaluation)
        failures.extend(_string_list(raw_evidence.get("failures") or case.get("failures")))
        failures.extend(_metric_threshold_failures(metrics, active_thresholds))
        failures = list(dict.fromkeys(failures))
        evaluation_payload = _ensure_mapping(evaluation)
        score = _coerce_score(evaluation_payload.get("score"))
        if score is None:
            score = min(metrics.values()) if metrics else (0.0 if failures else 1.0)
        passed_value = evaluation_payload.get("passed")
        if isinstance(passed_value, bool):
            passed = passed_value and not _metric_threshold_failures(metrics, active_thresholds)
        else:
            passed = not failures
        case_id = str(case.get("id") or case.get("case_id") or f"agent_report_case_{index}")
        raw_payload = {
            "case": copy.deepcopy(case),
            **copy.deepcopy(raw_evidence),
        }
        if evaluation and "agent_report_evaluation" not in raw_payload:
            raw_payload["agent_report_evaluation"] = copy.deepcopy(evaluation)
        records.append(
            AgentObservabilityRecord(
                index=index,
                source=_normalize_source(source),
                framework=_normalize_source(framework),
                run_id=case_id,
                candidate_id=candidate.id if candidate is not None else None,
                score=float(score),
                passed=bool(passed),
                failures=failures,
                metrics=metrics,
                raw=raw_payload,
                metadata={
                    "source_kind": _normalize_source(source),
                    "framework": _normalize_source(framework),
                    "case_id": case_id,
                    "case_metadata": copy.deepcopy(_ensure_mapping(case.get("metadata"))),
                },
            )
        )
    return AgentObservabilityWindow(
        source=_normalize_source(source),
        framework=_normalize_source(framework),
        candidate=candidate,
        records=records,
        required_metrics=thresholds,
        metadata={
            "kind": "agent_report_replay_cases",
            "case_count": len(records),
            **dict(metadata or {}),
        },
    )


def publish_futureagi_regression_dataset(
    dataset: AgentRegressionDataset,
    *,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    description: Optional[str] = None,
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
    dry_run: bool = False,
    client: Any = None,
    timeout: float = 30.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentDatasetSinkResult:
    """
    Publish regression cases to a Future AGI dataset.

    The preferred path uses the Future AGI `fi.datasets.Dataset` SDK. If that
    SDK surface is unavailable, the publisher falls back to the Future AGI HTTP
    dataset API through `ai-evaluation` auth primitives. Tests may inject a
    Future AGI-compatible client with `publish_regression_dataset()`,
    `create_regression_dataset()`, or `create_dataset()`.
    """
    active_name = dataset_name or dataset.name
    columns = _futureagi_dataset_columns()
    rows = dataset.to_futureagi_rows()
    active_endpoint = (
        fi_base_url or os.getenv("FI_BASE_URL") or "https://api.futureagi.com"
    ).rstrip("/")
    result_metadata = {
        "description": description,
        **dict(metadata or {}),
    }
    if dry_run:
        return _dataset_sink_result(
            provider="futureagi",
            dataset_name=active_name,
            dataset_id=dataset_id,
            case_count=len(rows),
            endpoint=active_endpoint,
            dry_run=True,
            status="planned",
            response={"columns": columns, "rows": rows},
            metadata=result_metadata,
        )

    if client is None:
        active_api_key = fi_api_key or os.getenv("FI_API_KEY")
        active_secret_key = fi_secret_key or os.getenv("FI_SECRET_KEY")
        if not active_api_key or not active_secret_key:
            return _dataset_sink_result(
                provider="futureagi",
                dataset_name=active_name,
                dataset_id=dataset_id,
                case_count=len(rows),
                endpoint=active_endpoint,
                status="failed",
                failures=[
                    "Future AGI publishing requires FI_API_KEY and FI_SECRET_KEY "
                    "or an injected Future AGI dataset client."
                ],
                metadata=result_metadata,
            )
        client = _load_futureagi_dataset_client(
            fi_api_key=active_api_key,
            fi_secret_key=active_secret_key,
            fi_base_url=active_endpoint,
            timeout=timeout,
        )
        if client is None:
            return _dataset_sink_result(
                provider="futureagi",
                dataset_name=active_name,
                dataset_id=dataset_id,
                case_count=len(rows),
                endpoint=active_endpoint,
                status="failed",
                failures=[
                    "Future AGI publishing requires the `futureagi` dataset SDK "
                    "or the `ai-evaluation` HTTP auth primitives."
                ],
                metadata=result_metadata,
            )

    try:
        response = _publish_futureagi_regression_dataset_with_client(
            client,
            dataset_name=active_name,
            dataset_id=dataset_id,
            columns=columns,
            rows=rows,
            metadata=result_metadata,
        )
    except Exception as exc:
        return _dataset_sink_result(
            provider="futureagi",
            dataset_name=active_name,
            dataset_id=dataset_id,
            case_count=len(rows),
            endpoint=active_endpoint,
            status="failed",
            failures=[f"Future AGI dataset publish failed: {exc}"],
            metadata=result_metadata,
        )

    response_payload = _safe_response_payload(response)
    return _dataset_sink_result(
        provider="futureagi",
        dataset_name=active_name,
        dataset_id=_response_id(response) or _response_id(response_payload) or dataset_id,
        case_count=len(rows),
        endpoint=active_endpoint,
        status="created" if dataset_id is None else "inserted",
        response=response_payload,
        metadata=result_metadata,
    )


def load_futureagi_regression_dataset(
    *,
    dataset_id: str,
    dataset_name: Optional[str] = None,
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
    client: Any = None,
    page_size: int = 100,
    max_pages: int = 100,
    timeout: float = 30.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegressionDataset:
    """
    Pull a Future AGI regression dataset back into replayable agent-opt cases.

    The loader reads the Future AGI dataset table API, maps column ids back to
    `publish_futureagi_regression_dataset()` column names, parses JSON/array
    cells, and reconstructs an `AgentRegressionDataset` that can be converted
    to an `AgentObservabilityWindow` for metric-based re-optimization.
    """

    if not dataset_id:
        raise ValueError("load_futureagi_regression_dataset requires dataset_id.")
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    active_endpoint = (fi_base_url or os.getenv("FI_BASE_URL") or "https://api.futureagi.com").rstrip("/")
    if client is None:
        active_api_key = fi_api_key or os.getenv("FI_API_KEY")
        active_secret_key = fi_secret_key or os.getenv("FI_SECRET_KEY")
        if not active_api_key or not active_secret_key:
            raise ValueError(
                "Future AGI dataset loading requires FI_API_KEY and FI_SECRET_KEY "
                "or an injected Future AGI dataset client."
            )
        client = _load_futureagi_dataset_reader_client(
            fi_api_key=active_api_key,
            fi_secret_key=active_secret_key,
            fi_base_url=active_endpoint,
            timeout=timeout,
        )
        if client is None:
            raise RuntimeError(
                "Future AGI dataset loading requires the `ai-evaluation` HTTP "
                "auth primitives."
            )

    payloads = _futureagi_dataset_payloads(
        client,
        dataset_id=dataset_id,
        page_size=page_size,
        max_pages=max_pages,
    )
    cases, table_metadata = _futureagi_regression_cases_from_payloads(
        payloads,
        dataset_id=dataset_id,
    )
    resolved_name = (
        dataset_name
        or str(table_metadata.get("dataset_name") or "").strip()
        or f"futureagi-regression-{dataset_id}"
    )
    return AgentRegressionDataset(
        name=resolved_name,
        source=_regression_cases_source(cases),
        framework=_regression_cases_framework(cases),
        cases=cases,
        metadata={
            "kind": "futureagi_regression_dataset",
            "dataset_id": dataset_id,
            "endpoint": active_endpoint,
            "page_count": len(payloads),
            "row_count": len(cases),
            "column_count": int(table_metadata.get("column_count") or 0),
            "futureagi_metadata": {
                key: value
                for key, value in table_metadata.items()
                if key != "column_count"
            },
            **dict(metadata or {}),
        },
    )


def build_futureagi_registry_replay_pack_manifest(
    dataset: AgentRegressionDataset,
    *,
    publish_result: Optional[AgentDatasetSinkResult] = None,
    registry_version: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegistryReplayPackManifest:
    """
    Build a version-pinned manifest for a Future AGI registry replay pack.

    The manifest is a retention record: it pins the registry version, Future AGI
    dataset id/name, selected case ids, coverage signal, and a deterministic
    case signature that a promotion gate can verify after live readback.
    """

    selection_metadata = _ensure_mapping(selection)
    result_metadata = _ensure_mapping(publish_result.metadata if publish_result else {})
    dataset_metadata = _ensure_mapping(dataset.metadata)
    active_registry_version = str(
        registry_version
        or dataset_metadata.get("registry_version")
        or result_metadata.get("registry_version")
        or selection_metadata.get("registry_version")
        or ""
    ).strip()
    if not active_registry_version:
        raise ValueError("registry_version is required for registry replay pack manifests.")

    dataset_id = (
        publish_result.dataset_id
        if publish_result is not None
        else _optional_str(dataset_metadata.get("dataset_id"))
    )
    dataset_name = (
        publish_result.dataset_name
        if publish_result is not None
        else dataset.name
    )
    case_ids = [str(case.id) for case in dataset.cases]
    case_signature = _registry_replay_case_signature(case_ids)
    required_presets, required_families = _registry_replay_requirements(selection_metadata)
    coverage_score = _first_float(
        selection_metadata.get("selected_coverage", {}).get("coverage_score")
        if isinstance(selection_metadata.get("selected_coverage"), Mapping)
        else None,
        selection_metadata.get("coverage_score"),
        dataset_metadata.get("coverage_score"),
        result_metadata.get("coverage_score"),
        0.0,
    )
    selection_complete = bool(
        selection_metadata.get("selection_complete")
        if "selection_complete" in selection_metadata
        else dataset_metadata.get("selection_complete")
        if "selection_complete" in dataset_metadata
        else result_metadata.get("selection_complete", False)
    )
    selected_positive_count = int(
        _first_float(
            selection_metadata.get("selected_positive_count"),
            dataset_metadata.get("selected_positive_count"),
            0,
        )
    )
    selected_negative_count = int(
        _first_float(
            selection_metadata.get("selected_negative_count"),
            dataset_metadata.get("selected_negative_count"),
            0,
        )
    )
    provider = publish_result.provider if publish_result is not None else "futureagi"
    manifest_name = name or f"{active_registry_version}:{dataset_name}"
    return AgentRegistryReplayPackManifest(
        name=manifest_name,
        provider=str(provider or "futureagi"),
        registry_version=active_registry_version,
        dataset_name=str(dataset_name),
        dataset_id=dataset_id,
        case_count=len(dataset.cases),
        case_ids=case_ids,
        case_signature=case_signature,
        retention_key=_registry_replay_retention_key(
            registry_version=active_registry_version,
            dataset_name=str(dataset_name),
            dataset_id=dataset_id,
            case_signature=case_signature,
        ),
        selection_complete=selection_complete,
        coverage_score=coverage_score,
        selected_positive_count=selected_positive_count,
        selected_negative_count=selected_negative_count,
        required_presets=required_presets,
        required_invariant_families=required_families,
        metadata={
            "dataset_source": dataset.source,
            "dataset_framework": dataset.framework,
            "dataset_metadata": copy.deepcopy(dataset.metadata),
            "publish_result_status": publish_result.status if publish_result else None,
            **dict(metadata or {}),
        },
    )


def check_futureagi_registry_replay_pack_promotion(
    manifest: AgentRegistryReplayPackManifest | Mapping[str, Any],
    *,
    registry_version: Optional[str] = None,
    dataset: Optional[AgentRegressionDataset] = None,
    dataset_id: Optional[str] = None,
    candidate: Optional[AgentCandidate] = None,
    optimizer_result: Any = None,
    optimizer_score: Optional[float] = None,
    min_coverage_score: float = 1.0,
    min_optimizer_score: float = 0.99,
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
    client: Any = None,
    page_size: int = 100,
    max_pages: int = 100,
    timeout: float = 30.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegistryReplayPackPromotionCheck:
    """
    Gate a selected registry replay pack before promotion.

    A pack is promotable only when the manifest's registry version matches the
    expected version, the pinned Future AGI dataset reads back with the same
    cases, coverage meets the required threshold, and optimizer replay evidence
    reaches `min_optimizer_score`.
    """

    if min_coverage_score < 0:
        raise ValueError("min_coverage_score must be non-negative.")
    if min_optimizer_score < 0:
        raise ValueError("min_optimizer_score must be non-negative.")

    active_manifest = _coerce_registry_replay_manifest(manifest)
    expected_registry_version = str(registry_version or active_manifest.registry_version)
    active_dataset = dataset
    active_dataset_id = dataset_id or active_manifest.dataset_id
    load_failure: Optional[str] = None
    if active_dataset is None and active_dataset_id:
        try:
            active_dataset = load_futureagi_regression_dataset(
                dataset_id=active_dataset_id,
                fi_api_key=fi_api_key,
                fi_secret_key=fi_secret_key,
                fi_base_url=fi_base_url,
                client=client,
                page_size=page_size,
                max_pages=max_pages,
                timeout=timeout,
                metadata={"promotion_gate": "registry_replay_pack"},
            )
        except Exception as exc:
            load_failure = f"Future AGI readback failed: {exc}"
    elif active_dataset is None:
        load_failure = "Future AGI dataset id is required for registry replay pack promotion."

    loaded_case_ids = [str(case.id) for case in active_dataset.cases] if active_dataset else []
    loaded_case_count = len(loaded_case_ids)
    loaded_case_signature = _registry_replay_case_signature(loaded_case_ids)
    replay_record_count = 0
    if active_dataset is not None:
        replay_window = active_dataset.to_observability_window(
            candidate=candidate,
            source="futureagi",
            framework=active_dataset.framework,
            metadata={"promotion_gate": "registry_replay_pack"},
        )
        replay_record_count = len(replay_window.records)

    active_optimizer_score = (
        float(optimizer_score)
        if optimizer_score is not None
        else _optimizer_result_score(optimizer_result)
    )
    failures: list[str] = []
    if load_failure:
        failures.append(load_failure)
    if active_manifest.provider.lower() != "futureagi":
        failures.append(
            f"registry replay pack provider '{active_manifest.provider}' is not Future AGI"
        )
    if not active_manifest.dataset_id and not dataset_id:
        failures.append("registry replay pack manifest does not pin a Future AGI dataset id")
    if active_manifest.registry_version != expected_registry_version:
        failures.append(
            "registry version mismatch: "
            f"manifest {active_manifest.registry_version} != expected {expected_registry_version}"
        )
    if not active_manifest.selection_complete:
        failures.append("registry replay pack selection is incomplete")
    if active_manifest.coverage_score < min_coverage_score:
        failures.append(
            f"coverage score {active_manifest.coverage_score:.4f} below {min_coverage_score:.4f}"
        )
    if active_dataset is not None:
        if loaded_case_count != active_manifest.case_count:
            failures.append(
                f"readback case count {loaded_case_count} != manifest {active_manifest.case_count}"
            )
        if loaded_case_signature != active_manifest.case_signature:
            failures.append("readback case signature does not match manifest")
        if replay_record_count != loaded_case_count:
            failures.append(
                f"replay record count {replay_record_count} != readback cases {loaded_case_count}"
            )
    if active_optimizer_score is None:
        failures.append("optimizer replay score is required for registry replay pack promotion")
    elif active_optimizer_score < min_optimizer_score:
        failures.append(
            f"optimizer replay score {active_optimizer_score:.4f} below {min_optimizer_score:.4f}"
        )

    check_metadata = {
        "loaded_dataset_metadata": copy.deepcopy(active_dataset.metadata) if active_dataset else {},
        **dict(metadata or {}),
    }
    return AgentRegistryReplayPackPromotionCheck(
        promotable=not failures,
        dataset_name=active_manifest.dataset_name,
        dataset_id=active_dataset_id,
        registry_version=active_manifest.registry_version,
        expected_registry_version=expected_registry_version,
        expected_case_count=active_manifest.case_count,
        loaded_case_count=loaded_case_count,
        expected_case_signature=active_manifest.case_signature,
        loaded_case_signature=loaded_case_signature,
        coverage_score=active_manifest.coverage_score,
        min_coverage_score=min_coverage_score,
        selection_complete=active_manifest.selection_complete,
        replay_record_count=replay_record_count,
        optimizer_score=active_optimizer_score,
        min_optimizer_score=min_optimizer_score,
        failures=failures,
        manifest=active_manifest,
        metadata=check_metadata,
    )


def compare_futureagi_registry_replay_pack_lineage(
    manifests: Sequence[AgentRegistryReplayPackManifest | Mapping[str, Any]],
    *,
    promotion_checks: Optional[
        Sequence[AgentRegistryReplayPackPromotionCheck | Mapping[str, Any]]
        | Mapping[str, AgentRegistryReplayPackPromotionCheck | Mapping[str, Any]]
    ] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegistryReplayPackLineageReport:
    """
    Compare version-pinned Future AGI registry replay packs.

    This is a local manifest comparison; it does not call Future AGI. Provide
    promotion checks from `check_futureagi_registry_replay_pack_promotion()` when
    readback and optimizer replay outcomes should be included in the lineage.
    """

    active_manifests = [_coerce_registry_replay_manifest(item) for item in manifests]
    if not active_manifests:
        raise ValueError("at least one registry replay-pack manifest is required.")
    checks_by_key = _registry_replay_promotion_checks_by_key(promotion_checks)
    entries = [
        _registry_replay_lineage_entry(
            manifest,
            checks_by_key=checks_by_key,
        )
        for manifest in active_manifests
    ]
    transitions = [
        _registry_replay_lineage_transition(previous, current)
        for previous, current in zip(entries, entries[1:])
    ]
    latest = entries[-1]
    best = max(
        entries,
        key=lambda entry: (
            1 if entry.promotion_promotable else 0,
            entry.optimizer_score if entry.optimizer_score is not None else float("-inf"),
            entry.coverage_score,
            entry.case_count,
        ),
    )
    drift_reasons = list(
        dict.fromkeys(
            reason
            for transition in transitions
            for reason in transition.drift_reasons
        )
    )
    return AgentRegistryReplayPackLineageReport(
        entry_count=len(entries),
        entries=entries,
        transitions=transitions,
        latest_registry_version=latest.registry_version,
        latest_dataset_id=latest.dataset_id,
        latest_promotable=latest.promotion_promotable,
        best_registry_version=best.registry_version,
        best_dataset_id=best.dataset_id,
        best_optimizer_score=best.optimizer_score,
        drift_detected=bool(drift_reasons),
        drift_reasons=drift_reasons,
        metadata={
            "kind": "futureagi_registry_replay_pack_lineage",
            **dict(metadata or {}),
        },
    )


def triage_futureagi_registry_replay_pack_regression(
    lineage: AgentRegistryReplayPackLineageReport | Mapping[str, Any],
    *,
    max_coverage_drop: float = 0.0,
    max_optimizer_score_drop: float = 0.02,
    require_latest_promotable: bool = True,
    require_readback_match: bool = True,
    block_on_selected_patch_change: bool = False,
    block_on_optimizer_backend_change: bool = False,
    block_on_case_signature_change: bool = False,
    block_on_retention_key_change: bool = False,
    block_on_required_contract_removal: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegistryReplayPackTriageReport:
    """
    Recommend whether replay-pack drift should block rollout.

    The triage is local and deterministic. It consumes a lineage report produced
    by `compare_futureagi_registry_replay_pack_lineage()` and turns coverage,
    optimizer-score, selected-patch, readback, and promotion drift into an
    auditable rollout decision.
    """

    if max_coverage_drop < 0:
        raise ValueError("max_coverage_drop must be non-negative.")
    if max_optimizer_score_drop < 0:
        raise ValueError("max_optimizer_score_drop must be non-negative.")

    active_lineage = _coerce_registry_replay_lineage_report(lineage)
    if not active_lineage.entries:
        raise ValueError("lineage report must contain at least one entry.")

    latest = active_lineage.entries[-1]
    baseline = active_lineage.entries[-2] if len(active_lineage.entries) > 1 else None
    latest_transition = active_lineage.transitions[-1] if active_lineage.transitions else None
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    if require_latest_promotable:
        if latest.promotion_promotable is False:
            blocking_reasons.append("latest_promotion_failed")
        elif latest.promotion_promotable is None:
            blocking_reasons.append("missing_latest_promotion_check")
    if require_readback_match and latest.readback_signature_matches is False:
        blocking_reasons.append("futureagi_readback_signature_mismatch")
    if (
        latest.loaded_case_count is not None
        and latest.loaded_case_count != latest.case_count
    ):
        blocking_reasons.append("futureagi_readback_case_count_mismatch")
    if (
        latest.replay_record_count is not None
        and latest.loaded_case_count is not None
        and latest.replay_record_count != latest.loaded_case_count
    ):
        blocking_reasons.append("futureagi_replay_record_count_mismatch")
    if require_latest_promotable and latest.optimizer_score is None:
        blocking_reasons.append("missing_latest_optimizer_score")
    if latest.failures and latest.promotion_promotable is not True:
        blocking_reasons.append("latest_promotion_failures")

    coverage_delta: Optional[float] = None
    optimizer_score_delta: Optional[float] = None
    if latest_transition is not None:
        coverage_delta = latest_transition.coverage_delta
        optimizer_score_delta = latest_transition.optimizer_score_delta
        if latest_transition.coverage_delta < -max_coverage_drop:
            blocking_reasons.append("coverage_regression")
        elif latest_transition.coverage_delta != 0:
            warnings.append("coverage_drift")
        if latest_transition.optimizer_score_delta is None:
            if latest.optimizer_score is None or (baseline and baseline.optimizer_score is None):
                warnings.append("missing_optimizer_score_delta")
        elif latest_transition.optimizer_score_delta < -max_optimizer_score_drop:
            blocking_reasons.append("optimizer_score_regression")
        elif latest_transition.optimizer_score_delta < 0:
            warnings.append("optimizer_score_drift")
        if latest_transition.selected_patch_changed:
            if block_on_selected_patch_change:
                blocking_reasons.append("selected_patch_changed")
            else:
                warnings.append("selected_patch_changed")
        if latest_transition.optimizer_backend_changed:
            if block_on_optimizer_backend_change:
                blocking_reasons.append("optimizer_backend_changed")
            else:
                warnings.append("optimizer_backend_changed")
        if latest_transition.case_signature_changed:
            if block_on_case_signature_change:
                blocking_reasons.append("case_signature_changed")
            else:
                warnings.append("case_signature_changed")
        if latest_transition.retention_key_changed:
            if block_on_retention_key_change:
                blocking_reasons.append("retention_key_changed")
            else:
                warnings.append("retention_key_changed")
        if latest_transition.promotion_status_changed:
            warnings.append("promotion_status_changed")
        if latest_transition.removed_required_presets:
            if block_on_required_contract_removal:
                blocking_reasons.append("required_presets_removed")
            else:
                warnings.append("required_presets_removed")
        if latest_transition.removed_invariant_families:
            if block_on_required_contract_removal:
                blocking_reasons.append("required_invariant_families_removed")
            else:
                warnings.append("required_invariant_families_removed")
        if (
            latest_transition.added_required_presets
            or latest_transition.added_invariant_families
        ):
            warnings.append("required_contract_expanded")

    best_optimizer_score_gap: Optional[float] = None
    if (
        latest.optimizer_score is not None
        and active_lineage.best_optimizer_score is not None
    ):
        best_optimizer_score_gap = round(
            latest.optimizer_score - active_lineage.best_optimizer_score,
            8,
        )
        if best_optimizer_score_gap < -max_optimizer_score_drop:
            blocking_reasons.append("latest_below_best_optimizer_score")

    blocking_reasons = _unique_strings(blocking_reasons)
    warnings = [
        warning
        for warning in _unique_strings(warnings)
        if warning not in blocking_reasons
    ]
    decision = "block" if blocking_reasons else ("review" if warnings else "promote")
    severity = _registry_replay_triage_severity(
        blocking_reasons=blocking_reasons,
        warnings=warnings,
    )
    recommendations = _registry_replay_triage_recommendations(
        blocking_reasons=blocking_reasons,
        warnings=warnings,
    )
    return AgentRegistryReplayPackTriageReport(
        decision=decision,
        severity=severity,
        block_rollout=bool(blocking_reasons),
        latest_registry_version=latest.registry_version,
        latest_dataset_id=latest.dataset_id,
        baseline_registry_version=baseline.registry_version if baseline else None,
        baseline_dataset_id=baseline.dataset_id if baseline else None,
        best_registry_version=active_lineage.best_registry_version,
        best_dataset_id=active_lineage.best_dataset_id,
        latest_promotable=latest.promotion_promotable,
        coverage_delta=coverage_delta,
        optimizer_score_delta=optimizer_score_delta,
        best_optimizer_score_gap=best_optimizer_score_gap,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        drift_reasons=list(active_lineage.drift_reasons),
        recommendations=recommendations,
        evidence={
            "thresholds": {
                "max_coverage_drop": max_coverage_drop,
                "max_optimizer_score_drop": max_optimizer_score_drop,
                "require_latest_promotable": require_latest_promotable,
                "require_readback_match": require_readback_match,
                "block_on_selected_patch_change": block_on_selected_patch_change,
                "block_on_optimizer_backend_change": block_on_optimizer_backend_change,
                "block_on_case_signature_change": block_on_case_signature_change,
                "block_on_retention_key_change": block_on_retention_key_change,
                "block_on_required_contract_removal": block_on_required_contract_removal,
            },
            "latest_entry": latest.model_dump(),
            "baseline_entry": baseline.model_dump() if baseline else None,
            "latest_transition": (
                latest_transition.model_dump() if latest_transition else None
            ),
            "latest_failures": list(latest.failures),
        },
        metadata={
            "kind": "futureagi_registry_replay_pack_regression_triage",
            **dict(metadata or {}),
        },
    )


def load_futureagi_experiment_history(
    *,
    experiment_id: str,
    candidate: Optional[AgentCandidate] = None,
    required_metrics: Optional[Mapping[str, float]] = None,
    required_trace_signals: Optional[Sequence[str]] = None,
    fi_api_key: Optional[str] = None,
    fi_secret_key: Optional[str] = None,
    fi_base_url: Optional[str] = None,
    client: Any = None,
    page_size: int = 100,
    max_pages: int = 20,
    timeout: float = 30.0,
    include_rows: bool = True,
    include_stats: bool = True,
    prefer_v2: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentObservabilityWindow:
    """
    Pull Future AGI experiment history into optimizer feedback.

    Unlike `load_futureagi_regression_dataset()`, this reads native Future AGI
    experiment detail/stats/row payloads instead of regression-pack rows. The
    resulting observability window can drive `AgentFeedbackOptimizer` through
    any metric-bound backend: deterministic, council, society, social-memory,
    curriculum, evolutionary, TPE, Pareto, or bandit.
    """

    if not experiment_id:
        raise ValueError("load_futureagi_experiment_history requires experiment_id.")
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    active_endpoint = (fi_base_url or os.getenv("FI_BASE_URL") or "https://api.futureagi.com").rstrip("/")
    if client is None:
        active_api_key = fi_api_key or os.getenv("FI_API_KEY")
        active_secret_key = fi_secret_key or os.getenv("FI_SECRET_KEY")
        if not active_api_key or not active_secret_key:
            raise ValueError(
                "Future AGI experiment-history loading requires FI_API_KEY and "
                "FI_SECRET_KEY or an injected Future AGI experiment client."
            )
        client = _load_futureagi_experiment_reader_client(
            fi_api_key=active_api_key,
            fi_secret_key=active_secret_key,
            fi_base_url=active_endpoint,
            timeout=timeout,
        )
        if client is None:
            raise RuntimeError(
                "Future AGI experiment-history loading requires the "
                "`ai-evaluation` HTTP auth primitives."
            )

    payload = _futureagi_experiment_payload(
        client,
        experiment_id=experiment_id,
        page_size=page_size,
        max_pages=max_pages,
        include_rows=include_rows,
        include_stats=include_stats,
        prefer_v2=prefer_v2,
    )
    thresholds = {
        str(key): float(value)
        for key, value in dict(required_metrics or {}).items()
    }
    required_signals = [_normalize_signal(item) for item in required_trace_signals or []]
    required_signals = [item for item in required_signals if item]
    experiment_metadata = _futureagi_experiment_metadata(payload, experiment_id=experiment_id)
    raw_records = _futureagi_experiment_observation_records(
        payload,
        experiment_id=experiment_id,
        experiment_metadata=experiment_metadata,
    )
    records = [
        _normalize_observability_record(
            raw_record,
            index=index,
            candidate=candidate,
            source="futureagi",
            framework=str(
                experiment_metadata.get("framework")
                or experiment_metadata.get("runtime")
                or "generic"
            ),
            required_metrics=thresholds,
            required_trace_signals=required_signals,
        )
        for index, raw_record in enumerate(raw_records, start=1)
    ]
    _penalize_missing_futureagi_experiment_metrics(records, thresholds)

    window_metadata = {
        "kind": "futureagi_experiment_history",
        "schema_version": FUTUREAGI_EXPERIMENT_HISTORY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_name": experiment_metadata.get("name"),
        "experiment_status": experiment_metadata.get("status"),
        "endpoint": active_endpoint,
        "record_count": len(records),
        "payload_sections": sorted(str(key) for key in payload.keys()),
        **dict(metadata or {}),
    }
    return AgentObservabilityWindow(
        source=_resolve_window_source(records, fallback="futureagi"),
        framework=_resolve_window_framework(records, fallback=str(window_metadata.get("framework") or "generic")),
        candidate=candidate,
        records=records,
        required_metrics=thresholds,
        required_trace_signals=required_signals,
        metadata=window_metadata,
    )


def build_agent_regression_dataset(
    windows: AgentObservabilityWindow | Sequence[AgentObservabilityWindow],
    *,
    name: str = "observability-regression",
    failed_only: bool = True,
    include_raw: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegressionDataset:
    """
    Convert normalized observability windows into durable regression cases.

    The resulting cases preserve the production signal that failed, the expected
    metric/trace thresholds a repaired candidate must satisfy, and export
    helpers for local replay plus Future AGI datasets.
    """

    normalized_windows = _regression_windows(windows)
    cases: list[AgentRegressionCase] = []
    for window_index, window in enumerate(normalized_windows, start=1):
        for record in window.records:
            if failed_only and record.passed:
                continue
            cases.append(
                _regression_case_from_observability_record(
                    record,
                    window=window,
                    window_index=window_index,
                    include_raw=include_raw,
                )
            )

    return AgentRegressionDataset(
        name=name,
        source=_regression_source(normalized_windows),
        framework=_regression_framework(normalized_windows),
        cases=cases,
        metadata={
            "kind": "observability_regression_dataset",
            "failed_only": failed_only,
            "include_raw": include_raw,
            "window_count": len(normalized_windows),
            "record_count": sum(len(window.records) for window in normalized_windows),
            "case_count": len(cases),
            **dict(metadata or {}),
        },
    )


def build_agent_regression_dataset_coverage_report(
    dataset: AgentRegressionDataset,
    *,
    target: Any = None,
    metric_path_hints: Optional[Mapping[str, Sequence[str]]] = None,
    tag_path_hints: Optional[Mapping[str, Sequence[str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRegressionDatasetCoverageReport:
    """Summarize replay-pack coverage before Future AGI publish or optimizer replay."""

    target_paths = _coverage_target_paths(target)
    target_path_set = set(target_paths)
    metric_hints = _coverage_path_hints(metric_path_hints)
    tag_hints = _coverage_path_hints(tag_path_hints)
    source_counts: dict[str, int] = {}
    framework_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    observed_metric_case_counts: dict[str, int] = {}
    required_metric_case_counts: dict[str, int] = {}
    failed_metric_case_counts: dict[str, int] = {}
    required_metrics: dict[str, float] = {}
    required_trace_signals: list[str] = []
    seen_required_signals: set[str] = set()
    trace_signal_case_counts: dict[str, int] = {}
    missing_trace_signal_case_counts: dict[str, int] = {}
    search_path_case_counts: dict[str, int] = {path: 0 for path in target_paths}
    failure_examples: dict[str, list[str]] = {}
    failed_case_count = 0
    passed_case_count = 0

    for case in dataset.cases:
        observability = _ensure_mapping(case.input.get("observability"))
        expected = _ensure_mapping(case.expected)
        source = str(observability.get("source") or case.metadata.get("source") or dataset.source)
        framework = str(
            observability.get("framework")
            or case.metadata.get("framework")
            or dataset.framework
        )
        _count(source_counts, source)
        _count(framework_counts, framework)

        passed_value = observability.get("passed")
        passed = bool(passed_value) if isinstance(passed_value, bool) else bool(case.metadata.get("passed"))
        if passed:
            passed_case_count += 1
        else:
            failed_case_count += 1

        tags = [str(tag) for tag in case.tags]
        for tag in tags:
            _count(tag_counts, tag)

        metrics = _float_mapping(observability.get("metrics"))
        for metric in sorted(metrics):
            _count(observed_metric_case_counts, metric)

        case_required_metrics = _float_mapping(expected.get("required_metrics"))
        for metric, threshold in sorted(case_required_metrics.items()):
            _count(required_metric_case_counts, metric)
            required_metrics[metric] = max(threshold, required_metrics.get(metric, threshold))
            observed = metrics.get(metric)
            if observed is None or observed < threshold:
                _count(failed_metric_case_counts, metric)

        trace_signals = {
            _normalize_signal(signal)
            for signal in _string_list(observability.get("trace_signals"))
            if _normalize_signal(signal)
        }
        for signal in sorted(trace_signals):
            _count(trace_signal_case_counts, signal)

        for signal in _string_list(expected.get("required_trace_signals")):
            normalized = _normalize_signal(signal)
            if not normalized:
                continue
            if normalized not in seen_required_signals:
                required_trace_signals.append(normalized)
                seen_required_signals.add(normalized)
            if normalized not in trace_signals:
                _count(missing_trace_signal_case_counts, normalized)

        failures = _string_list(observability.get("failures") or expected.get("previous_failures"))
        for failure in failures[:8]:
            family = _coverage_failure_family(failure)
            examples = failure_examples.setdefault(family, [])
            if case.id not in examples and len(examples) < 5:
                examples.append(case.id)

        for path in _coverage_search_path_hits(
            tags=tags,
            metrics=[*metrics.keys(), *case_required_metrics.keys()],
            failures=failures,
            target_paths=target_paths,
            target_path_set=target_path_set,
            metric_path_hints=metric_hints,
            tag_path_hints=tag_hints,
        ):
            search_path_case_counts[path] = search_path_case_counts.get(path, 0) + 1

    all_required_metrics = sorted(required_metrics)
    uncovered_required_metrics = [
        metric
        for metric in all_required_metrics
        if required_metric_case_counts.get(metric, 0) == 0
    ]
    uncovered_search_paths = [
        path for path in target_paths if search_path_case_counts.get(path, 0) == 0
    ]

    return AgentRegressionDatasetCoverageReport(
        dataset_name=dataset.name,
        source=dataset.source,
        framework=dataset.framework,
        case_count=len(dataset.cases),
        failed_case_count=failed_case_count,
        passed_case_count=passed_case_count,
        source_counts=dict(sorted(source_counts.items())),
        framework_counts=dict(sorted(framework_counts.items())),
        tag_counts=dict(sorted(tag_counts.items())),
        observed_metric_case_counts=dict(sorted(observed_metric_case_counts.items())),
        required_metric_case_counts=dict(sorted(required_metric_case_counts.items())),
        failed_metric_case_counts=dict(sorted(failed_metric_case_counts.items())),
        required_metrics={key: required_metrics[key] for key in all_required_metrics},
        required_trace_signals=required_trace_signals,
        trace_signal_case_counts=dict(sorted(trace_signal_case_counts.items())),
        missing_trace_signal_case_counts=dict(
            sorted(missing_trace_signal_case_counts.items())
        ),
        search_path_case_counts={
            key: search_path_case_counts[key]
            for key in [*target_paths, *sorted(path for path in search_path_case_counts if path not in target_path_set)]
        },
        uncovered_required_metrics=uncovered_required_metrics,
        uncovered_search_paths=uncovered_search_paths,
        failure_examples=dict(sorted(failure_examples.items())),
        metadata={
            "kind": "regression_dataset_coverage_report",
            "target_name": getattr(target, "name", None),
            "target_layers": list(getattr(target, "layers", []) or []),
            "metric_path_hints": {
                key: list(value) for key, value in metric_hints.items()
            },
            "tag_path_hints": {key: list(value) for key, value in tag_hints.items()},
            **dict(metadata or {}),
        },
    )


def _count(counts: dict[str, int], key: str) -> None:
    counts[str(key)] = counts.get(str(key), 0) + 1


def _coverage_target_paths(target: Any) -> list[str]:
    search_space = getattr(target, "search_space", None)
    if isinstance(search_space, Mapping):
        return [str(path) for path in search_space]
    return []


def _coverage_path_hints(
    value: Optional[Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for key, paths in dict(value or {}).items():
        hints[str(key)] = [str(path) for path in paths]
    return hints


def _coverage_search_path_hits(
    *,
    tags: Sequence[str],
    metrics: Sequence[str],
    failures: Sequence[str],
    target_paths: Sequence[str],
    target_path_set: set[str],
    metric_path_hints: Mapping[str, Sequence[str]],
    tag_path_hints: Mapping[str, Sequence[str]],
) -> list[str]:
    hits: set[str] = set()
    for metric in metrics:
        for path in metric_path_hints.get(str(metric), ()):
            if not target_path_set or path in target_path_set:
                hits.add(path)
    for tag in tags:
        for path in tag_path_hints.get(str(tag), ()):
            if not target_path_set or path in target_path_set:
                hits.add(path)

    if target_paths:
        text = " ".join([*tags, *metrics, *failures]).lower()
        text_tokens = set(_case_slug(text).split("-"))
        for path in target_paths:
            path_parts_list = [part.lower() for part in path.split(".") if part]
            matching_parts = (
                path_parts_list[1:] if len(path_parts_list) > 1 else path_parts_list
            )
            path_tokens = set(_case_slug(".".join(matching_parts)).split("-"))
            path_parts = set(matching_parts)
            if text_tokens.intersection(path_tokens) or text_tokens.intersection(path_parts):
                hits.add(path)

    return [path for path in target_paths if path in hits] + sorted(
        path for path in hits if path not in target_path_set
    )


def _coverage_failure_family(failure: Any) -> str:
    text = str(failure or "unknown")
    metric_match = _METRIC_FAILURE_RE.search(text)
    if metric_match:
        return f"metric:{metric_match.group(1)}"
    slug = _case_slug(text)
    return slug or "unknown"


_METRIC_FAILURE_RE = re.compile(r"metric '([^']+)'")


def _futureagi_dataset_columns() -> list[dict[str, str]]:
    return [dict(column) for column in FUTUREAGI_REGRESSION_DATASET_COLUMNS]


def _load_futureagi_dataset_client(
    *,
    fi_api_key: str,
    fi_secret_key: str,
    fi_base_url: str,
    timeout: float,
) -> Any:
    try:
        from fi.datasets import Dataset  # type: ignore
        from fi.datasets.types import (  # type: ignore
            Cell,
            Column,
            DatasetConfig,
            DataTypeChoices,
            ModelTypes,
            Row,
            SourceChoices,
        )

        return _FutureAGISDKDatasetPublisher(
            Dataset=Dataset,
            DatasetConfig=DatasetConfig,
            Column=Column,
            Row=Row,
            Cell=Cell,
            DataTypeChoices=DataTypeChoices,
            ModelTypes=ModelTypes,
            SourceChoices=SourceChoices,
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url,
            timeout=timeout,
        )
    except Exception:
        pass

    try:
        from fi.api.auth import APIKeyAuth  # type: ignore
        from fi.api.types import HttpMethod, RequestConfig  # type: ignore
        from fi.utils.routes import Routes  # type: ignore
    except Exception:
        return None

    return _FutureAGIHttpDatasetPublisher(
        APIKeyAuth=APIKeyAuth,
        HttpMethod=HttpMethod,
        RequestConfig=RequestConfig,
        Routes=Routes,
        fi_api_key=fi_api_key,
        fi_secret_key=fi_secret_key,
        fi_base_url=fi_base_url,
        timeout=timeout,
    )


def _load_futureagi_dataset_reader_client(
    *,
    fi_api_key: str,
    fi_secret_key: str,
    fi_base_url: str,
    timeout: float,
) -> Any:
    try:
        from fi.api.auth import APIKeyAuth  # type: ignore
        from fi.api.types import HttpMethod, RequestConfig  # type: ignore
        from fi.utils.routes import Routes  # type: ignore
    except Exception:
        return None

    return _FutureAGIHttpDatasetReader(
        APIKeyAuth=APIKeyAuth,
        HttpMethod=HttpMethod,
        RequestConfig=RequestConfig,
        Routes=Routes,
        fi_api_key=fi_api_key,
        fi_secret_key=fi_secret_key,
        fi_base_url=fi_base_url,
        timeout=timeout,
    )


def _load_futureagi_experiment_reader_client(
    *,
    fi_api_key: str,
    fi_secret_key: str,
    fi_base_url: str,
    timeout: float,
) -> Any:
    try:
        from fi.api.auth import APIKeyAuth  # type: ignore
        from fi.api.types import HttpMethod, RequestConfig  # type: ignore
    except Exception:
        return None

    return _FutureAGIHttpExperimentReader(
        APIKeyAuth=APIKeyAuth,
        HttpMethod=HttpMethod,
        RequestConfig=RequestConfig,
        fi_api_key=fi_api_key,
        fi_secret_key=fi_secret_key,
        fi_base_url=fi_base_url,
        timeout=timeout,
    )


def _publish_futureagi_regression_dataset_with_client(
    client: Any,
    *,
    dataset_name: str,
    dataset_id: Optional[str],
    columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Any:
    for method_name in (
        "publish_regression_dataset",
        "create_regression_dataset",
        "create_dataset",
    ):
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        try:
            return method(
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                columns=list(columns),
                rows=list(rows),
                metadata=dict(metadata),
            )
        except TypeError:
            try:
                return method(
                    name=dataset_name,
                    dataset_id=dataset_id,
                    columns=list(columns),
                    rows=list(rows),
                    metadata=dict(metadata),
                )
            except TypeError:
                return method(
                    dataset_name,
                    list(columns),
                    list(rows),
                )

    if all(callable(getattr(client, name, None)) for name in ("add_columns", "add_rows")):
        if dataset_id is None and callable(getattr(client, "create", None)):
            client.create()
            client.add_columns(_futureagi_http_columns(columns))
        client.add_rows(_futureagi_http_rows(rows, columns=columns))
        return client

    raise TypeError(
        "client must expose publish_regression_dataset(), "
        "create_regression_dataset(), create_dataset(), or Future AGI Dataset "
        "add_columns()/add_rows() methods"
    )


class _FutureAGISDKDatasetPublisher:
    def __init__(
        self,
        *,
        Dataset: Any,
        DatasetConfig: Any,
        Column: Any,
        Row: Any,
        Cell: Any,
        DataTypeChoices: Any,
        ModelTypes: Any,
        SourceChoices: Any,
        fi_api_key: str,
        fi_secret_key: str,
        fi_base_url: str,
        timeout: float,
    ) -> None:
        self.Dataset = Dataset
        self.DatasetConfig = DatasetConfig
        self.Column = Column
        self.Row = Row
        self.Cell = Cell
        self.DataTypeChoices = DataTypeChoices
        self.ModelTypes = ModelTypes
        self.SourceChoices = SourceChoices
        self.fi_api_key = fi_api_key
        self.fi_secret_key = fi_secret_key
        self.fi_base_url = fi_base_url
        self.timeout = timeout

    def publish_regression_dataset(
        self,
        *,
        dataset_name: str,
        dataset_id: Optional[str],
        columns: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        **_: Any,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "name": dataset_name,
            "model_type": self.ModelTypes.GENERATIVE_LLM,
        }
        if dataset_id:
            config_kwargs["id"] = dataset_id
        dataset_client = self.Dataset(
            dataset_config=self.DatasetConfig(**config_kwargs),
            fi_api_key=self.fi_api_key,
            fi_secret_key=self.fi_secret_key,
            fi_base_url=self.fi_base_url,
            timeout=self.timeout,
        )
        if dataset_id is None and _response_id(dataset_client.get_config()) is None:
            dataset_client = dataset_client.create()
        if dataset_id is None:
            dataset_client = dataset_client.add_columns(
                self._columns(columns)
            )
        if rows:
            dataset_client = dataset_client.add_rows(
                self._rows(rows, columns=columns)
            )
        config = dataset_client.get_config()
        return {
            "id": _response_id(config),
            "dataset_id": _response_id(config),
            "dataset_name": getattr(config, "name", dataset_name),
            "rows_added": len(rows),
            "columns": list(columns),
            "metadata": dict(metadata),
        }

    def _columns(self, columns: Sequence[Mapping[str, Any]]) -> list[Any]:
        sdk_columns = []
        for column in columns:
            data_type = self.DataTypeChoices(str(column["data_type"]))
            sdk_columns.append(
                self.Column(
                    name=str(column["name"]),
                    data_type=data_type,
                    source=self.SourceChoices.OTHERS,
                )
            )
        return sdk_columns

    def _rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        sdk_rows = []
        for index, row in enumerate(rows, start=1):
            cells = [
                self.Cell(
                    column_name=str(column["name"]),
                    value=_futureagi_cell_value(row.get(str(column["name"])), column),
                )
                for column in columns
            ]
            sdk_rows.append(self.Row(order=index, cells=cells))
        return sdk_rows


class _FutureAGIHttpDatasetPublisher:
    def __init__(
        self,
        *,
        APIKeyAuth: Any,
        HttpMethod: Any,
        RequestConfig: Any,
        Routes: Any,
        fi_api_key: str,
        fi_secret_key: str,
        fi_base_url: str,
        timeout: float,
    ) -> None:
        self.client = APIKeyAuth(
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url,
            timeout=timeout,
        )
        self.HttpMethod = HttpMethod
        self.RequestConfig = RequestConfig
        self.Routes = Routes
        self.base_url = fi_base_url.rstrip("/")
        self.timeout = int(timeout)

    def publish_regression_dataset(
        self,
        *,
        dataset_name: str,
        dataset_id: Optional[str],
        columns: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        response_payload: dict[str, Any] = {}
        active_dataset_id = dataset_id
        if active_dataset_id is None:
            response_payload["dataset"] = self._post(
                str(self.Routes.dataset_empty.value),
                {
                    "new_dataset_name": dataset_name,
                    "model_type": "GenerativeLLM",
                    "is_sdk": True,
                },
            )
            active_dataset_id = _response_id(response_payload["dataset"])
        if active_dataset_id is None:
            raise RuntimeError("Future AGI dataset creation did not return a dataset id")

        if dataset_id is None:
            response_payload["columns"] = self._post(
                str(self.Routes.dataset_add_columns.value).format(
                    dataset_id=active_dataset_id
                ),
                {"new_columns_data": _futureagi_http_columns(columns)},
            )
        response_payload["rows"] = self._post(
            str(self.Routes.dataset_add_rows.value).format(dataset_id=active_dataset_id),
            {"rows": _futureagi_http_rows(rows, columns=columns)},
        )
        response_payload.update(
            {
                "id": active_dataset_id,
                "dataset_id": active_dataset_id,
                "dataset_name": dataset_name,
                "rows_added": len(rows),
                "metadata": dict(metadata),
            }
        )
        return response_payload

    def _post(self, route: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.request(
            config=self.RequestConfig(
                method=self.HttpMethod.POST,
                url=f"{self.base_url}/{route}",
                json=dict(payload),
                timeout=self.timeout,
            )
        )
        return _http_response_payload(response)


class _FutureAGIHttpDatasetReader:
    def __init__(
        self,
        *,
        APIKeyAuth: Any,
        HttpMethod: Any,
        RequestConfig: Any,
        Routes: Any,
        fi_api_key: str,
        fi_secret_key: str,
        fi_base_url: str,
        timeout: float,
    ) -> None:
        self.client = APIKeyAuth(
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url,
            timeout=timeout,
        )
        self.HttpMethod = HttpMethod
        self.RequestConfig = RequestConfig
        self.Routes = Routes
        self.base_url = fi_base_url.rstrip("/")
        self.timeout = int(timeout)

    def fetch_regression_dataset(
        self,
        *,
        dataset_id: str,
        page_size: int,
        current_page_index: int,
    ) -> dict[str, Any]:
        route = str(self.Routes.dataset_table.value).format(dataset_id=dataset_id)
        response = self.client.request(
            config=self.RequestConfig(
                method=self.HttpMethod.GET,
                url=f"{self.base_url}/{route}",
                params={
                    "page_size": page_size,
                    "current_page_index": current_page_index,
                },
                timeout=self.timeout,
            )
        )
        return _http_response_payload(response)


class _FutureAGIHttpExperimentReader:
    def __init__(
        self,
        *,
        APIKeyAuth: Any,
        HttpMethod: Any,
        RequestConfig: Any,
        fi_api_key: str,
        fi_secret_key: str,
        fi_base_url: str,
        timeout: float,
    ) -> None:
        self.client = APIKeyAuth(
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url,
            timeout=timeout,
        )
        self.HttpMethod = HttpMethod
        self.RequestConfig = RequestConfig
        self.base_url = fi_base_url.rstrip("/")
        self.timeout = int(timeout)

    def fetch_experiment_history(
        self,
        *,
        experiment_id: str,
        page_size: int,
        max_pages: int,
        include_rows: bool,
        include_stats: bool,
        prefer_v2: bool,
    ) -> dict[str, Any]:
        history: dict[str, Any] = {"experiment_id": experiment_id}
        try:
            history["detail"] = self._get_first(
                [
                    f"model-hub/experiments/v2/{experiment_id}/",
                    "model-hub/experiments/",
                ]
                if prefer_v2
                else [
                    "model-hub/experiments/",
                    f"model-hub/experiments/v2/{experiment_id}/",
                ],
                params={"experiment_id": experiment_id},
            )
        except Exception as exc:
            history["detail_error"] = str(exc)
        if include_stats:
            try:
                history["stats"] = self._get_first(
                    [
                        f"model-hub/experiments/v2/{experiment_id}/stats/",
                        f"model-hub/experiments/{experiment_id}/stats/",
                    ]
                    if prefer_v2
                    else [
                        f"model-hub/experiments/{experiment_id}/stats/",
                        f"model-hub/experiments/v2/{experiment_id}/stats/",
                    ]
                )
            except Exception as exc:
                history["stats_error"] = str(exc)
        if include_rows:
            row_routes = (
                [
                    f"model-hub/experiments/v2/{experiment_id}/rows/",
                    f"model-hub/experiments/{experiment_id}/",
                ]
                if prefer_v2
                else [
                    f"model-hub/experiments/{experiment_id}/",
                    f"model-hub/experiments/v2/{experiment_id}/rows/",
                ]
            )
            pages = []
            try:
                for page_index in range(max_pages):
                    payload = self._get_first(
                        row_routes,
                        params={
                            "page_size": page_size,
                            "current_page_index": page_index,
                        },
                    )
                    pages.append(payload)
                    result = _futureagi_payload_result(payload)
                    total_pages = _futureagi_total_pages(result)
                    row_count = len(_futureagi_table_rows(result))
                    if total_pages is not None:
                        if page_index + 1 >= total_pages:
                            break
                    elif row_count < page_size:
                        break
                    else:
                        break
                history["rows"] = pages
            except Exception as exc:
                history["rows_error"] = str(exc)
        if "stats" not in history and "rows" not in history:
            try:
                history["list"] = self._get(
                    "model-hub/experiments/data/",
                    params={"page_size": page_size, "current_page_index": 0},
                )
            except Exception as exc:
                history["list_error"] = str(exc)
        return history

    def _get_first(
        self,
        routes: Sequence[str],
        params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for route in routes:
            try:
                return self._get(route, params=params)
            except Exception as exc:
                errors.append(f"{route}: {exc}")
        raise RuntimeError("; ".join(errors))

    def _get(
        self,
        route: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        response = self.client.request(
            config=self.RequestConfig(
                method=self.HttpMethod.GET,
                url=f"{self.base_url}/{route}",
                params=dict(params or {}),
                timeout=self.timeout,
            )
        )
        return _http_response_payload(response)


def _futureagi_http_columns(
    columns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(column["name"]),
            "data_type": str(column["data_type"]),
            "source": "OTHERS",
        }
        for column in columns
    ]


def _futureagi_http_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "order": index,
            "cells": [
                {
                    "column_name": str(column["name"]),
                    "value": _futureagi_cell_value(
                        row.get(str(column["name"])),
                        column,
                    ),
                }
                for column in columns
            ],
        }
        for index, row in enumerate(rows, start=1)
    ]


def _futureagi_cell_value(value: Any, column: Mapping[str, Any]) -> Any:
    if value is None:
        return ""
    data_type = str(column.get("data_type") or "text")
    if data_type in {"json", "array"}:
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _http_response_payload(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    ok = getattr(response, "ok", True)
    text = getattr(response, "text", "")
    if ok is False:
        raise RuntimeError(f"Future AGI API request failed with status {status_code}: {text[:500]}")
    try:
        payload = response.json()
    except Exception:
        return {"status_code": status_code, "text": text}
    return dict(payload) if isinstance(payload, Mapping) else {"value": payload}


def _dataset_sink_result(
    *,
    provider: str,
    dataset_name: str,
    case_count: int,
    status: str,
    dataset_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    dry_run: bool = False,
    failures: Optional[Sequence[str]] = None,
    response: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentDatasetSinkResult:
    return AgentDatasetSinkResult(
        provider=provider,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        case_count=case_count,
        endpoint=endpoint,
        dry_run=dry_run,
        status=status,
        failures=list(failures or []),
        response=dict(response or {}),
        metadata=dict(metadata or {}),
    )


def _response_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("id", "dataset_id", "datasetId"):
            if value.get(key) is not None:
                return str(value[key])
        for key in ("result", "data", "dataset"):
            nested = value.get(key)
            nested_id = _response_id(nested)
            if nested_id is not None:
                return nested_id
        return None
    for key in ("id", "dataset_id", "datasetId"):
        item = getattr(value, key, None)
        if item is not None:
            return str(item)
    return None


def _safe_response_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"items": list(value)}
    payload: dict[str, Any] = {}
    response_id = _response_id(value)
    if response_id is not None:
        payload["id"] = response_id
    if not payload:
        payload["repr"] = str(value)
    return payload


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_float(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _registry_replay_case_signature(case_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(case_id) for case_id in case_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _registry_replay_retention_key(
    *,
    registry_version: str,
    dataset_name: str,
    dataset_id: Optional[str],
    case_signature: str,
) -> str:
    dataset_key = dataset_id or dataset_name
    return f"{registry_version}:{dataset_key}:{case_signature[:16]}"


def _registry_replay_requirements(
    selection: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    presets: set[str] = set()
    families: set[str] = set()
    for item in _sequence_items(selection.get("required")):
        if not isinstance(item, Mapping):
            continue
        preset = _optional_str(item.get("preset"))
        family = _optional_str(item.get("invariant_family"))
        if preset:
            presets.add(preset)
        if family:
            families.add(family)
    for item in _sequence_items(selection.get("selected")):
        if not isinstance(item, Mapping):
            continue
        preset = _optional_str(item.get("preset"))
        family = _optional_str(item.get("invariant_family"))
        if preset:
            presets.add(preset)
        if family:
            families.add(family)
    return sorted(presets), sorted(families)


def _coerce_registry_replay_manifest(
    value: AgentRegistryReplayPackManifest | Mapping[str, Any],
) -> AgentRegistryReplayPackManifest:
    if isinstance(value, AgentRegistryReplayPackManifest):
        return value
    if isinstance(value, Mapping):
        return AgentRegistryReplayPackManifest(**dict(value))
    raise TypeError("manifest must be AgentRegistryReplayPackManifest or mapping.")


def _coerce_registry_replay_promotion_check(
    value: AgentRegistryReplayPackPromotionCheck | Mapping[str, Any],
) -> AgentRegistryReplayPackPromotionCheck:
    if isinstance(value, AgentRegistryReplayPackPromotionCheck):
        return value
    if isinstance(value, Mapping):
        return AgentRegistryReplayPackPromotionCheck(**dict(value))
    raise TypeError(
        "promotion check must be AgentRegistryReplayPackPromotionCheck or mapping."
    )


def _coerce_registry_replay_lineage_report(
    value: AgentRegistryReplayPackLineageReport | Mapping[str, Any],
) -> AgentRegistryReplayPackLineageReport:
    if isinstance(value, AgentRegistryReplayPackLineageReport):
        return value
    if isinstance(value, Mapping):
        return AgentRegistryReplayPackLineageReport(**dict(value))
    raise TypeError(
        "lineage must be AgentRegistryReplayPackLineageReport or mapping."
    )


def _registry_replay_promotion_checks_by_key(
    promotion_checks: Optional[
        Sequence[AgentRegistryReplayPackPromotionCheck | Mapping[str, Any]]
        | Mapping[str, AgentRegistryReplayPackPromotionCheck | Mapping[str, Any]]
    ],
) -> dict[str, AgentRegistryReplayPackPromotionCheck]:
    checks_by_key: dict[str, AgentRegistryReplayPackPromotionCheck] = {}
    if promotion_checks is None:
        return checks_by_key
    if isinstance(promotion_checks, Mapping):
        iterable = promotion_checks.values()
        explicit_keys = [str(key) for key in promotion_checks.keys()]
    else:
        iterable = promotion_checks
        explicit_keys = []
    for index, raw_check in enumerate(iterable):
        check = _coerce_registry_replay_promotion_check(raw_check)
        keys = [
            check.manifest.retention_key,
            check.dataset_id,
            check.registry_version,
            check.manifest.dataset_id,
            check.manifest.dataset_name,
        ]
        if index < len(explicit_keys):
            keys.append(explicit_keys[index])
        for key in keys:
            if key:
                checks_by_key[str(key)] = check
    return checks_by_key


def _registry_replay_lineage_entry(
    manifest: AgentRegistryReplayPackManifest,
    *,
    checks_by_key: Mapping[str, AgentRegistryReplayPackPromotionCheck],
) -> AgentRegistryReplayPackLineageEntry:
    check = _registry_replay_check_for_manifest(manifest, checks_by_key)
    check_metadata = _ensure_mapping(check.metadata if check else {})
    optimizer_backend = _optional_str(
        check_metadata.get("optimizer_backend")
        or check_metadata.get("selected_optimizer")
        or check_metadata.get("optimizer")
    )
    selected_patch_signature = _optional_str(
        check_metadata.get("selected_patch_signature")
        or check_metadata.get("selected_candidate_signature")
        or check_metadata.get("best_candidate_signature")
    )
    readback_signature_matches = None
    if check is not None:
        readback_signature_matches = (
            check.loaded_case_signature == check.expected_case_signature
        )
    return AgentRegistryReplayPackLineageEntry(
        registry_version=manifest.registry_version,
        dataset_name=manifest.dataset_name,
        dataset_id=manifest.dataset_id,
        retention_key=manifest.retention_key,
        case_count=manifest.case_count,
        case_signature=manifest.case_signature,
        coverage_score=manifest.coverage_score,
        selection_complete=manifest.selection_complete,
        required_presets=list(manifest.required_presets),
        required_invariant_families=list(manifest.required_invariant_families),
        promotion_promotable=check.promotable if check else None,
        loaded_case_count=check.loaded_case_count if check else None,
        replay_record_count=check.replay_record_count if check else None,
        readback_signature_matches=readback_signature_matches,
        optimizer_score=check.optimizer_score if check else None,
        min_optimizer_score=check.min_optimizer_score if check else None,
        optimizer_backend=optimizer_backend,
        selected_patch_signature=selected_patch_signature,
        failures=list(check.failures) if check else [],
        metadata={
            "manifest_name": manifest.name,
            "selected_positive_count": manifest.selected_positive_count,
            "selected_negative_count": manifest.selected_negative_count,
            "check_metadata": copy.deepcopy(check_metadata),
        },
    )


def _registry_replay_check_for_manifest(
    manifest: AgentRegistryReplayPackManifest,
    checks_by_key: Mapping[str, AgentRegistryReplayPackPromotionCheck],
) -> Optional[AgentRegistryReplayPackPromotionCheck]:
    for key in (
        manifest.retention_key,
        manifest.dataset_id,
        manifest.registry_version,
        manifest.dataset_name,
    ):
        if key and str(key) in checks_by_key:
            return checks_by_key[str(key)]
    return None


def _registry_replay_lineage_transition(
    previous: AgentRegistryReplayPackLineageEntry,
    current: AgentRegistryReplayPackLineageEntry,
) -> AgentRegistryReplayPackLineageTransition:
    coverage_delta = round(current.coverage_score - previous.coverage_score, 8)
    optimizer_score_delta: Optional[float] = None
    if previous.optimizer_score is not None and current.optimizer_score is not None:
        optimizer_score_delta = round(
            current.optimizer_score - previous.optimizer_score,
            8,
        )
    selected_patch_changed = _optional_change(
        previous.selected_patch_signature,
        current.selected_patch_signature,
    )
    optimizer_backend_changed = _optional_change(
        previous.optimizer_backend,
        current.optimizer_backend,
    )
    promotion_status_changed = _optional_bool_change(
        previous.promotion_promotable,
        current.promotion_promotable,
    )
    previous_presets = set(previous.required_presets)
    current_presets = set(current.required_presets)
    previous_families = set(previous.required_invariant_families)
    current_families = set(current.required_invariant_families)
    drift_reasons: list[str] = []
    if previous.case_signature != current.case_signature:
        drift_reasons.append("case_signature_changed")
    if previous.retention_key != current.retention_key:
        drift_reasons.append("retention_key_changed")
    if coverage_delta != 0:
        drift_reasons.append("coverage_score_changed")
    if optimizer_score_delta is not None and optimizer_score_delta != 0:
        drift_reasons.append("optimizer_score_changed")
    if selected_patch_changed:
        drift_reasons.append("selected_patch_changed")
    if optimizer_backend_changed:
        drift_reasons.append("optimizer_backend_changed")
    if promotion_status_changed:
        drift_reasons.append("promotion_status_changed")
    if previous_presets != current_presets:
        drift_reasons.append("required_presets_changed")
    if previous_families != current_families:
        drift_reasons.append("required_invariant_families_changed")
    return AgentRegistryReplayPackLineageTransition(
        from_registry_version=previous.registry_version,
        to_registry_version=current.registry_version,
        from_dataset_id=previous.dataset_id,
        to_dataset_id=current.dataset_id,
        case_count_delta=current.case_count - previous.case_count,
        coverage_delta=coverage_delta,
        optimizer_score_delta=optimizer_score_delta,
        case_signature_changed=previous.case_signature != current.case_signature,
        retention_key_changed=previous.retention_key != current.retention_key,
        selected_patch_changed=selected_patch_changed,
        optimizer_backend_changed=optimizer_backend_changed,
        promotion_status_changed=promotion_status_changed,
        added_required_presets=sorted(current_presets - previous_presets),
        removed_required_presets=sorted(previous_presets - current_presets),
        added_invariant_families=sorted(current_families - previous_families),
        removed_invariant_families=sorted(previous_families - current_families),
        drift_reasons=drift_reasons,
    )


def _optional_change(previous: Optional[str], current: Optional[str]) -> Optional[bool]:
    if previous is None or current is None:
        return None
    return previous != current


def _optional_bool_change(
    previous: Optional[bool],
    current: Optional[bool],
) -> Optional[bool]:
    if previous is None or current is None:
        return None
    return previous != current


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _registry_replay_triage_severity(
    *,
    blocking_reasons: Sequence[str],
    warnings: Sequence[str],
) -> str:
    critical = {
        "futureagi_readback_signature_mismatch",
        "futureagi_readback_case_count_mismatch",
        "futureagi_replay_record_count_mismatch",
        "latest_promotion_failed",
        "latest_promotion_failures",
        "missing_latest_promotion_check",
        "missing_latest_optimizer_score",
    }
    high = {
        "coverage_regression",
        "optimizer_score_regression",
        "latest_below_best_optimizer_score",
        "required_presets_removed",
        "required_invariant_families_removed",
    }
    if blocking_reasons:
        if any(reason in critical for reason in blocking_reasons):
            return "critical"
        if any(reason in high for reason in blocking_reasons):
            return "high"
        return "medium"
    if warnings:
        medium = {
            "case_signature_changed",
            "selected_patch_changed",
            "optimizer_backend_changed",
            "required_contract_expanded",
            "optimizer_score_drift",
        }
        if any(warning in medium for warning in warnings):
            return "medium"
        return "low"
    return "none"


def _registry_replay_triage_recommendations(
    *,
    blocking_reasons: Sequence[str],
    warnings: Sequence[str],
) -> list[str]:
    recommendation_by_reason = {
        "missing_latest_promotion_check": (
            "Run check_futureagi_registry_replay_pack_promotion() with Future AGI "
            "readback and optimizer replay before rollout."
        ),
        "missing_latest_optimizer_score": (
            "Attach optimizer replay score evidence before rollout."
        ),
        "latest_promotion_failed": (
            "Fix the latest promotion-gate failures before rollout."
        ),
        "latest_promotion_failures": (
            "Inspect latest promotion failures and repair the replay pack or candidate."
        ),
        "futureagi_readback_signature_mismatch": (
            "Republish or reload the Future AGI dataset until readback case signature "
            "matches the manifest."
        ),
        "futureagi_readback_case_count_mismatch": (
            "Republish or reload the Future AGI dataset until readback case count "
            "matches the manifest."
        ),
        "futureagi_replay_record_count_mismatch": (
            "Rebuild replay rows so every Future AGI dataset row becomes one "
            "optimizer replay record."
        ),
        "coverage_regression": (
            "Block rollout until replay-pack coverage recovers or the registry "
            "coverage loss is explicitly approved."
        ),
        "optimizer_score_regression": (
            "Block rollout and rerun curriculum or multi_interaction optimization "
            "against the latest Future AGI replay pack."
        ),
        "latest_below_best_optimizer_score": (
            "Compare the latest candidate with the best historical replay candidate "
            "before promotion."
        ),
        "required_presets_removed": (
            "Require explicit registry approval before removing preset coverage."
        ),
        "required_invariant_families_removed": (
            "Require explicit registry approval before removing invariant-family coverage."
        ),
        "selected_patch_changed": (
            "Review the selected patch diff and rerun staging replay with the new "
            "patch signature."
        ),
        "optimizer_backend_changed": (
            "Review backend-lineage evidence and confirm the new optimizer backend "
            "is expected for this replay pack."
        ),
        "case_signature_changed": (
            "Inspect added or removed case ids and confirm selection coverage still "
            "represents required registry families."
        ),
        "retention_key_changed": (
            "Record the retention-key change with the registry release metadata."
        ),
        "promotion_status_changed": (
            "Record promotion-status drift and compare latest promotion failures."
        ),
        "required_contract_expanded": (
            "Record the new required preset or invariant-family coverage in the "
            "registry release notes."
        ),
        "coverage_drift": (
            "Record coverage drift alongside the retained Future AGI replay pack."
        ),
        "optimizer_score_drift": (
            "Record optimizer-score drift and monitor the next replay run."
        ),
        "missing_optimizer_score_delta": (
            "Attach promotion checks for adjacent lineage entries to compare optimizer scores."
        ),
    }
    recommendations: list[str] = []
    for reason in list(blocking_reasons) + list(warnings):
        recommendation = recommendation_by_reason.get(reason)
        if recommendation:
            recommendations.append(recommendation)
    return _unique_strings(recommendations)


def _optimizer_result_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("final_score", "score", "optimizer_score"):
            if value.get(key) is not None:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    return None
        nested = value.get("result") or value.get("reoptimization_result")
        if nested is not None:
            return _optimizer_result_score(nested)
        return None
    for key in ("final_score", "score", "optimizer_score"):
        item = getattr(value, key, None)
        if item is not None:
            try:
                return float(item)
            except (TypeError, ValueError):
                return None
    return None


def _futureagi_experiment_payload(
    client: Any,
    *,
    experiment_id: str,
    page_size: int,
    max_pages: int,
    include_rows: bool,
    include_stats: bool,
    prefer_v2: bool,
) -> dict[str, Any]:
    if _futureagi_experiment_payload_like(client):
        payload = _load_payload(client)
        return dict(payload) if isinstance(payload, Mapping) else {"records": payload}

    method = getattr(client, "fetch_experiment_history", None)
    if not callable(method):
        raise TypeError(
            "client must expose fetch_experiment_history() or be a Future AGI "
            "experiment-history payload."
        )
    attempts = (
        lambda: method(
            experiment_id=experiment_id,
            page_size=page_size,
            max_pages=max_pages,
            include_rows=include_rows,
            include_stats=include_stats,
            prefer_v2=prefer_v2,
        ),
        lambda: method(
            experiment_id=experiment_id,
            page_size=page_size,
            max_pages=max_pages,
        ),
        lambda: method(experiment_id),
    )
    last_error: Optional[TypeError] = None
    for attempt in attempts:
        try:
            payload = _load_payload(attempt())
            return dict(payload) if isinstance(payload, Mapping) else {"records": payload}
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return {}


def _futureagi_experiment_payload_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in value
            for key in (
                "experiment",
                "experiment_id",
                "detail",
                "stats",
                "rows",
                "records",
                "variants",
                "rankings",
                "results",
                "history",
            )
        )
    return False


def _futureagi_experiment_metadata(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"id": experiment_id}
    for key in ("experiment", "detail"):
        section = payload.get(key)
        if not isinstance(section, Mapping):
            continue
        result = _ensure_mapping(_futureagi_payload_result(section))
        if not result:
            continue
        for field in ("id", "name", "status", "dataset", "dataset_id", "framework", "runtime"):
            value = result.get(field)
            if value is not None:
                metadata[field] = value
    for field in ("experiment_id", "experiment_name", "status", "framework", "runtime"):
        value = payload.get(field)
        if value is not None:
            metadata[field.replace("experiment_", "")] = value
    return metadata


def _futureagi_experiment_observation_records(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(
        _futureagi_explicit_history_records(
            payload,
            experiment_id=experiment_id,
            experiment_metadata=experiment_metadata,
        )
    )
    records.extend(
        _futureagi_experiment_stats_records(
            payload,
            experiment_id=experiment_id,
            experiment_metadata=experiment_metadata,
        )
    )
    records.extend(
        _futureagi_experiment_row_records(
            payload,
            experiment_id=experiment_id,
            experiment_metadata=experiment_metadata,
        )
    )
    return _dedupe_futureagi_observability_records(records)


def _futureagi_explicit_history_records(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("records", "history", "observations"):
        value = payload.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for index, item in enumerate(value, start=1):
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record.setdefault("id", f"{experiment_id}:history:{index}")
            records.append(
                _futureagi_experiment_record(
                    record,
                    experiment_id=experiment_id,
                    experiment_metadata=experiment_metadata,
                    source_section=key,
                    index=index,
                )
            )
    return records


def _futureagi_experiment_stats_records(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sections: list[tuple[str, Any]] = []
    for key in ("stats", "results", "comparisons", "list"):
        value = payload.get(key)
        if value is not None:
            sections.append((key, value))
    if payload.get("variants") is not None or payload.get("rankings") is not None:
        sections.append(("payload", payload))

    records: list[dict[str, Any]] = []
    for section_name, section in sections:
        result = _futureagi_payload_result(section)
        candidates = _futureagi_variant_rows(result)
        if section_name == "list":
            matching = [
                row
                for row in candidates
                if isinstance(row, Mapping)
                and str(row.get("id") or row.get("experiment_id") or "") == experiment_id
            ]
            candidates = matching or candidates
        for index, row in enumerate(candidates, start=1):
            if not isinstance(row, Mapping):
                continue
            metrics = _futureagi_metrics_from_mapping(row)
            if not metrics:
                metrics = _futureagi_status_metrics_from_mapping(row)
            if not metrics:
                continue
            candidate_id = _futureagi_variant_id(row, fallback=f"variant-{index}")
            record = {
                "id": f"{experiment_id}:variant:{candidate_id}",
                "run_id": f"{experiment_id}:variant:{candidate_id}",
                "candidate_id": str(candidate_id),
                "metrics": metrics,
                "score": _futureagi_record_score(metrics, row),
                "status": row.get("status") or row.get("state"),
                "raw_variant": dict(row),
            }
            records.append(
                _futureagi_experiment_record(
                    record,
                    experiment_id=experiment_id,
                    experiment_metadata=experiment_metadata,
                    source_section="stats",
                    index=index,
                )
            )
    return records


def _futureagi_experiment_row_records(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_pages = payload.get("rows")
    if raw_pages is None:
        raw_pages = payload.get("row_pages")
    if raw_pages is None and (
        "table" in payload or "column_config" in payload or "columnConfig" in payload
    ):
        raw_pages = [payload]
    if isinstance(raw_pages, Mapping):
        raw_pages = [raw_pages]
    if not isinstance(raw_pages, Sequence) or isinstance(raw_pages, (str, bytes, bytearray)):
        return []

    records: list[dict[str, Any]] = []
    for page_index, raw_page in enumerate(raw_pages):
        result = _futureagi_payload_result(raw_page)
        columns = _futureagi_table_columns(result)
        rows = _futureagi_table_rows(result)
        for row_index, row in enumerate(rows, start=1):
            values, row_metadata = _futureagi_row_values(row, columns=columns)
            metrics = _futureagi_metrics_from_row_values(values, columns=columns)
            if not metrics:
                continue
            row_id = row_metadata.get("row_id") or f"page-{page_index}-row-{row_index}"
            record = {
                "id": f"{experiment_id}:row:{row_id}",
                "run_id": f"{experiment_id}:row:{row_id}",
                "candidate_id": _futureagi_row_candidate_id(values, row),
                "metrics": metrics,
                "score": _futureagi_record_score(metrics, values),
                "raw_row": copy.deepcopy(row),
                "row_values": copy.deepcopy(values),
                "futureagi_row_id": row_id,
                "futureagi_row_order": row_metadata.get("order"),
            }
            records.append(
                _futureagi_experiment_record(
                    record,
                    experiment_id=experiment_id,
                    experiment_metadata=experiment_metadata,
                    source_section="rows",
                    index=len(records) + 1,
                )
            )
    return records


def _futureagi_experiment_record(
    record: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_metadata: Mapping[str, Any],
    source_section: str,
    index: int,
) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("source", "futureagi")
    normalized.setdefault("framework", experiment_metadata.get("framework") or "generic")
    normalized.setdefault("run_id", normalized.get("id") or f"{experiment_id}:{source_section}:{index}")
    normalized.setdefault("experiment_id", experiment_id)
    normalized.setdefault("experiment_name", experiment_metadata.get("name"))
    metadata = _ensure_mapping(normalized.get("metadata"))
    metadata.update(
        {
            "kind": "futureagi_experiment_history_record",
            "futureagi_experiment_id": experiment_id,
            "futureagi_experiment_name": experiment_metadata.get("name"),
            "futureagi_source_section": source_section,
            "futureagi_record_index": index,
        }
    )
    normalized["metadata"] = metadata
    return normalized


def _futureagi_variant_rows(result: Any) -> list[Any]:
    if isinstance(result, Mapping):
        for key in ("table_data", "tableData", "variants", "rankings", "comparisons", "results"):
            value = result.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
        table_rows = _futureagi_table_rows(result)
        if table_rows:
            return table_rows
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return list(result)
    return []


def _futureagi_metrics_from_row_values(
    values: Mapping[str, Any],
    *,
    columns: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    columns_by_name = {str(column.get("name") or ""): column for column in columns}
    metrics: dict[str, float] = {}
    for name, value in values.items():
        column = columns_by_name.get(str(name), {})
        if not _futureagi_column_looks_like_metric(name, column):
            continue
        score = _futureagi_metric_score(value)
        if score is not None:
            metrics[_normalize_metric_name(name)] = score
    return metrics


def _futureagi_metrics_from_mapping(row: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    nested_metrics = row.get("metrics") or row.get("metric_averages") or row.get("scores")
    if isinstance(nested_metrics, Mapping):
        for name, value in nested_metrics.items():
            score = _futureagi_metric_score(value)
            if score is not None:
                metrics[_normalize_metric_name(name)] = score
    for name, value in row.items():
        if not _futureagi_key_looks_like_metric(name):
            continue
        score = _futureagi_metric_score(value)
        if score is not None:
            metrics[_normalize_metric_name(name)] = score
    return metrics


def _futureagi_status_metrics_from_mapping(row: Mapping[str, Any]) -> dict[str, float]:
    status = str(row.get("status") or row.get("state") or "").strip().lower()
    if not status:
        return {}
    completed = {
        "completed",
        "complete",
        "success",
        "succeeded",
        "passed",
        "pass",
        "done",
    }
    return {"experiment_completed": 1.0 if status in completed else 0.0}


def _futureagi_metric_score(value: Any) -> Optional[float]:
    value = _futureagi_cell_payload_value(value)
    if isinstance(value, Mapping):
        for key in ("score", "value", "output", "cell_value", "cellValue", "average"):
            score = _futureagi_metric_score(value.get(key))
            if score is not None:
                return score
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = float(stripped.rstrip("%"))
        except ValueError:
            return None
        if stripped.endswith("%"):
            value = value / 100.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
        return max(0.0, min(numeric, 1.0))
    return None


def _futureagi_record_score(
    metrics: Mapping[str, float],
    raw: Mapping[str, Any],
) -> float:
    explicit = _futureagi_metric_score(
        raw.get("score")
        or raw.get("avg_score")
        or raw.get("average_score")
        or raw.get("overall_rating")
        or raw.get("overall")
    )
    if explicit is not None:
        return explicit
    return sum(metrics.values()) / len(metrics) if metrics else 1.0


def _futureagi_column_looks_like_metric(name: str, column: Mapping[str, Any]) -> bool:
    source = str(
        column.get("source")
        or column.get("origin_type")
        or column.get("originType")
        or column.get("type")
        or ""
    ).lower()
    if "evaluation" in source or "eval" in source or "score" in source:
        return True
    return _futureagi_key_looks_like_metric(name)


def _futureagi_key_looks_like_metric(name: Any) -> bool:
    normalized = _normalize_metric_name(name)
    if not normalized:
        return False
    excluded_tokens = (
        "id",
        "name",
        "dataset",
        "variant",
        "tokens",
        "token",
        "latency",
        "response_time",
        "duration",
        "runtime",
        "status",
        "order",
        "rank",
        "created",
        "updated",
    )
    if normalized in excluded_tokens or any(token == normalized for token in excluded_tokens):
        return False
    metric_tokens = (
        "score",
        "quality",
        "accuracy",
        "adherence",
        "outcome",
        "success",
        "safety",
        "correctness",
        "coverage",
        "grounding",
        "resilience",
        "coordination",
        "memory",
        "tool",
        "policy",
    )
    return any(token in normalized for token in metric_tokens)


def _normalize_metric_name(name: Any) -> str:
    return _case_slug(name).replace("-", "_")


def _futureagi_variant_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in (
        "candidate_id",
        "variant_id",
        "dataset_id",
        "id",
        "experiment_dataset_id",
        "experiment_dataset_name",
        "variant",
        "name",
    ):
        value = row.get(key)
        if value is not None:
            return _case_slug(value) or str(value)
    return fallback


def _futureagi_row_candidate_id(values: Mapping[str, Any], row: Any) -> Optional[str]:
    for source in (values, row if isinstance(row, Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("candidate_id", "variant_id", "experiment_dataset_id", "dataset_id"):
            value = source.get(key)
            if value is not None:
                return str(value)
    return None


def _dedupe_futureagi_observability_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = str(record.get("id") or record.get("run_id") or index)
        deduped[key] = dict(record)
    return list(deduped.values())


def _penalize_missing_futureagi_experiment_metrics(
    records: Sequence[AgentObservabilityRecord],
    required_metrics: Mapping[str, float],
) -> None:
    if not required_metrics:
        return
    required = set(required_metrics)
    for record in records:
        if record.passed or required & set(record.metrics):
            continue
        record.score = 0.0


def _futureagi_dataset_payloads(
    client: Any,
    *,
    dataset_id: str,
    page_size: int,
    max_pages: int,
) -> list[Any]:
    if _futureagi_payload_like(client):
        return [client]
    if isinstance(client, Sequence) and not isinstance(client, (str, bytes, bytearray)):
        if all(_futureagi_payload_like(item) for item in client):
            return list(client)

    payloads: list[Any] = []
    for page_index in range(max_pages):
        payload = _fetch_futureagi_dataset_page(
            client,
            dataset_id=dataset_id,
            page_size=page_size,
            current_page_index=page_index,
        )
        payloads.append(payload)
        result = _futureagi_payload_result(payload)
        total_pages = _futureagi_total_pages(result)
        row_count = len(_futureagi_table_rows(result))
        if total_pages is not None:
            if page_index + 1 >= total_pages:
                break
        elif row_count < page_size:
            break
        else:
            break
    return payloads


def _futureagi_payload_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in value
            for key in (
                "result",
                "table",
                "rows",
                "column_config",
                "columnConfig",
                "columns",
            )
        )
    return hasattr(value, "rows") and hasattr(value, "columns")


def _fetch_futureagi_dataset_page(
    client: Any,
    *,
    dataset_id: str,
    page_size: int,
    current_page_index: int,
) -> Any:
    for method_name in (
        "fetch_regression_dataset",
        "fetch_dataset_table",
        "get_dataset_table",
        "fetch_dataset",
        "get_dataset",
    ):
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        attempts = (
            lambda: method(
                dataset_id=dataset_id,
                page_size=page_size,
                current_page_index=current_page_index,
            ),
            lambda: method(
                dataset_id=dataset_id,
                page_size=page_size,
                page_index=current_page_index,
            ),
            lambda: method(
                dataset_id,
                page_size=page_size,
                current_page_index=current_page_index,
            ),
            lambda: method(dataset_id),
        )
        last_error: Optional[TypeError] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    raise TypeError(
        "client must expose fetch_regression_dataset(), fetch_dataset_table(), "
        "get_dataset_table(), fetch_dataset(), or get_dataset()."
    )


def _futureagi_regression_cases_from_payloads(
    payloads: Sequence[Any],
    *,
    dataset_id: str,
) -> tuple[list[AgentRegressionCase], dict[str, Any]]:
    cases: list[AgentRegressionCase] = []
    table_metadata: dict[str, Any] = {}
    max_column_count = 0
    for payload in payloads:
        result = _futureagi_payload_result(payload)
        columns = _futureagi_table_columns(result)
        max_column_count = max(max_column_count, len(columns))
        page_metadata = _futureagi_table_metadata(result)
        for key, value in page_metadata.items():
            if value is not None:
                table_metadata[key] = value
        for row in _futureagi_table_rows(result):
            cases.append(
                _futureagi_regression_case_from_row(
                    row,
                    columns=columns,
                    dataset_id=dataset_id,
                    index=len(cases) + 1,
                )
            )
    table_metadata["column_count"] = max_column_count
    return cases, table_metadata


def _futureagi_payload_result(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()
    if isinstance(payload, Mapping):
        for key in ("result", "data", "dataset_table", "datasetTable"):
            value = payload.get(key)
            if value is not None:
                return value
        return payload
    return payload


def _futureagi_table_columns(result: Any) -> list[dict[str, Any]]:
    raw_columns = _futureagi_get(
        result,
        "column_config",
        "columnConfig",
        "columns",
    )
    columns: list[dict[str, Any]] = []
    if isinstance(raw_columns, Sequence) and not isinstance(
        raw_columns,
        (str, bytes, bytearray),
    ):
        for raw_column in raw_columns:
            column_id = _futureagi_scalar(
                _futureagi_get(raw_column, "id", "column_id", "columnId")
            )
            name = _futureagi_scalar(
                _futureagi_get(raw_column, "name", "column_name", "columnName")
            )
            data_type = _futureagi_scalar(
                _futureagi_get(raw_column, "data_type", "dataType", "type")
            )
            if not name and column_id:
                name = column_id
            if name:
                columns.append(
                    {
                        "id": column_id or name,
                        "name": str(name),
                        "data_type": str(data_type or "text").lower(),
                    }
                )
    if columns:
        return columns
    return [
        {"id": column["name"], "name": column["name"], "data_type": column["data_type"]}
        for column in _futureagi_dataset_columns()
    ]


def _futureagi_table_rows(result: Any) -> list[Any]:
    rows = _futureagi_get(result, "table", "rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return list(rows)
    return []


def _futureagi_table_metadata(result: Any) -> dict[str, Any]:
    metadata = _futureagi_get(result, "metadata")
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    elif hasattr(metadata, "dict"):
        metadata = metadata.dict()
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _futureagi_total_pages(result: Any) -> Optional[int]:
    metadata = _futureagi_table_metadata(result)
    value = (
        metadata.get("total_pages")
        if "total_pages" in metadata
        else metadata.get("totalPages")
    )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _futureagi_regression_case_from_row(
    row: Any,
    *,
    columns: Sequence[Mapping[str, Any]],
    dataset_id: str,
    index: int,
) -> AgentRegressionCase:
    values, row_metadata = _futureagi_row_values(row, columns=columns)
    case_id = str(
        values.get("case_id")
        or row_metadata.get("row_id")
        or f"futureagi-regression-row-{index}"
    )
    observability = _ensure_mapping(values.get("observability"))
    if not observability:
        failures = _string_list(values.get("response"))
        observability = {
            "source": "futureagi",
            "framework": "generic",
            "run_id": case_id,
            "score": 0.0 if failures else 1.0,
            "passed": not failures,
            "failures": failures,
            "metrics": {},
            "trace_signals": [],
        }
    expected = _ensure_mapping(values.get("expected_response"))
    if not expected:
        expected = {
            "should_pass": True,
            "required_metrics": {},
            "required_trace_signals": [],
            "previous_score": _coerce_score(observability.get("score")),
            "previous_failures": _string_list(observability.get("failures")),
        }
    tags = _string_list(values.get("tags"))
    case_metadata = _ensure_mapping(values.get("metadata"))
    row_id = row_metadata.get("row_id")
    order = row_metadata.get("order")
    query = values.get("query")
    response = values.get("response")
    case_metadata.update(
        {
            "kind": "futureagi_regression_case",
            "futureagi_dataset_id": dataset_id,
            "futureagi_row_id": row_id,
            "futureagi_order": order,
            "futureagi_query": query,
            "futureagi_response": response,
        }
    )
    case_metadata.setdefault("dataset_case_id", case_id)
    if isinstance(observability, Mapping):
        case_metadata.setdefault("source", observability.get("source"))
        case_metadata.setdefault("framework", observability.get("framework"))
        case_metadata.setdefault("run_id", observability.get("run_id"))
        case_metadata.setdefault("candidate_id", observability.get("candidate_id"))

    return AgentRegressionCase(
        id=case_id,
        input={"observability": copy.deepcopy(dict(observability))},
        expected=copy.deepcopy(dict(expected)),
        tags=tags,
        metadata={
            key: value
            for key, value in case_metadata.items()
            if value is not None
        },
    )


def _futureagi_row_values(
    row: Any,
    *,
    columns: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    id_to_column = {
        str(column.get("id")): column
        for column in columns
        if column.get("id")
    }
    name_to_column = {
        str(column.get("name")): column
        for column in columns
        if column.get("name")
    }
    metadata = {
        "row_id": _futureagi_scalar(_futureagi_get(row, "row_id", "rowId", "id")),
        "order": _futureagi_get(row, "order"),
    }
    values: dict[str, Any] = {}

    cells = _futureagi_get(row, "cells")
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes, bytearray)):
        for cell in cells:
            column_id = _futureagi_scalar(
                _futureagi_get(cell, "column_id", "columnId", "column")
            )
            column = id_to_column.get(str(column_id)) or name_to_column.get(
                str(column_id)
            )
            if not column:
                continue
            column_name = str(column["name"])
            values[column_name] = _futureagi_parse_cell(
                _futureagi_get(cell, "value", "cell_value", "cellValue"),
                column=column,
            )
        return values, metadata

    if isinstance(row, Mapping):
        for column in columns:
            column_id = str(column.get("id") or "")
            column_name = str(column.get("name") or "")
            raw_cell = None
            if column_id and column_id in row:
                raw_cell = row[column_id]
            elif column_name and column_name in row:
                raw_cell = row[column_name]
            else:
                continue
            values[column_name] = _futureagi_parse_cell(raw_cell, column=column)
        if not values:
            for column_name in (
                "case_id",
                "query",
                "response",
                "expected_response",
                "observability",
                "tags",
                "metadata",
            ):
                if column_name in row:
                    column = name_to_column.get(column_name) or {
                        "name": column_name,
                        "data_type": _futureagi_column_data_type(column_name),
                    }
                    values[column_name] = _futureagi_parse_cell(
                        row[column_name],
                        column=column,
                    )
    return values, metadata


def _futureagi_parse_cell(value: Any, *, column: Mapping[str, Any]) -> Any:
    value = _futureagi_cell_payload_value(value)
    column_name = str(column.get("name") or "")
    data_type = str(
        column.get("data_type")
        or _futureagi_column_data_type(column_name)
        or "text"
    ).lower()
    if isinstance(value, str):
        stripped = value.strip()
        should_parse_json = (
            data_type in {"json", "array"}
            or column_name in {"expected_response", "observability", "tags", "metadata"}
            or stripped.startswith("{")
            or stripped.startswith("[")
        )
        if should_parse_json and stripped:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _futureagi_cell_payload_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, Mapping):
        for key in ("cell_value", "cellValue", "value"):
            if key in value:
                return value[key]
    return value


def _futureagi_column_data_type(column_name: str) -> str:
    for column in FUTUREAGI_REGRESSION_DATASET_COLUMNS:
        if column["name"] == column_name:
            return str(column["data_type"])
    return "text"


def _futureagi_get(value: Any, *keys: str) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return None
    for key in keys:
        item = getattr(value, key, None)
        if item is not None:
            return item
    return None


def _futureagi_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Sequence) and not isinstance(
                parsed,
                (str, bytes, bytearray),
            ):
                return [str(item) for item in parsed if item is not None]
        return [stripped]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw_value in value.items():
        score = _coerce_score(raw_value)
        if score is not None:
            metrics[str(key)] = score
    return metrics


def _observability_record_from_regression_case(
    case: AgentRegressionCase,
    *,
    index: int,
    candidate: Optional[AgentCandidate],
    source: str,
    framework: str,
) -> AgentObservabilityRecord:
    observability = _ensure_mapping(case.input.get("observability"))
    raw = observability.get("raw")
    if not isinstance(raw, Mapping):
        raw = copy.deepcopy(observability)
    failures = _string_list(
        observability.get("failures")
        or case.expected.get("previous_failures")
        or case.metadata.get("futureagi_response")
    )
    score = _coerce_score(observability.get("score"))
    if score is None:
        score = _coerce_score(case.expected.get("previous_score"))
    if score is None:
        score = 0.0 if failures else 1.0
    passed_value = observability.get("passed")
    if isinstance(passed_value, bool):
        passed = passed_value
    else:
        passed = not failures
    resolved_source = _normalize_source(
        observability.get("source")
        or case.metadata.get("source")
        or source
        or "futureagi"
    )
    resolved_framework = _normalize_source(
        observability.get("framework")
        or case.metadata.get("framework")
        or framework
        or "generic"
    )
    run_id = (
        observability.get("run_id")
        or case.metadata.get("run_id")
        or case.metadata.get("futureagi_row_id")
        or case.id
    )
    candidate_id = (
        observability.get("candidate_id")
        or case.metadata.get("candidate_id")
        or (candidate.id if candidate is not None else None)
    )
    return AgentObservabilityRecord(
        index=index,
        source=resolved_source,
        framework=resolved_framework,
        run_id=str(run_id) if run_id is not None else None,
        candidate_id=str(candidate_id) if candidate_id is not None else None,
        score=score,
        passed=passed,
        failures=failures,
        metrics=_float_mapping(observability.get("metrics")),
        trace_signals=_string_list(observability.get("trace_signals")),
        raw=copy.deepcopy(dict(raw)),
        metadata={
            "source_kind": resolved_source,
            "framework": resolved_framework,
            "regression_case_id": case.id,
            "regression_case_tags": list(case.tags),
            "regression_case_metadata": copy.deepcopy(case.metadata),
        },
    )


def _agent_report_case_raw_evidence(case: Mapping[str, Any]) -> dict[str, Any]:
    observability = _ensure_mapping(case.get("observability"))
    if not observability:
        observability = _ensure_mapping(_ensure_mapping(case.get("input")).get("observability"))
    raw = _ensure_mapping(observability.get("raw"))
    if not raw:
        raw = _ensure_mapping(case.get("raw"))
    return copy.deepcopy(raw)


def _agent_report_evaluation_metrics(evaluation: Any) -> dict[str, float]:
    payload = _ensure_mapping(evaluation)
    metrics = _float_mapping(_ensure_mapping(_ensure_mapping(payload.get("summary")).get("metric_averages")))
    if metrics:
        return metrics
    for case in _sequence_items(payload.get("cases")):
        case_dict = _ensure_mapping(case)
        for metric in _sequence_items(case_dict.get("metrics")):
            metric_dict = _ensure_mapping(metric)
            name = metric_dict.get("name")
            score = _coerce_score(metric_dict.get("score"))
            if name and score is not None:
                metrics[str(name)] = score
    return metrics


def _agent_report_evaluation_failures(evaluation: Any) -> list[str]:
    payload = _ensure_mapping(evaluation)
    failures: list[str] = []
    for finding in _sequence_items(payload.get("findings")):
        finding_dict = _ensure_mapping(finding)
        finding_type = finding_dict.get("type") or finding_dict.get("metric") or finding_dict.get("reason")
        if finding_type:
            failures.append(str(finding_type))
    for case in _sequence_items(payload.get("cases")):
        case_dict = _ensure_mapping(case)
        for finding in _sequence_items(case_dict.get("findings")):
            finding_dict = _ensure_mapping(finding)
            finding_type = finding_dict.get("type") or finding_dict.get("metric") or finding_dict.get("reason")
            if finding_type:
                failures.append(str(finding_type))
        for metric in _sequence_items(case_dict.get("metrics")):
            metric_dict = _ensure_mapping(metric)
            score = _coerce_score(metric_dict.get("score"))
            if score is not None and score < 1.0 and metric_dict.get("reason"):
                failures.append(str(metric_dict["reason"]))
    return list(dict.fromkeys(failures))


def _metric_threshold_failures(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    for name, threshold in thresholds.items():
        observed = metrics.get(name)
        if observed is None:
            failures.append(f"metric '{name}' missing from agent-report replay case")
        elif observed < threshold:
            failures.append(
                f"metric '{name}' below required threshold {threshold:.4f}: {observed:.4f}"
            )
    return failures


def _sequence_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _regression_dataset_required_metrics(
    cases: Sequence[AgentRegressionCase],
    *,
    override: Optional[Mapping[str, float]],
) -> dict[str, float]:
    if override is not None:
        return {str(key): float(value) for key, value in dict(override).items()}
    thresholds: dict[str, float] = {}
    for case in cases:
        expected = _ensure_mapping(case.expected)
        for key, value in _ensure_mapping(expected.get("required_metrics")).items():
            try:
                thresholds.setdefault(str(key), float(value))
            except (TypeError, ValueError):
                continue
    return thresholds


def _regression_dataset_required_trace_signals(
    cases: Sequence[AgentRegressionCase],
    *,
    override: Optional[Sequence[str]],
) -> list[str]:
    if override is not None:
        return [_normalize_signal(item) for item in override if _normalize_signal(item)]
    signals: list[str] = []
    seen: set[str] = set()
    for case in cases:
        expected = _ensure_mapping(case.expected)
        for signal in _string_list(expected.get("required_trace_signals")):
            normalized = _normalize_signal(signal)
            if normalized and normalized not in seen:
                signals.append(normalized)
                seen.add(normalized)
    return signals


def _regression_cases_source(cases: Sequence[AgentRegressionCase]) -> str:
    sources = sorted(
        {
            _normalize_source(
                _ensure_mapping(case.input.get("observability")).get("source")
                or case.metadata.get("source")
                or "futureagi"
            )
            for case in cases
        }
    )
    if not sources:
        return "futureagi"
    return sources[0] if len(sources) == 1 else "mixed"


def _regression_cases_framework(cases: Sequence[AgentRegressionCase]) -> str:
    frameworks = sorted(
        {
            _normalize_source(
                _ensure_mapping(case.input.get("observability")).get("framework")
                or case.metadata.get("framework")
                or "generic"
            )
            for case in cases
        }
    )
    if not frameworks:
        return "generic"
    return frameworks[0] if len(frameworks) == 1 else "mixed"


def _normalize_observability_record(
    record: dict[str, Any],
    *,
    index: int,
    candidate: Optional[AgentCandidate],
    source: str,
    framework: str,
    required_metrics: Mapping[str, float],
    required_trace_signals: Sequence[str],
) -> AgentObservabilityRecord:
    resolved_source = _resolve_source(record, fallback=source)
    resolved_framework = _resolve_framework(record, fallback=framework)
    trace_items = _trace_items(record)
    trace_signals = sorted(_trace_signals(record, trace_items))
    metrics = _extract_metrics(record)
    if trace_items:
        metrics.setdefault(
            "framework_trace_coverage",
            _trace_coverage(trace_signals, required_trace_signals),
        )
    if _has_transcript(record):
        metrics.setdefault(
            "framework_transcript_quality",
            0.0 if _has_error(record, trace_items) else 1.0,
        )
    if _has_error(record, trace_items):
        metrics.setdefault("runtime_success", 0.0)

    failures = _record_failures(
        record,
        metrics=metrics,
        trace_signals=trace_signals,
        required_metrics=required_metrics,
        required_trace_signals=required_trace_signals,
    )
    score = _record_score(record, metrics=metrics, failures=failures)
    passed = not failures
    return AgentObservabilityRecord(
        index=index,
        source=resolved_source,
        framework=resolved_framework,
        run_id=_first_string(record, "run_id", "id", "trace_id", "session_id", "room_name"),
        candidate_id=_candidate_id(record, candidate),
        score=score,
        passed=passed,
        failures=failures,
        metrics=metrics,
        trace_signals=trace_signals,
        raw=copy.deepcopy(record),
        metadata={
            "source_kind": resolved_source,
            "framework": resolved_framework,
            "trace_item_count": len(trace_items),
        },
    )


def _evaluation_from_observability_record(
    record: AgentObservabilityRecord,
    *,
    candidate: Optional[AgentCandidate],
) -> CandidateEvaluation:
    evaluation_candidate = candidate or AgentCandidate.from_config(
        record.raw.get("candidate_config")
        or record.raw.get("config")
        or {"observability": {"source": record.source, "framework": record.framework}},
        target_name=str(record.raw.get("target_name") or "observability-feedback"),
        metadata={
            "kind": "observability_feedback",
            "observability_source": record.source,
            "observability_framework": record.framework,
            "observability_run_id": record.run_id,
        },
    )
    return CandidateEvaluation(
        candidate=evaluation_candidate,
        score=record.score,
        reason="; ".join(record.failures),
        metadata={
            "agent_observability_feedback": record.model_dump(),
            "agent_report_evaluation": _agent_report_from_observability_record(record),
        },
    )


def _agent_report_from_observability_record(record: AgentObservabilityRecord) -> dict[str, Any]:
    return {
        "summary": {"metric_averages": dict(record.metrics)},
        "cases": [
            {
                "id": record.run_id or f"observability-{record.index}",
                "metrics": [
                    {
                        "name": name,
                        "score": score,
                        "reason": "; ".join(record.failures),
                    }
                    for name, score in record.metrics.items()
                ],
                "findings": [
                    {
                        "metric": name,
                        "score": score,
                        "evidence": "; ".join(record.failures),
                    }
                    for name, score in record.metrics.items()
                ],
            }
        ],
    }


def _regression_windows(
    windows: AgentObservabilityWindow | Sequence[AgentObservabilityWindow],
) -> list[AgentObservabilityWindow]:
    if isinstance(windows, AgentObservabilityWindow):
        return [windows]
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        return list(windows)
    raise TypeError("windows must be an AgentObservabilityWindow or a sequence of windows")


def _regression_case_from_observability_record(
    record: AgentObservabilityRecord,
    *,
    window: AgentObservabilityWindow,
    window_index: int,
    include_raw: bool,
) -> AgentRegressionCase:
    case_id = "-".join(
        item
        for item in (
            _case_slug(record.source),
            _case_slug(record.framework),
            _case_slug(record.run_id or f"record-{record.index}"),
            str(record.index),
        )
        if item
    )
    observability_input = {
        "source": record.source,
        "framework": record.framework,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "score": record.score,
        "passed": record.passed,
        "failures": list(record.failures),
        "metrics": dict(record.metrics),
        "trace_signals": list(record.trace_signals),
    }
    if include_raw:
        observability_input["raw"] = copy.deepcopy(record.raw)

    expected = {
        "should_pass": True,
        "required_metrics": dict(window.required_metrics),
        "required_trace_signals": list(window.required_trace_signals),
        "previous_score": record.score,
        "previous_failures": list(record.failures),
    }
    return AgentRegressionCase(
        id=case_id,
        input={"observability": observability_input},
        expected=expected,
        tags=_regression_tags(record, window=window),
        metadata={
            "kind": "observability_regression_case",
            "source": record.source,
            "framework": record.framework,
            "window_source": window.source,
            "window_framework": window.framework,
            "window_index": window_index,
            "record_index": record.index,
            "run_id": record.run_id,
            "candidate_id": record.candidate_id,
            "observed_score": record.score,
            "passed": record.passed,
            "failure_count": len(record.failures),
            "record_metadata": copy.deepcopy(record.metadata),
            "window_metadata": copy.deepcopy(window.metadata),
        },
    )


def _regression_tags(
    record: AgentObservabilityRecord,
    *,
    window: AgentObservabilityWindow,
) -> list[str]:
    tags = {
        "observability",
        f"source:{record.source}",
        f"framework:{record.framework}",
        "status:passed" if record.passed else "status:failed",
    }
    for metric, threshold in window.required_metrics.items():
        observed = record.metrics.get(metric)
        if observed is None or observed < threshold:
            tags.add(f"metric:{_case_slug(metric)}")
    present_signals = set(record.trace_signals)
    for signal in window.required_trace_signals:
        if signal not in present_signals:
            tags.add(f"missing_signal:{_case_slug(signal)}")
    if any("runtime error" in failure for failure in record.failures):
        tags.add("runtime:error")
    return sorted(tags)


def _regression_source(windows: Sequence[AgentObservabilityWindow]) -> str:
    sources = sorted({window.source for window in windows})
    return sources[0] if len(sources) == 1 else "mixed"


def _regression_framework(windows: Sequence[AgentObservabilityWindow]) -> str:
    frameworks = sorted({window.framework for window in windows})
    return frameworks[0] if len(frameworks) == 1 else "mixed"


def _case_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    chars = [char if char.isalnum() else "-" for char in text]
    return "-".join(part for part in "".join(chars).split("-") if part)[:80]


def _load_payload(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    if isinstance(payload, Path):
        return _parse_observability_text(payload.read_text())
    if isinstance(payload, str):
        if _looks_like_observability_text(payload):
            return _parse_observability_text(payload)
        try:
            path = Path(payload)
            if path.exists() and path.is_file():
                return _parse_observability_text(path.read_text())
        except OSError:
            pass
        return _parse_observability_text(payload)
    return payload


def _parse_observability_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        records = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def _looks_like_observability_text(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.startswith("{")
        or stripped.startswith("[")
        or "\n" in stripped
        or "\r" in stripped
    )


def _observation_records(payload: Any) -> list[Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return list(payload)
    if not isinstance(payload, Mapping):
        return [payload]
    if "resourceSpans" in payload or "resource_spans" in payload:
        return [payload]
    for key in ("runs", "traces", "sessions", "records", "items", "results"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [payload]


def _resolve_source(record: Mapping[str, Any], *, fallback: str) -> str:
    if fallback and fallback != "auto":
        return _normalize_source(fallback)
    explicit = _first_string(record, "source", "provider", "observability_source")
    if explicit:
        return _normalize_source(explicit)
    keys = {str(key).lower() for key in record}
    if {"feedback", "run_type", "dotted_order", "parent_run_id"} & keys:
        return "generic"
    if {"resourceSpans", "resource_spans"} & set(record):
        return "opentelemetry"
    if "span_data" in _json_text(record) or "trace_id" in keys:
        return "openai_agents"
    if {"room", "job", "session", "participant"} & keys or "makeSessionReport" in _json_text(record):
        return "livekit"
    return "generic"


def _resolve_framework(record: Mapping[str, Any], *, fallback: str) -> str:
    if fallback and fallback != "auto":
        return _normalize_source(fallback)
    explicit = _first_string(record, "framework", "runtime", "sdk", "provider")
    if explicit:
        return _normalize_source(explicit)
    text = _json_text(record)
    checks = (
        ("langgraph", ("langgraph", "stream_events")),
        ("langchain", ("langchain", "stream_events")),
        ("openai_agents", ("openai agents", "openai_agents", "span_data")),
        ("livekit", ("livekit", "agent_session", "room")),
        ("pipecat", ("pipecat", "frame")),
        ("crewai", ("crewai", "crew")),
        ("autogen", ("autogen", "groupchat")),
        ("opentelemetry", ("resourceSpans", "gen_ai.")),
    )
    for name, tokens in checks:
        if any(token.lower() in text for token in tokens):
            return name
    return "generic"


def _resolve_window_source(
    records: Sequence[AgentObservabilityRecord],
    *,
    fallback: str,
) -> str:
    if fallback and fallback != "auto":
        return _normalize_source(fallback)
    sources = sorted({record.source for record in records})
    return sources[0] if len(sources) == 1 else "mixed"


def _resolve_window_framework(
    records: Sequence[AgentObservabilityRecord],
    *,
    fallback: str,
) -> str:
    if fallback and fallback != "auto":
        return _normalize_source(fallback)
    frameworks = sorted({record.framework for record in records})
    return frameworks[0] if len(frameworks) == 1 else "mixed"


def _extract_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path in (
        ("metrics",),
        ("metric_averages",),
        ("scores",),
        ("outputs", "metrics"),
        ("outputs", "scores"),
        ("metadata", "metrics"),
        ("agent_report_evaluation", "summary", "metric_averages"),
        ("evaluation", "summary", "metric_averages"),
    ):
        value = _nested_get(record, path)
        if isinstance(value, Mapping):
            _merge_metric_mapping(metrics, value)

    feedback = record.get("feedback")
    if isinstance(feedback, Mapping):
        _merge_metric_mapping(metrics, feedback)
    elif isinstance(feedback, Sequence) and not isinstance(feedback, (str, bytes, bytearray)):
        for item in feedback:
            _merge_metric_item(metrics, item)

    for key in ("evaluations", "evaluation_results", "scores"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                _merge_metric_item(metrics, item)

    for report_key in ("agent_report_evaluation", "evaluation"):
        value = record.get(report_key)
        if isinstance(value, Mapping):
            _merge_agent_report_case_metrics(metrics, value)

    explicit_score = _coerce_score(record.get("score"))
    if explicit_score is not None and not metrics:
        metrics["score"] = explicit_score
    return metrics


def _merge_metric_mapping(metrics: dict[str, float], value: Mapping[str, Any]) -> None:
    for key, raw_score in value.items():
        score = _metric_score(raw_score)
        if score is not None:
            metrics[str(key)] = score


def _merge_metric_item(metrics: dict[str, float], item: Any) -> None:
    if not isinstance(item, Mapping):
        return
    name = item.get("key") or item.get("name") or item.get("metric")
    score = _metric_score(
        item.get("score", item.get("value", item.get("output")))
    )
    if name and score is not None:
        metrics[str(name)] = score


def _merge_agent_report_case_metrics(metrics: dict[str, float], report: Mapping[str, Any]) -> None:
    for case in report.get("cases", []) or []:
        if not isinstance(case, Mapping):
            continue
        for item in case.get("metrics", []) or []:
            _merge_metric_item(metrics, item)


def _metric_score(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        for key in ("score", "value", "output"):
            score = _coerce_score(value.get(key))
            if score is not None:
                return score
        return None
    return _coerce_score(value)


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return None


def _record_score(
    record: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> float:
    explicit = _coerce_score(record.get("score"))
    if explicit is not None:
        return explicit
    if metrics:
        return sum(metrics.values()) / len(metrics)
    return 0.0 if failures else 1.0


def _record_failures(
    record: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
    trace_signals: Sequence[str],
    required_metrics: Mapping[str, float],
    required_trace_signals: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    for name, threshold in required_metrics.items():
        observed = metrics.get(name)
        if observed is None:
            failures.append(f"metric '{name}' missing from observability record")
        elif observed < threshold:
            failures.append(
                f"metric '{name}' score {observed:.4f} below {threshold:.4f}"
            )
    missing_signals = [
        signal for signal in required_trace_signals if signal not in trace_signals
    ]
    if missing_signals:
        failures.append(
            "missing trace signal(s): " + ", ".join(sorted(missing_signals))
        )
    if _has_error(record, _trace_items(record)):
        failures.append("observability record contains runtime error signal")
    return failures


def _trace_items(record: Mapping[str, Any]) -> list[Any]:
    items: list[Any] = []
    for key in ("spans", "events", "session_events", "trace_events"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items.extend(value)
    trace = record.get("trace")
    if isinstance(trace, Mapping):
        for key in ("spans", "events"):
            value = trace.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                items.extend(value)
    report = record.get("report") or record.get("session_report") or record.get("session")
    if isinstance(report, Mapping):
        for key in ("events", "history", "conversation"):
            value = report.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                items.extend(value)
    items.extend(_otlp_spans(record))
    return items


def _otlp_spans(record: Mapping[str, Any]) -> list[Any]:
    resource_spans = record.get("resourceSpans") or record.get("resource_spans") or []
    spans: list[Any] = []
    for resource in resource_spans:
        if not isinstance(resource, Mapping):
            continue
        scope_spans = resource.get("scopeSpans") or resource.get("scope_spans") or []
        for scope in scope_spans:
            if not isinstance(scope, Mapping):
                continue
            scope_items = scope.get("spans") or []
            if isinstance(scope_items, Sequence) and not isinstance(scope_items, (str, bytes, bytearray)):
                spans.extend(scope_items)
    return spans


def _trace_signals(record: Mapping[str, Any], items: Sequence[Any]) -> set[str]:
    signals: set[str] = set()
    signal_items = list(items) or [record]
    for item in signal_items:
        text = _json_text(item)
        attributes = _attributes_text(item)
        combined = f"{text} {attributes}"
        if "invoke_agent" in combined or "agent" in combined:
            signals.add("agent")
        if "chat" in combined or "llm" in combined or "model" in combined:
            signals.add("model")
        if "execute_tool" in combined or "tool" in combined or "function_call" in combined:
            signals.add("tool")
        if "handoff" in combined or "delegate" in combined:
            signals.add("handoff")
        if "guardrail" in combined or "safety" in combined:
            signals.add("guardrail")
        if "message" in combined or "transcript" in combined or "conversation" in combined:
            signals.add("message")
        if "error" in combined or "exception" in combined or "failed" in combined:
            signals.add("error")
    return signals


def _attributes_text(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    attributes = item.get("attributes") or item.get("attrs") or {}
    if isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes, bytearray)):
        flattened = {}
        for attr in attributes:
            if not isinstance(attr, Mapping):
                continue
            key = attr.get("key")
            value = attr.get("value")
            flattened[str(key)] = value
        attributes = flattened
    return _json_text(attributes)


def _trace_coverage(
    trace_signals: Sequence[str],
    required_trace_signals: Sequence[str],
) -> float:
    if not required_trace_signals:
        return 1.0
    present = set(trace_signals)
    required = set(required_trace_signals)
    return len(present & required) / len(required)


def _has_transcript(record: Mapping[str, Any]) -> bool:
    text = _json_text(record)
    return any(
        token in text
        for token in (
            "transcript",
            "conversation_item",
            "user_input_transcribed",
            "assistant",
            "human",
            "message",
        )
    )


def _has_error(record: Mapping[str, Any], trace_items: Sequence[Any]) -> bool:
    for item in [record, *trace_items]:
        if not isinstance(item, Mapping):
            continue
        if item.get("error") or item.get("exception") or item.get("error.type"):
            return True
        status = item.get("status")
        if isinstance(status, Mapping):
            code = str(status.get("code") or status.get("status_code") or "").lower()
            if code in {"error", "2"}:
                return True
        elif str(status or "").lower() in {"error", "failed", "failure"}:
            return True
    return False


def _nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_string(record: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return str(value)
    return None


def _candidate_id(
    record: Mapping[str, Any],
    candidate: Optional[AgentCandidate],
) -> Optional[str]:
    return (
        _first_string(record, "candidate_id", "deployment_candidate_id")
        or (candidate.id if candidate is not None else None)
    )


def _normalize_source(value: Any) -> str:
    normalized = str(value or "generic").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "otel": "opentelemetry",
        "otlp": "opentelemetry",
        "traceai": "opentelemetry",
        "openai": "openai_agents",
        "openai_agent": "openai_agents",
        "livekit_agents": "livekit",
    }
    return aliases.get(normalized, normalized or "generic")


def _normalize_signal(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "llm": "model",
        "function": "tool",
        "function_call": "tool",
        "messages": "message",
        "transcript": "message",
    }
    return aliases.get(normalized, normalized)


def _json_text(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        return json.dumps(value, sort_keys=True, default=str).lower()
    except Exception:
        return str(value).lower()
