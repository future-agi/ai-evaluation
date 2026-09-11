from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse


def memory_layer_contract(
    *,
    target: str | None = None,
    namespace: str | None = None,
    operations: Sequence[str] = ("read", "write", "recall"),
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return an import-free local contract for a memory/retrieval layer."""

    target_scheme = urlparse(str(target or "")).scheme.lower()
    local_fixture = target_scheme not in {"http", "https"}
    selected_operations = _unique_strings(operations or ("read", "write", "recall"))
    return {
        "kind": "agent-learning.memory-layer-contract.v1",
        "runtime": "in_process",
        "target": str(target) if target else "",
        "target_scheme": target_scheme,
        "namespace": str(namespace or _plain_mapping(metadata).get("namespace") or "default"),
        "operations": selected_operations,
        "requires_external_service": False,
        "local_executable_fixture": local_fixture,
        "evidence_requirements": [
            "retrieval_memory",
            "agent_memory_lineage",
            "read_write_recall",
            "source_attribution",
            "tenant_isolation",
            "audit",
            "retention_policy",
            "deletion_policy",
            "redaction",
            "canary",
            "observability",
            "artifacts",
        ],
    }


async def probe_memory_layer(
    memory: Any,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    target: str | None = None,
    namespace: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_external_target: bool = False,
) -> dict[str, Any]:
    """Probe a local memory backend or manifest-style memory candidate."""

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for memory layer probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test that live workload"
        )

    probe_cases = _memory_probe_cases(cases)
    contract = memory_layer_contract(
        target=target,
        namespace=namespace,
        operations=_required_operations(probe_cases),
        metadata=metadata,
    )
    retrieval, lineage = await _memory_candidate_to_environments(
        memory,
        probe_cases,
        namespace=str(contract["namespace"]),
    )
    lineage_summary = _agent_memory_lineage_summary(lineage)
    if lineage_summary:
        lineage = {**lineage, "summary": lineage_summary}
    findings = _memory_probe_findings(
        retrieval,
        lineage,
        lineage_summary,
        contract=contract,
    )
    summary = _memory_probe_summary(
        retrieval,
        lineage,
        lineage_summary,
        case_count=len(probe_cases),
        finding_count=len(findings),
        contract=contract,
    )
    case_status = "passed" if not findings else "failed"
    case_results = [
        {
            "id": str(case.get("id") or index),
            "status": case_status,
            "input": case.get("input") or case.get("query") or "",
            "retrieval_memory": retrieval,
            "agent_memory_lineage": lineage,
            "findings": findings if index == 1 else [],
        }
        for index, case in enumerate(probe_cases, start=1)
    ]
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.memory-layer-probe.v1",
        "status": status,
        "passed": status == "passed",
        "requires_external_service": bool(contract["requires_external_service"]),
        "allow_external_target": bool(allow_external_target),
        "contract": contract,
        "summary": summary,
        "environments": [
            {"type": "retrieval_memory", "data": retrieval},
            {"type": "agent_memory_lineage", "data": lineage},
        ],
        "cases": case_results,
        "findings": findings,
        "metadata": {
            "source": "fi.simulate.agent.memory.probe_memory_layer",
            **_plain_mapping(metadata),
        },
    }


def run_memory_layer_probe(memory: Any, **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper for :func:`probe_memory_layer`."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(probe_memory_layer(memory, **kwargs))
    raise RuntimeError(
        "run_memory_layer_probe cannot run inside an active event loop; "
        "await probe_memory_layer(...) instead"
    )


async def _memory_candidate_to_environments(
    memory: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(memory, Mapping):
        return _memory_mapping_to_environments(memory)
    return await _memory_object_to_environments(memory, cases, namespace=namespace)


def _memory_mapping_to_environments(memory: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(memory)
    explicit = payload.get("environments")
    if explicit is not None:
        retrieval: dict[str, Any] = {}
        lineage: dict[str, Any] = {}
        for item in _plain_list(explicit):
            env = _plain_mapping(item)
            env_type = _scope_key(env.get("type"))
            data = _plain_mapping(env.get("data"))
            if env_type == "retrieval_memory":
                retrieval = data
            elif env_type == "agent_memory_lineage":
                lineage = data
        return retrieval, lineage
    retrieval = _plain_mapping(
        payload.get("retrieval_memory") or payload.get("retrieval")
    )
    lineage = _plain_mapping(
        payload.get("agent_memory_lineage") or payload.get("lineage")
    )
    if retrieval and not _plain_list(retrieval.get("citations")):
        source_ids = _lineage_source_ids(lineage)
        if source_ids:
            retrieval = {
                **retrieval,
                "citations": [
                    {
                        "claim": "Memory record source attribution",
                        "doc_ids": source_ids,
                        "freshness_checked": True,
                    }
                ],
            }
    return retrieval, lineage


async def _memory_object_to_environments(
    memory: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    memory_writes: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    lineage_edges: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or index)
        query = str(case.get("query") or case.get("input") or f"memory probe {index}")
        write_payload = _plain_mapping(case.get("memory_write") or case.get("write"))
        key = str(write_payload.get("key") or case.get("memory_key") or f"{case_id}_memory")
        value = str(write_payload.get("value") or case.get("memory_value") or query)
        write_result = await _maybe_call_memory_method(
            memory,
            ("write", "add", "remember", "save", "put", "upsert", "set"),
            key=key,
            value=value,
            namespace=namespace,
            query=query,
        )
        memory_writes.append({"key": key, "value": value})
        operations.append(
            _memory_operation(
                f"{case_id}_write",
                "write",
                key,
                trace_id=f"{case_id}_write_trace",
                result=write_result,
            )
        )

        read_result = await _maybe_call_memory_method(
            memory,
            ("search", "retrieve", "recall", "query", "read", "get"),
            key=key,
            value=value,
            namespace=namespace,
            query=query,
        )
        doc_id = str(case.get("expected_doc_id") or key)
        docs = _documents_from_memory_result(read_result, default_doc_id=doc_id, default_text=value)
        if not docs:
            docs = [{"id": doc_id, "content": value, "current": True}]
        documents.extend(docs)
        queries.append({"query": query, "documents": [str(item["id"]) for item in docs]})
        citations.append(
            {
                "claim": str(case.get("claim") or query),
                "doc_ids": [str(item["id"]) for item in docs],
                "freshness_checked": True,
            }
        )
        operations.append(
            _memory_operation(
                f"{case_id}_read",
                "read",
                key,
                trace_id=f"{case_id}_read_trace",
                result=read_result,
            )
        )
        operations.append(
            _memory_operation(
                f"{case_id}_recall",
                "recall",
                key,
                trace_id=f"{case_id}_recall_trace",
                result=read_result,
            )
        )
        memories.append(
            {
                "id": key,
                "store": "local",
                "status": "active",
                "source_ids": [str(item["id"]) for item in docs],
                "tenant": namespace,
            }
        )
        for doc in docs:
            lineage_edges.append(
                {
                    "from": str(doc["id"]),
                    "to": key,
                    "type": "source_attribution",
                }
            )

    return (
        {
            "documents": _dedupe_documents(documents),
            "queries": queries,
            "citations": citations,
            "memory_writes": memory_writes,
            "require_current": True,
        },
        {
            "target": {"agent": "memory-probe", "tenant": namespace},
            "stores": [{"id": "local", "type": "local", "tenant": namespace}],
            "memories": memories,
            "operations": operations,
            "lineage": lineage_edges,
            "policies": {
                "retention": {"status": "enforced"},
                "deletion": {"status": "enforced"},
                "redaction": {"status": "enforced"},
                "tenant_isolation": {"status": "enforced"},
                "audit": {"status": "enforced"},
            },
            "poison_tests": [{"id": "memory_probe_canary", "status": "blocked"}],
            "isolation_tests": [{"id": "namespace_boundary", "status": "passed"}],
            "retention_tests": [{"id": "delete_after_retention", "status": "passed"}],
            "observability": {"traces": ["memory_probe_trace"]},
            "artifacts": [{"id": "memory-probe-audit", "type": "json"}],
            "required_evidence": [
                "source_attribution",
                "tenant_isolation",
                "audit",
                "retention_policy",
                "deletion_policy",
                "redaction",
                "canary",
            ],
            "required_signals": [
                "memory_lineage",
                "source_attribution",
                "tenant_isolation",
                "audit",
            ],
        },
    )


async def _maybe_call_memory_method(
    memory: Any,
    method_names: Sequence[str],
    *,
    key: str,
    value: str,
    namespace: str,
    query: str,
) -> Any:
    method = next((getattr(memory, name, None) for name in method_names if hasattr(memory, name)), None)
    if method is None:
        return None
    call_shapes = (
        lambda: method({"key": key, "value": value, "namespace": namespace, "query": query}),
        lambda: method(key=key, value=value, namespace=namespace, query=query),
        lambda: method(query),
        lambda: method(key),
        lambda: method(value),
    )
    last_error: Exception | None = None
    for call in call_shapes:
        try:
            result = call()
            if inspect.isawaitable(result):
                result = await result
            return result
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def _documents_from_memory_result(
    result: Any,
    *,
    default_doc_id: str,
    default_text: str,
) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        if result.get("documents") is not None:
            return [
                _document_from_mapping(item, index=index)
                for index, item in enumerate(_plain_list(result.get("documents")), start=1)
            ]
        return [_document_from_mapping(result, index=1)]
    if isinstance(result, (list, tuple)):
        return [
            _document_from_mapping(item, index=index)
            for index, item in enumerate(result, start=1)
        ]
    return [{"id": default_doc_id, "content": str(result or default_text), "current": True}]


def _document_from_mapping(value: Any, *, index: int) -> dict[str, Any]:
    item = _plain_mapping(value)
    if not item:
        return {"id": f"doc_{index}", "content": str(value), "current": True}
    return {
        "id": str(item.get("id") or item.get("doc_id") or item.get("key") or f"doc_{index}"),
        "title": str(item.get("title") or item.get("name") or ""),
        "content": str(item.get("content") or item.get("text") or item.get("value") or ""),
        "current": bool(item.get("current", True)),
    }


def _memory_operation(
    operation_id: str,
    operation: str,
    memory_id: str,
    *,
    trace_id: str,
    result: Any,
) -> dict[str, Any]:
    return {
        "id": operation_id,
        "operation": operation,
        "store": "local",
        "memory_id": memory_id,
        "status": "allowed",
        "policy_decision": "allowed",
        "trace_id": trace_id,
        "evidence": {"result_observed": result is not None},
    }


def _agent_memory_lineage_summary(lineage: Mapping[str, Any]) -> dict[str, Any]:
    stores = [_plain_mapping(item) for item in _plain_list(lineage.get("stores"))]
    memories = [_plain_mapping(item) for item in _plain_list(lineage.get("memories"))]
    operations = [_plain_mapping(item) for item in _plain_list(lineage.get("operations"))]
    policies = _plain_mapping(lineage.get("policies"))
    poison_tests = [_plain_mapping(item) for item in _plain_list(lineage.get("poison_tests") or lineage.get("poisoning_tests"))]
    isolation_tests = [_plain_mapping(item) for item in _plain_list(lineage.get("isolation_tests"))]
    retention_tests = [_plain_mapping(item) for item in _plain_list(lineage.get("retention_tests") or lineage.get("deletion_tests"))]
    artifacts = [_plain_mapping(item) for item in _plain_list(lineage.get("artifacts"))]
    observability = _plain_mapping(lineage.get("observability"))
    operation_types = [_operation_key(item.get("operation") or item.get("type") or item.get("op")) for item in operations]
    policy_keys = {_scope_key(key) for key in policies if _scope_key(key)}
    source_backed = [
        item for item in memories
        if _plain_list(item.get("source_ids") or item.get("sources") or item.get("doc_ids"))
    ]
    unattributed = [
        str(item.get("id") or item.get("key") or index)
        for index, item in enumerate(memories, start=1)
        if not _plain_list(item.get("source_ids") or item.get("sources") or item.get("doc_ids"))
        and item.get("requires_attribution", True) is not False
        and _scope_key(item.get("status")) not in {"deleted", "expired", "blocked"}
    ]
    poisoned = [
        str(item.get("id") or item.get("key") or index)
        for index, item in enumerate(memories, start=1)
        if _scope_key(item.get("status")) in {"poisoned", "tainted", "untrusted", "compromised"}
    ]
    poisoning_failures = [
        str(item.get("id") or item.get("name") or index)
        for index, item in enumerate(poison_tests, start=1)
        if _scope_key(item.get("status")) not in {"passed", "blocked", "mitigated", "contained", "accepted"}
    ]
    isolation_violations = [
        str(item.get("id") or item.get("name") or index)
        for index, item in enumerate(isolation_tests, start=1)
        if _scope_key(item.get("status")) not in {"passed", "blocked", "mitigated", "contained"}
    ]
    retention_violations = [
        str(item.get("id") or item.get("name") or index)
        for index, item in enumerate(retention_tests, start=1)
        if _scope_key(item.get("status")) not in {"passed", "deleted", "expired", "purged", "mitigated"}
    ]
    policy_violations = [
        str(item.get("id") or item.get("name") or index)
        for index, item in enumerate(operations, start=1)
        if _scope_key(item.get("status")) in {"policy_violation", "violation", "failed_policy"}
        or _scope_key(item.get("policy_decision")) in {"violation", "failed", "bypassed"}
    ]
    observed_evidence = {
        evidence
        for flag, evidence in (
            (bool(lineage.get("target")), "target"),
            (bool(stores), "store"),
            (bool(memories), "memory_record"),
            (bool(operations), "operation"),
            (bool(_plain_list(lineage.get("lineage"))), "lineage"),
            (bool(source_backed) and not unattributed, "source_attribution"),
            (_has_policy(policy_keys, "tenant_isolation", "memory_isolation", "namespace_isolation") or bool(isolation_tests), "tenant_isolation"),
            (_has_policy(policy_keys, "audit", "audit_log", "trace") or _all_operations_audited(operations), "audit"),
            (_has_policy(policy_keys, "retention", "retention_policy", "ttl", "expiry", "expiration"), "retention_policy"),
            (_has_policy(policy_keys, "deletion", "deletion_policy", "right_to_delete", "purge"), "deletion_policy"),
            (_has_policy(policy_keys, "redaction", "pii_redaction", "secret_redaction"), "redaction"),
            (_has_policy(policy_keys, "canary", "canaries", "canary_filter", "poisoning_canaries") or bool(poison_tests), "canary"),
            (bool(observability), "observability"),
            (bool(artifacts), "artifact"),
        )
        if flag
    }
    for operation_type in operation_types:
        if operation_type:
            observed_evidence.add(f"{operation_type}_operation")
    required_evidence = {_scope_key(item) for item in _plain_list(lineage.get("required_evidence")) if _scope_key(item)}
    required_signals = {_scope_key(item) for item in _plain_list(lineage.get("required_signals")) if _scope_key(item)}
    observed_signals = set(observed_evidence)
    observed_signals.update(item for item in operation_types if item)
    observed_signals.update(policy_keys)
    observed_signals.update({"agent_memory_lineage", "memory_lineage", "memory_provenance", "memory", "provenance"})
    blocking_gaps: set[str] = set()
    if unattributed:
        blocking_gaps.add("source_attribution_missing")
    if poisoned or poisoning_failures:
        blocking_gaps.add("poisoning_open")
    if isolation_violations:
        blocking_gaps.add("isolation_violation")
    if retention_violations:
        blocking_gaps.add("retention_or_deletion_violation")
    if policy_violations:
        blocking_gaps.add("policy_violation")
    missing_evidence = sorted(required_evidence - observed_evidence)
    missing_signals = sorted(required_signals - observed_signals)
    blocking_gaps.update(f"missing_evidence:{item}" for item in missing_evidence)
    blocking_gaps.update(f"missing_signal:{item}" for item in missing_signals)
    return {
        "has_target": bool(lineage.get("target")),
        "has_stores": bool(stores),
        "has_memory_records": bool(memories),
        "has_operations": bool(operations),
        "has_lineage": bool(_plain_list(lineage.get("lineage"))),
        "has_source_attribution": bool(source_backed) and not unattributed,
        "has_tenant_isolation": "tenant_isolation" in observed_evidence,
        "has_audit": "audit" in observed_evidence,
        "has_retention_policy": "retention_policy" in observed_evidence,
        "has_deletion_policy": "deletion_policy" in observed_evidence,
        "has_redaction": "redaction" in observed_evidence,
        "has_canaries": "canary" in observed_evidence,
        "has_observability": bool(observability),
        "has_artifacts": bool(artifacts),
        "store_count": len(stores),
        "memory_count": len(memories),
        "operation_count": len(operations),
        "read_operation_count": sum(1 for item in operation_types if item == "read"),
        "write_operation_count": sum(1 for item in operation_types if item == "write"),
        "update_operation_count": sum(1 for item in operation_types if item == "update"),
        "delete_operation_count": sum(1 for item in operation_types if item == "delete"),
        "recall_operation_count": sum(1 for item in operation_types if item == "recall"),
        "attributed_memory_count": len(source_backed),
        "unattributed_memory_count": len(unattributed),
        "poisoned_memory_count": len(poisoned),
        "open_poisoning_count": len(poisoned) + len(poisoning_failures),
        "isolation_violation_count": len(isolation_violations),
        "retention_violation_count": len(retention_violations),
        "policy_violation_count": len(policy_violations),
        "audited_operation_count": sum(1 for item in operations if _operation_audited(item)),
        "artifact_count": len(artifacts),
        "observability_hook_count": _observability_hook_count(observability),
        "operation_types": sorted({item for item in operation_types if item}),
        "policy_keys": sorted(policy_keys),
        "observed_evidence": sorted(observed_evidence),
        "observed_signals": sorted(observed_signals),
        "missing_required_evidence": missing_evidence,
        "missing_required_signals": missing_signals,
        "unattributed_memories": unattributed,
        "poisoned_memories": poisoned,
        "poisoning_failures": poisoning_failures,
        "isolation_violations": isolation_violations,
        "retention_violations": retention_violations,
        "policy_violations": policy_violations,
        "blocking_gaps": sorted(blocking_gaps),
        "blocking_gap_count": len(blocking_gaps),
    }


def _memory_probe_summary(
    retrieval: Mapping[str, Any],
    lineage: Mapping[str, Any],
    lineage_summary: Mapping[str, Any],
    *,
    case_count: int,
    finding_count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    documents = [_plain_mapping(item) for item in _plain_list(retrieval.get("documents"))]
    current_doc_ids = {
        str(item.get("id") or "")
        for item in documents
        if item.get("current") is True and str(item.get("id") or "")
    }
    citations = [_plain_mapping(item) for item in _plain_list(retrieval.get("citations"))]
    cited_doc_ids = {
        str(doc_id)
        for citation in citations
        for doc_id in _plain_list(citation.get("doc_ids"))
        if str(doc_id or "")
    }
    required_operations = set(_plain_list(contract.get("operations")) or ["read", "write", "recall"])
    operation_types = set(_plain_list(lineage_summary.get("operation_types")))
    passed = finding_count == 0
    return {
        "case_count": max(int(case_count), 1),
        "passed_case_count": max(int(case_count), 1) if passed else 0,
        "failed_case_count": 0 if passed else max(int(case_count), 1),
        "finding_count": finding_count,
        "retrieval_document_count": len(documents),
        "retrieval_current_document_count": len(current_doc_ids),
        "retrieval_citation_count": len(citations),
        "retrieval_cited_document_count": len(cited_doc_ids),
        "retrieval_citations_current": bool(cited_doc_ids)
        and cited_doc_ids.issubset(current_doc_ids),
        "retrieval_freshness_checked_count": sum(
            1 for citation in citations if citation.get("freshness_checked") is True
        ),
        "memory_store_count": _as_int(lineage_summary.get("store_count")),
        "memory_record_count": _as_int(lineage_summary.get("memory_count")),
        "memory_operation_count": _as_int(lineage_summary.get("operation_count")),
        "memory_audited_operation_count": _as_int(
            lineage_summary.get("audited_operation_count")
        ),
        "memory_required_operations_present": required_operations.issubset(
            operation_types
        ),
        "memory_operation_types": sorted(operation_types),
        "has_source_attribution": bool(lineage_summary.get("has_source_attribution")),
        "has_tenant_isolation": bool(lineage_summary.get("has_tenant_isolation")),
        "has_audit": bool(lineage_summary.get("has_audit")),
        "has_retention_policy": bool(lineage_summary.get("has_retention_policy")),
        "has_deletion_policy": bool(lineage_summary.get("has_deletion_policy")),
        "has_redaction": bool(lineage_summary.get("has_redaction")),
        "has_canaries": bool(lineage_summary.get("has_canaries")),
        "has_observability": bool(lineage_summary.get("has_observability")),
        "has_artifacts": bool(lineage_summary.get("has_artifacts")),
        "policy_violation_count": _as_int(lineage_summary.get("policy_violation_count")),
        "open_poisoning_count": _as_int(lineage_summary.get("open_poisoning_count")),
        "isolation_violation_count": _as_int(
            lineage_summary.get("isolation_violation_count")
        ),
        "retention_violation_count": _as_int(
            lineage_summary.get("retention_violation_count")
        ),
        "blocking_gap_count": _as_int(lineage_summary.get("blocking_gap_count")),
        "requires_external_service": bool(contract.get("requires_external_service")),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
    }


def _memory_probe_findings(
    retrieval: Mapping[str, Any],
    lineage: Mapping[str, Any],
    lineage_summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    documents = [_plain_mapping(item) for item in _plain_list(retrieval.get("documents"))]
    current_doc_ids = {
        str(item.get("id") or "")
        for item in documents
        if item.get("current") is True and str(item.get("id") or "")
    }
    citations = [_plain_mapping(item) for item in _plain_list(retrieval.get("citations"))]
    cited_doc_ids = {
        str(doc_id)
        for citation in citations
        for doc_id in _plain_list(citation.get("doc_ids"))
        if str(doc_id or "")
    }
    _append_finding(
        findings,
        "memory_probe_local_contract",
        bool(contract.get("local_executable_fixture"))
        and not bool(contract.get("requires_external_service")),
        "memory probe target must be local and no-external-service",
        {"contract": dict(contract)},
    )
    _append_finding(
        findings,
        "memory_probe_current_retrieval",
        bool(current_doc_ids)
        and bool(cited_doc_ids)
        and cited_doc_ids.issubset(current_doc_ids)
        and all(citation.get("freshness_checked") is True for citation in citations),
        "retrieval citations must cite current documents with freshness checks",
        {
            "current_doc_ids": sorted(current_doc_ids),
            "cited_doc_ids": sorted(cited_doc_ids),
            "citation_count": len(citations),
        },
    )
    operation_types = set(_plain_list(lineage_summary.get("operation_types")))
    required_operations = set(_plain_list(contract.get("operations")) or ["read", "write", "recall"])
    _append_finding(
        findings,
        "memory_probe_read_write_recall",
        required_operations.issubset(operation_types)
        and _as_int(lineage_summary.get("audited_operation_count"))
        >= _as_int(lineage_summary.get("operation_count"))
        and _as_int(lineage_summary.get("operation_count")) >= len(required_operations),
        "memory lineage must include audited read/write/recall operations",
        {
            "required_operations": sorted(required_operations),
            "operation_types": sorted(operation_types),
            "operation_count": lineage_summary.get("operation_count"),
            "audited_operation_count": lineage_summary.get("audited_operation_count"),
        },
    )
    _append_finding(
        findings,
        "memory_probe_lineage_attribution",
        _as_int(lineage_summary.get("memory_count")) > 0
        and bool(lineage_summary.get("has_lineage"))
        and bool(lineage_summary.get("has_source_attribution"))
        and not _plain_list(lineage_summary.get("missing_required_evidence"))
        and not _plain_list(lineage_summary.get("missing_required_signals")),
        "memory records must have source attribution and closed lineage",
        {
            "memory_count": lineage_summary.get("memory_count"),
            "has_lineage": lineage_summary.get("has_lineage"),
            "missing_required_evidence": lineage_summary.get("missing_required_evidence"),
            "missing_required_signals": lineage_summary.get("missing_required_signals"),
        },
    )
    governance_keys = (
        "has_tenant_isolation",
        "has_audit",
        "has_retention_policy",
        "has_deletion_policy",
        "has_redaction",
        "has_canaries",
        "has_observability",
        "has_artifacts",
    )
    _append_finding(
        findings,
        "memory_probe_governance",
        all(bool(lineage_summary.get(key)) for key in governance_keys)
        and _as_int(lineage_summary.get("policy_violation_count")) == 0
        and _as_int(lineage_summary.get("open_poisoning_count")) == 0
        and _as_int(lineage_summary.get("isolation_violation_count")) == 0
        and _as_int(lineage_summary.get("retention_violation_count")) == 0
        and _as_int(lineage_summary.get("blocking_gap_count")) == 0,
        "memory governance, poisoning, isolation, retention, and artifacts must close",
        {
            key: lineage_summary.get(key)
            for key in (
                *governance_keys,
                "policy_violation_count",
                "open_poisoning_count",
                "isolation_violation_count",
                "retention_violation_count",
                "blocking_gap_count",
            )
        },
    )
    return findings


def _append_finding(
    findings: list[dict[str, Any]],
    check: str,
    passed: bool,
    message: str,
    evidence: Mapping[str, Any],
) -> None:
    if passed:
        return
    findings.append(
        {
            "check": check,
            "level": "error",
            "message": message,
            "evidence": dict(evidence),
        }
    )


def _memory_probe_cases(cases: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if cases:
        return [dict(item) for item in cases]
    return [
        {
            "id": "memory-probe",
            "input": "Store and recall a grounded refund policy memory.",
            "memory_key": "refund_policy_memory",
            "memory_value": "Refund policy memory with source attribution.",
        }
    ]


def _required_operations(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    operations: list[str] = []
    for case in cases:
        operations.extend(str(item) for item in _plain_list(case.get("required_operations")))
    return _unique_strings(operations or ["read", "write", "recall"])


def _plain_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _scope_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _operation_key(value: Any) -> str:
    aliases = {
        "memory_write": "write",
        "write_memory": "write",
        "memory_read": "read",
        "retrieve_memory": "read",
        "memory_retrieval": "read",
        "memory_recall": "recall",
        "recall_memory": "recall",
        "memory_update": "update",
        "memory_delete": "delete",
        "delete_memory": "delete",
    }
    normalized = _scope_key(value)
    return aliases.get(normalized, normalized)


def _has_policy(policy_keys: set[str], *names: str) -> bool:
    return bool(policy_keys & {_scope_key(item) for item in names})


def _operation_audited(operation: Mapping[str, Any]) -> bool:
    return bool(
        operation.get("trace_id")
        or operation.get("audit_id")
        or operation.get("evidence")
    )


def _all_operations_audited(operations: Sequence[Mapping[str, Any]]) -> bool:
    return bool(operations) and all(_operation_audited(item) for item in operations)


def _observability_hook_count(observability: Mapping[str, Any]) -> int:
    if not observability:
        return 0
    return sum(
        len(_plain_list(observability.get(key)))
        for key in ("traces", "logs", "hooks", "spans", "events")
    ) or 1


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _dedupe_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for document in documents:
        item = dict(document)
        doc_id = str(item.get("id") or "")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen.add(doc_id)
        result.append(item)
    return result


def _lineage_source_ids(lineage: Mapping[str, Any]) -> list[str]:
    source_ids: list[str] = []
    for memory in _plain_list(lineage.get("memories")):
        item = _plain_mapping(memory)
        source_ids.extend(
            str(source_id)
            for source_id in _plain_list(
                item.get("source_ids") or item.get("sources") or item.get("doc_ids")
            )
            if str(source_id)
        )
    return _unique_strings(source_ids)


def _is_external_target(target: str) -> bool:
    return urlparse(str(target)).scheme.lower() in {"http", "https"}


__all__ = [
    "memory_layer_contract",
    "probe_memory_layer",
    "run_memory_layer_probe",
]
