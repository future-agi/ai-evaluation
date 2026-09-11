from __future__ import annotations

import copy
import importlib
import sys
import traceback
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fi.simulate.environment import normalize_framework_import_manifest


def probe_framework_imports(
    targets: Sequence[str | Mapping[str, Any]] | str | Mapping[str, Any],
    *,
    name: str = "framework-import-runtime-probe",
    framework: str = "custom",
    adapter: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    observability: Mapping[str, Any] | None = None,
    artifacts: Iterable[Mapping[str, Any]] = (),
    required_sources: Iterable[str] = (),
    required_frameworks: Iterable[str] = (),
    required_export_types: Iterable[str] = (),
    required_signals: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe real Python imports and return framework-import manifest evidence.

    Targets are import specs such as ``"pkg.module"``, ``"pkg.module:factory"``,
    or mappings with ``module``, optional ``attribute``, optional ``invoke`` and
    optional ``args``/``kwargs`` fields. The probe imports modules for real and
    only invokes callables when a target explicitly sets ``invoke=True``.
    """

    target_specs = _target_specs(targets)
    if not target_specs:
        raise ValueError("targets must contain at least one import target")

    source_records = [
        _probe_target(item, index=index, default_framework=framework)
        for index, item in enumerate(target_specs, start=1)
    ]
    runtime_metadata = {
        "runtime_probe": {
            "target_count": len(source_records),
            "python_version": sys.version.split()[0],
            "policy": "import-only unless target.invoke is true",
        },
        **copy.deepcopy(dict(metadata or {})),
    }
    return normalize_framework_import_manifest(
        name=name,
        framework=framework,
        adapter=adapter,
        target=target,
        sources=source_records,
        observability=observability,
        artifacts=artifacts,
        required_sources=required_sources,
        required_frameworks=required_frameworks,
        required_export_types=required_export_types,
        required_signals=required_signals,
        metadata=runtime_metadata,
    )


def _target_specs(
    targets: Sequence[str | Mapping[str, Any]] | str | Mapping[str, Any],
) -> list[str | Mapping[str, Any]]:
    if isinstance(targets, (str, Mapping)):
        return [targets]
    return [item for item in targets if item not in (None, "", {}, [])]


def _probe_target(
    raw: str | Mapping[str, Any],
    *,
    index: int,
    default_framework: str,
) -> dict[str, Any]:
    spec = _normalize_target(raw, index=index, default_framework=default_framework)
    signals = {
        "framework_import",
        "runtime_import",
        "runtime_probe",
        "python_import",
        "module_import",
        *spec["signals"],
    }
    record: dict[str, Any] = {
        "id": spec["id"],
        "name": spec["name"],
        "framework": spec["framework"],
        "export_type": spec["export_type"],
        "module": spec["module"],
        "attribute": spec.get("attribute"),
        "status": "failed",
        "record_count": 0,
        "signals": sorted(signal for signal in signals if signal),
        "metadata": copy.deepcopy(spec["metadata"]),
    }
    if not spec["module"]:
        record["error"] = "target is missing module"
        record["signals"] = sorted({*record["signals"], "import_error"})
        return record

    try:
        module = importlib.import_module(spec["module"])
        record["path"] = str(getattr(module, "__file__", spec["module"]) or spec["module"])
        obj: Any = module
        if spec.get("attribute"):
            obj = _resolve_attribute(module, str(spec["attribute"]))
            signals.update({"attribute", "symbol", _key(str(spec["attribute"]))})
        if spec["require_callable"] or spec["invoke"]:
            if not callable(obj):
                raise TypeError(f"{spec['target']} is not callable")
            signals.add("callable")
        if spec["invoke"]:
            result = obj(*spec["args"], **spec["kwargs"])
            signals.update({"runtime_call", "call_succeeded"})
            if spec["has_expected_result"] and result != spec["expected_result"]:
                raise AssertionError(
                    "call result did not match expected_result "
                    f"({result!r} != {spec['expected_result']!r})"
                )
            record["call_result_type"] = type(result).__name__
        record["status"] = "passed"
        record["record_count"] = 1
        record["signals"] = sorted(signal for signal in signals if signal)
        record["object_type"] = type(obj).__name__
    except Exception as exc:  # noqa: BLE001 - probes report failures as evidence.
        record["error"] = str(exc) or type(exc).__name__
        record["exception_type"] = type(exc).__name__
        record["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        record["signals"] = sorted({*record["signals"], "import_error"})
    return record


def _normalize_target(
    raw: str | Mapping[str, Any],
    *,
    index: int,
    default_framework: str,
) -> dict[str, Any]:
    if isinstance(raw, str):
        item: dict[str, Any] = {"target": raw}
    else:
        item = copy.deepcopy(dict(raw))
    target_text = str(
        item.get("target")
        or item.get("path")
        or item.get("import")
        or item.get("module")
        or ""
    )
    parsed_module, parsed_attribute = _split_import_target(target_text)
    module = str(item.get("module") or parsed_module or "")
    callable_field = item.get("callable")
    attribute = (
        item.get("attribute")
        or item.get("attr")
        or item.get("symbol")
        or item.get("export")
        or (callable_field if isinstance(callable_field, str) else None)
        or parsed_attribute
    )
    target_ref = f"{module}:{attribute}" if attribute else module
    framework = str(item.get("framework") or item.get("runtime") or default_framework)
    export_type = str(item.get("export_type") or item.get("type") or "probe_suite")
    source_id = str(item.get("id") or item.get("name") or f"runtime_import_{index}")
    return {
        "id": _key(source_id),
        "name": str(item.get("name") or target_ref or source_id),
        "framework": framework,
        "export_type": export_type,
        "module": module,
        "attribute": str(attribute) if attribute else "",
        "target": target_ref,
        "invoke": _truthy(item.get("invoke") or item.get("call")),
        "require_callable": _truthy(callable_field) or _truthy(item.get("require_callable")),
        "args": list(_as_sequence(item.get("args"))),
        "kwargs": copy.deepcopy(dict(item.get("kwargs") or {})),
        "has_expected_result": "expected_result" in item or "expected_return" in item,
        "expected_result": item.get("expected_result", item.get("expected_return")),
        "signals": _key_list(item.get("signals")),
        "metadata": copy.deepcopy(dict(item.get("metadata") or {})),
    }


def _split_import_target(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if ":" not in text:
        return text, ""
    module, attribute = text.split(":", 1)
    return module.strip(), attribute.strip()


def _resolve_attribute(module: Any, attribute: str) -> Any:
    obj = module
    for part in attribute.split("."):
        if not part:
            continue
        obj = getattr(obj, part)
    return obj


def _as_sequence(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _key_list(value: Any) -> list[str]:
    return sorted({_key(item) for item in _as_sequence(value) if _key(item)})


def _key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
