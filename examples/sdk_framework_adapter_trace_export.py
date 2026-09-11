import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalTraceExportAgent"


@dataclass(frozen=True)
class FrameworkTraceExportResponse:
    content: str
    framework: str
    trace_export: dict[str, Any]
    adapter_required_signals: list[str]


def _otel_attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        otel_value = {"boolValue": value}
    elif isinstance(value, int):
        otel_value = {"intValue": str(value)}
    elif isinstance(value, float):
        otel_value = {"doubleValue": value}
    elif isinstance(value, list):
        otel_value = {
            "arrayValue": {
                "values": [
                    _otel_attr("item", item)["value"]
                    for item in value
                ]
            }
        }
    elif isinstance(value, dict):
        otel_value = {
            "kvlistValue": {
                "values": [
                    _otel_attr(str(item_key), item_value)
                    for item_key, item_value in value.items()
                ]
            }
        }
    else:
        otel_value = {"stringValue": str(value)}
    return {"key": key, "value": otel_value}


class LocalTraceExportAgent:
    """Local framework adapter that emits an OTLP-shaped trace export."""

    def run(self, text: str) -> str:
        assert text
        return "Weak trace response without normalized framework span evidence."

    async def execute_task(self, payload: dict[str, Any]) -> FrameworkTraceExportResponse:
        assert payload["metadata"]["framework"] == "langgraph"
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        trace_export = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _otel_attr("service.name", "local-langgraph-refund"),
                            _otel_attr("telemetry.sdk.name", "opentelemetry"),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "agent-learning.local-trace-export",
                                "version": "1.0.0",
                            },
                            "spans": [
                                {
                                    "traceId": trace_id,
                                    "spanId": "b7ad6b7169203331",
                                    "name": "langgraph refund model chat",
                                    "kind": "SPAN_KIND_INTERNAL",
                                    "startTimeUnixNano": "1710000000000000000",
                                    "endTimeUnixNano": "1710000000035000000",
                                    "attributes": [
                                        _otel_attr("gen_ai.operation.name", "chat"),
                                        _otel_attr("gen_ai.request.model", "local-refund-model"),
                                        _otel_attr("gen_ai.usage.input_tokens", 78),
                                        _otel_attr("gen_ai.usage.output_tokens", 24),
                                        _otel_attr("signals", ["model", "latency", "cost"]),
                                    ],
                                },
                                {
                                    "traceId": trace_id,
                                    "spanId": "c54f77e99b734a0d",
                                    "parentSpanId": "b7ad6b7169203331",
                                    "name": "tool call policy_lookup",
                                    "kind": "SPAN_KIND_INTERNAL",
                                    "startTimeUnixNano": "1710000000040000000",
                                    "endTimeUnixNano": "1710000000055000000",
                                    "attributes": [
                                        _otel_attr("gen_ai.operation.name", "execute_tool"),
                                        _otel_attr("gen_ai.tool.name", "policy_lookup"),
                                        _otel_attr(
                                            "gen_ai.tool.arguments",
                                            {"case_id": "refund-42", "market": "us"},
                                        ),
                                        _otel_attr(
                                            "gen_ai.tool.result",
                                            {
                                                "policy": "standard_refund",
                                                "eligible": True,
                                            },
                                        ),
                                        _otel_attr("signals", ["tool", "latency"]),
                                    ],
                                },
                                {
                                    "traceId": trace_id,
                                    "spanId": "f1d2d2f924e986ac",
                                    "parentSpanId": "b7ad6b7169203331",
                                    "name": "langgraph checkpoint refund decision",
                                    "kind": "SPAN_KIND_INTERNAL",
                                    "startTimeUnixNano": "1710000000060000000",
                                    "endTimeUnixNano": "1710000000064000000",
                                    "attributes": [
                                        _otel_attr("gen_ai.operation.name", "checkpoint"),
                                        _otel_attr("checkpoint.operation", "write"),
                                        _otel_attr("state.decision", "approved refund"),
                                        _otel_attr("signals", ["state", "checkpoint", "latency"]),
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        return FrameworkTraceExportResponse(
            content=(
                "Framework trace export adapter approved refund after model, "
                "policy_lookup tool, and checkpoint spans."
            ),
            framework="langgraph",
            trace_export=trace_export,
            adapter_required_signals=["model", "tool", "state", "latency", "cost"],
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-trace-export-run",
        framework="langgraph",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "trace-export-refund",
                "input": "Approve the refund with full framework trace evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["policy_lookup"],
                "required_events": [
                    "framework_trace_span",
                    "framework_trace",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_trace",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-trace-export"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_trace_export_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-trace-export.json"
    )
    run(destination)
