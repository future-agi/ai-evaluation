from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from fi.alk import optimize, simulate


TASK_DESCRIPTION = (
    "Evaluate whether the selected orchestration-stack agent approves the refund "
    "with current policy grounding, recorded trace evidence, memory provenance, "
    "and critic-reviewed reconciliation."
)
EXPECTED_RESULT = (
    "The optimized stack approves refund, records trace evidence, uses current "
    "policy grounding, keeps memory provenance, and emits critic-reviewed "
    "reconciliation."
)
SUCCESS_CRITERIA = [
    "approves refund",
    "records trace",
    "current policy grounding",
    "memory provenance",
    "critic-reviewed reconciliation",
]


def _orchestration_example() -> Any:
    example_path = Path(__file__).with_name("sdk_orchestration_optimization.py")
    spec = importlib.util.spec_from_file_location(
        "sdk_orchestration_optimization_for_trinity_probe",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_probe_optimization(endpoint: str) -> dict[str, Any]:
    orchestration_example = _orchestration_example()
    return optimize.optimize_trinity_stack_probe(
        name="sdk-trinity-stack-probe-optimization",
        endpoint=endpoint,
        stack_candidates=[
            orchestration_example.weak_stack(),
            orchestration_example.strong_stack(),
        ],
        agent_candidates=[
            orchestration_example.weak_agent(),
            orchestration_example.strong_agent(),
        ],
        evaluation_config=orchestration_example.evaluation_config(),
        task_description=TASK_DESCRIPTION,
        expected_result=EXPECTED_RESULT,
        success_criteria=SUCCESS_CRITERIA,
        metadata={"cookbook": "sdk-trinity-stack-probe-optimization"},
    )


def build_manifest(endpoint: str) -> dict[str, Any]:
    return optimize.build_trinity_run_manifest_from_probe_optimization(
        build_probe_optimization(endpoint),
        name="sdk-trinity-stack-probe-promotion-run",
        metadata={"cookbook": "sdk-trinity-stack-probe-optimization"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    with _local_trinity_evaluation_hook() as endpoint:
        manifest_path = output.with_suffix(".manifest.json")
        simulate.write_manifest_file(build_manifest(endpoint), manifest_path)
        result = asyncio.run(simulate.run_manifest_file(manifest_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


@contextmanager
def _local_trinity_evaluation_hook() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TrinityHookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/eval/trinity-stack"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _TrinityHookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/eval/trinity-stack":
            self._write_json(404, {"content": "unknown evaluation hook"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._write_json(
                400,
                {
                    "content": "trinity hook received invalid json",
                    "success": False,
                    "error": "invalid json",
                },
            )
            return

        answer = " ".join(
            str(message.get("content") or "")
            for message in payload.get("case", {}).get("messages", [])
            if message.get("role") == "assistant"
        ).lower()
        required_terms = [
            "approves refund",
            "records trace",
            "current policy grounding",
            "memory provenance",
            "critic-reviewed reconciliation",
        ]
        matched = [term for term in required_terms if term in answer]
        leaked = "sk-" in answer or "api_key" in answer
        score = 1.0 if len(matched) == len(required_terms) and not leaked else 0.2
        self._write_json(
            200,
            {
                "metrics": [
                    {
                        "name": "external_task_quality",
                        "score": score,
                        "reason": (
                            "Local trinity judge accepted the selected stack agent."
                            if score == 1.0
                            else "Local trinity judge rejected incomplete stack evidence."
                        ),
                        "details": {
                            "matched_terms": matched,
                            "missing_terms": [
                                term for term in required_terms if term not in matched
                            ],
                            "secret_leak_detected": leaked,
                            "verdict": "accepted" if score == 1.0 else "rejected",
                        },
                    }
                ],
                "success": True,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-trinity-stack-probe-optimization.json"
    )
    run(destination)
