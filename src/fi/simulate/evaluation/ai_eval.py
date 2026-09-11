from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import os
import base64
import json

from fi.simulate.simulation.models import TestReport


def evaluate_agent_report(
    report: TestReport,
    *,
    config: Mapping[str, Any] | None = None,
    threshold: float = 0.7,
    attach: bool = True,
) -> Any:
    """
    Evaluate a simulation report locally with Agent Learning eval metrics.

    This is the no-cloud path for the trinity loop:
    Agent Learning simulation captures messages/tool calls/events/artifacts,
    Agent Learning evals score the agent trajectory and pentest signals, and
    Agent Learning optimization can optimize against the attached numeric
    scores.

    Returns the AgentReportEvaluation object. When `attach=True`,
    each TestCaseResult receives an `evaluation["agent_report"]` payload and
    aggregate summary is copied into `result.metadata["agent_report_summary"]`.
    """

    try:
        from fi.evals.metrics.agents import evaluate_agent_report as evaluate
    except Exception as e:  # pragma: no cover - import error clarity
        raise RuntimeError(
            "Agent Learning Kit eval metrics are required. "
            "Reinstall `agent-learning-kit`."
        ) from e

    evaluation = evaluate(report, config=dict(config or {}), threshold=threshold)
    if attach:
        _attach_agent_report_evaluation(report, evaluation)
    return evaluation


def evaluate_report(
    report: TestReport,
    *,
    eval_templates: Iterable[str] | None = ("task_completion", "tone", "is_helpful"),
    eval_specs: Sequence[dict] | None = None,
    model_name: str = "turing_flash",
    api_key: str | None = None,
    secret_key: str | None = None,
    extra_inputs: Mapping[str, str] | None = None,
) -> TestReport:
    """
    Evaluate each test case transcript using Agent Learning evals.

    - Templates like "task_completion" will receive input and output fields
      mapped from persona and transcript.
    - "tone" will receive the whole transcript as input.

    Docs: https://docs.futureagi.com/future-agi/get-started/evaluation/running-your-first-eval#evaluate-using-sdk
    """

    try:
        from fi.evals import Evaluator
    except Exception as e:  # pragma: no cover - import error clarity
        raise RuntimeError(
            "Agent Learning Kit eval engine is required. Reinstall `agent-learning-kit`."
        ) from e

    evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key)

    for result in report.results:
        persona = result.persona
        transcript = result.transcript

        scores: dict[str, dict] = {}

        def resolve_source(key: str) -> str | None:
            if key == "transcript":
                return transcript
            if key == "messages":
                return json.dumps(result.messages, default=str)
            if key == "tool_calls":
                return json.dumps(result.tool_calls, default=str)
            if key == "artifacts":
                return json.dumps([_model_to_dict(item) for item in result.artifacts], default=str)
            if key == "events":
                return json.dumps([_model_to_dict(item) for item in result.events], default=str)
            if key == "metadata":
                return json.dumps(result.metadata, default=str)
            if key == "persona":
                return json.dumps(persona.persona, default=str)
            if key == "persona.situation":
                return persona.situation
            if key == "persona.outcome":
                return persona.outcome
            if key == "audio_input_path":
                val = getattr(result, "audio_input_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            if key == "audio_output_path":
                val = getattr(result, "audio_output_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            if key == "audio_combined_path":
                val = getattr(result, "audio_combined_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            return None

        def _encode_audio_inputs(inputs: dict[str, str]) -> dict[str, str]:
            """Strict encoding: if a value is a local audio file path, replace that value with base64.

            - Never rename keys or add aliases.
            - Do not add extra fields (no audio_mime or data URI).
            """
            audio_exts = {".wav", ".ogg", ".mp3", ".m4a", ".flac", ".aac"}
            for k, v in list(inputs.items()):
                if isinstance(v, str) and os.path.exists(v):
                    _, ext = os.path.splitext(v.lower())
                    if ext in audio_exts:
                        try:
                            with open(v, "rb") as f:
                                data = f.read()
                            inputs[k] = base64.b64encode(data).decode("ascii")
                        except Exception:
                            # Leave as-is on read failure
                            pass
            return inputs

        # If eval_specs provided, use explicit mappings per template
        if eval_specs:
            for spec in eval_specs:
                template = spec.get("template")
                mapping: Mapping[str, str] = spec.get("map", {})  # desired_input_key -> source_key
                if not template:
                    continue
                inputs: dict[str, str] = {}
                for dest, source in mapping.items():
                    val = resolve_source(source)
                    if val is not None:
                        inputs[dest] = val
                if extra_inputs:
                    inputs.update(extra_inputs)
                inputs = _encode_audio_inputs(inputs)
                try:
                    ev = evaluator.evaluate(eval_templates=template, inputs=inputs, model_name=model_name)
                    item = ev.eval_results[0] if ev and getattr(ev, "eval_results", None) else None
                    scores[template] = {
                        "output": getattr(item, "output", None),
                        "reason": getattr(item, "reason", None),
                        "score": getattr(item, "score", None),
                    }
                except Exception as e:
                    scores[template] = {"error": str(e), "inputs": inputs}
        else:
            # Fallback: simple built-ins by template name
            for template in (eval_templates or []):
                inputs: dict[str, str] = {}
                if template == "tone":
                    inputs = {"input": transcript}
                elif template == "task_completion":
                    inputs = {"input": persona.situation, "output": transcript}
                elif template == "is_helpful":
                    inputs = {"input": transcript}
                else:
                    inputs = {"input": transcript}

                if extra_inputs:
                    inputs.update(extra_inputs)
                inputs = _encode_audio_inputs(inputs)

                try:
                    ev = evaluator.evaluate(eval_templates=template, inputs=inputs, model_name=model_name)
                    item = ev.eval_results[0] if ev and getattr(ev, "eval_results", None) else None
                    scores[template] = {
                        "output": getattr(item, "output", None),
                        "reason": getattr(item, "reason", None),
                        "score": getattr(item, "score", None),
                    }
                except Exception as e:
                    scores[template] = {"error": str(e)}

        result.evaluation = scores

    return report


def _model_to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [_model_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_model_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_to_dict(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _model_to_dict(item) for key, item in vars(value).items()}
    return value


def _attach_agent_report_evaluation(report: TestReport, evaluation: Any) -> None:
    cases = getattr(evaluation, "cases", []) or []
    summary = _model_to_dict(getattr(evaluation, "summary", {}))
    aggregate = {
        "score": getattr(evaluation, "score", None),
        "passed": getattr(evaluation, "passed", None),
        "threshold": getattr(evaluation, "threshold", None),
        "summary": summary,
    }
    for index, result in enumerate(report.results):
        case = cases[index] if index < len(cases) else None
        metrics = getattr(case, "metrics", []) if case is not None else []
        payload = {
            **aggregate,
            "case_score": getattr(case, "score", None) if case is not None else None,
            "case_passed": getattr(case, "passed", None) if case is not None else None,
            "metrics": [_model_to_dict(metric) for metric in metrics],
            "findings": _model_to_dict(getattr(case, "findings", [])) if case is not None else [],
        }
        result.evaluation = dict(result.evaluation or {})
        result.evaluation["agent_report"] = payload
        result.metadata = dict(result.metadata or {})
        result.metadata["agent_report_summary"] = aggregate
