from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .evidence import score_simulation_evidence
from .targets import AgentCandidate, CandidateEvaluation


class SimulationEvaluator:
    """
    Evaluates an AgentCandidate through simulate-sdk and optionally ai-evaluation.

    The bridge is dependency-light by design:
    - pass fake runner/evaluate functions in tests,
    - or install `agent-simulate` and `ai-evaluation` for real runs.
    """

    def __init__(
        self,
        *,
        agent_factory: Optional[Callable[[AgentCandidate], Any]] = None,
        scenario: Any = None,
        topic: Optional[str] = None,
        runner: Any = None,
        runner_cls: Any = None,
        runner_kwargs: Optional[
            Mapping[str, Any] | Callable[[AgentCandidate], Mapping[str, Any]]
        ] = None,
        eval_specs: Optional[Iterable[Dict[str, Any]]] = None,
        eval_templates: Optional[Iterable[str]] = None,
        evaluate_report_fn: Optional[Callable[..., Any]] = None,
        evaluate_report_kwargs: Optional[Dict[str, Any]] = None,
        agent_report_config: Optional[
            Mapping[str, Any] | Callable[[AgentCandidate], Mapping[str, Any]]
        ] = None,
        agent_report_threshold: float = 0.7,
        use_agent_report_evaluator: bool = False,
        report_scorer: Optional[Callable[[Any, AgentCandidate], float]] = None,
        evidence_scorer_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.scenario = scenario
        self.topic = topic
        self.runner = runner
        self.runner_cls = runner_cls
        self.runner_kwargs = runner_kwargs or {}
        self.eval_specs = list(eval_specs) if eval_specs is not None else None
        self.eval_templates = list(eval_templates) if eval_templates is not None else None
        self.evaluate_report_fn = evaluate_report_fn
        self.evaluate_report_kwargs = evaluate_report_kwargs or {}
        self.agent_report_config = agent_report_config
        self.agent_report_threshold = agent_report_threshold
        self.use_agent_report_evaluator = use_agent_report_evaluator
        self.report_scorer = report_scorer
        self.evidence_scorer_config = (
            dict(evidence_scorer_config) if evidence_scorer_config is not None else None
        )

    def evaluate_candidate(self, candidate: AgentCandidate) -> CandidateEvaluation:
        agent = self._build_agent(candidate)
        runner = self._get_runner()
        run_kwargs = {
            **self._runner_kwargs(candidate),
            **candidate.config.get("simulation", {}),
        }
        if self.scenario is not None:
            run_kwargs["scenario"] = self.scenario
        if self.topic is not None:
            run_kwargs["topic"] = self.topic

        report = _run_sync(
            runner.run_test(
                agent_callback=agent,
                **run_kwargs,
            )
        )

        if self.eval_specs is not None or self.eval_templates is not None:
            report = self._evaluate_report(report)

        agent_report_evaluation = None
        if self.use_agent_report_evaluator or self.agent_report_config is not None:
            agent_report_evaluation = self._evaluate_agent_report(report, candidate)

        evidence_evaluation = None
        if self.evidence_scorer_config is not None and agent_report_evaluation is None:
            evidence_evaluation = score_simulation_evidence(
                report,
                candidate=candidate,
                config=self.evidence_scorer_config,
            )

        score_source = (
            agent_report_evaluation
            if agent_report_evaluation is not None
            else evidence_evaluation
            if evidence_evaluation is not None
            else report
        )
        score = self._score_report(score_source, candidate)
        metadata = {"source": "simulate-sdk"}
        if agent_report_evaluation is not None:
            metadata["agent_report_evaluation"] = _dump_model(agent_report_evaluation)
        if evidence_evaluation is not None:
            metadata["simulation_evidence_score"] = evidence_evaluation.metadata.get(
                "simulation_evidence_score"
            )
        return CandidateEvaluation(
            candidate=candidate,
            score=score,
            report=report,
            metadata=metadata,
        )

    def _build_agent(self, candidate: AgentCandidate) -> Any:
        if self.agent_factory is not None:
            return self.agent_factory(candidate)
        agent = candidate.config.get("agent_callback") or candidate.config.get("agent")
        if agent is None:
            raise ValueError(
                "SimulationEvaluator needs an agent_factory or candidate.config['agent_callback']."
            )
        return agent

    def _get_runner(self) -> Any:
        if self.runner is not None:
            return self.runner
        if self.runner_cls is not None:
            return self.runner_cls()
        try:
            from fi.simulate import TestRunner
        except Exception as exc:  # pragma: no cover - import clarity
            raise RuntimeError(
                "agent-simulate is required for SimulationEvaluator unless runner/runner_cls is provided."
            ) from exc
        return TestRunner()

    def _evaluate_report(self, report: Any) -> Any:
        evaluate_report = self.evaluate_report_fn
        if evaluate_report is None:
            try:
                from fi.simulate.evaluation import evaluate_report as imported
            except Exception as exc:  # pragma: no cover - import clarity
                raise RuntimeError(
                    "simulate-sdk evaluate_report is required unless evaluate_report_fn is provided."
                ) from exc
            evaluate_report = imported

        kwargs = dict(self.evaluate_report_kwargs)
        if self.eval_specs is not None:
            kwargs["eval_specs"] = self.eval_specs
        if self.eval_templates is not None:
            kwargs["eval_templates"] = self.eval_templates
        return evaluate_report(report, **kwargs)

    def _evaluate_agent_report(
        self,
        report: Any,
        candidate: AgentCandidate,
    ) -> Any:
        config = self._agent_report_config(candidate)
        try:
            from fi.simulate.evaluation import evaluate_agent_report
        except Exception:
            try:
                from fi.evals.metrics.agents import evaluate_agent_report
            except Exception as exc:  # pragma: no cover - import clarity
                raise RuntimeError(
                    "SimulationEvaluator local agent report scoring requires "
                    "simulate-sdk with evaluate_agent_report or ai-evaluation>=1.1."
                ) from exc

        return evaluate_agent_report(
            report,
            config=config,
            threshold=self.agent_report_threshold,
        )

    def _agent_report_config(self, candidate: AgentCandidate) -> Dict[str, Any]:
        config = self.agent_report_config
        if callable(config):
            return dict(config(candidate))
        return dict(config or {})

    def _runner_kwargs(self, candidate: AgentCandidate) -> Dict[str, Any]:
        config = self.runner_kwargs
        if callable(config):
            return dict(config(candidate))
        return dict(config or {})

    def _score_report(self, report: Any, candidate: AgentCandidate) -> float:
        if self.report_scorer is not None:
            return float(self.report_scorer(report, candidate))
        direct_score = _coerce_score(getattr(report, "score", None))
        if direct_score is not None:
            return direct_score
        scores = list(_iter_report_scores(report))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


def _run_sync(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "SimulationEvaluator.evaluate_candidate() was called from a running event loop. "
        "Pass a synchronous runner or call it outside the event loop for now."
    )


def _iter_report_scores(report: Any):
    for result in getattr(report, "results", []) or []:
        evaluation = getattr(result, "evaluation", None)
        if not isinstance(evaluation, dict):
            continue
        for item in evaluation.values():
            if isinstance(item, dict):
                for key in ("score", "output", "value"):
                    value = item.get(key)
                    score = _coerce_score(value)
                    if score is not None:
                        yield score
                        break
            else:
                score = _coerce_score(item)
                if score is not None:
                    yield score


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pass", "passed", "true", "yes"}:
            return 1.0
        if lowered in {"fail", "failed", "false", "no"}:
            return 0.0
        try:
            return max(0.0, min(1.0, float(lowered)))
        except ValueError:
            return None
    return None


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value
