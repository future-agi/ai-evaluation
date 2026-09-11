from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fi.simulate.agent.wrapper import AgentWrapper, SimulationArtifact, SimulationEvent
from fi.simulate.environment import EnvironmentAdapter
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    EvidencePolicy,
    SimulationSpec,
    SimulatorPolicySpec,
    new_run_id,
)
from fi.simulate.results.base import ResultSink
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.simulation.engines.base import BaseEngine
from fi.simulate.simulation.models import Persona, Scenario, TestReport
from fi.simulate.simulation.synthetic import SyntheticDataGenerator


class LocalTextEngine(BaseEngine):
    async def run(
        self,
        *,
        scenario: Scenario | None = None,
        agent_callback: Callable[..., Any] | AgentWrapper | Any | None = None,
        topic: str | None = None,
        num_scenarios: int = 3,
        max_turns: int = 6,
        min_turns: int = 2,
        attacks: Iterable[str] | None = None,
        modality: str = "text",
        artifacts: list[SimulationArtifact | dict[str, Any]] | None = None,
        events: list[SimulationEvent | dict[str, Any]] | None = None,
        environment: EnvironmentAdapter | Iterable[EnvironmentAdapter] | None = None,
        auto_execute_tools: bool = True,
        stop_when: Callable[[list[dict[str, Any]], Persona], bool] | None = None,
        agent_wrapper_kwargs: dict[str, Any] | None = None,
        result_sink: ResultSink | None = None,
        **kwargs: Any,
    ) -> TestReport:
        if agent_callback is None:
            raise ValueError("LocalTextEngine requires an 'agent_callback'.")
        if scenario is None:
            if not topic:
                raise ValueError("LocalTextEngine requires either 'scenario' or 'topic'.")
            scenario = SyntheticDataGenerator().generate(
                topic,
                num_personas=num_scenarios,
                seed=kwargs.get("seed"),
                task=kwargs.get("task", topic),
                include_adversarial=kwargs.get("include_adversarial", True),
                include_edge_cases=kwargs.get("include_edge_cases", True),
            )
        config: dict[str, Any] = {
            "max_turns": max_turns,
            "min_turns": min_turns,
            "modality": modality,
        }
        if attacks is not None:
            config["attacks"] = list(attacks)
        spec = SimulationSpec(
            run_id=str(kwargs.get("run_id") or new_run_id()),
            environment=EnvironmentSpec(
                adapter="chat",
                world_kind="conversation",
                config=config,
            ),
            target=AgentEndpointSpec(adapter="callable"),
            simulator=SimulatorPolicySpec(adapter="synthetic_user"),
            scenario=scenario,
            evidence=EvidencePolicy(),
        )
        report = await SimulationRunner().run(
            spec,
            target=agent_callback,
            result_sink=result_sink,
            artifacts=artifacts,
            events=events,
            environment=environment,
            auto_execute_tools=auto_execute_tools,
            stop_when=stop_when,
            agent_wrapper_kwargs=agent_wrapper_kwargs,
        )
        if report.failure is not None:
            raise RuntimeError(report.failure.message)
        return report.to_legacy(include_runtime_metadata=False)
