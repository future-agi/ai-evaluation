"""Voice environment plugin (canonical plan §3/§7.2-7.4).

This is the registry-facing wrapper that finally routes voice through the same
``SimulationRunner`` spine as chat. It does *not* reimplement the voice engine —
it hydrates the typed inputs from ``spec.environment.config`` and drives the
existing, working ``run_voice_simulation`` (LiveKit engine), returning the legacy
``TestReport`` the runner converts uniformly.

The voice config is secret-free by construction: providers are referenced by
``api_key_env`` / ``api_secret_env`` (env var *names*), never raw values, so the
whole config embeds inside the validated ``SimulationSpec`` without tripping
``_reject_resolved_secrets``. Secrets reach the child process through the
environment, resolved by the runner activity.

Distinct from :class:`fi.simulate.environment.VoiceEnvironment`, which is the
deterministic *replay* adapter (a sync ``EnvironmentAdapter`` fixture) — a
different role, kept under its existing public name.
"""

from __future__ import annotations

import inspect
import logging

from fi.simulate.environments.base import EnvironmentManifest
from fi.simulate.registry import register_environment
from fi.simulate.runtime.capabilities import EndpointCapabilities
from fi.simulate.runtime.failures import FailureStage, SimulationFailure
from fi.simulate.runtime.report import SimulationReport
from fi.simulate.runtime.run import RunStatus, TestCaseStatus
from fi.simulate.simulation.models import TestReport

_logger = logging.getLogger(__name__)

# Kwargs VoiceEnvironmentPlugin.run already binds explicitly when it calls
# run_voice_simulation (see below) — reserved so a same-named key surviving in
# hosted voice.params cannot collide with them at the call site.
_PLUGIN_OWNED = frozenset(
    {
        "agent_definition",
        "livekit_runtime",
        "scenario",
        "simulator",
        "simulation_run_id",
        "on_case_complete",
        "on_case_start",
    }
)


def _filter_hosted_voice_params(params: dict, run_voice_simulation) -> dict:
    """Drop platform-sent keys run_voice_simulation can't bind: it is
    keyword-only with a closed parameter list, so one stray key would raise
    TypeError at hydration and kill the whole hosted run. (A **kwargs sink, if
    one is ever added, accepts everything, so nothing is dropped.)"""
    parameters = inspect.signature(run_voice_simulation).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(params)
    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    } - _PLUGIN_OWNED
    dropped = set(params) - accepted
    if dropped:
        _logger.warning("hosted_voice_params_ignored", extra={"keys": sorted(dropped)})
    return {key: value for key, value in params.items() if key in accepted}


@register_environment("voice")
class VoiceEnvironmentPlugin:
    manifest = EnvironmentManifest(
        name="voice",
        world_kinds=["voice_telephony", "voice"],
        capabilities=EndpointCapabilities(
            audio=True,
            streaming=True,
            interruption=True,
            recording=True,
            transcript_events=True,
            web_rtc=True,
        ),
    )

    async def run(
        self,
        spec,
        *,
        target=None,
        artifacts=None,
        events=None,
        environment=None,
        auto_execute_tools: bool = True,
        stop_when=None,
        agent_wrapper_kwargs=None,
        on_case_complete=None,
        on_case_start=None,
    ) -> TestReport:
        from fi.simulate import voice as voice_api
        from fi.simulate.agent.definition import (
            AgentDefinition,
            LiveKitSimulatorRuntime,
            SimulatorAgentDefinition,
        )

        config = dict(spec.environment.config or {})
        agent_definition = AgentDefinition.model_validate(config["agent_definition"])
        livekit_runtime = (
            LiveKitSimulatorRuntime.model_validate(config["livekit_runtime"])
            if config.get("livekit_runtime")
            else None
        )
        simulator = (
            SimulatorAgentDefinition.model_validate(config["simulator"])
            if config.get("simulator")
            else None
        )
        params = _filter_hosted_voice_params(
            dict(config.get("params") or {}), voice_api.run_voice_simulation
        )

        # When streaming, score the case's goal (if any) BEFORE the case is
        # submitted, so the streamed payload carries ``goal_machine`` metadata
        # identically to the post-run report — then forward to the runner's sink
        # callback. Absent streaming, the report-level pass below is authoritative.
        streamed_callback = None
        if on_case_complete is not None:
            scenario = spec.scenario

            async def streamed_callback(index, case):  # noqa: ANN001
                self._attach_goal_to_case(scenario, case)
                await on_case_complete(index, case)

        report = await voice_api.run_voice_simulation(
            agent_definition=agent_definition,
            livekit_runtime=livekit_runtime,
            scenario=spec.scenario,
            simulator=simulator,
            simulation_run_id=spec.run_id,
            on_case_complete=streamed_callback,
            on_case_start=on_case_start,
            **params,
        )
        self._attach_goal_machine(spec.scenario, report)
        return report

    @classmethod
    def _attach_goal_machine(cls, scenario, report: TestReport) -> None:
        """Voice world-contract (plan §1.9, settle-only). A declared
        ``scenario.goal`` is scored over each case transcript at episode end and
        attached as metadata — the same idiom chat uses. No declared goal ⇒ no-op
        (byte-identical). Voice has no per-turn hook, so this scores the run; it
        does not (and must not) early-stop a live call or fail the run.

        Idempotent: overwrites ``goal_machine`` with the same value the streaming
        path already wrote, so streamed and reconciled cases stay identical.
        """
        for case in report.results:
            cls._attach_goal_to_case(scenario, case)

    @staticmethod
    def _attach_goal_to_case(scenario, case) -> None:
        """Score ``scenario.goal`` over one case's transcript, in place."""
        goal = getattr(scenario, "goal", None)
        if goal is None:
            return
        from fi.simulate.simulation import goal_machine

        verification = getattr(scenario, "verification", None)
        settle = goal_machine.evaluate_settle(
            goal,
            verification,
            environment_state={},
            world_status={},
            messages=getattr(case, "messages", None) or [],
        )
        case.metadata["goal_machine"] = {
            "states_reached": settle.get("states_reached", []),
            "stop_reason": None,
            "checks": settle.get("checks", []),
        }

    def finalize_run_status(self, report: SimulationReport) -> SimulationReport:
        """A voice run whose only case(s) failed is a failed job, not COMPLETED.

        ``from_legacy`` carries per-case status but the overall status is the
        environment's to decide (plan §3: environments enforce terminal
        conditions). Chat keeps COMPLETED; only voice downgrades.
        """
        failed = [
            case
            for case in report.test_cases
            if case.status is not TestCaseStatus.COMPLETED
        ]
        if not report.test_cases or len(failed) != len(report.test_cases):
            return report
        failure = failed[0].failure or SimulationFailure(
            stage=FailureStage.RUNNING,
            code="voice_run_failed",
            message="voice simulation failed",
            retryable=False,
        )
        return SimulationReport.model_validate(
            {
                **report.model_dump(exclude={"report_hash"}),
                "status": RunStatus.FAILED.value,
                "failure": failure.model_dump(),
            }
        )


__all__ = ["VoiceEnvironmentPlugin"]
