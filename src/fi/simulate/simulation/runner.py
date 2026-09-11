from typing import Optional, Callable
import os

from fi.simulate.agent.definition import (
    AgentDefinition,
    LiveKitSimulatorRuntime,
    SimulatorAgentDefinition,
)
from fi.simulate.simulation.models import Scenario, TestReport
from fi.simulate.simulation.engines import (
    LiveKitEngine,
    BaseEngine,
    CloudEngine,
    LocalTextEngine,
)


class TestRunner:
    """
    Main entry point for running agent simulations.

    Supports three execution modes:
    1. Local mode (LiveKit): Uses LiveKit to connect to deployed agents
    2. Cloud mode (Backend API): Uses Future AGI backend for orchestrated testing
    3. Local text mode: Runs a self-contained text simulator against an agent callback

    The mode is automatically determined based on the arguments provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize the TestRunner.

        Args:
            api_key: Optional API key for cloud mode. If not provided, will check FI_API_KEY env var.
            secret_key: Optional Secret key for cloud mode. If not provided, will check FI_SECRET_KEY env var.
            api_url: Optional API URL for cloud mode. If not provided, will check FI_BASE_URL env var.
        """
        self.api_key = api_key or os.environ.get("FI_API_KEY")
        self.secret_key = secret_key or os.environ.get("FI_SECRET_KEY")
        self.api_url = api_url or os.environ.get("FI_BASE_URL")
        self._engine: Optional[BaseEngine] = None

    async def run_test(
        self,
        # --- Local Mode Arguments (LiveKit) ---
        agent_definition: Optional[AgentDefinition] = None,
        livekit_runtime: Optional[LiveKitSimulatorRuntime] = None,
        scenario: Optional[Scenario] = None,
        simulator: Optional[SimulatorAgentDefinition] = None,
        # --- Cloud Mode Arguments (Backend API) ---
        run_id: Optional[str] = None,
        run_test_name: Optional[str] = None,
        agent_callback: Optional[Callable] = None,
        # --- Shared Arguments ---
        num_scenarios: int = 1,
        topic: Optional[str] = None,
        simulation_run_id: Optional[str] = None,
        record_audio: bool = False,
        recorder_sample_rate: int = 8000,
        recorder_join_delay: float = 0.2,
        min_turn_messages: int = 8,
        max_seconds: float = 45.0,
        **kwargs,
    ) -> TestReport:
        """
        Run a test simulation.

        Mode is determined by arguments:
        - If `run_id` or `run_test_name` is provided → Cloud mode (Backend API)
        - If `agent_definition` is provided → Local mode (LiveKit)

        Args:
            agent_definition: Target-agent configuration for local mode
            livekit_runtime: FutureAGI LiveKit simulator runtime
            scenario: Test scenario for local mode
            simulator: Simulator configuration for local mode
            run_id: Run ID from platform for cloud mode
            run_test_name: Run test name (alternative to run_id) - will fetch ID from backend
            agent_callback: User's agent function to wrap for cloud mode
            num_scenarios: Number of scenarios to generate (local mode only)
            topic: Topic for scenario generation (local mode only)
            simulation_run_id: Optional stable run ID for local LiveKit mode
            record_audio: Whether to record audio
            recorder_sample_rate: Audio sample rate
            recorder_join_delay: Delay before recorder joins
            min_turn_messages: Minimum turn messages
            max_seconds: Maximum test duration
            **kwargs: Additional arguments passed to engine

        Returns:
            TestReport with results from all test cases
        """
        # Dispatch to appropriate engine
        if run_id is not None or run_test_name is not None:
            # Cloud mode - Use CloudEngine
            timeout = kwargs.pop("timeout", 120.0)  # Default 120s for LLM operations
            engine = CloudEngine(
                self.api_key, self.secret_key, self.api_url, timeout=timeout
            )
            return await engine.run(
                run_id=run_id,
                run_test_name=run_test_name,
                agent_callback=agent_callback,
                **kwargs,
            )
        elif agent_callback is not None:
            # Local text mode - no backend, LiveKit, or model dependency required
            engine = LocalTextEngine()
            return await engine.run(
                scenario=scenario,
                agent_callback=agent_callback,
                num_scenarios=num_scenarios,
                topic=topic,
                run_id=simulation_run_id,
                **kwargs,
            )

        elif agent_definition is not None:
            # Local mode - use LiveKit engine
            if LiveKitEngine is None:
                raise ImportError(
                    "LiveKit mode requires the LiveKit dependency, but it is not available in this environment. "
                    "Install it (and compatible plugins) then retry. Cloud mode (run_id/run_test_name) does not "
                    "require LiveKit."
                )
            engine = LiveKitEngine()
            return await engine.run(
                agent_definition=agent_definition,
                livekit_runtime=livekit_runtime,
                scenario=scenario,
                simulator=simulator,
                num_scenarios=num_scenarios,
                topic=topic,
                record_audio=record_audio,
                recorder_sample_rate=recorder_sample_rate,
                recorder_join_delay=recorder_join_delay,
                min_turn_messages=min_turn_messages,
                max_seconds=max_seconds,
                run_id=simulation_run_id,
                **kwargs,
            )
        else:
            raise ValueError(
                "Must provide either 'agent_definition' (Local/LiveKit mode) "
                "'agent_callback' with a scenario/topic (Local text mode), "
                "or 'run_id'/'run_test_name' (Cloud/Backend API mode)."
            )
