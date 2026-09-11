
from abc import ABC, abstractmethod
from typing import Optional
from fi.simulate.agent.definition import AgentDefinition, SimulatorAgentDefinition
from fi.simulate.simulation.models import Scenario, TestReport

class BaseEngine(ABC):
    """
    Abstract base class for simulation engines.
    """
    
    @abstractmethod
    async def run(
        self, 
        agent_definition: Optional[AgentDefinition] = None,
        scenario: Optional[Scenario] = None,
        simulator: Optional[SimulatorAgentDefinition] = None,
        **kwargs
    ) -> TestReport:
        pass

