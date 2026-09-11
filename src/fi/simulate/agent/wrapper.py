from typing import List, Dict, Union, Any, Optional, Literal
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


ArtifactType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "screenshot",
    "browser_dom",
    "file",
    "json",
    "trace",
]


class SimulationArtifact(BaseModel):
    """
    Modality-neutral artifact carried through a simulation.

    Use `uri` or `path` for large media, `data` for small inline payloads, and
    `metadata` for framework-specific details like sample rate, viewport, page
    URL, or image dimensions.
    """

    type: ArtifactType
    uri: Optional[str] = None
    path: Optional[str] = None
    data: Optional[Any] = None
    mime_type: Optional[str] = None
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationEvent(BaseModel):
    """Normalized event for tools, memory, browser/CUA actions, voice states, and framework spans."""

    type: str
    name: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInput(BaseModel):
    """
    Input data passed to the user's agent wrapper during a simulation step.
    """
    thread_id: str
    messages: List[Dict[str, Any]]  # Full conversation history: [{"role": "user", "content": "..."}]
    new_message: Optional[Dict[str, Any]] = None # The latest message to respond to
    
    # Metadata for execution context (useful for logging/debugging)
    execution_id: Optional[str] = None
    turn_index: Optional[int] = None
    scenario_name: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None
    situation: Optional[str] = None
    expected_outcome: Optional[str] = None
    modality: Optional[str] = None
    artifacts: List[SimulationArtifact] = Field(default_factory=list)
    events: List[SimulationEvent] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    """
    Standardized response from the user's agent.
    """
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_responses: Optional[List[Dict[str, Any]]] = None  # Tool role messages with results
    artifacts: List[SimulationArtifact] = Field(default_factory=list)
    events: List[SimulationEvent] = Field(default_factory=list)
    memory_updates: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentWrapper(ABC):
    """
    Base class for wrapping user agents to work with the simulation SDK.
    Users should implement the `call` method.
    """
    
    @abstractmethod
    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        """
        Process the input and return the agent's response.
        
        Args:
            input: The AgentInput object containing message history and context.
            
        Returns:
            A string (content only) or AgentResponse object.
        """
        pass
