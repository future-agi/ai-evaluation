from __future__ import annotations


def _tool_calls() -> list[dict]:
    return [
        {
            "id": "framework_status",
            "name": "framework_trace_status",
            "arguments": {},
        }
    ]


class LocalLangChainRunnable:
    async def ainvoke(self, payload: dict) -> dict:
        assert payload["metadata"]["framework"] == "langchain"
        assert payload["metadata"]["cookbook"] == "multi-framework-simulation"
        return {
            "content": (
                "LangChain-style runnable completed the support workflow with "
                "framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "langchain"},
        }


class LocalLangGraphApp:
    async def ainvoke(self, payload: dict) -> dict:
        assert payload["metadata"]["framework"] == "langgraph"
        assert payload["metadata"]["cookbook"] == "multi-framework-simulation"
        return {
            "content": (
                "LangGraph-style graph completed the stateful refund workflow "
                "with framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "langgraph"},
        }


class LocalLlamaIndexChatEngine:
    async def achat(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "LlamaIndex-style chat engine completed retrieval-grounded "
                "answering with framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "llamaindex"},
        }


class LocalOpenAIAgentsRunner:
    def run(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "OpenAI Agents-style runner completed the handoff task with "
                "framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "openai_agents"},
        }


class LocalAutoGenAgent:
    def run(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "AutoGen-style agent chat completed the group task with "
                "framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "autogen"},
        }


class LocalCrewAICrew:
    def kickoff(self, payload: dict) -> dict:
        assert payload["metadata"]["framework"] == "crewai"
        assert payload["metadata"]["cookbook"] == "multi-framework-simulation"
        return {
            "content": (
                "CrewAI-style crew completed manager and worker coordination "
                "with framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "crewai"},
        }


class LocalPydanticAIAgent:
    def run(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "PydanticAI-style agent completed typed task execution with "
                "framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "pydantic_ai"},
        }


class LocalPipecatPipeline:
    def process(self, payload: dict) -> dict:
        assert payload["metadata"]["framework"] == "pipecat"
        assert payload["modality"] == "voice"
        return {
            "content": (
                "Pipecat-style voice pipeline completed the voice handoff with "
                "framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "pipecat"},
        }


class LocalLiveKitAgent:
    def respond(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "LiveKit-style realtime agent completed the voice room response "
                "with framework trace evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "livekit"},
        }


class LocalCustomRefundOrchestrator:
    def run(self, text: str) -> dict:
        assert text
        return {
            "content": (
                "Custom refund orchestrator queued the task, but it did not "
                "complete policy verification or emit framework tool evidence."
            ),
            "tool_calls": [],
            "metadata": {"framework_conformance": "incomplete"},
        }

    def execute_task(self, payload: dict) -> dict:
        assert payload["metadata"]["framework"] == "custom_refund_orchestrator"
        assert payload["metadata"]["cookbook"] == "multi-framework-simulation"
        return {
            "content": (
                "Custom refund orchestrator approved the task through the "
                "execute_task custom framework adapter with dict input and "
                "framework_trace_status tool evidence."
            ),
            "tool_calls": _tool_calls(),
            "metadata": {"framework_conformance": "custom_refund_orchestrator"},
        }


def build_langchain_agent() -> LocalLangChainRunnable:
    return LocalLangChainRunnable()


def build_langgraph_agent() -> LocalLangGraphApp:
    return LocalLangGraphApp()


def build_llamaindex_chat_engine() -> LocalLlamaIndexChatEngine:
    return LocalLlamaIndexChatEngine()


def build_openai_agents_runner() -> LocalOpenAIAgentsRunner:
    return LocalOpenAIAgentsRunner()


def build_autogen_agent() -> LocalAutoGenAgent:
    return LocalAutoGenAgent()


def build_crewai_crew() -> LocalCrewAICrew:
    return LocalCrewAICrew()


def build_pydantic_ai_agent() -> LocalPydanticAIAgent:
    return LocalPydanticAIAgent()


def build_pipecat_pipeline() -> LocalPipecatPipeline:
    return LocalPipecatPipeline()


def build_livekit_agent() -> LocalLiveKitAgent:
    return LocalLiveKitAgent()


def build_custom_refund_orchestrator() -> LocalCustomRefundOrchestrator:
    return LocalCustomRefundOrchestrator()
