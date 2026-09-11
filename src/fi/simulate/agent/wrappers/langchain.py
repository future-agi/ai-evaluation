from typing import Any, Union
from fi.simulate.agent.wrapper import AgentWrapper, AgentInput, AgentResponse

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    # LangChain is optional. Fallback message classes keep wrappers and tests
    # usable for mocked/local simulations without installing langchain-core.
    class _FallbackMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage(_FallbackMessage):
        pass

    class AIMessage(_FallbackMessage):
        pass

    class SystemMessage(_FallbackMessage):
        pass

    LANGCHAIN_INSTALLED = False
else:
    LANGCHAIN_INSTALLED = True

class LangChainAgentWrapper(AgentWrapper):
    """
    Wrapper for LangChain Runnable or Chain agents.
    """
    def __init__(self, agent: Any, system_prompt: str = None):
        """
        Args:
            agent: A LangChain Runnable (chain, agent executor) that accepts input.
                   It is expected to accept a dictionary with "messages" or "input".
            system_prompt: Optional system prompt to prepend to message history.
        """
        self.agent = agent
        self.system_prompt = system_prompt

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:

        # Convert history to LangChain messages
        lc_messages = []
        
        if self.system_prompt:
            lc_messages.append(SystemMessage(content=self.system_prompt))
            
        for msg in input.messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
        
        # Invoke the agent
        # We try standard keys used in LC agents
        inputs = {
            "messages": lc_messages,
            "input": input.new_message.get("content") if input.new_message else "",
            "chat_history": lc_messages[:-1] if lc_messages else []
        }
        
        # Support both ainvoke and invoke
        if hasattr(self.agent, "ainvoke"):
            response = await self.agent.ainvoke(inputs)
        else:
            response = self.agent.invoke(inputs)
            
        # Parse response
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "output" in response:
            return response["output"]
            
        return str(response)
