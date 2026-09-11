from typing import Any, Union
from fi.simulate.agent.wrapper import AgentWrapper, AgentInput, AgentResponse

class GeminiAgentWrapper(AgentWrapper):
    """
    Wrapper for Google Gemini (Generative AI) agents.
    Supports google-generativeai SDK.
    """
    def __init__(self, model: Any, system_prompt: str = None):
        """
        Args:
            model: An instance of google.generativeai.GenerativeModel
            system_prompt: Optional system instructions. 
                           Note: Ideally configure system_instruction on the model itself.
                           If provided here, it will be prepended as a user message.
        """
        self.model = model
        self.system_prompt = system_prompt

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        # Convert internal messages to Gemini format (Content objects)
        # Note: Gemini SDK manages chat history via ChatSession usually,
        # but for stateless call we pass full history if supported, 
        # or we might need to reconstruct a chat session.
        
        # Simple reconstruction of history for a chat session
        history = []
        
        if self.system_prompt:
            # Prepend system prompt as a user message for context
            history.append({"role": "user", "parts": [f"System Instruction: {self.system_prompt}"]})
            # Add a dummy model acknowledgement to keep turns valid (User -> Model -> User)
            history.append({"role": "model", "parts": ["Understood."]})
            
        for msg in input.messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            
            # Gemini typically expects history excluding the last message which is passed to send_message
            history.append({"role": role, "parts": [content]})
            
        if not history:
            raise ValueError("No messages provided to Gemini wrapper")

        # The last user message is the prompt
        last_turn = history.pop()
        if last_turn["role"] != "user":
            # If the last message wasn't user, something is weird in the flow,
            # but we can try to send empty or handle it. 
            # Ideally simulator sends User message last.
            prompt = ""
        else:
            prompt = last_turn["parts"][0]

        # Start a chat with the history
        chat = self.model.start_chat(history=history)
        
        # Check if async generation is supported (google-generativeai >= 0.3.0 has send_message_async)
        if hasattr(chat, "send_message_async"):
            response = await chat.send_message_async(prompt)
        else:
            # Fallback to sync
            response = chat.send_message(prompt)
            
        return response.text
