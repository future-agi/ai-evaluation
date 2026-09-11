import litellm
from typing import Dict

from ..types import LLMMessage
from ..base.base_generator import BaseGenerator

# logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LiteLLMGenerator(BaseGenerator):
    """
    A Generator that uses LiteLLM to call any supported language model.
    """

    def __init__(self, model: str, prompt_template: str):
        """
        Initializes the LiteLLMGenerator.

        Args:
            model: The name of the model to use (e.g., "gpt-4o-mini").
            prompt_template: A string template for the prompt, with placeholders
                             in f-string format (e.g., "Summarize this: {text}").
        """
        self.model = model
        self.prompt_template = prompt_template
        # LiteLLM is stateless, so no further setup is needed here.

    def generate(self, prompt_vars: Dict[str, str], **litellm_kwargs) -> str:
        """
        Fills the prompt template and calls the LiteLLM API.

        Args:
            prompt_vars: A dictionary of variables to fill the prompt template.
            litellm_kwargs: Any litellm supported kwargs

        Returns:
            The string content of the model's response.
        """
        prompt = self.prompt_template.format(**prompt_vars)

        messages = [LLMMessage(role="user", content=prompt)]
        messages_for_litellm = [msg.model_dump(exclude_none=True) for msg in messages]

        try:
            # litellm allows us to drop any params which may not be supported by the model
            litellm.drop_params = True
            response = litellm.completion(
                model=self.model, messages=messages_for_litellm, **litellm_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            # Basic error handling
            print(f"An error occurred with LiteLLM: {e}")
            return ""

    @property
    def model_name(self) -> str:
        return self.model

    def get_prompt_template(self) -> str:
        """Returns the current prompt template."""
        return self.prompt_template

    def set_prompt_template(self, template: str):
        """Updates the prompt template."""
        self.prompt_template = template
