import json
from typing import Any, Dict, List, Type
from pydantic import BaseModel, create_model
from jinja2 import Environment, BaseLoader

from ...base_llm_metric import BaseLLMJudgeMetric
from ..types import CustomInput, DefaultJudgeOutput
from ....llm.base_llm_provider import LLMProvider
from .prompts import DEFAULT_USER_PROMPT_TEMPLATE


class CustomLLMJudge(BaseLLMJudgeMetric[CustomInput]):
    """
    A smart, user-configurable LLM-as-a-judge metric that prioritizes ease of use.

    For the most common use cases, the user only needs to provide their
    grading criteria. The judge provides sensible defaults for the prompt
    template and output format, which can be optionally overridden for
    advanced customization.
    """

    @property
    def metric_name(self) -> str:
        return self.config.get("name", "custom_llm_judge")

    def __init__(self, provider: LLMProvider, config: Dict[str, Any], **litellm_kwargs):
        # The ONLY required key is now 'grading_criteria'
        if "grading_criteria" not in config:
            raise ValueError(
                "CustomLLMJudge config must contain a 'grading_criteria' key."
            )

        super().__init__(provider, config, **litellm_kwargs)

        # Explicitly set the input model, as this class is generic
        self.input_model = CustomInput

        # Smartly decide which Pydantic model to uset
        self._output_model = DefaultJudgeOutput

    @property
    def output_pydantic_model(self) -> Type[BaseModel]:
        return self._output_model

    def _create_prompt_messages(self, inputs: CustomInput) -> List[Dict[str, str]]:
        jinja_env = Environment(loader=BaseLoader())
        jinja_env.filters["tojson"] = json.dumps

        # Use the user-provided template if it exists, otherwise use the default
        template_str = self.config.get(
            "user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE
        )
        template = jinja_env.from_string(template_str)

        render_context = {
            "grading_criteria": self.config["grading_criteria"],
            "few_shot_examples": self.config.get("few_shot_examples", []),
            "task_input": inputs.model_dump(),
        }

        user_prompt = template.render(render_context)

        system_prompt = self.config.get(
            "system_prompt",
            "You are an expert AI evaluator. Follow the user's instructions and output format precisely.",
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _normalize_score(self, parsed_output: BaseModel) -> Dict[str, Any]:
        """Normalizes the score from the validated Pydantic output."""
        output_dict = parsed_output.model_dump()

        # Prioritize finding a "score" field, as it's our default
        score_val = output_dict.get("score", 1.0)

        return {"output": float(score_val), "reason": json.dumps(output_dict, indent=2)}
