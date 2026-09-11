import json
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import IterationHistory, OptimizationResult
from ..utils.early_stopping import EarlyStoppingConfig, EarlyStoppingChecker
import logging

logger = logging.getLogger(__name__)
# ==============================================================================
# Prompts and Pydantic Models for the Teacher LLM (Meta-Model)
# ==============================================================================


META_PROMPT_TEMPLATE = """
You are a world-class expert in prompt engineering. Your task is to diagnose and optimize a given prompt based on its performance on a set of test cases.

### Current Prompt
The following is the current prompt being evaluated:
---
{current_prompt}
---

### Previous Failed Attempts
You have already tried the following prompts, but they performed worse than the current one. Analyze why they failed to avoid repeating mistakes.
---
{other_attempts}
---

### Performance Data
The current prompt was run on a set of examples, and here are the results. Pay close attention to the examples with low scores.
---
{annotated_results}
---

### Task Description
{task_description}

### Your Task
Think step-by-step to generate an improved prompt:
1.  **Analyze Failures:** Deeply analyze the failing examples. What patterns do you see? Is the prompt too vague, too restrictive, or missing key instructions?
2.  **Formulate a Hypothesis:** Based on your analysis, state a clear hypothesis for how to improve the prompt. For example, "My hypothesis is that adding a chain-of-thought instruction will improve reasoning on multi-step problems."
3.  **Generate Improved Prompt:** Rewrite the *entire* prompt, implementing your hypothesis. The new prompt should be a complete replacement for the current one.

Return ONLY a valid JSON object with two keys: "hypothesis" (your string hypothesis) and "improved_prompt" (the complete new prompt string).
"""


class MetaPromptOutput(BaseModel):
    hypothesis: str = Field(
        description="The hypothesis for why the new prompt will be better."
    )
    improved_prompt: str = Field(description="The complete, new, improved prompt.")


# ==============================================================================
# The MetaPrompt Optimizer Class
# ==============================================================================


class MetaPromptOptimizer(BaseOptimizer):
    """
    Optimizes a prompt by using a powerful "teacher" LLM to analyze its
    performance and rewrite it. This is inspired by the `promptim` library.
    """

    def __init__(
        self,
        teacher_generator: LiteLLMGenerator,
        task_model: Optional[str] = None,
    ):
        """
        Initializes the MetaPrompt Optimizer.

        Args:
            teacher_generator: A powerful generator (e.g., GPT-4o, Claude 3 Opus)
                used to analyze performance and generate new prompts.
            task_model: Model used to run candidate prompts while scoring them.
                Defaults to the teacher generator's model.
        """
        self.teacher = teacher_generator
        self.task_model = task_model

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        task_description: str = "I want to improve my prompt.",
        num_rounds: Optional[int] = 5,
        eval_subset_size: Optional[int] = 40,
        early_stopping: Optional[EarlyStoppingConfig] = None,
    ) -> OptimizationResult:
        logger.info("--- Starting Meta-Prompt Optimization ---")

        # Initialize early stopping checker
        checker = None
        if early_stopping and early_stopping.is_enabled():
            checker = EarlyStoppingChecker(early_stopping)
            logger.info(f"Early stopping enabled: {early_stopping}")

        if not initial_prompts:
            raise ValueError("Initial prompts list cannot be empty.")

        current_prompt = initial_prompts[0]
        best_prompt = current_prompt
        best_score = -1.0
        history: List[IterationHistory] = []
        previous_attempts = set()

        for round_num in range(num_rounds):
            logger.info(
                f"\n--- Starting Optimization Round {round_num + 1}/{num_rounds} ---"
            )
            logger.info(f"Current best prompt:\n{current_prompt}")

            # 1. Evaluate the current prompt on a subset of data
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            iteration_history = self._score_prompt(
                current_prompt, evaluator, data_mapper, eval_subset
            )

            if not iteration_history:
                logger.warning("Evaluation of current prompt failed. Skipping round.")
                continue

            history.append(iteration_history)
            current_score = iteration_history.average_score

            if current_score > best_score:
                best_score = current_score
                best_prompt = current_prompt
                logger.info(f"New best score found: {best_score:.4f}")

            # Check early stopping
            if checker:
                num_evals = len(eval_subset)
                if checker.should_stop(current_score, num_evals):
                    logger.info(
                        f"Early stopping triggered: {checker.get_state()['stop_reason']}"
                    )
                    break

            # 2. Use the teacher model to generate a new, improved prompt
            annotated_results_str = self._format_results(iteration_history, eval_subset)

            # Format previous attempts for the meta-prompt
            other_attempts_str = (
                "\n---\n".join(list(previous_attempts)) if previous_attempts else "N/A"
            )

            meta_prompt = META_PROMPT_TEMPLATE.format(
                current_prompt=current_prompt,
                other_attempts=other_attempts_str,
                annotated_results=annotated_results_str,
                task_description=task_description,
            )

            logger.debug("Generating new prompt with meta-prompt...")
            new_prompt_json = self.teacher.generate(
                prompt_vars={"prompt": meta_prompt},
                response_format={"type": "json_object"},
            )

            try:
                parsed_output = MetaPromptOutput.model_validate_json(new_prompt_json)
                logger.info(f"Teacher's Hypothesis: {parsed_output.hypothesis}")
                previous_attempts.add(current_prompt)
                current_prompt = parsed_output.improved_prompt
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(
                    f"Failed to parse new prompt from teacher model, keeping current prompt. Error: {e}"
                )

        final_best_generator = LiteLLMGenerator(self.teacher.model_name, best_prompt)

        # Build result with early stopping metadata
        return OptimizationResult(
            best_generator=final_best_generator,
            history=history,
            final_score=best_score,
            early_stopped=checker.get_state()["stopped"] if checker else False,
            stop_reason=checker.get_state()["stop_reason"] if checker else None,
            total_iterations=len(history),
            total_evaluations=(
                checker.get_state()["total_evaluations"]
                if checker
                else sum(len(h.individual_results) for h in history)
            ),
        )

    def _score_prompt(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> IterationHistory | None:
        """Scores a single prompt and returns its history."""
        try:
            temp_generator = LiteLLMGenerator(
                self.task_model or self.teacher.model_name, prompt
            )
            generated_outputs = [
                temp_generator.generate(example) for example in dataset
            ]
            eval_inputs = [
                data_mapper.map(gen_out, ex)
                for gen_out, ex in zip(generated_outputs, dataset)
            ]
            results = evaluator.evaluate(eval_inputs)
            avg_score = (
                sum(res.score for res in results) / len(results) if results else 0.0
            )
            return IterationHistory(
                prompt=prompt, average_score=avg_score, individual_results=results
            )
        except Exception as e:
            logger.error(f"Failed to score prompt: {e}")
            return None

    def _format_results(
        self, iteration_history: IterationHistory, dataset: List[Dict[str, Any]]
    ) -> str:
        """Formats the evaluation results into a string for the meta-prompt."""
        formatted_lines = []
        for i, result in enumerate(iteration_history.individual_results):
            example_input = dataset[i]
            formatted_lines.append(f"Example {i + 1}:")
            formatted_lines.append(
                f"  Input: {json.dumps(example_input, ensure_ascii=False)}"
            )
            formatted_lines.append(f"  Score: {result.score:.2f}")
            formatted_lines.append(f"  Reason: {result.reason}")
            formatted_lines.append("---")
        return "\n".join(formatted_lines)
