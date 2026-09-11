import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ValidationError

from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import IterationHistory, OptimizationResult
from ..utils.early_stopping import EarlyStoppingConfig, EarlyStoppingChecker

GET_GRADIENTS_PROMPT = """
You are an expert in prompt engineering. I'm trying to write a zero-shot classifier prompt.
My current prompt is:
---
{prompt}
---
This prompt performed poorly on the following examples:
---
{error_examples}
---
Provide {num_feedbacks} distinct reasons why the prompt could have failed. Each reason should be a concise critique.
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the critiques).
"""

APPLY_GRADIENT_PROMPT = """
You are an expert in prompt engineering. I'm trying to improve a zero-shot classifier prompt.
My current prompt is:
---
{prompt}
---
It performed poorly on these examples:
---
{error_examples}
---
A key reason for the failure is the following critique: "{feedback}"
Based on this critique, generate {num_new_prompts} different, improved versions of the prompt.
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the new prompts).
"""

PARAPHRASE_PROMPT = """
Generate {num_variations} semantic paraphrases of the following prompt. The meaning should be identical, but the wording should be different.
---
{prompt}
---
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the paraphrased prompts).
"""


class GradientVariations(BaseModel):
    variations: List[str] = Field(description="A list of generated text strings.")


class ProTeGi(BaseOptimizer):
    """
    A corrected and robust implementation of the ProTeGi optimizer.
    """

    def __init__(
        self,
        teacher_generator: LiteLLMGenerator,
        num_gradients: int = 4,
        errors_per_gradient: int = 4,
        prompts_per_gradient: int = 1,
        beam_size: int = 4,
        task_model: Optional[str] = None,
    ):
        self.teacher = teacher_generator
        self.task_model = task_model
        self.num_gradients = num_gradients
        self.errors_per_gradient = errors_per_gradient
        self.prompts_per_gradient = prompts_per_gradient
        self.beam_size = beam_size
        logging.info("--- ProTeGi Optimizer Initialized ---")

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        early_stopping: Optional[EarlyStoppingConfig] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        num_rounds = kwargs.get("num_rounds", 3)
        eval_subset_size = kwargs.get("eval_subset_size", 32)

        # Initialize early stopping checker
        checker = None
        if early_stopping and early_stopping.is_enabled():
            checker = EarlyStoppingChecker(early_stopping)
            logging.info(f"Early stopping enabled: {early_stopping}")

        beam = set(initial_prompts)
        best_overall_score = -1.0
        best_overall_prompt = initial_prompts[0] if initial_prompts else ""
        history: List[IterationHistory] = []

        for round_num in range(num_rounds):
            logging.info(
                f"\n--- Starting Optimization Round {round_num + 1}/{num_rounds} ---"
            )

            # 1. EXPANSION: Generate new candidates from the current beam
            current_prompts = list(beam)
            logging.info(
                f"Expanding {len(current_prompts)} prompts into new candidates..."
            )
            expanded_prompts = self._expand_candidates(
                current_prompts, evaluator, data_mapper, dataset
            )

            # The candidate pool for this round is the union of the old beam and new prompts
            candidate_pool = beam.union(expanded_prompts)
            logging.info(
                f"Candidate pool for this round has {len(candidate_pool)} unique prompts."
            )

            # 2. SELECTION: Score all candidates in the pool
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            iteration_history = self._score_candidates(
                list(candidate_pool), evaluator, data_mapper, eval_subset
            )
            history.extend(iteration_history)

            # 3. BEAM UPDATE: Select the top N prompts for the next round
            sorted_history = sorted(
                iteration_history, key=lambda x: x.average_score, reverse=True
            )
            if not sorted_history:
                logging.warning("No successful evaluations in this round. Halting.")
                break

            beam = {item.prompt for item in sorted_history[: self.beam_size]}
            best_round_score = sorted_history[0].average_score
            best_round_prompt = sorted_history[0].prompt

            logging.info(f"Best score in round {round_num + 1}: {best_round_score:.4f}")
            logging.info(f"New beam selected with {len(beam)} prompts.")

            if best_round_score > best_overall_score:
                best_overall_score = best_round_score
                best_overall_prompt = best_round_prompt

            # Check early stopping
            if checker:
                num_evals = len(candidate_pool) * len(eval_subset)
                if checker.should_stop(best_round_score, num_evals):
                    logging.info(
                        f"Early stopping triggered: {checker.get_state()['stop_reason']}"
                    )
                    break

        final_best_generator = LiteLLMGenerator(
            self.teacher.model_name, best_overall_prompt
        )

        # Build result with early stopping metadata
        return OptimizationResult(
            best_generator=final_best_generator,
            history=history,
            final_score=best_overall_score,
            early_stopped=checker.get_state()["stopped"] if checker else False,
            stop_reason=checker.get_state()["stop_reason"] if checker else None,
            total_iterations=len(history),
            total_evaluations=(
                checker.get_state()["total_evaluations"]
                if checker
                else sum(len(h.individual_results) for h in history)
            ),
        )

    def _expand_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> Set[str]:
        new_prompts = set()
        for i, prompt in enumerate(prompts):
            logging.debug(f"--> Expanding prompt {i + 1}/{len(prompts)}...")
            errors = self._get_errors(prompt, evaluator, data_mapper, dataset)
            if not errors:
                logging.debug("Prompt produced no errors. No expansion.")
                continue

            critiques = self._get_gradients(prompt, errors)
            logging.debug(f"Generated {len(critiques)} critiques (gradients).")

            for feedback in critiques:
                generated = self._apply_gradient(prompt, errors, feedback)
                if generated:
                    logging.debug(
                        f"Generated {len(generated)} new prompts from critique: '{feedback[:50]}...'"
                    )
                    new_prompts.update(generated)
        return new_prompts

    def _get_errors(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        sample_size: int = 32,
    ) -> List[Dict[str, Any]]:
        subset = random.sample(dataset, min(len(dataset), sample_size))
        temp_generator = LiteLLMGenerator(
                self.task_model or self.teacher.model_name, prompt
            )

        generated_outputs = [temp_generator.generate(example) for example in subset]
        eval_inputs = [
            data_mapper.map(gen_out, ex)
            for gen_out, ex in zip(generated_outputs, subset)
        ]
        results = evaluator.evaluate(eval_inputs)

        errors = [subset[i] for i, res in enumerate(results) if res.score < 0.5]
        logging.debug(
            f"Found {len(errors)} errors with score < 0.5 from a subset of {len(subset)}."
        )
        return errors

    def _get_gradients(self, prompt: str, errors: List[Dict[str, Any]]) -> List[str]:
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        critique_prompt = GET_GRADIENTS_PROMPT.format(
            prompt=prompt,
            error_examples=json.dumps(error_sample, indent=2, ensure_ascii=False),
            num_feedbacks=self.num_gradients,
        )
        response_text = self.teacher.generate(
            prompt_vars={"prompt": critique_prompt},
            response_format={"type": "json_object"},
        )
        return self._parse_variations_from_json(response_text)

    def _apply_gradient(
        self, prompt: str, errors: List[Dict[str, Any]], feedback: str
    ) -> List[str]:
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        rewrite_prompt = APPLY_GRADIENT_PROMPT.format(
            prompt=prompt,
            error_examples=json.dumps(error_sample, indent=2, ensure_ascii=False),
            feedback=feedback,
            num_new_prompts=self.prompts_per_gradient,
        )
        response_text = self.teacher.generate(
            prompt_vars={"prompt": rewrite_prompt},
            response_format={"type": "json_object"},
        )
        return self._parse_variations_from_json(response_text)

    def _score_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[IterationHistory]:
        histories = []
        for i, prompt in enumerate(prompts):
            logging.info(
                f"--> Scoring prompt {i + 1}/{len(prompts)}: '{prompt[:100]}...'"
            )
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
            logging.info(f"    Average score: {avg_score:.4f}")
            histories.append(
                IterationHistory(
                    prompt=prompt, average_score=avg_score, individual_results=results
                )
            )
        return histories

    @staticmethod
    def _parse_variations_from_json(text: str) -> List[str]:
        text = text.strip()

        try:
            data = json.loads(text)
            return GradientVariations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError):
            pass

        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                return GradientVariations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError):
            pass

        try:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = text[start_index : end_index + 1]
                data = json.loads(json_str)
                return GradientVariations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(
                f"Failed to parse teacher model JSON response after all fallbacks: {e}"
            )
            logging.debug(f"Raw problematic output that failed parsing:\n{text}")
            return []

        # If no JSON object is found at all
        logging.warning("Could not find any JSON in the teacher's response.")
        logging.debug(f"Raw response with no JSON:\n{text}")
        return []
