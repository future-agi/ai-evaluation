import json
import logging
import random
import re
from typing import Any, Dict, List, Set, Optional

from pydantic import BaseModel, ValidationError

from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import IterationHistory, OptimizationResult
from ..utils.early_stopping import EarlyStoppingConfig, EarlyStoppingChecker

logger = logging.getLogger(__name__)

MUTATE_PROMPT = """
You are an expert in prompt engineering. You will be given a task description and different styles known as meta prompts. Your task is to generate {num_variations} diverse variations of the following instruction by adaptively mixing meta prompt while keeping similar semantic meaning.

[Task Description]: {task_description}
[Meta Prompts]: {meta_prompts}
[Prompt Instruction]: {prompt_instruction}

Return ONLY a valid JSON object with a single key "variations" containing a list of the new prompt strings.
"""

CRITIQUE_PROMPT = """
You are an expert prompt engineering analyst. My current prompt is:
---
{instruction}
---
This prompt performed poorly on the following examples:
---
{examples}
---
Provide a detailed critique explaining the potential reasons for failure.

Return ONLY a valid JSON object with a single key "variations" containing a list with ONE string: your critique.
"""

REFINE_PROMPT = """
You are an expert prompt engineer. My current prompt is:
---
{instruction}
---
It failed on these examples:
---
{examples}
---
Here is a critique of the prompt's weaknesses: "{critique}"

Based on this critique, write {steps_per_sample} different, improved versions of the prompt.

Return ONLY a valid JSON object with a single key "variations" containing a list of the new prompt strings.
"""


class Variations(BaseModel):
    variations: List[str]


class PromptWizardOptimizer(BaseOptimizer):
    """
    An adapter for the PromptWizard optimization algorithm, using a multi-stage
    process of mutation, critique, and refinement for prompt instructions.
    """

    def __init__(
        self,
        teacher_generator: LiteLLMGenerator,
        mutate_rounds: int = 3,
        refine_iterations: int = 2,
        beam_size: int = 1,
        task_model: Optional[str] = None,
    ):
        self.teacher = teacher_generator
        self.task_model = task_model
        self.mutate_rounds = mutate_rounds
        self.refine_iterations = refine_iterations
        self.beam_size = beam_size
        self.thinking_styles = THINKING_STYLES
        logger.info("--- PromptWizard Optimizer Initialized ---")
        logger.debug(
            f"Initialized with: mutate_rounds={mutate_rounds}, "
            f"refine_iterations={refine_iterations}, beam_size={beam_size}"
        )

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        task_description: str = "No task description given.",
        early_stopping: Optional[EarlyStoppingConfig] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        eval_subset_size = kwargs.get("eval_subset_size", 25)
        logger.info("--- Starting PromptWizard Optimization ---")
        logger.debug(f"Task: {task_description}")
        logger.debug(f"Initial prompts count: {len(initial_prompts)}")
        logger.debug(f"Dataset size: {len(dataset)}")
        logger.debug(f"Evaluation subset size: {eval_subset_size}")

        # Initialize early stopping checker
        checker = None
        if early_stopping and early_stopping.is_enabled():
            checker = EarlyStoppingChecker(early_stopping)
            logger.info(f"Early stopping enabled: {early_stopping}")

        if not initial_prompts:
            raise ValueError("Initial prompts list cannot be empty.")

        current_best_instruction = initial_prompts[0]
        history: List[IterationHistory] = []
        logger.info(f"Initial best instruction: '{current_best_instruction[:100]}...'")

        for i in range(self.refine_iterations):
            logger.info(
                f"\n--- Instruction Refinement Iteration {i + 1}/{self.refine_iterations} ---"
            )

            # 1. Mutate
            logger.info("Step 1: Mutating instruction...")
            mutated_prompts = self._mutate_instruction(
                current_best_instruction, task_description
            )
            candidate_pool = {current_best_instruction, *mutated_prompts}
            logger.info(f"Generated {len(mutated_prompts)} unique new prompts.")
            logger.debug(f"Candidate pool size: {len(candidate_pool)}")

            # 2. Score
            logger.info("Step 2: Scoring candidate prompts...")
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            logger.debug(f"Scoring against a subset of {len(eval_subset)} examples.")
            iteration_history = self._score_candidates(
                list(candidate_pool), evaluator, data_mapper, eval_subset
            )
            history.extend(iteration_history)

            sorted_by_score = sorted(
                iteration_history, key=lambda x: x.average_score, reverse=True
            )
            top_prompts_this_round = [
                item.prompt for item in sorted_by_score[: self.beam_size]
            ]
            logger.info(f"Top {self.beam_size} prompts selected for refinement.")
            for idx, p in enumerate(top_prompts_this_round):
                score = sorted_by_score[idx].average_score
                logger.debug(f"  - Prompt (Score: {score:.4f}): '{p[:100]}...'")

            # Check early stopping
            if checker:
                best_round_score = sorted_by_score[0].average_score
                num_evals = len(candidate_pool) * len(eval_subset)
                if checker.should_stop(best_round_score, num_evals):
                    logger.info(
                        f"Early stopping triggered: {checker.get_state()['stop_reason']}"
                    )
                    break

            # 3. Critique and Refine
            logger.info("Step 3: Critiquing and refining top prompts...")
            refined_prompts = set()
            for prompt_to_refine in top_prompts_this_round:
                errors = self._get_errors(
                    prompt_to_refine, evaluator, data_mapper, dataset
                )
                if errors:
                    logger.debug(
                        f"Found {len(errors)} errors for prompt: '{prompt_to_refine[:100]}...'"
                    )
                    refined = self._critique_and_refine(prompt_to_refine, errors)
                    if refined:
                        logger.debug("Successfully refined prompt.")
                        refined_prompts.add(refined)
                else:
                    logger.debug(
                        f"No errors found for prompt, skipping refinement: '{prompt_to_refine[:100]}...'"
                    )

            # Determine the best instruction for the next iteration
            if refined_prompts:
                logger.info(
                    f"Scoring {len(refined_prompts)} refined prompts to find new best."
                )
                final_candidates_this_round = {
                    current_best_instruction,
                    *refined_prompts,
                }
                final_history = self._score_candidates(
                    list(final_candidates_this_round),
                    evaluator,
                    data_mapper,
                    eval_subset,
                )
                history.extend(final_history)
                current_best_instruction = sorted(
                    final_history, key=lambda x: x.average_score, reverse=True
                )[0].prompt
            else:
                logger.info("No prompts were refined, carrying over previous best.")
                current_best_instruction = top_prompts_this_round[0]

            logger.info(
                f"Best instruction after iteration {i + 1}: '{current_best_instruction[:100]}...'"
            )

        logger.info("--- PromptWizard Optimization Finished ---")
        final_history = sorted(history, key=lambda x: x.average_score, reverse=True)
        best_prompt = final_history[0].prompt
        best_score = final_history[0].average_score
        logger.info(f"Final best prompt (Score: {best_score:.4f}): '{best_prompt}'")

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

    def _mutate_instruction(
        self,
        base_instruction: str,
        task_description: str,
    ) -> Set[str]:
        logger.debug(
            f"Entering mutation phase for instruction: '{base_instruction[:100]}...'"
        )
        all_variations = set()
        temp_generator = LiteLLMGenerator(
            self.task_model or self.teacher.model_name, "{prompt}"
        )
        for i in range(self.mutate_rounds):
            logger.debug(f"Mutation round {i + 1}/{self.mutate_rounds}")
            prompt = MUTATE_PROMPT.format(
                num_variations=len(self.thinking_styles),
                task_description=task_description,
                meta_prompts="\n".join(self.thinking_styles),
                prompt_instruction=base_instruction,
            )
            response_text = temp_generator.generate(
                {"prompt": prompt}, response_format={"type": "json_object"}
            )
            variations = self._parse_variations_from_json(response_text)
            logger.debug(f"Generated {len(variations)} variations in this round.")
            all_variations.update(variations)
        logger.debug(f"Total unique variations from mutation: {len(all_variations)}")
        return all_variations

    def _critique_and_refine(
        self,
        prompt: str,
        errors: List[Dict[str, Any]],
    ) -> str | None:
        logger.debug(f"Entering critique and refine for: '{prompt[:100]}...'")
        error_str = json.dumps(errors, indent=2, ensure_ascii=False)

        logger.debug("Generating critique...")
        critique_prompt = CRITIQUE_PROMPT.format(instruction=prompt, examples=error_str)
        critique_response = self.teacher.generate(
            {"prompt": critique_prompt}, response_format={"type": "json_object"}
        )
        critiques = self._parse_variations_from_json(critique_response)
        if not critiques:
            logger.warning("Critique generation failed, skipping refinement.")
            return None
        critique_text = "\n".join([critique for critique in critiques])
        logger.debug(f"Generated critique: '{critique_text[:100]}...'")

        logger.debug("Refining prompt based on critique...")
        refine_prompt = REFINE_PROMPT.format(
            instruction=prompt,
            examples=error_str,
            critique=critique_text,
            steps_per_sample=1,
        )
        refined_text = self.teacher.generate(
            {"prompt": refine_prompt}, response_format={"type": "json_object"}
        )

        refined_prompts = self._parse_variations_from_json(refined_text)
        if refined_prompts:
            logger.debug(f"Refined prompt: '{refined_prompts[0][:100]}...'")
            return refined_prompts[0]
        else:
            logger.warning("Refinement step produced no new prompts.")
            return None

    # Helper methods (shared with ProTeGi, kept here for encapsulation)
    def _get_errors(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        sample_size: int = 10,
    ) -> List[Dict[str, Any]]:
        logger.debug(f"Getting errors for prompt: '{prompt[:100]}...'")
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
        errors = [
            {"inputs": subset[i], "output": generated_outputs[i], "score": res.score}
            for i, res in enumerate(results)
            if res.score < 0.5
        ]
        logger.debug(f"Found {len(errors)} examples with score < 0.5.")
        return errors

    def _score_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[IterationHistory]:
        logger.debug(f"Scoring {len(prompts)} candidate prompts.")
        histories = []
        for i, prompt in enumerate(prompts):
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
            logger.debug(
                f"  - Scored prompt {i + 1}/{len(prompts)} (Avg Score: {avg_score:.4f}): '{prompt[:100]}...'"
            )
            histories.append(
                IterationHistory(
                    prompt=prompt, average_score=avg_score, individual_results=results
                )
            )
        return histories

    @staticmethod
    def _parse_variations_from_json(text: str) -> List[str]:
        text = text.strip()

        # --- Stage 1: Try to parse the entire string as JSON ---
        try:
            data = json.loads(text)
            return Variations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError):
            # This is expected if there's extra text, so we continue.
            pass

        # --- Stage 2: Look for a JSON markdown code block ---
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                return Variations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError):
            pass

        # --- Stage 3: Greedy fallback to find the first '{' and last '}' ---
        try:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = text[start_index : end_index + 1]
                data = json.loads(json_str)
                return Variations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError) as e:
            # If all parsing attempts fail, log the error and return empty.
            logging.error(
                f"Failed to parse teacher model JSON response after all fallbacks: {e}"
            )
            logging.debug(f"Raw problematic output that failed parsing:\n{text}")
            return []

        # If no JSON object is found at all
        logging.warning("Could not find any JSON in the teacher's response.")
        logging.debug(f"Raw response with no JSON:\n{text}")
        return []


# Static list of thinking styles from PromptWizard's config
THINKING_STYLES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one.",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "Critical Thinking: Analyze the problem from different perspectives, questioning assumptions.",
    "Try creative thinking, generate innovative and out-of-the-box ideas.",
    "Use systems thinking: Consider the problem as part of a larger interconnected system.",
    "Let's think step by step.",
]
