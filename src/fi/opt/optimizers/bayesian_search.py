import optuna
import logging
import random
import json
import re
import time
from typing import List, Dict, Any, Optional, Callable
from ..base.base_optimizer import BaseOptimizer
from ..types import OptimizationResult, IterationHistory, EvaluationResult
from ..datamappers import BasicDataMapper
from ..generators.litellm import LiteLLMGenerator
from ..base.evaluator import Evaluator
from ..utils.early_stopping import EarlyStoppingConfig, EarlyStoppingChecker


TEACHER_SYSTEM_PROMPT = (
    """
You are an expert prompt engineer with deep knowledge of few-shot learning and template design. Your task is to analyze a sample of dataset items and create an optimal Python .format() string template for few-shot examples.

ANALYSIS REQUIREMENTS:
1. Examine the structure and content of the provided dataset examples
2. Identify all available field names/keys in the examples
3. Determine which fields represent inputs vs. expected outputs
4. Design a template that clearly demonstrates the input-output relationship

TEMPLATE DESIGN PRINCIPLES:
- Use ONLY field names that actually exist in the provided examples
- Include both input and output fields to enable effective few-shot learning
- Create clear, readable formatting that helps models understand the pattern
- Use descriptive labels (e.g., "Input:", "Output:", "Question:", "Answer:")
- Ensure the template is concise yet informative
- Maintain consistent formatting across examples

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
    "example_template": "your_template_string_here"
}

The template string must:
- Use Python .format() syntax with curly braces for field substitution
- Include clear labels for input and output sections
- Be ready to use without any modifications
- Work for all examples in the dataset

Example of a well-formed template:
"Question: {question}\nAnswer: {answer}"
or
"Prompt: {prompt}\nExpected Response: {response}\n---"

DO NOT include any explanations, comments, or additional text - only the JSON object.
    """
).strip()


class BayesianSearchOptimizer(BaseOptimizer):
    """
    An optimizer that uses Bayesian optimization (via Optuna) to find the
    best prompt by intelligently selecting few-shot examples.
    """

    def __init__(
        self,
        # Few-shot search space
        min_examples: int = 2,
        max_examples: int = 8,
        allow_repeats: bool = False,
        fixed_example_indices: Optional[List[int]] = None,
        # Trials and randomness
        n_trials: int = 10,
        seed: int = 42,
        # Inference/generation config
        inference_model_name: str = "gpt-4o-mini",
        inference_model_kwargs: Optional[Dict[str, Any]] = None,
        # Example formatting and prompt construction
        example_template: Optional[str] = None,
        example_template_fields: Optional[List[str]] = None,
        field_aliases: Optional[Dict[str, str]] = None,
        example_separator: str = "\n",
        few_shot_position: str = "append",  # "prepend" | "append"
        prompt_builder: Optional[Callable[[str, List[str]], str]] = None,
        example_formatter: Optional[Callable[[Dict[str, Any]], str]] = None,
        few_shot_title: Optional[str] = None,
        # Teacher-guided template inference (optional)
        infer_example_template_via_teacher: bool = False,
        teacher_model_name: str = "gpt-5",
        teacher_model_kwargs: Optional[Dict[str, Any]] = None,
        template_infer_n_samples: int = 8,
        teacher_system_prompt: str = TEACHER_SYSTEM_PROMPT,
        teacher_infer_max_retries: int = 2,
        teacher_infer_retry_sleep: float = 0.5,
        # Evaluation controls
        eval_subset_size: Optional[int] = None,
        eval_subset_strategy: str = "random",  # "random" | "first" | "all"
        score_aggregator: Optional[Callable[[List[EvaluationResult]], float]] = None,
        # Optuna controls
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        direction: str = "maximize",
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
    ):
        # Search space
        self.min_examples = min_examples
        self.max_examples = max_examples
        self.allow_repeats = allow_repeats
        self.fixed_example_indices = fixed_example_indices or []
        # Trials and randomness
        self.n_trials = n_trials
        self.seed = seed
        # Inference/generation
        self.inference_model_name = inference_model_name
        self.inference_model_kwargs = inference_model_kwargs or {}
        # Formatting/building
        self.example_template = example_template
        self.example_template_fields = example_template_fields
        self.field_aliases = field_aliases or {}
        self.example_separator = example_separator
        self.few_shot_position = few_shot_position
        self.prompt_builder = prompt_builder
        self.example_formatter = example_formatter
        self.few_shot_title = few_shot_title
        # Teacher-guided template inference
        self.infer_example_template_via_teacher = infer_example_template_via_teacher
        self.teacher_model_name = teacher_model_name
        # default kwargs for gpt-5 style models
        default_teacher_kwargs: Dict[str, Any] = {
            "temperature": 1.0,
            "max_tokens": 16000,
        }
        self.teacher_model_kwargs = {
            **default_teacher_kwargs,
            **(teacher_model_kwargs or {}),
        }
        self.template_infer_n_samples = template_infer_n_samples
        self.teacher_system_prompt = teacher_system_prompt
        self.teacher_infer_max_retries = max(0, int(teacher_infer_max_retries))
        self.teacher_infer_retry_sleep = max(0.0, float(teacher_infer_retry_sleep))
        # Evaluation
        self.eval_subset_size = eval_subset_size
        self.eval_subset_strategy = eval_subset_strategy
        self.score_aggregator = score_aggregator or self._default_score_aggregator
        # Optuna
        self.sampler = sampler or optuna.samplers.TPESampler(seed=self.seed)
        self.pruner = pruner
        self.direction = direction
        self.storage = storage
        self.study_name = study_name
        # runtime state
        self._runtime_example_template: Optional[str] = None

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        early_stopping: Optional[EarlyStoppingConfig] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        logging.info("--- Starting Bayesian Search Optimization ---")

        # Initialize early stopping checker
        checker = None
        if early_stopping and early_stopping.is_enabled():
            checker = EarlyStoppingChecker(early_stopping)
            logging.info(f"Early stopping enabled: {early_stopping}")

        if not initial_prompts:
            raise ValueError("Initial prompts list cannot be empty.")

        initial_prompt = initial_prompts[0]
        history: List[IterationHistory] = []

        # Optionally infer the example template via a teacher model from a sample of the dataset
        self._runtime_example_template = None
        if self.infer_example_template_via_teacher:
            try:
                self._runtime_example_template = self._infer_example_template(dataset)
                logging.info(
                    f"Inferred example template via teacher model: \n {self._runtime_example_template}"
                )
            except Exception as e:
                logging.warning(f"Falling back to default example_template. Error: {e}")
                self._runtime_example_template = None

        def objective(trial: optuna.Trial) -> float:
            # Suggest number of few-shot examples
            n_examples = trial.suggest_int(
                "n_examples", self.min_examples, self.max_examples
            )

            # Use a single seed to derive indices, avoiding dynamic value spaces in Optuna
            example_seed = trial.suggest_int("example_seed", 0, 2_000_000_000)
            rng = random.Random(example_seed)

            # Honor fixed indices first
            selected_indices: List[int] = list(self.fixed_example_indices)
            remaining_needed = max(0, n_examples - len(selected_indices))

            if remaining_needed > 0:
                if self.allow_repeats:
                    # Repeats allowed: sample with replacement
                    more = [
                        rng.randrange(len(dataset)) for _ in range(remaining_needed)
                    ]
                    selected_indices.extend(more)
                else:
                    # Unique sampling: sample without replacement from remaining pool
                    pool = [
                        i for i in range(len(dataset)) if i not in set(selected_indices)
                    ]
                    take = min(remaining_needed, len(pool))
                    selected_indices.extend(rng.sample(pool, take))

            # Format the selected examples for few-shot
            demo_examples = [dataset[i] for i in selected_indices]
            example_strings = [self._format_example(ex) for ex in demo_examples]
            few_shot_block = self._build_few_shot_block(example_strings)

            # Build the full prompt
            full_prompt = self._build_prompt(initial_prompt, few_shot_block)

            # Score the prompt
            iteration_history = self._score_prompt(
                full_prompt, evaluator, data_mapper, dataset
            )

            if not iteration_history:
                trial.set_user_attr("prompt", full_prompt)
                return 0.0

            history.append(iteration_history)
            avg_score = iteration_history.average_score
            trial.set_user_attr("prompt", full_prompt)
            logging.info(
                f"Trial {trial.number}: Score={avg_score:.4f}, Num Examples={len(selected_indices)}"
            )

            # Check early stopping
            if checker:
                eval_size = len(self._select_eval_subset(dataset))
                if checker.should_stop(avg_score, eval_size):
                    logging.info(
                        f"Early stopping triggered: {checker.get_state()['stop_reason']}"
                    )
                    trial.study.stop()

            return avg_score

        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=bool(self.storage and self.study_name),
        )

        try:
            study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            logging.info(f"Optimization stopped: {e}")

        # Check if any trials completed before accessing best_trial
        if not history:
            raise RuntimeError(
                "Optimization stopped before any trials completed successfully"
            )

        best_prompt = study.best_trial.user_attrs.get("prompt", initial_prompt)
        best_generator = LiteLLMGenerator(self.inference_model_name, best_prompt)

        # Build result with early stopping metadata
        return OptimizationResult(
            best_generator=best_generator,
            history=history,
            final_score=float(study.best_value)
            if study.best_value is not None
            else 0.0,
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
    ) -> Optional[IterationHistory]:
        try:
            eval_dataset = self._select_eval_subset(dataset)
            temp_generator = LiteLLMGenerator(self.inference_model_name, prompt)

            generated_outputs = [
                temp_generator.generate(example, **self.inference_model_kwargs)
                for example in eval_dataset
            ]
            eval_inputs = [
                data_mapper.map(gen_out, ex)
                for gen_out, ex in zip(generated_outputs, eval_dataset)
            ]
            results = evaluator.evaluate(eval_inputs)
            avg_score = self.score_aggregator(results)
            return IterationHistory(
                prompt=prompt, average_score=avg_score, individual_results=results
            )
        except Exception as e:
            logging.error(f"Failed to score prompt: {e}")
            return None

    def _infer_example_template(self, dataset: List[Dict[str, Any]]) -> str:
        sample_size = min(self.template_infer_n_samples, max(1, len(dataset)))
        sample = (
            random.sample(dataset, sample_size)
            if len(dataset) > sample_size
            else dataset
        )

        # Build a minimal payload: include keys and a few short examples limited to those keys
        keys: List[str] = sorted({k for ex in sample for k in ex.keys()})
        trimmed_examples: List[Dict[str, Any]] = [
            {k: str(ex.get(k, ""))[:500] for k in keys} for ex in sample
        ]
        user_payload = json.dumps(
            {"keys": keys, "examples": trimmed_examples}, ensure_ascii=False
        )

        prompt_template = (
            f"{self.teacher_system_prompt}\n\n"
            "Available keys:\n{keys}\n\n"
            "Examples (JSON):\n{examples_json}\n\n"
            'Respond ONLY with a JSON object like {{"example_template": "..."}}.'
        )

        teacher = LiteLLMGenerator(self.teacher_model_name, prompt_template)

        last_err: Optional[Exception] = None
        for attempt in range(self.teacher_infer_max_retries + 1):
            try:
                content = teacher.generate(
                    {"keys": ", ".join(keys), "examples_json": user_payload},
                    response_format={"type": "json_object"},
                    **self.teacher_model_kwargs,
                )
                template = self._parse_example_template_from_content(content)
                if template:
                    return template
                raise ValueError("Missing or empty 'example_template' in response")
            except Exception as e:
                last_err = e
                if attempt < self.teacher_infer_max_retries:
                    time.sleep(self.teacher_infer_retry_sleep)
                else:
                    break
        raise RuntimeError(f"Teacher template inference failed: {last_err}")

    @staticmethod
    def _parse_example_template_from_content(content: str) -> Optional[str]:
        # First try strict JSON
        try:
            data = json.loads(content)
            tmpl = data.get("example_template")
            if isinstance(tmpl, str) and tmpl.strip():
                return tmpl
        except Exception:
            pass
        # Try to extract JSON object containing example_template
        try:
            match = re.search(
                r"\{[\s\S]*?\"example_template\"\s*:\s*\"[\s\S]*?\"[\s\S]*?\}", content
            )
            if match:
                obj = json.loads(match.group(0))
                tmpl = obj.get("example_template")
                if isinstance(tmpl, str) and tmpl.strip():
                    return tmpl
        except Exception:
            pass
        return None

    def _select_eval_subset(
        self, dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.eval_subset_size or self.eval_subset_size >= len(dataset):
            return dataset
        size = max(1, self.eval_subset_size)
        if self.eval_subset_strategy == "first":
            return dataset[:size]
        elif self.eval_subset_strategy == "random":
            return random.sample(dataset, size)
        else:
            return dataset

    def _format_example(self, example: Dict[str, Any]) -> str:
        if self.example_formatter:
            return self.example_formatter(example)
        template = self._runtime_example_template or self.example_template
        if template:
            try:
                return template.format(**example)
            except Exception:
                pass
        # Fallbacks when no template or failed formatting
        if self.example_template_fields:
            lines: List[str] = []
            for key in self.example_template_fields:
                if key in example:
                    label = self.field_aliases.get(key, key)
                    lines.append(f"{label}: {example[key]}")
            if lines:
                return "\n".join(lines)
        # Final fallback: JSON dump of the example
        return json.dumps(example, ensure_ascii=False)

    def _build_few_shot_block(self, example_strings: List[str]) -> str:
        block = self.example_separator.join(example_strings)
        if self.few_shot_title:
            return f"{self.few_shot_title}\n{block}"
        return block

    def _build_prompt(self, base_prompt: str, few_shot_block: str) -> str:
        if self.prompt_builder:
            return self.prompt_builder(base_prompt, [few_shot_block])
        if not few_shot_block:
            return base_prompt
        # Escape braces in few-shot block to avoid str.format collisions
        safe_block = self._escape_braces(few_shot_block)
        if self.few_shot_position == "prepend":
            return f"{safe_block}\n\n---\n\n{base_prompt}"
        # default append
        return f"{base_prompt}\n\n---\n\n{safe_block}\n\n---"

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _default_score_aggregator(results: List[EvaluationResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / max(1, len(results))
