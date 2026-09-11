import json
import os
from typing import List, Dict, Any, Union, Optional
import logging
from ..types import EvaluationResult

from fi.evals import Evaluator as FAGIEvaluator
from fi.evals.metrics.base_metric import BaseMetric
from fi.evals.metrics.base_llm_metric import BaseLLMJudgeMetric
from fi.evals.llm.providers.litellm import LiteLLMProvider

logger = logging.getLogger(__name__)


class Evaluator:
    """
    A unified evaluator that seamlessly handles all evaluation
    backends: heuristics, Custom LLM-as-a-judge, or the FutureAGI platform.
    """

    def __init__(
        self,
        # Option 1: For local evaluation (Heuristics or LLM Judge)
        metric: Optional[Union[BaseMetric, BaseLLMJudgeMetric]] = None,
        # Option 2: For FutureAGI evaluation
        eval_template: Optional[str] = None,
        eval_model_name: Optional[str] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        # Optional: For LLM-as-a-judge if a specific provider is needed
        provider: Optional[LiteLLMProvider] = None,
    ):
        """
        Initializes the unified evaluator.

        To use local metrics (heuristics or LLM-as-a-judge):
            - Provide an instantiated `metric` object.
            - If the metric is an LLM judge, you can optionally pass a `provider`,
              otherwise it will default to a LiteLLMProvider using environment variables.

        To use the FutureAGI online platform:
            - Provide an `eval_template` name (e.g., "summary_quality").
            - Provide a `model_name` for the evaluation (e.g., "turing_flash").
            - Provide your `fi_api_key`.
        """
        self._strategy: str = ""
        self._metric_instance: Optional[BaseMetric] = None
        self._online_client: Optional[FAGIEvaluator] = None
        self._online_eval_template: Optional[str] = None
        self._online_model_name: Optional[str] = None

        if metric:
            # --- LOCAL EVALUATION ---
            if not isinstance(metric, BaseMetric):
                raise TypeError(
                    "The 'metric' argument must be an instance of a class inheriting from BaseMetric."
                )

            # If it's an LLM judge that hasn't been given a provider, create a default one.
            if isinstance(metric, BaseLLMJudgeMetric) and metric.provider is None:
                metric.provider = provider or LiteLLMProvider()

            self._strategy = "local"
            self._metric_instance = metric
            logger.info(
                "Initialized Evaluator with local metric: %s", metric.__class__.__name__
            )

        elif eval_template and eval_model_name:
            # --- FAGI EVALUATION PATH (FutureAGI Platform) ---
            self._strategy = "fagi"
            api_key = fi_api_key or os.getenv("FI_API_KEY")
            secret = fi_secret_key or os.getenv("FI_SECRET_KEY")
            if not api_key or not secret:
                raise ValueError(
                    "To use the FutureAGI platform, you must provide an 'fi_api_key' and 'fi_secret_key' or set the FI_API_KEY and FI_SECRET_KEY environment variable."
                )

            self._online_client = FAGIEvaluator(
                fi_api_key=api_key, fi_secret_key=secret
            )
            self._online_eval_template = eval_template
            self._online_model_name = eval_model_name
            logger.info(
                "Initialized Evaluator for online evaluation with template '%s' and model '%s'.",
                eval_template,
                eval_model_name,
            )

        else:
            raise ValueError(
                "Invalid configuration. You must provide either a local 'metric' object "
                "or the 'eval_template' and 'model_name' for online evaluation."
            )

    def evaluate(self, inputs: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """
        Runs a batch evaluation using the configured strategy.
        """
        logger.info(
            "Starting evaluation for %d inputs using '%s' strategy.",
            len(inputs),
            self._strategy,
        )
        if self._strategy == "local":
            return self._evaluate_local(inputs)
        elif self._strategy == "fagi":
            return self._evaluate_online(inputs)
        else:
            # This should never be reached
            raise RuntimeError("Evaluator is not configured with a valid strategy.")

    def _evaluate_local(self, inputs: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Handles evaluation using local BaseMetric instances."""
        logger.info(
            "Running local evaluation with metric: %s",
            self._metric_instance.__class__.__name__,
        )
        try:
            batch_result = self._metric_instance.evaluate(inputs)
            results: List[EvaluationResult] = []
            for i, (single_input, res) in enumerate(
                zip(inputs, batch_result.eval_results)
            ):
                logger.debug(
                    f"Evaluating input #{i + 1}: {json.dumps(single_input, indent=2, ensure_ascii=False)}"
                )
                if res and isinstance(res.output, (int, float)):
                    score = max(0.0, min(1.0, float(res.output)))
                    reason = res.reason or ""
                    results.append(EvaluationResult(score=score, reason=reason))
                    logger.info(
                        f"Input #{i + 1} evaluated successfully. Score: {score:.4f}\nReason: {reason}"
                    )
                else:
                    reason = "Local evaluation failed or returned invalid output."
                    if res:
                        reason = res.reason or reason
                    results.append(EvaluationResult(score=0.0, reason=reason))
                    logger.warning(
                        "Input #%d evaluation failed. Reason: %s", i + 1, reason
                    )
        except Exception as e:
            logger.error(f"Local evaluation failed for batch: {e}", exc_info=True)
            # Return failing results for all inputs
            results = [
                EvaluationResult(score=0.0, reason=f"Local evaluation failed: {e}")
                for _ in inputs
            ]

        logger.info("Local evaluation completed. Returning %d results.", len(results))
        return results

    def _evaluate_online(self, inputs: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Handles evaluation using the FutureAGI platform."""
        results: List[EvaluationResult] = []
        # for some reason the online evaluator takes single input and not a list of inputs.
        for i, single_input in enumerate(inputs):
            try:
                batch_result = self._online_client.evaluate(
                    eval_templates=self._online_eval_template,
                    inputs=single_input,
                    model_name=self._online_model_name,
                )
                # logger.debug(
                #     f"Online evaluation result: {json.dumps(batch_result.model_dump_json(), indent=2, ensure_ascii=False)}"
                # )
                eval_res = (
                    batch_result.eval_results[0]
                    if batch_result and batch_result.eval_results
                    else None
                )
                # logger.debug(f"Online evaluation result: {eval_res}")
                if eval_res and isinstance(eval_res.output, (int, float)):
                    score = max(0.0, min(1.0, float(eval_res.output)))
                    results.append(
                        EvaluationResult(score=score, reason=eval_res.reason or "")
                    )
                    logger.info(
                        f"Input #{i + 1} evaluated successfully. Score: {score:.4f}\nReason: {eval_res.reason}"
                    )
                elif eval_res and isinstance(eval_res.output, str):
                    if eval_res.output.lower() in ["pass", "passed", "true"]:
                        results.append(
                            EvaluationResult(score=1.0, reason=eval_res.reason or "")
                        )
                    else:
                        results.append(
                            EvaluationResult(score=0.0, reason=eval_res.reason or "")
                        )
                    logger.info(
                        "Input #%d evaluated successfully. Reason: %s",
                        i + 1,
                        eval_res.reason,
                    )
                else:
                    reason = "Online evaluation failed or returned invalid output."
                    if eval_res:
                        reason = eval_res.reason or reason
                    results.append(EvaluationResult(score=0.0, reason=reason))
                    logger.warning(
                        f"Could not evaluate input #{i + 1}, adding 0 score."
                    )

            except Exception as e:
                logger.error(f"API call failed for input #{i + 1}: {e}", exc_info=True)
                results.append(
                    EvaluationResult(score=0.0, reason=f"API call failed: {e}")
                )
        logger.info("Online evaluation completed. Returning %d results.", len(results))
        return results
