import inspect
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import defaultdict
import os

from requests import Response

from fi.api.auth import APIKeyAuth, ResponseHandler
from fi.api.types import HttpMethod, RequestConfig
from fi.evals.templates import EvalTemplate
from fi.evals.types import BatchRunResult, EvalResult, EvalResultMetric
from fi.testcases import TestCase, MLLMImage, MLLMAudio
from fi.utils.errors import InvalidAuthError
from fi.utils.routes import Routes
from fi.utils.utils import get_keys_from_env, get_base_url_from_env


class EvalResponseHandler(ResponseHandler[BatchRunResult, None]):
    """Handles responses for evaluation requests"""

    @classmethod
    def _parse_success(cls, response: Response) -> BatchRunResult:
        return cls.convert_to_batch_results(response.json())

    @classmethod
    def _handle_error(cls, response: Response) -> None:
        if response.status_code == 400:
            raise Exception(
                f"Evaluation failed with a 400 Bad Request. Please check your input data and evaluation configuration. Response: {response.text}"
            )
        elif response.status_code == 403:
            raise InvalidAuthError()
        else:
            raise Exception(
                f"Error in evaluation: {response.status_code}, response: {response.text}"
            )

    @classmethod
    def convert_to_batch_results(cls, response: Dict[str, Any]) -> BatchRunResult:
        """
        Convert API response to BatchRunResult

        Args:
            response: Raw API response dictionary

        Returns:
            BatchRunResult containing evaluation results
        """
        eval_results = []

        for result in response.get("result", {}):
            for evaluation in result.get("evaluations", []):
                new_metadata = {}
                if evaluation.get("metadata"):
                    if isinstance(evaluation.get("metadata"), dict):
                        metadata = evaluation.get("metadata")
                    elif isinstance(evaluation.get("metadata"), str):
                        metadata = json.loads(evaluation.get("metadata"))
                    else:
                        metadata = {}
                    new_metadata["usage"] = metadata.get("usage", {})
                    new_metadata["cost"] = metadata.get("cost", {})
                    new_metadata["explanation"] = metadata.get("explanation", {})
                eval_results.append(
                    EvalResult(
                        data=evaluation.get("data"),
                        failure=evaluation.get("failure"),
                        reason=evaluation.get("reason", ""),
                        runtime=evaluation.get("runtime", 0),
                        metadata=new_metadata,
                        metrics=[
                            EvalResultMetric(id=metric["id"], value=metric["value"])
                            for metric in evaluation.get("metrics", [])
                        ],
                    )
                )

        return BatchRunResult(eval_results=eval_results)


class EvalInfoResponseHandler(ResponseHandler[dict, None]):
    """Handles responses for evaluation info requests"""

    @classmethod
    def _parse_success(cls, response: Response) -> dict:
        data = response.json()
        if "result" in data:
            return data["result"]
        else:
            raise Exception(f"Failed to get evaluation info: {data}")

    @classmethod
    def _handle_error(cls, response: Response) -> None:
        if response.status_code == 400:
            response.raise_for_status()
        if response.status_code == 403:
            raise InvalidAuthError()
        raise Exception(f"Failed to get evaluation info: {response.status_code}")


class Evaluator(APIKeyAuth):
    """Client for evaluating LLM test cases"""

    def __init__(
        self,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Eval Client

        Args:
            fi_api_key: API key
            fi_secret_key: Secret key
            fi_base_url: Base URL

        Keyword Args:
            timeout: Optional timeout value in seconds (default: 200)
            max_queue_bound: Optional maximum queue size (default: 5000)
            max_workers: Optional maximum number of workers (default: 8)
        """
        super().__init__(fi_api_key, fi_secret_key, fi_base_url, **kwargs)
        self._max_workers = kwargs.get("max_workers", 8)  # Default to 8 if not provided

    def evaluate(
        self,
        eval_templates: Union[str, type[EvalTemplate]],
        inputs: Union[
            TestCase,
            List[TestCase],
            Dict[str, Any],
            List[Dict[str, Any]],
        ],
        timeout: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> BatchRunResult:
        """
        Run a single or batch of evaluations independently

        Args:
            eval_templates: Evaluation name string (e.g., "Factual Accuracy")
            inputs: Single test case or list of test cases
            timeout: Optional timeout value for the evaluation
            model_name: Optional model name to use for the evaluation for Future AGI Agents

        Returns:
            BatchRunResult containing evaluation results

        Raises:
            ValidationError: If the inputs do not match the evaluation templates
            Exception: If the API request fails
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        def _extract_name(t) -> str | None:
            if isinstance(t, str):
                return t
            if isinstance(t, EvalTemplate):
                return t.eval_name
            if inspect.isclass(t) and issubclass(t, EvalTemplate):
                return t.eval_name
            return None
        eval_name = _extract_name(
            eval_templates[0] if isinstance(eval_templates, list) else eval_templates
        )
        if eval_name is None:
            raise TypeError(
                "Unsupported eval_templates argument. "
                "Expect eval template class/obj or name str."
            )

        

        transformed_api_inputs = defaultdict(list)

        if not inputs:
            # Allows an empty transformed_api_inputs if the original inputs list was empty.
            pass
        else:
            for tc in inputs:
                dumped_tc: Dict[str, Any]
                # Handle both TestCase objects and raw dictionaries
                if isinstance(tc, TestCase):
                    # TestCase object
                    dumped_tc = tc.model_dump(exclude_none=True, by_alias=False)
                elif isinstance(tc, dict):
                    # Raw dictionary
                    processed_tc = {}
                    for key, value in tc.items():
                        if isinstance(value, str):
                            original_value = value # Keep a copy for fallback
                            try:
                                mllm_image = MLLMImage(url=value)
                                processed_tc[key] = mllm_image.url
                                logging.debug(f"Processed key '{key}' as MLLMImage.")
                                continue 
                            except ValueError as e_img: 
                                logging.debug(f"Key '{key}' not processed as MLLMImage (Error: {e_img}). Trying MLLMAudio.")
                                try:
                                    mllm_audio = MLLMAudio(url=value)
                                    if not mllm_audio.is_plain_text:
                                        processed_tc[key] = mllm_audio.url
                                        logging.debug(f"Processed key '{key}' as MLLMAudio.")
                                        continue 
                                    else:
                                        logging.debug(f"Key '{key}' treated as plain text by MLLMAudio. Using original value.")
                                        processed_tc[key] = original_value
                                except ValueError as e_audio: 
                                    logging.debug(f"Key '{key}' not processed as MLLMAudio (Error: {e_audio}). Using original value.")
                                    processed_tc[key] = original_value
                                except Exception as e_gen_audio: 
                                    logging.warning(f"Unexpected error processing key '{key}' as MLLMAudio (Error: {e_gen_audio}). Using original value.")
                                    processed_tc[key] = original_value
                            except Exception as e_gen_image: 
                                logging.warning(f"Unexpected error processing key '{key}' as MLLMImage (Error: {e_gen_image}). Using original value.")
                                processed_tc[key] = original_value
                        else:
                            # Not a string, keep original value
                            processed_tc[key] = value
                    dumped_tc = processed_tc
                else:
                    raise TypeError(
                        f"Invalid input type: {type(tc)}. Each input must be a TestCase object or a dictionary with valid keys."
                    )
                
                for key, value in dumped_tc.items():
                    transformed_api_inputs[key].append(value)

        final_api_payload = {
            "eval_name": eval_name,
            "inputs": dict(transformed_api_inputs),
            "model": model_name,
        }

        
        all_results = []
        failed_inputs = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit the batch only once
            future = executor.submit(
                self.request,
                config=RequestConfig(
                    method=HttpMethod.POST,
                    url=f"{self._base_url}/{Routes.evaluatev2.value}",
                    json=final_api_payload,
                    timeout=timeout or self._default_timeout,
                ),
                response_handler=EvalResponseHandler,
            )
            future_to_input = {future: inputs}  # map single future to all inputs

            for future in as_completed(future_to_input):
                try:
                    response: BatchRunResult = future.result(timeout=timeout or self._default_timeout)
                    all_results.extend(response.eval_results)
                except TimeoutError:
                    input_case = future_to_input[future]
                    logging.error(f"Evaluation timed out for input: {input_case}")
                    failed_inputs.append(input_case)
                
                except Exception as exc:
                    input_case = future_to_input[future]
                    logging.error(f"Evaluation failed for input {input_case}: {str(exc)}")
                    failed_inputs.append(input_case)

        if failed_inputs:
            logging.warning(f"Failed to evaluate {len(failed_inputs)} inputs out of {len(inputs)} total inputs")

        return BatchRunResult(eval_results=all_results)

    def _validate_inputs(
        self,
        inputs: List[TestCase],
        eval_objects: List[EvalTemplate],
    ):
        """
        Validate the inputs against the evaluation templates

        Args:
            inputs: List of test cases to validate
            eval_objects: List of evaluation templates to validate against

        Returns:
            bool: True if validation passes

        Raises:
            Exception: If validation fails or templates don't share common tags
        """

        # First validate that all eval objects share at least one common tag
        if len(eval_objects) > 1:
            # Get sets of tags for each eval object
            tag_sets = [set(obj.eval_tags) for obj in eval_objects]

            # Find intersection of all tag sets
            common_tags = set.intersection(*tag_sets)

            if not common_tags:
                template_names = [obj.name for obj in eval_objects]
                raise Exception(
                    f"Evaluation templates {template_names} must share at least one common tag. "
                    f"Current tags for each template: {[list(tags) for tags in tag_sets]}"
                )

        # Then validate each eval object's required inputs
        for eval_object in eval_objects:
            eval_object.validate_input(inputs)

        return True

    def _get_eval_configs(
        self, eval_templates: Union[str, List[str]]
    ) -> List[EvalTemplate]:
        if isinstance(eval_templates, str):
            eval_templates = [eval_templates]

        for template in eval_templates:
            eval_info = self._get_eval_info(template)
            template.eval_id = eval_info["eval_id"]
            template.name = eval_info["name"]
            template.description = eval_info["description"]
            template.eval_tags = eval_info["eval_tags"]
            template.required_keys = eval_info["config"]["required_keys"]
            template.output = eval_info["config"]["output"]
            template.eval_type_id = eval_info["config"]["eval_type_id"]
            template.config_schema = (
                eval_info["config"]["config"] if "config" in eval_info["config"] else {}
            )
            template.criteria = eval_info["criteria"]
            template.choices = eval_info["choices"]
            template.multi_choice = eval_info["multi_choice"]
        return eval_templates

    @lru_cache(maxsize=100)
    def _get_eval_info(self, eval_name: str) -> Dict[str, Any]:
        url = (
            self._base_url
            + "/"
            + Routes.get_eval_templates.value
        )
        response = self.request(
            config=RequestConfig(method=HttpMethod.GET, url=url),
            response_handler=EvalInfoResponseHandler,
        )
        eval_info = next((item for item in response if item["name"] == eval_name), None)
        if eval_info is None:
            raise KeyError(f"Evaluation template '{eval_name}' not found in registry")
        if not eval_info:
            raise Exception(f"Evaluation template with name '{eval_name}' not found")
        return eval_info

    def list_evaluations(self):
        """
        Fetch information about all available evaluation templates by getting eval_info
        for each template class defined in templates.py.

        Returns:
            List[Dict[str, Any]]: List of evaluation template information dictionaries
        """
        config = RequestConfig(method=HttpMethod.GET,
                                url=f"{self._base_url}/{Routes.get_eval_templates.value}")
                                
        response = self.request(config=config, response_handler=EvalInfoResponseHandler)

        return response
evaluate = lambda eval_templates, inputs, timeout=None: Evaluator().evaluate(eval_templates, inputs, timeout)

list_evaluations = lambda: Evaluator().list_evaluations()