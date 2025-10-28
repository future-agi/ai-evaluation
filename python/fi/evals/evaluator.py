import inspect
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pandas as pd
from requests import Response

from fi.api.auth import APIKeyAuth, ResponseHandler
from fi.api.types import HttpMethod, RequestConfig
from fi.evals.templates import EvalTemplate
from fi.evals.types import BatchRunResult, EvalResult
from fi.testcases import TestCase
from fi.utils.errors import InvalidAuthError
from fi.utils.routes import Routes

try:
    from opentelemetry import trace
    from fi_instrumentation.otel import check_custom_eval_config_exists
    from opentelemetry import trace as otel_trace_api
except ImportError:
    pass


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
                        name=evaluation.get("name", ""),
                        output=evaluation.get("output", None),
                        reason=evaluation.get("reason", ""),
                        runtime=evaluation.get("runtime", 0),
                        output_type=evaluation.get("outputType", ""),
                        eval_id=evaluation.get("evalId", ""),
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
            langfuse_secret_key: Optional Langfuse secret key
            langfuse_public_key: Optional Langfuse public key
            langfuse_host: Optional Langfuse host
        """
        super().__init__(fi_api_key, fi_secret_key, fi_base_url, **kwargs)
        self._max_workers = kwargs.get("max_workers", 8)  # Default to 8 if not provided
        
        # Handle Langfuse credentials
        self.langfuse_secret_key = kwargs.get("langfuse_secret_key") or os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_public_key = kwargs.get("langfuse_public_key") or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = kwargs.get("langfuse_host") or os.getenv("LANGFUSE_HOST")


    def evaluate(
        self,
        eval_templates: Union[str, type[EvalTemplate]],
        inputs: Dict[str, Any],
        timeout: Optional[int] = None,
        model_name: Optional[str] = None,
        custom_eval_name: Optional[str] = None,
        trace_eval: Optional[bool] = False,
        platform: Optional[str] = None,
        is_async: Optional[bool] = False,
        error_localizer: Optional[bool] = False,
        **kwargs,
    ) -> BatchRunResult:
        """
        Run a single or batch of evaluations independently

        Args:
            eval_templates: Evaluation name string (e.g., "Factual Accuracy")
            inputs: Single test case or list of test cases
            timeout: Optional timeout value for the evaluation
            model_name: Optional model name to use for the evaluation for Future AGI Agents
            span_id: Optional span_id to attach to the evaluation. If not provided, it will be retrieved from the OpenTelemetry context if available.
            custom_eval_name: Optional custom evaluation name to use for the evaluation. If not provided, eval will not be added to the span.
        Returns:
            BatchRunResult containing evaluation results

        Raises:
            ValidationError: If the inputs do not match the evaluation templates
            Exception: If the API request fails
        """
        if platform:
            if isinstance(eval_templates, str) and isinstance(inputs, dict) and custom_eval_name:
                return self._configure_evaluations(
                    eval_templates=eval_templates,
                    inputs=inputs,
                    platform=platform,
                    custom_eval_name=custom_eval_name,
                    model_name=model_name,
                    **kwargs
                )
            else:
                raise ValueError("Invalid arguments for platform configuration")


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

        span_id = None
        project_name = None
        if trace_eval:
            if not custom_eval_name:
                trace_eval = False
                logging.warning("Failed to trace the evaluation. Please set the custom_eval_name.")
            else:
                try:
                    from opentelemetry import trace
                    from fi_instrumentation.otel import check_custom_eval_config_exists

                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        span_context = current_span.get_span_context()
                        if span_context.is_valid:
                            span_id = format(span_context.span_id, "016x")
                            tracer_provider = trace.get_tracer_provider()
                            if hasattr(tracer_provider, "resource"):
                                attributes = tracer_provider.resource.attributes
                                project_name = attributes.get("project_name")
                    
                    if project_name:
                        eval_tags = [
                            {
                                "custom_eval_name": custom_eval_name,
                                "eval_name": eval_name,
                                "mapping": {},
                                "config": {},
                            }
                        ]
                        custom_eval_exists = check_custom_eval_config_exists(
                            project_name=project_name,
                            eval_tags=eval_tags,
                        )

                        if custom_eval_exists:
                            trace_eval = False
                            logging.warning("Failed to trace the evaluation. Custom eval configuration with the same name already exists for this project")
                    else:
                        trace_eval = False
                        logging.warning(
                            "Could not determine project_name from OpenTelemetry context. "
                            "Skipping check for existing custom eval configuration."
                        )

                except ImportError:
                    logging.exception(
                        "Future AGI SDK not found. "
                        "Please install 'fi-instrumentation-otel' to automatically enrich the evaluation with project context."
                    )
                    return

        if eval_name is None:
            raise TypeError(
                "Unsupported eval_templates argument. "
                "Expect eval template class/obj or name str."
            )

        final_api_payload = {
            "eval_name": eval_name,
            "inputs": inputs,
            "model": model_name,
            "span_id": span_id,
            "custom_eval_name": custom_eval_name,
            "trace_eval": trace_eval,
            "is_async": is_async,
            "error_localizer": error_localizer,
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


    def get_eval_result(self, eval_id: str):
        """
        Get the result of an evaluation by its ID
        """
        url = f"{self._base_url}/{Routes.get_eval_result.value}"
        response = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=url,
                params={"eval_id": eval_id},
                timeout=self._default_timeout,
            ),
        )

        return response.json()


    def _configure_evaluations(
        self,
        eval_templates: str,
        inputs: Dict[str, Any],
        platform: str,
        custom_eval_name: str,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Configure evaluations on a specified platform.

        This will not return any evaluation results, but rather a
        confirmation message from the backend.

        Args:
            eval_config: The evaluation configuration dictionary.
            platform: The platform to which the evaluations should be sent.
            timeout: Optional timeout for the API request.
            **kwargs: Additional configuration parameters to be sent with the request.

        Returns:
            A dictionary containing the backend's response message.
        """
        try:
            from fi.evals.otel_utils import _get_current_otel_span
            
            if platform == "langfuse":
                kwargs["langfuse_secret_key"] = self.langfuse_secret_key
                kwargs["langfuse_public_key"] = self.langfuse_public_key
                kwargs["langfuse_host"] = self.langfuse_host

            current_span = _get_current_otel_span()
            if current_span:
                span_context = current_span.get_span_context()
                if span_context.is_valid:
                    span_id = format(span_context.span_id, "016x")
                    trace_id = format(span_context.trace_id, "032x")
                    kwargs["span_id"] = span_id
                    kwargs["trace_id"] = trace_id
                
            # Check if span_id and trace_id are present in kwargs
            if "span_id" not in kwargs or "trace_id" not in kwargs:
                logging.warning(
                    "span_id and/or trace_id not found in kwargs ."
                    "Please run this function within a span context."
                )
                return

            api_payload = {
                "eval_config": {
                    "eval_templates": eval_templates,
                    "inputs": inputs,
                    "model_name": model_name
                },
                "custom_eval_name": custom_eval_name,
                "platform": platform,
                **kwargs,
            }
            
            response = self.request(
                config=RequestConfig(
                    method=HttpMethod.POST,
                    url=f"{self._base_url}/{Routes.configure_evaluations.value}",
                    json=api_payload,
                    timeout=self._default_timeout,
                ),
            )

            if response.status_code != 200:
                logging.warning(
                    f"Received non-200 status code from backend: {response.status_code}. "
                    f"Response: {response.text}"
                )

            return response.json()
        
        except ImportError:
            logging.exception(
                "Future AGI SDK not found. "
                "Please install 'fi-instrumentation-otel' to use these evaluations."
            )
            return


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
    

    def evaluate_pipeline(
            self,
            project_name: str,
            version : str,
            eval_data : List[Dict[str, Any]],
    ):
        api_payload = {
            "project_name": project_name,
            "version": version,
            "eval_data": eval_data
        }

        response = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.evaluate_pipeline.value}",
                json=api_payload,
                timeout=self._default_timeout,
            ),
        )
    
        return response.json()
    
    
    def get_pipeline_results(
            self,
            project_name: str,
            versions : List[str],
    ):
        
        if not isinstance(versions, list) or not all(isinstance(v, str) for v in versions):
            raise TypeError("versions must be a list of strings")
        
        api_payload = {
            "project_name": project_name,
            "versions": ",".join(versions),
        }

        response = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=f"{self._base_url}/{Routes.evaluate_pipeline.value}",
                params=api_payload,
                timeout=self._default_timeout,
            ),
        )

        return response.json()


evaluate = lambda eval_templates, inputs, timeout=None: Evaluator().evaluate(eval_templates, inputs, timeout)

list_evaluations = lambda: Evaluator().list_evaluations()



