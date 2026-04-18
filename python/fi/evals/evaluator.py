import inspect
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from requests import Response

from fi.api.auth import APIKeyAuth, ResponseHandler
from fi.api.types import HttpMethod, RequestConfig
from fi.evals.execution import Execution, _normalize_status
from fi.evals.templates import EvalTemplate
from fi.evals.types import BatchRunResult, EvalResult
from fi.utils.errors import InvalidAuthError
from fi.utils.routes import Routes

try:
    from opentelemetry import trace
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

        # The revamped backend (post 2026-04-12) returns pure snake_case:
        #   {"result": [{"evaluations": [
        #       {"name", "reason", "runtime", "output", "output_type",
        #        "eval_id", "model"?, "error_localizer_enabled"?,
        #        "error_localizer"?}
        #   ]}]}
        # Async / error-localization paths may return the eval wrapped in
        # {"eval_status": "...", "result": <eval>} — handle that too.
        for result in response.get("result", []) or []:
            if isinstance(result, dict) and "evaluations" in result:
                entries = result.get("evaluations", []) or []
            else:
                entries = [result] if isinstance(result, dict) else []

            for evaluation in entries:
                if not isinstance(evaluation, dict):
                    continue
                eval_results.append(
                    EvalResult(
                        name=evaluation.get("name", ""),
                        output=evaluation.get("output", evaluation.get("value")),
                        reason=evaluation.get("reason", ""),
                        runtime=evaluation.get("runtime", 0),
                        output_type=evaluation.get("output_type", ""),
                        eval_id=evaluation.get("eval_id", ""),
                        model=evaluation.get("model"),
                        error_localizer_enabled=evaluation.get(
                            "error_localizer_enabled"
                        ),
                        error_localizer=evaluation.get("error_localizer"),
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
        eval_config: Optional[Dict[str, Any]] = None,
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

                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        span_context = current_span.get_span_context()
                        if span_context.is_valid:
                            span_id = format(span_context.span_id, "016x")
                            tracer_provider = trace.get_tracer_provider()
                            if hasattr(tracer_provider, "resource"):
                                attributes = tracer_provider.resource.attributes
                                project_name = attributes.get("project_name")

                    if not project_name:
                        trace_eval = False
                        logging.warning(
                            "Could not determine project_name from OpenTelemetry context. "
                            "Skipping trace_eval for this evaluation."
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

        # Dynamic registry: filter user-supplied inputs to only the keys the
        # backend currently accepts for this eval. The api rejects supersets
        # (e.g. {output,input,context} for a template that only wants
        # {output}), so this can't be a pass-through. If the registry fetch
        # fails or the name is unknown, leave inputs untouched.
        if kwargs.get("skip_input_mapping") is not True and isinstance(inputs, dict):
            try:
                from fi.evals.core.cloud_registry import map_inputs_to_backend
                inputs = map_inputs_to_backend(
                    eval_name,
                    inputs,
                    base_url=self._base_url,
                    api_key=self._fi_api_key,
                    secret_key=self._fi_secret_key,
                )
            except Exception as exc:
                logging.debug("Dynamic input mapping skipped: %s", exc)

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

        if eval_config:
            final_api_payload["config"] = {"params": eval_config}

        
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
                    all_results.append(
                        EvalResult(
                            name=eval_name,
                            output=None,
                            reason=f"Evaluation timed out after {timeout or self._default_timeout}s",
                            runtime=0,
                        )
                    )
                except Exception as exc:
                    input_case = future_to_input[future]
                    logging.error(f"Evaluation failed for input {input_case}: {str(exc)}")
                    failed_inputs.append(input_case)
                    all_results.append(
                        EvalResult(
                            name=eval_name,
                            output=None,
                            reason=str(exc),
                            runtime=0,
                        )
                    )

        if failed_inputs:
            logging.warning(f"Failed to evaluate {len(failed_inputs)} inputs out of {len(inputs)} total inputs")

        # Automatically enrich current span with evaluation results
        result = BatchRunResult(eval_results=all_results)
        try:
            from fi.evals.otel.enrichment import enrich_span_with_batch_result, is_auto_enrichment_enabled
            if is_auto_enrichment_enabled():
                enriched_count = enrich_span_with_batch_result(result)
                if enriched_count > 0:
                    logging.debug(f"Enriched active span with {enriched_count} evaluation results")
        except ImportError:
            pass  # OTEL enrichment not available
        except Exception as e:
            logging.debug(f"Failed to enrich span with evaluation results: {e}")

        return result


    def get_eval_result(self, eval_id: str):
        """
        Get the raw evaluation status payload by ID (unparsed).

        For a higher-level handle that understands the status envelope
        and can be awaited, see :py:meth:`get_execution`.
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

    # ------------------------------------------------------------------
    # Async submission / execution handles
    # ------------------------------------------------------------------

    def submit(
        self,
        eval_templates: Union[str, type[EvalTemplate]],
        inputs: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        custom_eval_name: Optional[str] = None,
        trace_eval: bool = False,
        error_localizer: bool = False,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Execution:
        """
        Submit an eval for async execution and return an :class:`Execution`
        handle immediately. Use ``handle.wait()`` or
        :py:meth:`get_execution` to poll for completion.

        This is the non-blocking equivalent of :py:meth:`evaluate` —
        internally it always sets ``is_async=True`` so the backend records
        the evaluation and starts a worker without holding the HTTP
        connection open.
        """

        def _extract_name(t: Any) -> Optional[str]:
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

        payload = {
            "eval_name": eval_name,
            "inputs": inputs,
            "model": model_name,
            "span_id": kwargs.get("span_id"),
            "custom_eval_name": custom_eval_name,
            "trace_eval": trace_eval,
            "is_async": True,
            "error_localizer": error_localizer,
        }

        response = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.evaluatev2.value}",
                json=payload,
                timeout=timeout or self._default_timeout,
            ),
        )
        body = response.json() if hasattr(response, "json") else {}

        # Backend responds with:
        #   {"status": true,  "result": [{"evaluations": [{name, output_type, eval_id}]}]}
        # on success, or
        #   {"status": false, "result": {...error dict...}}
        # on validation failure.
        if not body.get("status", True):
            raise RuntimeError(
                f"Async submit rejected by backend: {body.get('result')}"
            )

        results = body.get("result") or []
        if not isinstance(results, list) or not results:
            raise RuntimeError(
                f"Async submit did not return a result list (response: {body})"
            )
        evaluations = results[0].get("evaluations") or []
        first_eval = evaluations[0] if evaluations else {}
        execution_id = first_eval.get("eval_id")
        if not execution_id:
            raise RuntimeError(
                f"Async submit did not return an eval_id (response: {body})"
            )

        handle = Execution(
            id=str(execution_id),
            kind="eval",
            status="pending",
        )
        handle._refresher = lambda eid=execution_id: self._refresh_eval_execution(eid)
        return handle

    def get_execution(self, execution_id: str) -> Execution:
        """
        Fetch the latest state of an async single-eval execution by ID.

        Returns an :class:`Execution` handle with an attached refresher
        closure — call ``handle.wait()`` to block until completion.
        """
        handle = self._refresh_eval_execution(execution_id)
        handle._refresher = (
            lambda eid=execution_id: self._refresh_eval_execution(eid)
        )
        return handle

    def _refresh_eval_execution(self, execution_id: str) -> Execution:
        url = f"{self._base_url}/{Routes.get_eval_result.value}"
        response = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=url,
                params={"eval_id": execution_id},
                timeout=self._default_timeout,
            ),
        )
        body = response.json() if hasattr(response, "json") else {}
        # Envelope: {"status": true, "result": {"eval_status": ..., "result": <body|str>}}
        payload = body.get("result") or {}
        status = _normalize_status(payload.get("eval_status"))
        raw_result = payload.get("result")
        error_message = payload.get("error_message")

        parsed_result: Any = None
        error_localizer: Optional[Dict[str, Any]] = None
        if isinstance(raw_result, dict):
            # Completed / failed state — full eval record.
            parsed_result = EvalResult(
                name=raw_result.get("name", ""),
                output=raw_result.get("output", raw_result.get("value")),
                reason=raw_result.get("reason"),
                runtime=raw_result.get("runtime", 0),
                output_type=raw_result.get("output_type"),
                eval_id=str(raw_result.get("eval_id", execution_id)),
                model=raw_result.get("model"),
                error_localizer_enabled=raw_result.get("error_localizer_enabled"),
                error_localizer=raw_result.get("error_localizer"),
            )
            error_localizer = raw_result.get("error_localizer")
            error_message = raw_result.get("error_message") or error_message
        # else: raw_result is a human-readable string like "Evaluation is
        # being processed." — leave parsed_result as None.

        return Execution(
            id=str(execution_id),
            kind="eval",
            status=status,
            result=parsed_result,
            error_message=error_message,
            error_localizer=error_localizer,
        )


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
        inputs: Dict[str, Any],
        eval_objects: List[EvalTemplate],
    ) -> bool:
        """Backward-compatible stub. Client-side input validation now
        happens dynamically via :mod:`fi.evals.core.cloud_registry`
        against the api's ``required_keys``. Any mismatch surfaces as a
        backend 400 with a clear error message.
        """
        return True

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


# Top-level convenience for the common "list everything" case.
# The main ``evaluate()`` entrypoint is imported from ``fi.evals.core``.
list_evaluations = lambda: Evaluator().list_evaluations()



