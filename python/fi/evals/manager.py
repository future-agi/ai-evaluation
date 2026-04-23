"""
Management client for the revamped eval template API.

Covers the `/model-hub/eval-templates/*` surface introduced in the backend
Evals Revamp (Phases 1–7): list, create, detail, update, delete, versions,
and composite evals. For execution of single evals use
``fi.evals.Evaluator.evaluate``; this client is for managing the templates
themselves.
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from requests import Response

from fi.api.auth import APIKeyAuth, ResponseHandler
from fi.api.types import HttpMethod, RequestConfig
from fi.evals.execution import Execution
from fi.evals.manager_types import (
    BulkDeleteResponse,
    CompositeCreateResponse,
    CompositeDetailResponse,
    CompositeExecutionResponse,
    GroundTruthConfigResponse,
    GroundTruthDataResponse,
    GroundTruthDeleteResponse,
    GroundTruthListResponse,
    GroundTruthRoleMappingResponse,
    GroundTruthSearchResponse,
    GroundTruthStatusResponse,
    GroundTruthUploadResponse,
    GroundTruthVariableMappingResponse,
    PlaygroundRunResponse,
    TemplateChartsResponse,
    TemplateCreateResponse,
    TemplateDetailResponse,
    TemplateDuplicateResponse,
    TemplateFeedbackListResponse,
    TemplateListResponse,
    TemplateUpdateResponse,
    TemplateUsageResponse,
    VersionCreateResponse,
    VersionListResponse,
    VersionRestoreResponse,
    VersionSetDefaultResponse,
)
from fi.utils.errors import InvalidAuthError
from fi.utils.routes import Routes


logger = logging.getLogger(__name__)


class _JsonResponseHandler(ResponseHandler[Dict[str, Any], None]):
    """Return the ``result`` payload from a standard backend response."""

    @classmethod
    def _parse_success(cls, response: Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise Exception(f"Non-JSON response: {response.text}")
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    @classmethod
    def _handle_error(cls, response: Response) -> None:
        if response.status_code == 403:
            raise InvalidAuthError()
        if response.status_code == 404:
            raise Exception(f"Not found: {response.text}")
        raise Exception(
            f"Eval template API error {response.status_code}: {response.text}"
        )


class EvalTemplateManager(APIKeyAuth):
    """
    Client for managing eval templates on Future AGI.

    Example::

        from fi.evals import EvalTemplateManager

        mgr = EvalTemplateManager()
        t = mgr.create_template(
            name="is-polite",
            eval_type="llm",
            instructions="Is the {{output}} polite? Answer yes or no.",
            output_type="pass_fail",
            pass_threshold=0.7,
        )
        print(t.id)           # attribute access (autocomplete ✓)
        print(t["id"])         # dict access (backwards compat ✓)
        mgr.delete_template(t.id)
    """

    def _typed_request(self, config: RequestConfig, response_type: Any = None) -> Any:
        """Make a request and parse the response into a typed model."""
        raw = self.request(config=config, response_handler=_JsonResponseHandler)
        if response_type is None or isinstance(raw, str):
            return raw
        return response_type.model_validate(raw)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def list_templates(
        self,
        *,
        page: int = 0,
        page_size: int = 25,
        search: Optional[str] = None,
        owner_filter: Literal["all", "user", "system"] = "all",
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Literal["name", "updated_at", "created_at"] = "updated_at",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> TemplateListResponse:
        """List eval templates with pagination, search, and filters.

        Returns a dict with ``items``, ``total``, ``page``, ``page_size``.
        """
        payload: Dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "owner_filter": owner_filter,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        if search is not None:
            payload["search"] = search
        if filters is not None:
            payload["filters"] = filters

        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_list.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateListResponse.model_validate(raw)

    def get_template(self, template_id: str) -> TemplateDetailResponse:
        """Fetch a single eval template by UUID."""
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_detail.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateDetailResponse.model_validate(raw)

    def create_template(
        self,
        *,
        name: str,
        instructions: str = "",
        eval_type: Literal["llm", "code", "agent"] = "llm",
        model: str = "turing_large",
        output_type: Literal["pass_fail", "percentage", "deterministic"] = "pass_fail",
        pass_threshold: float = 0.5,
        choice_scores: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        check_internet: bool = False,
        code: Optional[str] = None,
        code_language: Optional[Literal["python", "javascript"]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[Literal["auto", "agent", "quick"]] = None,
        tools: Optional[Dict[str, Any]] = None,
        knowledge_bases: Optional[List[str]] = None,
        data_injection: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
        is_draft: bool = False,
    ) -> TemplateCreateResponse:
        """
        Create a single eval template.

        The ``name`` must match ``[a-z0-9_-]+`` and cannot start/end with a
        separator (backend constraint). For ``eval_type="code"`` supply
        ``code``; otherwise supply ``instructions`` that reference at least
        one ``{{variable}}`` unless ``data_injection`` is enabled.
        """
        payload: Dict[str, Any] = {
            "name": name,
            "is_draft": is_draft,
            "eval_type": eval_type,
            "instructions": instructions,
            "model": model,
            "output_type": output_type,
            "pass_threshold": pass_threshold,
            "tags": list(tags or []),
            "check_internet": check_internet,
        }
        if choice_scores is not None:
            payload["choice_scores"] = choice_scores
        if description is not None:
            payload["description"] = description
        if code is not None:
            payload["code"] = code
        if code_language is not None:
            payload["code_language"] = code_language
        if messages is not None:
            payload["messages"] = messages
        if few_shot_examples is not None:
            payload["few_shot_examples"] = few_shot_examples
        if mode is not None:
            payload["mode"] = mode
        if tools is not None:
            payload["tools"] = tools
        if knowledge_bases is not None:
            payload["knowledge_bases"] = knowledge_bases
        if data_injection is not None:
            payload["data_injection"] = data_injection
        if summary is not None:
            payload["summary"] = summary

        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_create_v2.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateCreateResponse.model_validate(raw)

    def update_template(
        self, template_id: str, **fields: Any
    ) -> TemplateUpdateResponse:
        """
        Update mutable fields on an eval template (PUT).

        Accepts any subset of: ``name``, ``eval_type``, ``instructions``,
        ``model``, ``output_type``, ``pass_threshold``, ``choice_scores``,
        ``multi_choice``, ``description``, ``tags``, ``check_internet``,
        ``code``, ``code_language``, ``messages``, ``few_shot_examples``,
        ``mode``, ``tools``, ``knowledge_bases``, ``data_injection``,
        ``summary``, ``error_localizer_enabled``, ``publish``.
        """
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_update_v2.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PUT,
                url=url,
                json={k: v for k, v in fields.items() if v is not None},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateUpdateResponse.model_validate(raw)

    def delete_template(self, template_id: str) -> Dict[str, Any]:
        """Soft-delete an eval template."""
        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_delete.value}",
                json={"eval_template_id": template_id},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    def bulk_delete_templates(self, template_ids: List[str]) -> BulkDeleteResponse:
        """Soft-delete multiple eval templates in one call."""
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_bulk_delete.value}",
                json={"template_ids": list(template_ids)},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return BulkDeleteResponse.model_validate(raw)

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def list_versions(self, template_id: str) -> VersionListResponse:
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_list.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return VersionListResponse.model_validate(raw)

    def create_version(
        self,
        template_id: str,
        *,
        criteria: Optional[str] = None,
        model: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> VersionCreateResponse:
        """
        Snapshot the current template state as a new version. Each param is
        optional — any value left ``None`` is copied from the live template.
        """
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_create.value.format(template_id=template_id)
        )
        payload: Dict[str, Any] = {}
        if criteria is not None:
            payload["criteria"] = criteria
        if model is not None:
            payload["model"] = model
        if config_snapshot is not None:
            payload["config_snapshot"] = config_snapshot

        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return VersionCreateResponse.model_validate(raw)

    def set_default_version(
        self, template_id: str, version_id: str
    ) -> VersionSetDefaultResponse:
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_set_default.value.format(
                template_id=template_id, version_id=version_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PUT, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return VersionSetDefaultResponse.model_validate(raw)

    def restore_version(
        self, template_id: str, version_id: str
    ) -> VersionRestoreResponse:
        """
        Create a new version on top of the template whose contents match
        the target version. The old version is left untouched.
        """
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_restore.value.format(
                template_id=template_id, version_id=version_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return VersionRestoreResponse.model_validate(raw)

    # ------------------------------------------------------------------
    # Composite evals
    # ------------------------------------------------------------------

    def create_composite(
        self,
        *,
        name: str,
        child_template_ids: List[str],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        aggregation_enabled: bool = True,
        aggregation_function: Literal[
            "weighted_avg", "avg", "min", "max", "pass_rate"
        ] = "weighted_avg",
        child_weights: Optional[Dict[str, float]] = None,
        composite_child_axis: Literal[
            "", "pass_fail", "percentage", "choices", "code"
        ] = "",
    ) -> CompositeCreateResponse:
        """
        Create a composite eval from existing eval templates. ``child_weights``
        maps template_id → weight (default 1.0). Setting
        ``composite_child_axis`` enforces a homogeneity check — every child
        must match that output axis (e.g. all ``pass_fail``).
        """
        payload: Dict[str, Any] = {
            "name": name,
            "child_template_ids": list(child_template_ids),
            "aggregation_enabled": aggregation_enabled,
            "aggregation_function": aggregation_function,
            "composite_child_axis": composite_child_axis,
            "tags": list(tags or []),
        }
        if description is not None:
            payload["description"] = description
        if child_weights is not None:
            payload["child_weights"] = child_weights

        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.composite_eval_create.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return CompositeCreateResponse.model_validate(raw)

    def get_composite(self, template_id: str) -> CompositeDetailResponse:
        url = (
            f"{self._base_url}/"
            + Routes.composite_eval_detail.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return CompositeDetailResponse.model_validate(raw)

    def update_composite(
        self,
        template_id: str,
        **fields: Any,
    ) -> CompositeDetailResponse:
        """PATCH the composite. Same field names as ``create_composite``."""
        url = (
            f"{self._base_url}/"
            + Routes.composite_eval_detail.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PATCH,
                url=url,
                json={k: v for k, v in fields.items() if v is not None},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return CompositeDetailResponse.model_validate(raw)

    def submit_composite(
        self,
        template_id: str,
        *,
        mapping: Dict[str, Any],
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        error_localizer: bool = False,
        input_data_types: Optional[Dict[str, str]] = None,
        span_context: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Execution:
        """
        Submit a composite eval for non-blocking execution.

        The backend's composite execute endpoint is still synchronous,
        so this method offloads the HTTP call to a background thread
        inside the calling process and returns an :class:`Execution`
        handle immediately. Use ``handle.wait()`` to block until the
        run finishes, or poll ``handle.is_done()`` yourself.

        IMPORTANT: The execution ID is client-local — if this process
        dies or you drop the handle, the in-flight work is lost. For
        cross-process resumable runs, use
        :py:meth:`fi.evals.Evaluator.submit` against a single eval.
        """
        handle = Execution(
            id=str(uuid.uuid4()),
            kind="composite",
            status="pending",
        )
        # Keep a reference so the thread isn't GC'd while running.
        handle._refresher = lambda h=handle: h  # type: ignore[attr-defined]

        def _runner() -> None:
            handle.status = "processing"
            try:
                result = self.execute_composite(
                    template_id,
                    mapping=mapping,
                    model=model,
                    config=config,
                    error_localizer=error_localizer,
                    input_data_types=input_data_types,
                    span_context=span_context,
                    trace_context=trace_context,
                    session_context=session_context,
                )
                handle.result = result
                if isinstance(result, dict):
                    handle.error_localizer = result.get("error_localizer_results")
                handle.status = "completed"
            except Exception as exc:  # noqa: BLE001
                handle.error_message = f"{exc.__class__.__name__}: {exc}"
                handle.status = "failed"
                logger.exception(
                    "Composite execution %s failed: %s", handle.id, exc
                )

        thread = threading.Thread(
            target=_runner, daemon=True, name=f"fi-composite-{handle.id[:8]}"
        )
        thread.start()
        return handle

    def execute_composite(
        self,
        template_id: str,
        *,
        mapping: Dict[str, Any],
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        error_localizer: bool = False,
        input_data_types: Optional[Dict[str, str]] = None,
        span_context: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> CompositeExecutionResponse:
        """
        Run every child eval in the composite with a single input mapping
        and return per-child results plus the aggregated score.
        """
        payload: Dict[str, Any] = {
            "mapping": mapping,
            "config": config or {},
            "error_localizer": error_localizer,
            "input_data_types": input_data_types or {},
        }
        if model is not None:
            payload["model"] = model
        if span_context is not None:
            payload["span_context"] = span_context
        if trace_context is not None:
            payload["trace_context"] = trace_context
        if session_context is not None:
            payload["session_context"] = session_context

        url = (
            f"{self._base_url}/"
            + Routes.composite_eval_execute.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return CompositeExecutionResponse.model_validate(raw)

    # ------------------------------------------------------------------
    # Ground Truth (Phase 9)
    # ------------------------------------------------------------------

    def list_ground_truth(self, template_id: str) -> GroundTruthListResponse:
        """List ground-truth datasets attached to an eval template."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_list.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthListResponse.model_validate(raw)

    def upload_ground_truth(
        self,
        template_id: str,
        *,
        name: str,
        columns: List[str],
        data: List[Dict[str, Any]],
        description: str = "",
        file_name: str = "",
        variable_mapping: Optional[Dict[str, str]] = None,
        role_mapping: Optional[Dict[str, str]] = None,
    ) -> GroundTruthUploadResponse:
        """
        Upload a ground-truth dataset as a JSON body (``columns`` +
        ``data`` rows). For file-upload mode (CSV/XLSX) use the HTTP API
        directly — the SDK does not wrap multipart uploads yet.
        """
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "file_name": file_name,
            "columns": list(columns),
            "data": list(data),
        }
        if variable_mapping is not None:
            payload["variable_mapping"] = variable_mapping
        if role_mapping is not None:
            payload["role_mapping"] = role_mapping

        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_upload.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthUploadResponse.model_validate(raw)

    def get_ground_truth_config(self, template_id: str) -> GroundTruthConfigResponse:
        """Fetch the ground-truth config block from an eval template."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_config.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthConfigResponse.model_validate(raw)

    def set_ground_truth_config(
        self,
        template_id: str,
        *,
        enabled: bool = True,
        ground_truth_id: Optional[str] = None,
        mode: Literal["auto", "manual", "disabled"] = "auto",
        max_examples: int = 3,
        similarity_threshold: float = 0.7,
        injection_format: Literal[
            "structured", "conversational", "xml"
        ] = "structured",
    ) -> GroundTruthConfigResponse:
        """
        Update the ground-truth configuration on an eval template
        (few-shot retrieval settings used at eval time).
        """
        payload = {
            "enabled": enabled,
            "ground_truth_id": ground_truth_id,
            "mode": mode,
            "max_examples": max_examples,
            "similarity_threshold": similarity_threshold,
            "injection_format": injection_format,
        }
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_config.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PUT,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthConfigResponse.model_validate(raw)

    def set_ground_truth_variable_mapping(
        self, ground_truth_id: str, variable_mapping: Dict[str, str]
    ) -> GroundTruthVariableMappingResponse:
        """Map eval variables → ground-truth column names."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_mapping.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PUT,
                url=url,
                json={"variable_mapping": variable_mapping},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthVariableMappingResponse.model_validate(raw)

    def set_ground_truth_role_mapping(
        self, ground_truth_id: str, role_mapping: Dict[str, str]
    ) -> GroundTruthRoleMappingResponse:
        """
        Map semantic roles (``input``, ``expected_output``, ``score``,
        ``reasoning``) → ground-truth column names.
        """
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_role_mapping.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.PUT,
                url=url,
                json={"role_mapping": role_mapping},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthRoleMappingResponse.model_validate(raw)

    def get_ground_truth_data(
        self,
        ground_truth_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> GroundTruthDataResponse:
        """Paginated ground-truth rows (1-based page)."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_data.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=url,
                params={"page": page, "page_size": page_size},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthDataResponse.model_validate(raw)

    def get_ground_truth_status(self, ground_truth_id: str) -> GroundTruthStatusResponse:
        """Embedding generation status / progress for a ground-truth dataset."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_status.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthStatusResponse.model_validate(raw)

    def trigger_ground_truth_embeddings(
        self, ground_truth_id: str
    ) -> Dict[str, Any]:
        """Kick off async embedding generation for a ground-truth dataset."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_embed.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

    def search_ground_truth(
        self,
        ground_truth_id: str,
        *,
        query: str,
        max_results: int = 3,
    ) -> GroundTruthSearchResponse:
        """
        Retrieve the most similar ground-truth rows to ``query``. Requires
        embeddings to be ``completed`` — call
        :py:meth:`trigger_ground_truth_embeddings` first and poll
        :py:meth:`get_ground_truth_status`.
        """
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_search.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json={"query": query, "max_results": max_results},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthSearchResponse.model_validate(raw)

    def delete_ground_truth(self, ground_truth_id: str) -> GroundTruthDeleteResponse:
        """Soft-delete a ground-truth dataset."""
        url = (
            f"{self._base_url}/"
            + Routes.ground_truth_delete.value.format(
                ground_truth_id=ground_truth_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.DELETE,
                url=url,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return GroundTruthDeleteResponse.model_validate(raw)

    # ------------------------------------------------------------------
    # Usage stats, feedback, and 30-day charts (Phase 10)
    # ------------------------------------------------------------------

    def get_template_usage(
        self,
        template_id: str,
        *,
        page: int = 0,
        page_size: int = 25,
        period: Literal["30m", "6h", "1d", "7d", "30d", "90d"] = "30d",
        version: Optional[str] = None,
    ) -> TemplateUsageResponse:
        """Paginated usage stats + run logs for an eval template."""
        params: Dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "period": period,
        }
        if version is not None:
            params["version"] = version
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_usage.value.format(template_id=template_id)
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=url,
                params=params,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateUsageResponse.model_validate(raw)

    def list_template_feedback(
        self,
        template_id: str,
        *,
        page: int = 0,
        page_size: int = 25,
    ) -> TemplateFeedbackListResponse:
        """Paginated user feedback on an eval template's runs."""
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_feedback_list.value.format(
                template_id=template_id
            )
        )
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.GET,
                url=url,
                params={"page": page, "page_size": page_size},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateFeedbackListResponse.model_validate(raw)

    def get_template_charts(
        self, template_ids: List[str]
    ) -> TemplateChartsResponse:
        """
        30-day sparkline data (run counts + error rates) for the given
        templates — the same payload that powers the eval-list table.
        """
        url = f"{self._base_url}/{Routes.eval_template_list_charts.value}"
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json={"template_ids": list(template_ids)},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateChartsResponse.model_validate(raw)

    # ------------------------------------------------------------------
    # Duplicate + playground
    # ------------------------------------------------------------------

    def duplicate_template(
        self, template_id: str, name: str
    ) -> TemplateDuplicateResponse:
        """
        Clone an existing user-owned eval template under a new ``name``.
        Returns ``{message, eval_template_id}``.
        """
        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_duplicate.value}",
                json={"eval_template_id": template_id, "name": name},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return TemplateDuplicateResponse.model_validate(raw)

    def run_playground(
        self,
        template_id: str,
        *,
        mapping: Dict[str, Any],
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        input_data_types: Optional[Dict[str, str]] = None,
        error_localizer: bool = False,
        kb_id: Optional[str] = None,
        # Auto-context: pass either a resolved dict, or let the server
        # resolve via the matching id.
        row_context: Optional[Dict[str, Any]] = None,
        span_context: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
        call_context: Optional[Dict[str, Any]] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> PlaygroundRunResponse:
        """
        Run a saved eval template via the playground endpoint.

        Unlike :py:meth:`fi.evals.Evaluator.evaluate` — which identifies
        the eval by name — this identifies by ``template_id`` and
        exercises the full runtime config path, including auto-context
        injection (row / span / trace / session / call). Useful for
        test-drive workflows that want to run a specific template
        version against a specific trace or dataset row.
        """
        payload: Dict[str, Any] = {
            "template_id": template_id,
            "mapping": mapping,
            "error_localizer": error_localizer,
        }
        if model is not None:
            payload["model"] = model
        if config is not None:
            payload["config"] = config
        if input_data_types is not None:
            payload["input_data_types"] = input_data_types
        if kb_id is not None:
            payload["kb_id"] = kb_id
        for key, value in (
            ("row_context", row_context),
            ("span_context", span_context),
            ("trace_context", trace_context),
            ("session_context", session_context),
            ("call_context", call_context),
            ("span_id", span_id),
            ("trace_id", trace_id),
            ("session_id", session_id),
            ("call_id", call_id),
        ):
            if value is not None:
                payload[key] = value

        raw = self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_playground.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
        return PlaygroundRunResponse.model_validate(raw)
