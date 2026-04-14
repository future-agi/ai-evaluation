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
        mgr.delete_template(t["id"])
    """

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
    ) -> Dict[str, Any]:
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

        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_list.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    def get_template(self, template_id: str) -> Dict[str, Any]:
        """Fetch a single eval template by UUID."""
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_detail.value.format(template_id=template_id)
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

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
    ) -> Dict[str, Any]:
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

        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_create_v2.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    def update_template(
        self, template_id: str, **fields: Any
    ) -> Dict[str, Any]:
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
        return self.request(
            config=RequestConfig(
                method=HttpMethod.PUT,
                url=url,
                json={k: v for k, v in fields.items() if v is not None},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

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

    def bulk_delete_templates(self, template_ids: List[str]) -> Dict[str, Any]:
        """Soft-delete multiple eval templates in one call."""
        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.eval_template_bulk_delete.value}",
                json={"template_ids": list(template_ids)},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def list_versions(self, template_id: str) -> Dict[str, Any]:
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_list.value.format(template_id=template_id)
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

    def create_version(
        self,
        template_id: str,
        *,
        criteria: Optional[str] = None,
        model: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    def set_default_version(
        self, template_id: str, version_id: str
    ) -> Dict[str, Any]:
        url = (
            f"{self._base_url}/"
            + Routes.eval_template_version_set_default.value.format(
                template_id=template_id, version_id=version_id
            )
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.PUT, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

    def restore_version(
        self, template_id: str, version_id: str
    ) -> Dict[str, Any]:
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
        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

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
    ) -> Dict[str, Any]:
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

        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self._base_url}/{Routes.composite_eval_create.value}",
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

    def get_composite(self, template_id: str) -> Dict[str, Any]:
        url = (
            f"{self._base_url}/"
            + Routes.composite_eval_detail.value.format(template_id=template_id)
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.GET, url=url, timeout=self._default_timeout
            ),
            response_handler=_JsonResponseHandler,
        )

    def update_composite(
        self,
        template_id: str,
        **fields: Any,
    ) -> Dict[str, Any]:
        """PATCH the composite. Same field names as ``create_composite``."""
        url = (
            f"{self._base_url}/"
            + Routes.composite_eval_detail.value.format(template_id=template_id)
        )
        return self.request(
            config=RequestConfig(
                method=HttpMethod.PATCH,
                url=url,
                json={k: v for k, v in fields.items() if v is not None},
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )

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
    ) -> Dict[str, Any]:
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
        return self.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=url,
                json=payload,
                timeout=self._default_timeout,
            ),
            response_handler=_JsonResponseHandler,
        )
