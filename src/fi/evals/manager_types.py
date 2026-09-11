"""
Typed response models for :class:`~fi.evals.manager.EvalTemplateManager`.

Every model inherits from :class:`_APIResponse` which supports both
attribute access (``result.id``) and dict access (``result["id"]``)
for backwards compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Dict-compatible base (Stripe / OpenAI SDK pattern)
# ---------------------------------------------------------------------------

class _APIResponse(BaseModel):
    """Base for all manager response models.

    Supports both attribute and dict-style access so existing code
    using ``result["id"]`` keeps working while new code gets IDE
    autocomplete via ``result.id``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            if self.model_extra and key in self.model_extra:
                return self.model_extra[key]
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (
            bool(self.model_extra) and key in self.model_extra
        )


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class TemplateListItem(_APIResponse):
    id: str
    name: str
    template_type: Optional[str] = None
    eval_type: Optional[str] = None
    output_type: Optional[str] = None
    owner: Optional[str] = None
    created_by_name: Optional[str] = None
    version_count: Optional[int] = None
    current_version: Optional[str] = None
    last_updated: Optional[str] = None
    thirty_day_chart: Optional[List[Any]] = None
    thirty_day_error_rate: Optional[List[Any]] = None
    thirty_day_run_count: Optional[int] = None
    tags: Optional[List[str]] = None


class VersionItem(_APIResponse):
    id: str
    version_number: int
    is_default: bool = False
    criteria: Optional[str] = None
    model: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    created_by_name: Optional[str] = None
    created_at: Optional[str] = None


class CompositeChildSummary(_APIResponse):
    child_id: str
    child_name: str
    order: int
    eval_type: Optional[str] = None
    pinned_version_id: Optional[str] = None
    pinned_version_number: Optional[int] = None
    weight: float = 1.0
    required_keys: Optional[List[str]] = None


class CompositeChildResult(_APIResponse):
    child_id: str
    child_name: str
    order: int
    score: Optional[Any] = None
    output: Optional[Any] = None
    reason: Optional[str] = None
    output_type: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None
    log_id: Optional[str] = None
    weight: float = 1.0
    error_localizer_result: Optional[Dict[str, Any]] = None


class GroundTruthItem(_APIResponse):
    id: str
    name: str
    description: Optional[str] = None
    file_name: Optional[str] = None
    columns: List[str] = []
    row_count: int = 0
    variable_mapping: Optional[Dict[str, str]] = None
    role_mapping: Optional[Dict[str, str]] = None
    embedding_status: str = "pending"
    embedded_row_count: Optional[int] = None
    storage_type: Optional[str] = None
    created_at: Optional[str] = None


class GroundTruthConfig(_APIResponse):
    enabled: bool = False
    ground_truth_id: Optional[str] = None
    mode: Optional[str] = None
    max_examples: Optional[int] = None
    similarity_threshold: Optional[float] = None
    injection_format: Optional[str] = None


# ---------------------------------------------------------------------------
# Template CRUD responses
# ---------------------------------------------------------------------------

class TemplateCreateResponse(_APIResponse):
    id: str
    name: str
    version: str


class TemplateDetailResponse(_APIResponse):
    id: str
    name: str
    description: Optional[str] = None
    template_type: Optional[str] = None
    eval_type: Optional[str] = None
    instructions: Optional[str] = None
    model: Optional[str] = None
    output_type: Optional[str] = None
    pass_threshold: Optional[float] = None
    choice_scores: Optional[Dict[str, float]] = None
    choices: Optional[Any] = None
    multi_choice: bool = False
    code: Optional[str] = None
    code_language: Optional[str] = None
    required_keys: Optional[List[str]] = None
    owner: Optional[str] = None
    created_by_name: Optional[str] = None
    version_count: Optional[int] = None
    current_version: Optional[str] = None
    tags: Optional[List[str]] = None
    check_internet: bool = False
    error_localizer_enabled: bool = False
    aggregation_enabled: Optional[bool] = None
    aggregation_function: Optional[str] = None
    composite_child_axis: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TemplateUpdateResponse(_APIResponse):
    id: str
    name: Optional[str] = None
    updated: bool = True


class TemplateDuplicateResponse(_APIResponse):
    message: str
    eval_template_id: str


class TemplateListResponse(_APIResponse):
    items: List[TemplateListItem] = []
    total: int = 0
    page: int = 0
    page_size: int = 25


class BulkDeleteResponse(_APIResponse):
    deleted_count: int = 0


# ---------------------------------------------------------------------------
# Versioning responses
# ---------------------------------------------------------------------------

class VersionListResponse(_APIResponse):
    template_id: str
    versions: List[VersionItem] = []
    total: int = 0


class VersionCreateResponse(_APIResponse):
    id: str
    version_number: int
    is_default: bool = False


class VersionSetDefaultResponse(_APIResponse):
    id: str
    version_number: int
    is_default: bool = True


class VersionRestoreResponse(_APIResponse):
    id: str
    version_number: int
    is_default: bool = False
    restored_from: Optional[int] = None


# ---------------------------------------------------------------------------
# Composite responses
# ---------------------------------------------------------------------------

class CompositeCreateResponse(_APIResponse):
    id: str
    name: str
    template_type: str = "composite"
    aggregation_enabled: bool = True
    aggregation_function: Optional[str] = None
    composite_child_axis: Optional[str] = None
    children: List[CompositeChildSummary] = []


class CompositeDetailResponse(_APIResponse):
    id: str
    name: str
    description: Optional[str] = None
    template_type: str = "composite"
    aggregation_enabled: bool = True
    aggregation_function: Optional[str] = None
    composite_child_axis: Optional[str] = None
    children: List[CompositeChildSummary] = []
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CompositeExecutionResponse(_APIResponse):
    composite_id: str
    composite_name: Optional[str] = None
    aggregation_enabled: bool = True
    aggregation_function: Optional[str] = None
    aggregate_score: Optional[float] = None
    aggregate_pass: Optional[bool] = None
    children: List[CompositeChildResult] = []
    summary: Optional[Any] = None
    error_localizer_results: Optional[Dict[str, Any]] = None
    total_children: int = 0
    completed_children: int = 0
    failed_children: int = 0


# ---------------------------------------------------------------------------
# Ground Truth responses
# ---------------------------------------------------------------------------

class GroundTruthListResponse(_APIResponse):
    template_id: str
    items: List[GroundTruthItem] = []
    total: int = 0


class GroundTruthUploadResponse(_APIResponse):
    id: str
    name: str
    row_count: int = 0
    columns: List[str] = []
    embedding_status: str = "pending"


class GroundTruthDataResponse(_APIResponse):
    id: str
    page: int = 1
    page_size: int = 50
    total_rows: int = 0
    total_pages: int = 1
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []


class GroundTruthStatusResponse(_APIResponse):
    id: str
    embedding_status: str = "pending"
    embedded_row_count: Optional[int] = None
    total_rows: int = 0
    progress_percent: Optional[float] = None


class GroundTruthVariableMappingResponse(_APIResponse):
    id: str
    variable_mapping: Dict[str, str] = {}


class GroundTruthRoleMappingResponse(_APIResponse):
    id: str
    role_mapping: Dict[str, str] = {}
    embedding_status: Optional[str] = None


class GroundTruthConfigResponse(_APIResponse):
    ground_truth: GroundTruthConfig


class GroundTruthSearchResponse(_APIResponse):
    query: str
    results: List[Dict[str, Any]] = []
    total: int = 0


class GroundTruthDeleteResponse(_APIResponse):
    deleted: bool = True
    id: str


# ---------------------------------------------------------------------------
# Usage / Feedback / Charts responses
# ---------------------------------------------------------------------------

class TemplateUsageResponse(_APIResponse):
    template_id: str
    stats: Dict[str, Any] = {}
    chart: List[Any] = []
    logs: Optional[Any] = None  # paginated dict {items, page, page_size}


class TemplateFeedbackListResponse(_APIResponse):
    template_id: str
    items: List[Dict[str, Any]] = []
    total: int = 0
    page: int = 0
    page_size: int = 25


class TemplateChartsResponse(_APIResponse):
    charts: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Playground response
# ---------------------------------------------------------------------------

class PlaygroundRunResponse(_APIResponse):
    output: Optional[Any] = None
    reason: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    output_type: Optional[str] = None
    log_id: Optional[str] = None
