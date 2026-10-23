"""Observe completed tools from a source-owned event API or response body."""

from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


def _result_error_path(value: str) -> str:
    if value and (
        any(character.isspace() or character in "[]" for character in value)
        or (value.startswith("$") and not value.startswith("$."))
        or not all(value.removeprefix("$.").split("."))
    ):
        raise ValueError("invalid tool result error path")
    return value


ResultErrorPath = Annotated[str, AfterValidator(_result_error_path)]


class CompletedToolFailureEvent(BaseModel):
    """A source-declared failed terminal outcome for a tool action."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=128)
    action_id_path: str = Field(min_length=1, max_length=256)
    result_path: str = Field(min_length=1, max_length=256)


class CompletedToolEvents(BaseModel):
    """Paired completions; response IDs are stable and unique within a thread."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["endpoint", "response"] = "endpoint"
    correlation_header: str = Field(default="", max_length=128)
    path: str = Field(
        default="", max_length=1024,
        description="Same-server relative event endpoint containing exactly one {{correlation_id}} placeholder, replaced by the correlation_header response value. Empty for source=response.",
    )
    items_path: str = Field(min_length=1, max_length=256)
    next_page_path: str = Field(default="", max_length=256)
    page_parameter: str = Field(default="", max_length=128)
    max_pages: int = Field(default=10, ge=1, le=20)
    kind_path: str = Field(min_length=1, max_length=256)
    action_kind: str = Field(min_length=1, max_length=128)
    observation_kind: str = Field(min_length=1, max_length=128)
    action_id_path: str = Field(min_length=1, max_length=256)
    observation_action_id_path: str = Field(min_length=1, max_length=256)
    name_path: str = Field(min_length=1, max_length=256)
    arguments_path: str = Field(min_length=1, max_length=256)
    result_path: str = Field(min_length=1, max_length=256)
    failure_events: list[CompletedToolFailureEvent] = Field(
        default_factory=list, max_length=8,
        description="Source-declared failure terminal kinds, paired by the same action identifier as observations. Each emits ok=false and requires a non-null error result.",
    )
    error_path: str = Field(default="", max_length=256)
    result_error_path: ResultErrorPath = Field(
        default="", max_length=256,
        description="Source-declared nullable boolean/string error field relative to the tool result, decoded once if JSON text. Use dot-separated keys or array indices, optionally prefixed with $. Missing, null, false or empty string means no reported error; true or nonempty string means failure. Other values or invalid traversal reject the evidence. The original result is preserved.",
    )
    status_path: str = Field(default="", max_length=256)
    completed_status: str = Field(default="", max_length=128)

    @field_validator("correlation_header")
    @classmethod
    def _header(cls, value: str) -> str:
        if value and (
            not re.fullmatch(r"[A-Za-z0-9-]+", value)
            or any(
                part in value.lower()
                for part in ("authorization", "cookie", "token", "secret", "api-key")
            )
        ):
            raise ValueError(
                "completed events require a non-credential correlation header"
            )
        return value

    @field_validator("page_parameter")
    @classmethod
    def _parameter(cls, value: str) -> str:
        if value and not re.fullmatch(r"[A-Za-z0-9_]+", value):
            raise ValueError("invalid completed events page parameter")
        return value

    @model_validator(mode="after")
    def _endpoint(self) -> "CompletedToolEvents":
        kinds = [self.action_kind, self.observation_kind, *(event.kind for event in self.failure_events)]
        if len(kinds) != len(set(kinds)):
            raise ValueError("completed events require distinct event kinds")
        if self.source == "response":
            if (
                any(
                    (
                        self.correlation_header,
                        self.path,
                        self.next_page_path,
                        self.page_parameter,
                    )
                )
                or not self.status_path
                or not self.completed_status
            ):
                raise ValueError(
                    "response events require completion status and no endpoint fields"
                )
            if not (self.error_path or self.result_error_path or self.failure_events):
                raise ValueError("response events require a tool outcome mapping")
            return self
        if (
            not all(
                (
                    self.correlation_header,
                    self.path,
                    self.next_page_path,
                    self.page_parameter,
                    self.error_path,
                )
            )
            or self.status_path
            or self.completed_status
        ):
            raise ValueError(
                "endpoint events require correlation, pagination and error fields"
            )
        parts = urlsplit(self.path)
        if (
            not self.path.startswith("/")
            or self.path.startswith("//")
            or parts.scheme
            or parts.netloc
            or parts.fragment
            or "\\" in self.path
            or re.findall(r"{{(.*?)}}", self.path) != ["correlation_id"]
            or self.action_kind == self.observation_kind
        ):
            raise ValueError(
                "completed events path must be a same-server relative endpoint containing "
                "exactly one {{correlation_id}} placeholder, with no scheme, host, fragment or backslash"
            )
        return self


class CompletedToolRows(BaseModel):
    """Completed rows for this request only; cumulative histories require event IDs."""

    model_config = ConfigDict(extra="forbid")

    items_path: str = Field(min_length=1, max_length=256)
    name_path: str = Field(min_length=1, max_length=256)
    arguments_path: str = Field(min_length=1, max_length=256)
    result_path: str = Field(min_length=1, max_length=256)
    id_path: str = Field(default="", max_length=256)
    error_path: str = Field(default="", max_length=256)
    result_error_path: ResultErrorPath = Field(
        default="", max_length=256,
        description="Source-declared nullable boolean/string error field relative to the tool result, decoded once if JSON text. Use dot-separated keys or array indices, optionally prefixed with $. Missing, null, false or empty string means no reported error; true or nonempty string means failure. Other values or invalid traversal reject the evidence. The original result is preserved.",
    )
    max_items: int = Field(default=128, ge=1, le=2000)
