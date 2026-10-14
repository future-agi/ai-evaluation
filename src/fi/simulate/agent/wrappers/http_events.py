"""Declarative observation of completed tools through a source-owned event API."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompletedToolEvents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_header: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=1024)
    items_path: str = Field(min_length=1, max_length=256)
    next_page_path: str = Field(min_length=1, max_length=256)
    page_parameter: str = Field(min_length=1, max_length=128)
    max_pages: int = Field(default=10, ge=1, le=20)
    kind_path: str = Field(min_length=1, max_length=256)
    action_kind: str = Field(min_length=1, max_length=128)
    observation_kind: str = Field(min_length=1, max_length=128)
    action_id_path: str = Field(min_length=1, max_length=256)
    observation_action_id_path: str = Field(min_length=1, max_length=256)
    name_path: str = Field(min_length=1, max_length=256)
    arguments_path: str = Field(min_length=1, max_length=256)
    result_path: str = Field(min_length=1, max_length=256)
    error_path: str = Field(min_length=1, max_length=256)

    @field_validator("correlation_header")
    @classmethod
    def _header(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9-]+", value) or any(
            part in value.lower()
            for part in ("authorization", "cookie", "token", "secret", "api-key")
        ):
            raise ValueError(
                "completed events require a non-credential correlation header"
            )
        return value

    @field_validator("page_parameter")
    @classmethod
    def _parameter(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+", value):
            raise ValueError("invalid completed events page parameter")
        return value

    @model_validator(mode="after")
    def _endpoint(self) -> "CompletedToolEvents":
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
                "completed events require a relative correlated endpoint and distinct event kinds"
            )
        return self
