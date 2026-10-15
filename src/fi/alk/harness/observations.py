"""Frozen HTTP read operations. Source review establishes their server-side semantics."""

import json
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ObservationParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["string", "integer"]
    max_length: int | None = Field(default=None, ge=1, le=4096)
    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.type == "string":
            if (
                self.max_length is None
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ValueError("observation_string_bounds_required")
        elif (
            self.max_length is not None
            or self.minimum is None
            or self.maximum is None
            or self.minimum > self.maximum
        ):
            raise ValueError("observation_integer_bounds_required")
        return self

    def accepts(self, value):
        if self.type == "string":
            return type(value) is str and len(value) <= self.max_length
        return type(value) is int and self.minimum <= value <= self.maximum


class HttpObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    capability: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    path: str = Field(max_length=2048)
    body: dict[str, Any]
    parameters: dict[str, ObservationParameter] = Field(
        default_factory=dict, max_length=8
    )

    @model_validator(mode="after")
    def template(self):
        parsed = urlsplit(self.path)
        if (
            not self.path.startswith("/")
            or self.path.startswith("//")
            or parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or "\\" in self.path
            or "{{" in self.path
            or any(ord(c) < 32 for c in self.path)
        ):
            raise ValueError("observation_fixed_relative_path_required")
        if any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
            for name in self.parameters
        ):
            raise ValueError("observation_parameter_name_invalid")
        used = set()

        def visit(value, depth=0):
            if depth > 16:
                raise ValueError("observation_template_too_deep")
            if isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str) or "{{" in key:
                        raise ValueError("observation_literal_keys_required")
                    visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    visit(child, depth + 1)
            elif isinstance(value, str) and ("{{" in value or "}}" in value):
                match = re.fullmatch(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}", value)
                if not match:
                    raise ValueError("observation_whole_leaf_parameter_required")
                used.add(match[1])

        visit(self.body)
        if used != set(self.parameters):
            raise ValueError("observation_parameters_mismatch")
        if len(json.dumps(self.body, allow_nan=False).encode()) > 65536:
            raise ValueError("observation_template_too_large")
        return self

    def render(self, parameters):
        if not isinstance(parameters, dict) or set(parameters) != set(self.parameters):
            raise ValueError("observation_parameters_mismatch")
        if any(
            not spec.accepts(parameters[name]) for name, spec in self.parameters.items()
        ):
            raise ValueError("observation_parameter_invalid")

        def visit(value):
            if isinstance(value, dict):
                return {key: visit(child) for key, child in value.items()}
            if isinstance(value, list):
                return [visit(child) for child in value]
            if isinstance(value, str) and value.startswith("{{"):
                return parameters[value[2:-2]]
            return value

        return visit(self.body)
