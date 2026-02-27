"""Configuration schema definitions for fi-evaluation.yaml."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIConfig(BaseModel):
    """API configuration settings."""
    base_url: str = Field(
        default="https://api.futureagi.com",
        description="Base URL for the Future AGI API"
    )


class DefaultsConfig(BaseModel):
    """Default settings for evaluations."""
    model: str = Field(
        default="gpt-4o",
        description="Default model for LLM-as-judge evaluations"
    )
    timeout: int = Field(
        default=200,
        description="Default timeout in seconds"
    )
    parallel_workers: int = Field(
        default=8,
        description="Number of parallel workers for evaluation"
    )


class EvaluationConfig(BaseModel):
    """Configuration for a single evaluation."""
    name: str = Field(..., description="Name of this evaluation")
    template: Optional[str] = Field(
        default=None,
        description="Single evaluation template to use"
    )
    templates: Optional[List[str]] = Field(
        default=None,
        description="List of evaluation templates to use"
    )
    data: str = Field(..., description="Path to test data file")
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional configuration for the evaluation"
    )

    @field_validator("templates", "template")
    @classmethod
    def validate_template_presence(cls, v, info):
        """Ensure at least one template specification method is used."""
        return v


class AssertionConfig(BaseModel):
    """Configuration for evaluation assertions."""
    template: Optional[str] = Field(
        default=None,
        description="Template to assert on (mutually exclusive with 'global')"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="List of assertion conditions (e.g., 'pass_rate >= 0.85')"
    )
    on_fail: str = Field(
        default="error",
        description="Action on failure: error, warn, or skip"
    )
    is_global: bool = Field(
        default=False,
        alias="global",
        description="If true, assertion applies globally across all templates"
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("on_fail")
    @classmethod
    def validate_on_fail(cls, v):
        """Validate on_fail value."""
        valid_values = ["warn", "error", "skip"]
        if v not in valid_values:
            raise ValueError(f"on_fail must be one of: {valid_values}")
        return v


class ThresholdOverrides(BaseModel):
    """Per-template threshold overrides."""
    model_config = ConfigDict(extra="allow")


class ThresholdsConfig(BaseModel):
    """Threshold shortcuts for assertions."""
    default_pass_rate: Optional[float] = Field(
        default=None,
        description="Default pass rate threshold for all templates"
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first assertion failure"
    )
    overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-template threshold overrides"
    )


class OutputConfig(BaseModel):
    """Output configuration settings."""
    format: str = Field(
        default="json",
        description="Output format: json, table, csv, html"
    )
    path: str = Field(
        default="./results/",
        description="Path to save results"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in output"
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate output format."""
        valid_formats = ["json", "table", "csv", "html"]
        if v not in valid_formats:
            raise ValueError(f"format must be one of: {valid_formats}")
        return v


class FIEvaluationConfig(BaseModel):
    """Root configuration schema for fi-evaluation.yaml."""
    version: str = Field(
        default="1.0",
        description="Configuration file version"
    )
    api: Optional[APIConfig] = Field(
        default=None,
        description="API configuration"
    )
    defaults: Optional[DefaultsConfig] = Field(
        default=None,
        description="Default evaluation settings"
    )
    evaluations: List[EvaluationConfig] = Field(
        ...,
        description="List of evaluation configurations"
    )
    output: Optional[OutputConfig] = Field(
        default=None,
        description="Output configuration"
    )
    assertions: Optional[List[AssertionConfig]] = Field(
        default=None,
        description="Assertions to run on evaluation results"
    )
    thresholds: Optional[ThresholdsConfig] = Field(
        default=None,
        description="Threshold shortcuts for assertions"
    )

    def get_defaults(self) -> DefaultsConfig:
        """Get defaults config, creating one if not present."""
        return self.defaults or DefaultsConfig()

    def get_output_config(self) -> OutputConfig:
        """Get output config, creating one if not present."""
        return self.output or OutputConfig()

    def get_api_config(self) -> APIConfig:
        """Get API config, creating one if not present."""
        return self.api or APIConfig()
