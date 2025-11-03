import json
from typing import Any, Dict, List, Optional, Union
import re

# This would ideally be in a separate helper file
from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from ..base_metric import BaseMetric
from ...types import TextMetricInput, JsonMetricInput


class ContainsJson(BaseMetric[TextMetricInput]):
    """Checks if the response text contains a valid JSON object or array."""

    @property
    def metric_name(self) -> str:
        return "contains_json"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        text = inputs.response.strip()
        # Simple regex to find potential JSON candidates
        json_candidates = re.findall(r"\{.*\}|\[.*\]", text, re.DOTALL)
        for candidate in json_candidates:
            try:
                json.loads(candidate)
                return {
                    "output": 1.0,
                    "reason": "A valid JSON entity was found in the response.",
                }
            except json.JSONDecodeError:
                continue
        return {"output": 0.0, "reason": "No valid JSON entity found in the response."}


class IsJson(BaseMetric[TextMetricInput]):
    """Checks if the entire response text is a single, valid JSON object."""

    @property
    def metric_name(self) -> str:
        return "is_json"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        try:
            json.loads(inputs.response)
            return {"output": 1.0, "reason": "Response is a valid JSON object."}
        except json.JSONDecodeError as e:
            return {
                "output": 0.0,
                "reason": f"Response is not a valid JSON object: {e}",
            }


class JsonSchema(BaseMetric[JsonMetricInput]):
    """Validates the `response` against a provided JSON schema."""

    @property
    def metric_name(self) -> str:
        return "json_schema"

    def compute_one(self, inputs: JsonMetricInput) -> Dict[str, Any]:
        if not inputs.schema:
            raise ValueError("JsonSchema metric requires 'schema' to be provided.")
        try:
            actual_data = (
                json.loads(inputs.response)
                if isinstance(inputs.response, str)
                else inputs.response
            )
        except json.JSONDecodeError as e:
            return {"output": 0.0, "reason": f"Actual JSON is invalid: {e}"}
        try:
            schema_data = (
                json.loads(inputs.schema)
                if isinstance(inputs.schema, str)
                else inputs.schema
            )
        except json.JSONDecodeError as e:
            return {"output": 0.0, "reason": f"Schema JSON is invalid: {e}"}
        try:
            validate(instance=actual_data, schema=schema_data)
            return {"output": 1.0, "reason": "JSON conforms to the schema."}
        except JsonSchemaValidationError as e:
            return {
                "output": 0.0,
                "reason": f"JSON schema validation failed: {e.message}",
            }
