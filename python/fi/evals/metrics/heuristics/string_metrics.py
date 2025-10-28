import re
from typing import Any, Dict, List, Optional

from ..base_metric import BaseMetric
from ...types import TextMetricInput
import requests


class Regex(BaseMetric[TextMetricInput]):
    """Checks if a regex pattern is found in the response text."""

    @property
    def metric_name(self) -> str:
        return "regex"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern = self.config.get("pattern")
        if not self.pattern:
            raise ValueError("Regex metric requires a 'pattern' in its config.")

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        match = re.search(self.pattern, inputs.response)
        if match:
            return {
                "output": 1.0,  # Using 1.0 for success (True)
                "reason": f"Regex pattern '{self.pattern}' found in response.",
            }
        else:
            return {
                "output": 0.0,  # Using 0.0 for failure (False)
                "reason": f"Regex pattern '{self.pattern}' not found in response.",
            }


class Contains(BaseMetric[TextMetricInput]):
    @property
    def metric_name(self) -> str:
        return "contains"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.keyword = self.config.get("keyword")
        self.case_sensitive = self.config.get("case_sensitive", False)
        if not self.keyword:
            raise ValueError("Contains metric requires a 'keyword' config.")

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        text, kw = (
            (inputs.response, self.keyword)
            if self.case_sensitive
            else (inputs.response.lower(), self.keyword.lower())
        )
        is_present = kw in text
        return {
            "output": 1.0 if is_present else 0.0,
            "reason": f"Keyword '{self.keyword}' found"
            if is_present
            else f"Keyword '{self.keyword}' not found",
        }


class _BaseContainsKeywords(BaseMetric[TextMetricInput]):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.keywords = self.config.get("keywords")
        self.case_sensitive = self.config.get("case_sensitive", False)
        if not self.keywords or not isinstance(self.keywords, list):
            raise ValueError(
                f"{self.metric_name} metric requires a 'keywords' list in config."
            )

    def _get_found_keywords(self, text: str) -> List[str]:
        text_to_check = text if self.case_sensitive else text.lower()
        found = [
            kw
            for kw in self.keywords
            if (kw if self.case_sensitive else kw.lower()) in text_to_check
        ]
        return found


class ContainsAll(_BaseContainsKeywords):
    @property
    def metric_name(self) -> str:
        return "contains_all"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        found = self._get_found_keywords(inputs.response)
        if len(found) == len(self.keywords):
            return {
                "output": 1.0,
                "reason": f"All {len(self.keywords)} keywords found.",
            }
        missing = [kw for kw in self.keywords if kw not in found]
        return {"output": 0.0, "reason": f"Missing keywords: {', '.join(missing)}"}


class ContainsAny(_BaseContainsKeywords):
    """Checks if the response text contains any of the provided keywords."""

    @property
    def metric_name(self) -> str:
        return "contains_any"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        found_keywords = self._get_found_keywords(inputs.response)
        if found_keywords:
            return {
                "output": 1.0,
                "reason": f"Found keywords: {', '.join(found_keywords)}",
            }
        return {"output": 0.0, "reason": "No keywords found in response."}


class ContainsNone(_BaseContainsKeywords):
    """Checks if the response text contains none of the provided keywords."""

    @property
    def metric_name(self) -> str:
        return "contains_none"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        found_keywords = self._get_found_keywords(inputs.response)
        if not found_keywords:
            return {"output": 1.0, "reason": "No forbidden keywords found."}
        return {
            "output": 0.0,
            "reason": f"Found forbidden keywords: {', '.join(found_keywords)}",
        }


def _standardize_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"http://{url}"


class OneLine(BaseMetric[TextMetricInput]):
    """Checks if the text is a single line."""

    @property
    def metric_name(self) -> str:
        return "one_line"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        is_one_line = "\n" not in inputs.response.strip()
        return {
            "output": 1.0 if is_one_line else 0.0,
            "reason": "Response is a single line."
            if is_one_line
            else "Response contains multiple lines.",
        }


class ContainsEmail(Regex):
    """Checks if the text contains an email address."""

    @property
    def metric_name(self) -> str:
        return "contains_email"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Pass the specific regex pattern to the parent Regex class
        super().__init__({"pattern": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"})


class IsEmail(Regex):
    """Checks if the entire text is a valid email address."""

    @property
    def metric_name(self) -> str:
        return "is_email"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            {"pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"}
        )


class ContainsLink(Regex):
    """Checks if the text contains a link."""

    @property
    def metric_name(self) -> str:
        return "contains_link"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__({"pattern": r"(?!.*@)(?:https?://)?(?:www\.)?\S+\.\S+"})


class ContainsValidLink(BaseMetric[TextMetricInput]):
    """Checks if the text contains a link that returns a 2xx status code."""

    @property
    def metric_name(self) -> str:
        return "contains_valid_link"

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        pattern = r"(?!.*@)(?:https?://)?(?:www\.)?\S+\.\S+"
        match = re.search(pattern, inputs.response)
        if not match:
            return {"output": 0.0, "reason": "No link found in response."}

        url = _standardize_url(match.group(0))
        try:
            response = requests.head(url, timeout=5)
            if 200 <= response.status_code < 300:
                return {
                    "output": 1.0,
                    "reason": f"Valid link '{url}' found (Status: {response.status_code})",
                }
            else:
                return {
                    "output": 0.0,
                    "reason": f"Invalid link '{url}' found (Status: {response.status_code})",
                }
        except requests.RequestException as e:
            return {
                "output": 0.0,
                "reason": f"Unreachable link '{url}' found. Error: {e}",
            }


class Equals(BaseMetric[TextMetricInput]):
    """Checks if the response text exactly matches the expected text."""

    @property
    def metric_name(self) -> str:
        return "equals"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("Equals metric requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError("Equals metric requires 'expected_response' to be a string.")
        resp, expected = (
            (inputs.response, inputs.expected_response)
            if self.case_sensitive
            else (inputs.response.lower(), inputs.expected_response.lower())
        )
        return {
            "output": 1.0 if resp == expected else 0.0,
            "reason": "Response matches expected text."
            if resp == expected
            else "Response does not match.",
        }


class StartsWith(BaseMetric[TextMetricInput]):
    """Checks if the response text starts with the expected text."""

    @property
    def metric_name(self) -> str:
        return "starts_with"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("StartsWith metric requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError("StartsWith requires 'expected_response' to be a string.")
        resp, prefix = (
            (inputs.response, inputs.expected_response)
            if self.case_sensitive
            else (inputs.response.lower(), inputs.expected_response.lower())
        )
        starts = resp.startswith(prefix)
        return {
            "output": 1.0 if starts else 0.0,
            "reason": f"Response starts with '{inputs.expected_response}'."
            if starts
            else f"Response does not start with '{inputs.expected_response}'.",
        }


class EndsWith(BaseMetric[TextMetricInput]):
    """Checks if the response text ends with the expected text."""

    @property
    def metric_name(self) -> str:
        return "ends_with"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("EndsWith metric requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError("EndsWith requires 'expected_response' to be a string.")
        resp, suffix = (
            (inputs.response, inputs.expected_response)
            if self.case_sensitive
            else (inputs.response.lower(), inputs.expected_response.lower())
        )
        ends = resp.endswith(suffix)
        return {
            "output": 1.0 if ends else 0.0,
            "reason": f"Response ends with '{inputs.expected_response}'."
            if ends
            else f"Response does not end with '{inputs.expected_response}'.",
        }


# --- Length Metrics ---


class LengthLessThan(BaseMetric[TextMetricInput]):
    """Checks if text length is less than a max_length."""

    @property
    def metric_name(self) -> str:
        return "length_less_than"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_length = self.config.get("max_length")
        if not isinstance(self.max_length, int):
            raise ValueError(
                "LengthLessThan metric requires an integer 'max_length' config."
            )

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        is_less = len(inputs.response) < self.max_length
        return {
            "output": 1.0 if is_less else 0.0,
            "reason": f"Length {len(inputs.response)} < {self.max_length}"
            if is_less
            else f"Length {len(inputs.response)} >= {self.max_length}",
        }


class LengthGreaterThan(BaseMetric[TextMetricInput]):
    """Checks if text length is greater than a min_length."""

    @property
    def metric_name(self) -> str:
        return "length_greater_than"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_length = self.config.get("min_length")
        if not isinstance(self.min_length, int):
            raise ValueError(
                "LengthGreaterThan metric requires an integer 'min_length' config."
            )

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        is_greater = len(inputs.response) > self.min_length
        return {
            "output": 1.0 if is_greater else 0.0,
            "reason": f"Length {len(inputs.response)} > {self.min_length}"
            if is_greater
            else f"Length {len(inputs.response)} <= {self.min_length}",
        }


class LengthBetween(BaseMetric[TextMetricInput]):
    """Checks if text length is between a min_length and max_length."""

    @property
    def metric_name(self) -> str:
        return "length_between"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_length = self.config.get("min_length")
        self.max_length = self.config.get("max_length")
        if not isinstance(self.min_length, int) or not isinstance(self.max_length, int):
            raise ValueError(
                "LengthBetween requires integer 'min_length' and 'max_length' configs."
            )

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        length = len(inputs.response)
        is_between = self.min_length <= length <= self.max_length
        reason = (
            f"Length {length} is between [{self.min_length}, {self.max_length}]"
            if is_between
            else f"Length {length} is not between [{self.min_length}, {self.max_length}]"
        )
        return {"output": 1.0 if is_between else 0.0, "reason": reason}
