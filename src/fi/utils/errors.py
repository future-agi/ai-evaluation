"""Error hierarchy for the SDK.

Trimmed to the classes actually referenced by ai-evaluation. If a new
exception is needed, add it here instead of reaching back into a
separate package.
"""
from collections.abc import Iterable
from typing import List, Optional, Union

from .constants import API_KEY_ENVVAR_NAME, SECRET_KEY_ENVVAR_NAME


class SDKException(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        self.custom_message = message
        self.__cause__ = cause
        super().__init__(message or self.get_message())

    def __str__(self) -> str:
        return self.custom_message or self.get_message()

    def get_message(self) -> str:
        if self.__cause__:
            return f"An SDK error occurred, caused by: {self.__cause__}"
        return "An unknown error occurred in the SDK."

    def get_error_code(self) -> str:
        return "UNKNOWN_SDK_ERROR"


class MissingAuthError(SDKException):
    def __init__(
        self,
        fi_api_key: Optional[str],
        fi_secret_key: Optional[str],
        cause: Optional[Exception] = None,
    ) -> None:
        self.missing_api_key = fi_api_key is None
        self.missing_secret_key = fi_secret_key is None
        super().__init__(cause=cause)

    def get_message(self) -> str:
        missing = []
        if self.missing_api_key:
            missing.append("'fi_api_key'")
        if self.missing_secret_key:
            missing.append("'fi_secret_key'")
        return (
            "FI Client could not obtain credentials. Pass fi_api_key and "
            "fi_secret_key directly or set env vars:\n"
            f" - {API_KEY_ENVVAR_NAME}\n"
            f" - {SECRET_KEY_ENVVAR_NAME}\n"
            f"Missing: {', '.join(missing)}"
        )

    def get_error_code(self) -> str:
        return "MISSING_FI_CLIENT_AUTHENTICATION"


class InvalidAuthError(SDKException):
    """Raised when the api rejects credentials (bad/expired key)."""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=message
            or "Invalid FI Client Authentication, please check your API key and secret key.",
            cause=cause,
        )

    def get_error_code(self) -> str:
        return "INVALID_FI_CLIENT_AUTHENTICATION"


class InvalidValueType(SDKException):
    def __init__(
        self,
        value_name: str,
        value: Union[bool, int, float, str],
        correct_type: str,
        cause: Optional[Exception] = None,
    ) -> None:
        self.value_name = value_name
        self.value = value
        self.correct_type = correct_type
        super().__init__(cause=cause)

    def get_message(self) -> str:
        return (
            f"{self.value_name} with value {self.value!r} is of type "
            f"{type(self.value).__name__}, but expected from {self.correct_type}."
        )

    def get_error_code(self) -> str:
        return "INVALID_VALUE_TYPE"


class MissingRequiredKey(SDKException):
    def __init__(self, field_name: str, missing_key: str, cause: Optional[Exception] = None) -> None:
        self.field_name = field_name
        self.missing_key = missing_key
        super().__init__(
            message=(
                f"Missing required key '{missing_key}' in {field_name}. "
                "Please check your configuration or API documentation."
            ),
            cause=cause,
        )

    def get_error_code(self) -> str:
        return "MISSING_REQUIRED_KEY"


class MissingRequiredConfigForEvalTemplate(SDKException):
    def __init__(
        self,
        missing_key: str,
        eval_template_name: str,
        cause: Optional[Exception] = None,
    ) -> None:
        self.missing_key = missing_key
        self.eval_template_name = eval_template_name
        super().__init__(
            message=(
                f"Missing required config '{missing_key}' for eval template "
                f"'{eval_template_name}'."
            ),
            cause=cause,
        )

    def get_error_code(self) -> str:
        return "MISSING_EVAL_TEMPLATE_CONFIG"


class InvalidAdditionalHeaders(SDKException):
    def __init__(self, invalid_headers: Iterable, cause: Optional[Exception] = None) -> None:
        self.invalid_header_names = invalid_headers
        super().__init__(cause=cause)

    def get_message(self) -> str:
        return (
            "Found invalid additional header, cannot use reserved headers named: "
            f"{', '.join(map(str, self.invalid_header_names))}."
        )

    def get_error_code(self) -> str:
        return "INVALID_ADDITIONAL_HEADERS"


class DatasetNotFoundError(SDKException):
    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=message or "No existing dataset found for current dataset name.",
            cause=cause,
        )

    def get_error_code(self) -> str:
        return "DATASET_NOT_FOUND"


class FileNotFoundException(SDKException):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.file_path = file_path
        super().__init__(message=message or self._default_message(), cause=cause)

    def _default_message(self) -> str:
        if isinstance(self.file_path, list):
            head = ", ".join(map(str, self.file_path[:3]))
            if len(self.file_path) > 3:
                head += f", and {len(self.file_path) - 3} more"
            return f"Files not found: {head}."
        return f"File not found: {self.file_path}."

    def get_error_code(self) -> str:
        return "FILE_NOT_FOUND"


class UnsupportedFileType(SDKException):
    def __init__(
        self,
        file_ext: str,
        file_name: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.file_ext = file_ext
        self.file_name = file_name
        super().__init__(
            message=message or f"Unsupported file type: '.{file_ext}' for file '{file_name}'.",
            cause=cause,
        )

    def get_error_code(self) -> str:
        return "UNSUPPORTED_FILE_TYPE"
