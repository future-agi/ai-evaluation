from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Union, List

from .constants import (
    API_KEY_ENVVAR_NAME,
    MAX_NUMBER_OF_EMBEDDINGS,
    SECRET_KEY_ENVVAR_NAME,
)
from .types import Environments, ModelTypes


class SDKException(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        self.custom_message = message  # Store custom message if provided
        self.__cause__ = cause
        super().__init__(message or self.get_message()) # Use get_message if no specific message is passed

    def __str__(self) -> str:
        """Default string representation for all exceptions."""
        # If a specific message was passed to constructor, use it. Otherwise, use get_message().
        return self.custom_message or self.get_message()

    def get_message(self) -> str:
        """Return a human-readable error message. Subclasses should override this."""
        if self.__cause__:
            return f"An SDK error occurred, caused by: {self.__cause__}"
        return "An unknown error occurred in the SDK."

    def get_error_code(self) -> str:
        """Return a machine-readable error code. Subclasses should override this."""
        return "UNKNOWN_SDK_ERROR"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.get_message()}', code='{self.get_error_code()}')"


# Custom Dataset Exceptions
class DatasetError(SDKException):
    """Base exception for all dataset operations."""

    def get_message(self) -> str:
        return "Invalid Dataset Operation."

    def get_error_code(self) -> str:
        return "DATASET_OPERATION_INVALID"

    def __repr__(self) -> str: 
        return "Invalid Dataset Operation"


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found."""

    def get_message(self) -> str:
        return "No Existing Dataset Found for Current Dataset Name."

    def get_error_code(self) -> str:
        return "DATASET_NOT_FOUND"

    def __repr__(self) -> str:
        return "No_Dataset_Found"

class DatasetAuthError(DatasetError):  # For 401
    """Raised when authentication fails for dataset operations (e.g., invalid API key)."""

    def get_message(self) -> str:
        return "Invalid Dataset Authentication, please check your API key and Secret key."

    def get_error_code(self) -> str:
        return "DATASET_AUTH_ERROR"
    
    def __repr__(self) -> str:
        return "Invalid_Dataset_Authentication"


class DatasetValidationError(DatasetError):  # For 400
    """Raised when input data fails validation."""

    def get_message(self) -> str:
        return "Invalid Dataset Validation, please check your input data."

    def get_error_code(self) -> str:
        return "DATASET_VALIDATION_ERROR"

    def __repr__(self) -> str:
        return "Invalid_Dataset_Validation"


class RateLimitError(DatasetError):  # For 429
    """Raised when API rate limits are exceeded."""

    def get_message(self) -> str:
        return "Rate Limit Exceeded, please contact FutureAGI support at support@futureagi.com or check your current plan."

    def get_error_code(self) -> str:
        return "RATE_LIMIT_EXCEEDED"

    def __repr__(self) -> str:
        return "Rate_Limit_Exceeded"


class ServerError(DatasetError):  # For 500 and other 5xx
    """Raised for server-side errors."""

    def get_message(self) -> str:
        return "Internal Server Error, please contact FutureAGI support at support@futureagi.com."

    def get_error_code(self) -> str:
        return "SERVER_ERROR"

    def __repr__(self) -> str:
        return "Server_Error"
  
class UnexpectedDataFormatError(DatasetError):
    """Raised when the data format is unexpected."""
    
    def get_message(self) -> str:
        return "Messages Data Format is unexpected, please send in correct format for example:\n\n" \
            "messages = [\n" \
            "    {\n" \
            "        'role': 'user',\n" \
            "        'content': [\n" \
            "            {\n" \
            "                'type': 'text',\n" \
            "                'text': 'What is the capital of France?'\n" \
            "            }\n" \
            "        ]\n" \
            "    },\n" \
            "    {\n" \
            "        'role': 'assistant',\n" \
            "        'content': [\n" \
            "            {\n" \
            "                'type': 'text',\n" \
            "                'text': 'The capital of France is Paris.'\n" \
            "            }\n" \
            "        ]\n" \
            "    }\n" \
            "]\n\n" \
            "Note: You can also use simple string format for content:\n" \
            "messages = [{'role': 'user', 'content': 'What is the capital of France?'}]\n" \
            "The SDK will automatically convert it to the expected format."
    
    def get_error_code(self) -> str:
        return "UNEXPECTED_DATA_FORMAT"
    
    def __repr__(self) -> str:
        return "Unexpected_Data_Format"


class ServiceUnavailableError(DatasetError):  # For 503
    """Raised when the service is temporarily unavailable."""

    def get_message(self) -> str:
        return "Service Unavailable, please try again later."

    def get_error_code(self) -> str:
        return "SERVICE_UNAVAILABLE"

    def __repr__(self) -> str:
        return "Service_Unavailable"


class MissingAuthError(SDKException):
    def __init__(self, fi_api_key: Optional[str], fi_secret_key: Optional[str], cause: Optional[Exception] = None) -> None:
        self.missing_api_key = fi_api_key is None
        self.missing_secret_key = fi_secret_key is None
        super().__init__(cause=cause)

    def get_message(self) -> str:
        missing_list = []
        if self.missing_api_key:
            missing_list.append("'fi_api_key'")
        if self.missing_secret_key:
            missing_list.append("'fi_secret_key'")

        return (
            "FI Client could not obtain credentials. You can pass your fi_api_key and fi_secret_key "
            "directly to the FI Client, or you can set environment variables which will be read if the "
            "keys are not directly passed. "
            "To set the environment variables use the following variable names: \n"
            f" - {API_KEY_ENVVAR_NAME} for the api key\n"
            f" - {SECRET_KEY_ENVVAR_NAME} for the secret key\n"
            f"Missing: {', '.join(missing_list)}"
        )

    def get_error_code(self) -> str:
        return "MISSING_FI_CLIENT_AUTHENTICATION"
    
    def __repr__(self) -> str: # Custom repr might be useful here
        return f"MissingAuthError(missing_api_key={self.missing_api_key}, missing_secret_key={self.missing_secret_key})"


class InvalidAuthError(SDKException):
    """
    Exception raised when API authentication fails due to invalid credentials.
    
    This may be due to:
    - Incorrect API key or secret
    - Expired credentials
    - Insufficient permissions
    """
    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(message=message or "Invalid FI Client Authentication, please check your API key and secret key.", cause=cause)

    def get_message(self) -> str:
        # The message is set in __init__, but we can provide a default fallback if needed.
        return self.custom_message or "Invalid FI Client Authentication, please check your API key and secret key."

    def get_error_code(self) -> str:
        return "INVALID_FI_CLIENT_AUTHENTICATION"


class InvalidAdditionalHeaders(SDKException):
    """Exception raised when additional headers are invalid."""

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


class InvalidNumberOfEmbeddings(SDKException):
    """Exception raised when the number of embeddings is invalid."""

    def __init__(self, number_of_embeddings: int, cause: Optional[Exception] = None) -> None:
        self.number_of_embeddings = number_of_embeddings
        super().__init__(cause=cause)

    def get_message(self) -> str:
        return (
            f"The schema contains {self.number_of_embeddings} different embeddings when a maximum of "
            f"{MAX_NUMBER_OF_EMBEDDINGS} is allowed."
        )

    def get_error_code(self) -> str:
        return "INVALID_NUMBER_OF_EMBEDDINGS"


class InvalidValueType(SDKException):
    """Exception raised when the value type is invalid."""

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
            f"{self.value_name} with value {self.value!r} is of type {type(self.value).__name__}, "
            f"but expected from {self.correct_type}."
        )

    def get_error_code(self) -> str:
        return "INVALID_VALUE_TYPE"


class InvalidSupportedType(SDKException):
    """Exception raised when the supported type is invalid."""

    def __init__(
        self,
        value_name: str,
        value: Union[ModelTypes, Environments],
        correct_type: str, 
        cause: Optional[Exception] = None,
    ) -> None:
        self.value_name = value_name
        self.value = value
        self.correct_type = correct_type # e.g. "ModelTypes" or "Environments" or specific values
        super().__init__(cause=cause)

    def get_message(self) -> str:
        return (
            f"{self.value_name} with value {self.value!r} is not supported as of now. " # Corrected "noy" to "not"
            f"Supported types/values are: {self.correct_type}."
        )

    def get_error_code(self) -> str:
        return "UNSUPPORTED_TYPE_OR_VALUE"


class MissingRequiredKey(SDKException):
    def __init__(self, field_name: str, missing_key: str, cause: Optional[Exception] = None) -> None:
        self.field_name = field_name
        self.missing_key = missing_key
        message = (
            f"Missing required key '{self.missing_key}' in {self.field_name}. "
            "Please check your configuration or API documentation."
        )
        super().__init__(message=message, cause=cause)


    def get_error_code(self) -> str:
        return "MISSING_REQUIRED_KEY"


class MissingRequiredConfigForEvalTemplate(SDKException):
    def __init__(self, missing_key: str, eval_template_name: str, cause: Optional[Exception] = None) -> None:
        self.missing_key = missing_key
        self.eval_template_name = eval_template_name
        message = f"Missing required config '{self.missing_key}' for eval template '{self.eval_template_name}'."
        super().__init__(message=message, cause=cause)


    def get_error_code(self) -> str:
        return "MISSING_EVAL_TEMPLATE_CONFIG"


class FileNotFoundException(SDKException):
    def __init__(self, file_path: Union[str, List[str]], message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        self.file_path = file_path
        # Allow custom message, otherwise generate one
        super().__init__(message=message or self._generate_message(), cause=cause)

    def _generate_message(self) -> str:
        if isinstance(self.file_path, list):
            # Truncate long lists for brevity in messages
            display_paths = self.file_path[:3]
            paths_str = ', '.join(map(str, display_paths))
            if len(self.file_path) > 3:
                paths_str += f", and {len(self.file_path) - 3} more"
            return f"Files not found: {paths_str}."
        return f"File not found: {self.file_path}."
    

    def get_error_code(self) -> str:
        return "FILE_NOT_FOUND"


class UnsupportedFileType(SDKException):
    def __init__(self, file_ext: str, file_name: str, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        self.file_ext = file_ext
        self.file_name = file_name
        super().__init__(message=message or f"Unsupported file type: '.{self.file_ext}' for file '{self.file_name}'.", cause=cause)


    def get_error_code(self) -> str:
        return "UNSUPPORTED_FILE_TYPE"

# Prompt template errors
class TemplateAlreadyExists(SDKException):
    """Raised when attempting to create a template that already exists"""
    def __init__(self, template_name: str, message: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        self.template_name = template_name
        super().__init__(message=message or f"Template '{self.template_name}' already exists. Please use a different name to create a new template.", cause=cause)


    def get_error_code(self) -> str:
        return "TEMPLATE_ALREADY_EXISTS"

class TemplateNotFound(SDKException):
    """Raised when a prompt template cannot be located by the given name or ID."""

    def __init__(self, template_name: str, cause: Optional[Exception] = None) -> None:
        self.template_name = template_name
        super().__init__(
            message=f"Prompt template '{template_name}' not found.", cause=cause
        )

    def get_error_code(self) -> str:
        return "TEMPLATE_NOT_FOUND"
