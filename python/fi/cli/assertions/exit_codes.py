"""Exit codes for CLI operations."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Exit codes for CI/CD integration.

    These codes allow CI systems to distinguish between different
    types of failures and take appropriate actions.
    """
    SUCCESS = 0                    # All evaluations and assertions passed
    EVALUATION_ERROR = 1           # Error during evaluation execution
    ASSERTION_FAILED = 2           # One or more assertions failed
    ASSERTION_WARNING = 3          # Assertions passed but with warnings (--strict mode)
    CONFIG_ERROR = 4               # Configuration file error
    DATA_ERROR = 5                 # Test data file error
    API_ERROR = 6                  # API connection/authentication error
    TIMEOUT_ERROR = 7              # Evaluation timeout
    UNKNOWN_ERROR = 99             # Unknown error
